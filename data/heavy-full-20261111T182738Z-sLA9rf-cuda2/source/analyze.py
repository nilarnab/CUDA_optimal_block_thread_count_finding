#!/usr/bin/env python3
"""Aggregate schema-3 single-launch observations without hiding missing actions."""
from collections import Counter,defaultdict
import csv
import json
import math
from pathlib import Path
import statistics as S
from catalog import reference

IDENTITY=('trial_id','context_id','kernel_id','m','n','k','seed','input_seed_index','round','block_x','block_y','grid_x','grid_y')
ACTION=('block_x','block_y','grid_x','grid_y')
def key(r):return tuple(str(r[k]) for k in IDENTITY)
def write_csv(path,rows,fields):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def analyze(root):
    root=Path(root);meta=json.loads((root/'metadata.json').read_text());plan=json.loads((root/'heavy_planned.json').read_text())
    issues=json.loads((root/'collection_errors.json').read_text()) if (root/'collection_errors.json').exists() else []
    rows=[]
    if (root/'heavy_trials.csv').exists():
        with (root/'heavy_trials.csv').open(newline='') as f:
            reader=csv.DictReader(f)
            if len(reader.fieldnames or [])!=len(set(reader.fieldnames or [])):raise ValueError('Duplicate columns')
            rows=list(reader)
    expected=Counter(key(j) for j in plan);observed=Counter(key(j) for j in rows)
    if observed!=expected:issues.append('Recorded identities differ from plan')
    if any(v!=1 for v in expected.values()) or any(v!=1 for v in observed.values()):issues.append('Duplicate trial identity')
    groups=defaultdict(list);jobs=defaultdict(list);by_context=defaultdict(list)
    for j in plan:jobs[(j['context_id'],*(j[a] for a in ACTION))].append(j)
    for r in rows:groups[(r['context_id'],*(int(r[a]) for a in ACTION))].append(r)
    measurements=[]
    for group,planned in jobs.items():
        rr=groups[group];values=[]
        for r in rr:
            if r['status']=='ok':
                try:
                    v=float(r['kernel_ms'])
                    if not math.isfinite(v) or v<=0:raise ValueError()
                    values.append(v)
                except (ValueError,TypeError):issues.append('Invalid successful timing '+str(r['trial_id']))
        full=len(values)==len(planned) and Counter(key(r) for r in rr)==Counter(key(j) for j in planned)
        median=S.median(values) if values else None;mad=S.median(abs(v-median) for v in values) if values else None
        first=planned[0]
        r={k:first[k] for k in ('session_id','context_id','kernel_id','split_group','m','n','k','input_seed_index','seed','runtime_tier','grid_policy')+ACTION}
        r.update(valid=full,repeats_ok=len(values),repeats_planned=len(planned),median_ms=median,mad_ms=mad,relative_mad=mad/median if median else None)
        measurements.append(r);by_context[first['context_id']].append(r)
    rewards=[];episodes=[]
    for cid,options in by_context.items():
        target=reference(options);baseline=target['median_ms'] if target['valid'] else None
        valid=[r for r in options if r['valid']];best=min((r['median_ms'] for r in valid),default=None)
        for r in valid:
            if baseline is not None:rewards.append(dict(r,reward_log_vs_reference=math.log(baseline/r['median_ms']),regret=r['median_ms']/best-1,
                reference_ms=baseline,assumed_prediction_ms=meta['assumed_prediction_ms'],net_speedup_assuming_prediction=baseline/(r['median_ms']+meta['assumed_prediction_ms']),
                saved_ms_minus_assumed_prediction=baseline-r['median_ms']-meta['assumed_prediction_ms']))
        episodes.append(dict(context_id=cid,**{k:target[k] for k in ('session_id','kernel_id','split_group','m','n','k','input_seed_index','seed','runtime_tier')},
                             reference_ms=baseline,reference_action={k:target[k] for k in ACTION},
                             actions=[dict(**{k:r[k] for k in ACTION+('grid_policy','valid')},median_ms=r['median_ms'] if r['valid'] else None) for r in options]))
    fields=list(measurements[0]) if measurements else ['context_id','valid','median_ms']
    write_csv(root/'heavy_measurements.csv',measurements,fields)
    extra=['reward_log_vs_reference','regret','reference_ms','assumed_prediction_ms','net_speedup_assuming_prediction','saved_ms_minus_assumed_prediction']
    write_csv(root/'heavy_action_table.csv',rewards,fields+extra)
    with (root/'replay.jsonl').open('w') as f:
        for ep in episodes:f.write(json.dumps(ep,allow_nan=False)+'\n')
    calibrations=json.loads((root/'calibration.json').read_text())
    probes_ok=sum(r['status']=='ok' for r in calibrations)
    failures=sum(r['status']!='ok' for r in rows)
    complete=bool(plan) and expected==observed and not failures and not issues
    status='COMPLETE' if complete else 'CALIBRATED' if meta['mode']=='calibrate' and probes_ok and not issues else 'INCOMPLETE'
    reached={r['kernel_id'] for r in calibrations if r['status']=='ok' and 1000<=r['kernel_ms']<=5000}
    summary=dict(status=status,heavy_protocol_revision=1,planned_trials=len(plan),recorded_trials=len(rows),failed_trials=failures,missing_trials=sum((expected-observed).values()),
                 contexts=len(by_context),actions=len(measurements),valid_actions=sum(r['valid'] for r in measurements),usable_reward_rows=len(rewards),
                 reference_seconds_contexts=sum(r['reference_ms'] is not None and r['reference_ms']>=1000 for r in episodes),
                 complete_contexts=sum(all(a['valid'] for a in r['actions']) for r in episodes),noisy_actions_over_5pct=sum(r['relative_mad'] is not None and r['relative_mad']>.05 for r in measurements),
                 successful_calibrations=probes_ok,failed_or_memory_limited_calibrations=len(calibrations)-probes_ok,
                 kernels_without_selected_shapes=[k for k in meta['kernels'] if k not in {r['kernel_id'] for r in episodes}],
                 seconds_target_unreached=[k for k in meta['kernels'] if k not in reached],validation_errors=issues)
    for mark in ('COMPLETE','INCOMPLETE','CALIBRATED'):(root/mark).unlink(missing_ok=True)
    (root/status).write_text(status+'\n');(root/'heavy_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (root/'heavy_summary.md').write_text('# Heavy pilot results\n\n'+status+'\n\n'+'\n'.join('- '+k+': '+str(v) for k,v in summary.items() if k!='validation_errors')+
        '\n\nSingle-launch CUDA event times; different protocol from v1/v2. Large outputs use 64 CPU samples plus all-output finite and guard checks. Invalid replay latencies are null. Net speedups use an assumed prediction cost, not an executed model. Seconds contexts are counted using measured reference medians. Inspect calibration.json for unreached targets and probe failures.\n')
    return summary

if __name__=='__main__':
    import sys
    result=analyze(Path(sys.argv[1]));print(json.dumps(result,indent=2));sys.exit(0 if result['status'] in ('COMPLETE','CALIBRATED') else 1)
