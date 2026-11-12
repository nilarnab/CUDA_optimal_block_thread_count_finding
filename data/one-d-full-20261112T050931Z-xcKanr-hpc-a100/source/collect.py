#!/usr/bin/env python3
"""Calibrate twelve 1D kernels, then measure the original sixteen block sizes."""
import argparse
from collections import Counter
import csv
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import random
import secrets
import shutil
import subprocess
import sys
from catalog import KINDS,GROUPS,SMOKE,LADDERS,actions,reference,counts,device_bytes,tier
from analyze import analyze
from legacy_format import export_legacy

KEYS=('trial_id','round','block_x','block_y','grid_x','grid_y')
FIELDS=['session_id','trial_id','context_id','kernel_id','split_group','m','n','k','input_seed_index','seed','reference_calibration_ms','runtime_tier','round',
        'block_x','block_y','grid_x','grid_y','grid_policy','status','kernel_ms','iterations','warmup_launches','query_wall_ms',
        'correctness_scope','checked_outputs','output_elements','input_elements','device_bytes','max_abs_error','max_scaled_error',
        'cpu_setup_reference_ms','timestamp_unix_ns','error']

def validate_sources(source):
    manifest=json.loads((source/'provenance.json').read_text())
    for row in manifest['sources']:
        p=(source/row['path']).resolve()
        if not p.is_relative_to(source.resolve()):raise ValueError('Invalid source path')
        data=p.read_bytes()
        if hashlib.sha256(data).hexdigest()!=row['sha256'] or hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()!=row['git_blob_sha1']:
            raise ValueError('Source provenance mismatch: '+row['path'])
    if (source/'kernels/includes.h').read_bytes():raise ValueError('Original includes.h must remain empty')
    return manifest

def accept(record,job):
    if isinstance(record,dict) and record.get('status')=='failed':raise ValueError(record.get('error','worker failed'))
    if not isinstance(record,dict) or any(type(record.get(k)) is not int or record[k]!=job[k] for k in KEYS):raise ValueError('Worker result identity mismatch')
    if record.get('status')!='ok' or record.get('correctness')!='passed' or record.get('guards')!='passed':raise ValueError(record.get('error','correctness/guard failure'))
    for field in ('kernel_ms','query_wall_ms','cpu_setup_reference_ms'):
        v=record.get(field)
        if type(v) not in (int,float) or not math.isfinite(v) or v<=0:raise ValueError('Invalid '+field)
    for field in ('max_abs_error','max_scaled_error'):
        v=record.get(field)
        if type(v) not in (int,float) or not math.isfinite(v) or v<0:raise ValueError('Invalid '+field)
    if record['max_scaled_error']>1 or type(record.get('iterations')) is not int or record['iterations']!=1:raise ValueError('Incorrect single-launch protocol')
    a,b,out=counts(job['kernel_id'],job['m'],job['n'],job['k'])
    for field,want in (('input_elements',a+b),('output_elements',out),('device_bytes',device_bytes(job['kernel_id'],job['m'],job['n'],job['k']))):
        if type(record.get(field)) is not int or record[field]!=want:raise ValueError('Incorrect '+field)
    full=out<=4096
    if record.get('correctness_scope')!=('full_cpu_reference' if full else '64_sampled_cpu_reference') or type(record.get('checked_outputs')) is not int or record['checked_outputs']!=(out if full else 64):raise ValueError('Incorrect validation scope')
    if type(record.get('timestamp_unix_ns')) is not int or record['timestamp_unix_ns']<=0 or type(record.get('warmup_launches')) is not int or record['warmup_launches'] not in (0,1):raise ValueError('Invalid timestamp/warmup')
    return record

def run_worker(root,device,first,jobs,stem,cap,timeout):
    folder=root/'workers';folder.mkdir(exist_ok=True)
    path=folder/(stem+'.jobs');path.write_text(''.join('{trial_id} {block_x} {block_y} {grid_x} {grid_y} {round}\n'.format(**j) for j in jobs))
    output=folder/(stem+'.jsonl');error=folder/(stem+'.stderr');reason=''
    with output.open('x') as stdout,error.open('x') as stderr:
        process=subprocess.Popen([str(root/'benchmark'),'--worker',str(device),first['kernel_id'],str(first['m']),str(first['n']),str(first['k']),str(first['seed']),str(path),str(cap)],stdout=stdout,stderr=stderr)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:reason='worker_timeout';process.kill();process.wait()
        except BaseException:
            process.kill();process.wait();raise
    seen=Counter();records={};issues=[];valid_ids={j['trial_id'] for j in jobs}
    for number,line in enumerate(output.read_text().splitlines(),1):
        try:
            r=json.loads(line);i=r['trial_id']
            if type(i) is not int or i not in valid_ids:raise ValueError('Unknown trial ID')
            seen[i]+=1;records[i]=r
        except (ValueError,KeyError,TypeError) as exc:issues.append(stem+' line '+str(number)+': '+str(exc))
    if process.returncode and not reason:reason='worker_failed: '+error.read_text()[-1000:].strip()
    accepted=[]
    for job in jobs:
        try:
            if seen[job['trial_id']]!=1:raise ValueError(reason or 'missing or duplicate worker result')
            r=accept(records[job['trial_id']],job)
            accepted.append(dict(job,**{k:v for k,v in r.items() if k not in job}))
        except (ValueError,TypeError) as exc:accepted.append(dict(job,status='failed',error=str(exc)))
    return accepted,issues

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('mode',choices=('smoke','quick','calibrate','full','seconds'));p.add_argument('device',type=int)
    p.add_argument('--kernels',default=','.join(KINDS));p.add_argument('--seeds',type=int);p.add_argument('--repeats',type=int)
    p.add_argument('--max-device-gib',type=float,default=2);p.add_argument('--probe-timeout',type=int,default=180);p.add_argument('--worker-timeout',type=int,default=3600)
    p.add_argument('--assumed-prediction-ms',type=float,default=.8)
    return p

def collect(args):
    root=args.root.resolve();source=root/'source';provenance=validate_sources(source)
    names=args.kernels.split(',')
    if len(set(names))!=len(names) or any(k not in KINDS for k in names):raise ValueError('Unknown or duplicate kernel IDs')
    seeds=args.seeds if args.seeds is not None else 1
    rounds=args.repeats if args.repeats is not None else 1 if args.mode=='smoke' else 10 if args.mode=='full' else 3
    if not 1<=seeds<=10 or not 1<=rounds<=20 or not math.isfinite(args.max_device_gib) or not .01<=args.max_device_gib<=32 or min(args.probe_timeout,args.worker_timeout)<1 or not math.isfinite(args.assumed_prediction_ms) or args.assumed_prediction_ms<0:raise ValueError('Invalid collection limits')
    cap=int(args.max_device_gib*2**30)
    result=subprocess.run([str(root/'benchmark'),'--describe',str(args.device)],capture_output=True,text=True,timeout=60,check=True)
    device=json.loads(result.stdout);session_seed=secrets.randbits(63);rng=random.Random(session_seed)
    for name in names:actions(name,*SMOKE[name],device)
    metadata=dict(csv_format='ls_cat_pilot_first_version',heavy_protocol_revision=1,collection_profile='twelve_1d_kernels_16_actions',session_id=root.name,started_utc=datetime.now(timezone.utc).isoformat(),mode=args.mode,kernels=names,device=device,grid_search='coverage',
                  session_seed=session_seed,input_seeds=seeds,repeats=rounds,max_device_bytes=cap,probe_timeout_s=args.probe_timeout,worker_timeout_s=args.worker_timeout,
                  assumed_prediction_ms=args.assumed_prediction_ms,timing_method='one kernel launch between CUDA events; one warmup per action per worker',
                  correctness='full CPU output reference for <=4096 output elements; otherwise 64 deterministic CPU samples plus all-output finite scan and both guards',
                  grouping='Keep every shape/seed/session of a split_group together when evaluating unseen source',provenance=provenance)
    (root/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    telemetry=None;handles=[];issues=[];calibrations=[];selected=[];plan=[]
    if shutil.which('nvidia-smi'):
        handles=[(root/'telemetry.csv').open('w'),(root/'telemetry-stderr.log').open('w')]
        try:telemetry=subprocess.Popen(['nvidia-smi','--id='+device['pci_bus_id'],'--query-gpu=timestamp,uuid,pci.bus_id,temperature.gpu,clocks.sm,clocks.mem,utilization.gpu,power.draw','--format=csv','--loop=1'],stdout=handles[0],stderr=handles[1])
        except OSError as exc:handles[1].write(str(exc))
    try:
        for name in names:
            targets=[500] if args.mode=='seconds' else [5] if args.mode=='quick' else [5,50,500,1000]
            ladder=[SMOKE[name]] if args.mode=='smoke' else LADDERS[name]
            found=set();fallback=None;shape_count=len(selected)
            for m,n,k in ladder:
                opts=actions(name,m,n,k,device);ref=reference(opts)
                info=dict(kernel_id=name,m=m,n=n,k=k,split_group=GROUPS[name],seed=session_seed,input_seed_index=0,round=0,trial_id=0,**ref)
                if device_bytes(name,m,n,k)>min(cap,int(device.get('free_bytes',cap)*.8)):
                    calibrations.append(dict(info,status='memory_limit',error='planned buffers exceed cap/free-memory budget'));continue
                print('Calibrating {} m={} n={} k={}'.format(name,m,n,k),flush=True)
                rows,errors=run_worker(root,args.device,info,[info],'probe_'+str(len(calibrations)),cap,args.probe_timeout);issues.extend(errors)
                row=rows[0];calibrations.append(row)
                (root/'calibration.json').write_text(json.dumps(calibrations,indent=2)+'\n')
                if row['status']!='ok':print('  '+row['error'],flush=True);continue
                ms=row['kernel_ms'];print('  reference {:.3f} ms ({})'.format(ms,tier(ms)),flush=True)
                shape=dict(kernel_id=name,m=m,n=n,k=k,reference_calibration_ms=ms,runtime_tier=tier(ms))
                if args.mode=='smoke':selected.append(shape);break
                if ms>2000:break
                if fallback is None or ms>fallback['reference_calibration_ms']:fallback=shape
                achieved={t for t in targets if ms>=t}
                if achieved-found:
                    selected.append(shape)
                    found.update(achieved)
                if set(targets).issubset(found):break
            # Retain fast families as explicitly labeled controls when they cannot reach 5 ms.
            if len(selected)==shape_count and fallback is not None and args.mode not in ('seconds','calibrate'):
                selected.append(fallback)
        (root/'calibration.json').write_text(json.dumps(calibrations,indent=2)+'\n')
        if args.mode=='calibrate':selected=[]
        rng.shuffle(selected)
        for shape in selected:
            for seed_index in range(seeds):
                tag='{}:{}:{}:{}:{}'.format(shape['kernel_id'],shape['m'],shape['n'],shape['k'],seed_index)
                seed=session_seed^int.from_bytes(hashlib.sha256(tag.encode()).digest()[:8],'little')
                cid=tag.replace(':','_');opts=actions(shape['kernel_id'],shape['m'],shape['n'],shape['k'],device)
                jobs=[dict(shape,session_id=root.name,context_id=cid,split_group=GROUPS[shape['kernel_id']],input_seed_index=seed_index,seed=seed,round=r,**a) for r in range(rounds) for a in opts]
                random.Random(seed).shuffle(jobs)
                for j in jobs:j['trial_id']=len(plan);plan.append(j)
        (root/'heavy_planned.json').write_text(json.dumps(plan,indent=2)+'\n');(root/'selected_shapes.json').write_text(json.dumps(selected,indent=2)+'\n')
        estimate=sum(j['reference_calibration_ms'] for j in plan)/60000
        print('{} selected shapes, {} trials; {:.1f} minutes timed work at reference speed (other actions and setup can take longer).'.format(len(selected),len(plan),estimate),flush=True)
        with (root/'heavy_trials.csv').open('x',newline='') as f:
            writer=csv.DictWriter(f,FIELDS,extrasaction='ignore');writer.writeheader();f.flush()
            contexts=list(dict.fromkeys(j['context_id'] for j in plan))
            for index,cid in enumerate(contexts):
                jobs=[j for j in plan if j['context_id']==cid]
                print('{}/{} {}: {} trials'.format(index+1,len(contexts),cid,len(jobs)),flush=True)
                records,errors=run_worker(root,args.device,jobs[0],jobs,cid,cap,args.worker_timeout);issues.extend(errors)
                writer.writerows(records);f.flush();print('  {} accepted, {} failed'.format(sum(r['status']=='ok' for r in records),sum(r['status']!='ok' for r in records)),flush=True)
    finally:
        if telemetry and telemetry.poll() is None:
            telemetry.terminate()
            try:telemetry.wait(timeout=5)
            except subprocess.TimeoutExpired:telemetry.kill();telemetry.wait()
        for f in handles:f.close()
        (root/'collection_errors.json').write_text(json.dumps(issues,indent=2)+'\n')
        if (root/'heavy_planned.json').exists():
            summary=analyze(root);export_legacy(root,summary)
        else:summary={'status':'INTERRUPTED_DURING_CALIBRATION'};(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    return 0 if summary['status'] in ('COMPLETE','CALIBRATED') else 1

if __name__=='__main__':
    try:sys.exit(collect(parser().parse_args()))
    except KeyboardInterrupt:print('Interrupted; retain the run directory and raw worker logs.',file=sys.stderr);sys.exit(130)
    except (ValueError,OSError,KeyError,subprocess.SubprocessError) as exc:print('Collection failed: '+str(exc),file=sys.stderr);sys.exit(1)
