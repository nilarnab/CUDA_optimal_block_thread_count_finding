"""Export exactly the first pilot's CSV headers and its original aggregation rules."""
import csv
import hashlib
import json
from pathlib import Path
import legacy_analyze

FIELDS=['kernel_id','repo_id','source_sha256','size','block_x','block_y','round','seed','status','kernel_ms','batch_ms','iterations',
        'grid_x','grid_y','grid_z','input_elements','output_elements','max_abs_error','warmup_launches','elapsed_wall_s','error','detail_json']
SOURCES=dict(gemm=('104','kernels/matrixMultiply.cu'),segmented_sum=('10040','kernels/gpuSum.cu'),mean=('10194','kernels/mean_kernel.cu'),
             variance=('10194','kernels/variance_kernel.cu'),syrk=('polybench_acc','kernels/syrk_kernel.cuh'),nbody=('nvidia_adapted','nbody.cuh'))

def identifier(r,meta):
    # v1 omits grid and input-seed identity from its grouping key; aliases prevent silent collisions.
    result=r['kernel_id'] if r['kernel_id']=='nbody' else r['kernel_id']+'__k'+str(r['k'])
    if r['kernel_id']=='gemm' and r['m']!=r['n']:result+='__n'+str(r['n'])
    if meta.get('grid_search')=='expanded':result+='__grid_'+r['grid_policy']
    if meta['input_seeds']>1:result+='__s'+str(r['input_seed_index'])
    return result

def export_legacy(root,heavy_summary):
    root=Path(root);plan=json.loads((root/'heavy_planned.json').read_text());meta=json.loads((root/'metadata.json').read_text());contexts={};mapping={};compatible=[]
    for j in plan:
        alias=identifier(j,meta)
        mapping[int(j['trial_id'])]=(alias,j['m'])
        repo,path=SOURCES[j['kernel_id']]
        contexts[(alias,j['m'])]=dict(kernel_id=alias,base_kernel_id=j['kernel_id'],m=j['m'],n=j['n'],k=j['k'],size=j['m'],input_seed_index=j['input_seed_index'],seed=j['seed'],
                             grid_policy=j['grid_policy'],split_group=j['split_group'],runtime_tier=j['runtime_tier'],repo_id=repo,source_path=path,
                             source_sha256=hashlib.sha256((root/'source'/path).read_bytes()).hexdigest())
        compatible.append(dict(kernel_id=alias,size=j['m'],block_x=j['block_x'],block_y=j['block_y'],round=j['round'],seed=j['seed']))
    (root/'planned.json').write_text(json.dumps(compatible,indent=2)+'\n')
    (root/'contexts.json').write_text(json.dumps(list(contexts.values()),indent=2)+'\n')
    with (root/'heavy_trials.csv').open(newline='') as src,(root/'trials.csv').open('w',newline='') as dst:
        writer=csv.DictWriter(dst,FIELDS,extrasaction='ignore');writer.writeheader()
        for r in csv.DictReader(src):
            context_key=mapping[int(r['trial_id'])];alias=context_key[0];ctx=contexts[context_key]
            row={k:r[k] for k in FIELDS if k in r};row.update(kernel_id=alias,size=r['m'],repo_id=ctx['repo_id'],source_sha256=ctx['source_sha256'],grid_z=1)
            row['batch_ms']=r['kernel_ms'] if r['status']=='ok' else ''
            row['elapsed_wall_s']=float(r['query_wall_ms'])/1000 if r.get('query_wall_ms') else ''
            detail=dict(r,base_kernel_id=ctx['base_kernel_id'],timing_method='single_launch_cuda_events',correctness='passed' if r['status']=='ok' else 'unavailable')
            for k in ('trial_id','m','n','k','input_seed_index','seed','round','block_x','block_y','grid_x','grid_y','iterations','warmup_launches','checked_outputs','input_elements','output_elements','device_bytes','timestamp_unix_ns'):
                if detail.get(k):detail[k]=int(detail[k])
            for k in ('kernel_ms','query_wall_ms','reference_calibration_ms','max_abs_error','max_scaled_error','cpu_setup_reference_ms'):
                if detail.get(k):detail[k]=float(detail[k])
            row['detail_json']=json.dumps(detail,separators=(',',':'))
            writer.writerow(row)
    if not plan:
        # Calibration-only mode intentionally has no v1 trial dataset to analyze.
        legacy_analyze.write_csv(root/'measurements.csv',legacy_analyze.MEASUREMENT_FIELDS,[])
        legacy_analyze.write_csv(root/'action_table.csv',legacy_analyze.ACTION_FIELDS,[])
        legacy_analyze.write_text(root/'summary.json',json.dumps(heavy_summary,indent=2)+'\n')
        legacy_analyze.write_text(root/'summary.md','# Heavy pilot\n\n'+heavy_summary['status']+'; see heavy_summary.md and calibration.json.\n')
        return
    summary=legacy_analyze.analyze(root)
    if heavy_summary['status']!='COMPLETE':
        summary['status']='INCOMPLETE';summary['validation_errors'].extend(heavy_summary['validation_errors'])
        (root/'COMPLETE').unlink(missing_ok=True);(root/'INCOMPLETE').write_text('Heavy audit incomplete; see heavy_summary.json.\n')
        legacy_analyze.write_text(root/'summary.json',json.dumps(summary,indent=2)+'\n')
        legacy_analyze.write_text(root/'summary.md',legacy_analyze.markdown(summary))
