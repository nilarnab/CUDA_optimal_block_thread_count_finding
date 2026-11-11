import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import collect
from analyze import analyze
from catalog import KINDS,SMOKE,actions,counts,device_bytes,reference
import legacy_format
import legacy_analyze

SOURCE=Path(__file__).resolve().parent
DEVICE=dict(sm_count=68,max_threads_per_block=1024,max_block_dim_x=1024,max_block_dim_y=1024,
            max_grid_dim_x=2147483647,max_grid_dim_y=65535,free_bytes=10*2**30,pci_bus_id='0000:83:00.0',
            kernels={k:dict(max_threads_per_block=1024) for k in KINDS})

def record(j):
    a,b,out=counts(j['kernel_id'],j['m'],j['n'],j['k'])
    return dict(**{k:j[k] for k in collect.KEYS},status='ok',correctness='passed',guards='passed',kernel_ms=10.,query_wall_ms=12.,
                cpu_setup_reference_ms=1.,iterations=1,warmup_launches=1,max_abs_error=0.,max_scaled_error=0.,
                input_elements=a+b,output_elements=out,device_bytes=device_bytes(j['kernel_id'],j['m'],j['n'],j['k']),
                correctness_scope='full_cpu_reference' if out<=4096 else '64_sampled_cpu_reference',checked_outputs=min(out,64) if out>4096 else out,
                timestamp_unix_ns=123456)

FAKE='''#!/usr/bin/env python3
import json,sys
from pathlib import Path
if sys.argv[1]=='--describe':
 print(Path(__file__).with_name('fake_device.json').read_text());sys.exit(0)
_,_,device,name,m,n,k,seed,jobs,cap=sys.argv
m,n,k=map(int,(m,n,k))
if name=='nbody':a,b,out=4*m,0,3*m
elif name in ('gemm','syrk'):a,b,out=m*k,k*n if name=='gemm' else 0,m*n
else:a,b,out=m*k,m if name=='variance' else 0,m
for line in Path(jobs).read_text().splitlines():
 ident,bx,by,gx,gy,round=map(int,line.split())
 print(json.dumps(dict(trial_id=ident,block_x=bx,block_y=by,grid_x=gx,grid_y=gy,round=round,status='ok',correctness='passed',guards='passed',
  kernel_ms=1000.0,query_wall_ms=1001.0,cpu_setup_reference_ms=1.0,iterations=1,warmup_launches=1,max_abs_error=0.0,max_scaled_error=0.0,
  input_elements=a+b,output_elements=out,device_bytes=4*(a+b+out+32),correctness_scope='full_cpu_reference' if out<=4096 else '64_sampled_cpu_reference',
  checked_outputs=out if out<=4096 else 64,timestamp_unix_ns=123456)),flush=True)
'''

class Pipeline(unittest.TestCase):
    def test_provenance(self):
        self.assertEqual(len(collect.validate_sources(SOURCE)['sources']),6)

    def test_corrupt_original_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'source';shutil.copytree(SOURCE,p,ignore=shutil.ignore_patterns('__pycache__','runs'))
            (p/'kernels/gpuSum.cu').write_text('corrupt')
            with self.assertRaises(ValueError):collect.validate_sources(p)

    def test_counts_memory(self):
        self.assertEqual(device_bytes('gemm',8192,8192,16384),int(1.25*2**30)+128)
        with self.assertRaises(ValueError):counts('syrk',32,64,128)
        with self.assertRaises(ValueError):counts('gemm',16384,16384,1048576)

    def test_action_contracts(self):
        for name in KINDS:
            for m,n,k in [SMOKE[name],(8192,8192,4096) if name in ('gemm','syrk') else (8192,1,1) if name=='nbody' else (4096,1,65536)]:
                aa=actions(name,m,n,k,DEVICE,smoke=True)
                keys=[tuple(a[f] for f in ('block_x','block_y','grid_x','grid_y')) for a in aa]
                self.assertEqual(len(keys),len(set(keys)))
                self.assertEqual(reference(aa)['grid_policy'],'coverage')
                for a in aa:
                    bx,by,gx,gy=[a[f] for f in ('block_x','block_y','grid_x','grid_y')]
                    if name in ('gemm','syrk'):self.assertGreaterEqual(gx*bx,n);self.assertGreaterEqual(gy*by,m)
                    elif name!='nbody':self.assertGreaterEqual(gx*gy*bx,m)

    def test_protocol(self):
        j=dict(kernel_id='gemm',m=17,n=19,k=23,trial_id=0,round=0,block_x=16,block_y=16,grid_x=2,grid_y=2)
        collect.accept(record(j),j)
        for field,bad in [('kernel_ms',float('nan')),('iterations',64),('trial_id',False),('device_bytes',0),('guards','failed'),('checked_outputs',0)]:
            r=record(j);r[field]=bad
            with self.assertRaises(ValueError):collect.accept(r,j)

    def test_end_to_end_and_invalid_export(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'run';root.mkdir();shutil.copytree(SOURCE,root/'source',ignore=shutil.ignore_patterns('__pycache__','runs'))
            (root/'benchmark').write_text(FAKE);(root/'benchmark').chmod(0o755);(root/'fake_device.json').write_text(json.dumps(DEVICE))
            args=collect.parser().parse_args([str(root),'smoke','0'])
            with patch.object(collect.shutil,'which',return_value=None):self.assertEqual(collect.collect(args),0)
            summary=json.loads((root/'heavy_summary.json').read_text());self.assertEqual(summary['contexts'],6)
            self.assertEqual(summary['recorded_trials'],summary['planned_trials']);self.assertEqual(summary['failed_trials'],0)
            with (root/'trials.csv').open() as f:self.assertEqual(csv.DictReader(f).fieldnames,legacy_format.FIELDS)
            with (root/'measurements.csv').open() as f:self.assertEqual(csv.DictReader(f).fieldnames,list(legacy_analyze.MEASUREMENT_FIELDS))
            with (root/'action_table.csv').open() as f:self.assertEqual(csv.DictReader(f).fieldnames,list(legacy_analyze.ACTION_FIELDS))
            # The untouched first-version analyzer accepts the generated plan/trials.
            self.assertEqual(legacy_analyze.analyze(root)['status'],'COMPLETE')
            with (root/'heavy_trials.csv').open() as f:rows=list(csv.DictReader(f));fields=list(rows[0])
            rows[0]['status']='failed';rows[0]['kernel_ms']=''
            with (root/'heavy_trials.csv').open('w',newline='') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
            summary=analyze(root);self.assertEqual(summary['status'],'INCOMPLETE')
            eps=[json.loads(x) for x in (root/'replay.jsonl').read_text().splitlines()]
            invalid=[a for ep in eps for a in ep['actions'] if not a['valid']]
            self.assertEqual(len(invalid),1);self.assertIsNone(invalid[0]['median_ms'])
            rows.pop()
            with (root/'heavy_trials.csv').open('w',newline='') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
            self.assertEqual(analyze(root)['missing_trials'],1)

    def test_seconds_calibration_selects_measured_shape(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'run';root.mkdir();shutil.copytree(SOURCE,root/'source',ignore=shutil.ignore_patterns('__pycache__','runs'))
            (root/'benchmark').write_text(FAKE);(root/'benchmark').chmod(0o755);(root/'fake_device.json').write_text(json.dumps(DEVICE))
            args=collect.parser().parse_args([str(root),'seconds','0','--kernels','gemm,nbody','--seeds','1','--repeats','1'])
            with patch.object(collect.shutil,'which',return_value=None):self.assertEqual(collect.collect(args),0)
            shapes=json.loads((root/'selected_shapes.json').read_text());self.assertEqual(len(shapes),2)
            self.assertTrue(all(s['runtime_tier']=='1_to_5_seconds' for s in shapes))

    def test_extra_grids_seeds_keep_unique_first_version_keys(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'run';root.mkdir();shutil.copytree(SOURCE,root/'source',ignore=shutil.ignore_patterns('__pycache__','runs'))
            (root/'benchmark').write_text(FAKE);(root/'benchmark').chmod(0o755);(root/'fake_device.json').write_text(json.dumps(DEVICE))
            args=collect.parser().parse_args([str(root),'smoke','0','--kernels','gemm,nbody','--seeds','2','--grid-search','expanded'])
            with patch.object(collect.shutil,'which',return_value=None):self.assertEqual(collect.collect(args),0)
            summary=legacy_analyze.analyze(root)
            self.assertEqual(summary['status'],'COMPLETE');self.assertEqual(summary['duplicate_observed_keys'],0)
            with (root/'trials.csv').open() as f:rows=list(csv.DictReader(f))
            for r in rows:
                self.assertEqual(float(r['batch_ms']),float(r['kernel_ms']))
                self.assertEqual(int(r['iterations']),1)
                self.assertEqual(int(r['size']),json.loads(r['detail_json'])['m'])

    def test_calibration_only_preserves_empty_first_version_headers(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'run';root.mkdir();shutil.copytree(SOURCE,root/'source',ignore=shutil.ignore_patterns('__pycache__','runs'))
            (root/'benchmark').write_text(FAKE);(root/'benchmark').chmod(0o755);(root/'fake_device.json').write_text(json.dumps(DEVICE))
            args=collect.parser().parse_args([str(root),'calibrate','0','--kernels','gemm'])
            with patch.object(collect.shutil,'which',return_value=None):self.assertEqual(collect.collect(args),0)
            self.assertTrue((root/'CALIBRATED').exists())
            with (root/'trials.csv').open() as f:
                reader=csv.DictReader(f);self.assertEqual(reader.fieldnames,legacy_format.FIELDS);self.assertEqual(list(reader),[])

    def test_timeout_is_retained_as_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'benchmark').write_text('#!/usr/bin/env python3\nimport time\ntime.sleep(10)\n');(root/'benchmark').chmod(0o755)
            job=dict(kernel_id='gemm',m=17,n=19,k=23,seed=1,trial_id=0,round=0,block_x=16,block_y=16,grid_x=2,grid_y=2)
            rows,issues=collect.run_worker(root,0,job,[job],'timeout',1000000,1)
            self.assertEqual(issues,[]);self.assertEqual(rows[0]['status'],'failed');self.assertEqual(rows[0]['error'],'worker_timeout')
            self.assertNotIn('kernel_ms',rows[0])

if __name__=='__main__':unittest.main()
