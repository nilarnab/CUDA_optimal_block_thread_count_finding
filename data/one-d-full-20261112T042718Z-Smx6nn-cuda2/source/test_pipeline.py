import contextlib
import csv
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import collect
from analyze import analyze
from catalog import KINDS, ELEMENTWISE, TENSOR_OUTPUT, SMOKE, LADDERS, THREAD_COUNTS, actions, counts, device_bytes
import legacy_format
import legacy_analyze

SOURCE=Path(__file__).resolve().parent
DEVICE=dict(sm_count=68,max_threads_per_block=1024,max_block_dim_x=1024,max_block_dim_y=1024,
            max_grid_dim_x=2147483647,max_grid_dim_y=65535,free_bytes=10*2**30,pci_bus_id='0000:83:00.0',
            kernels={k:dict(max_threads_per_block=1024) for k in KINDS})
FAKE='''#!/usr/bin/env python3
import json,sys
from pathlib import Path
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'source'))
from catalog import counts
if sys.argv[1]=='--describe':
 print((root/'fake_device.json').read_text());sys.exit(0)
_,_,device,name,m,n,k,seed,jobs,cap=sys.argv
m,n,k=map(int,(m,n,k));a,b,out=counts(name,m,n,k)
timings=json.loads((root/'fake_timings.json').read_text())
value=timings.get(name,1000.0)
if isinstance(value,list):value=value[min(m,len(value)-1)]
if isinstance(value,dict):value=value['probe' if Path(jobs).name.startswith('probe_') else 'sweep']
for line in Path(jobs).read_text().splitlines():
 ident,bx,by,gx,gy,round=map(int,line.split())
 row=dict(trial_id=ident,block_x=bx,block_y=by,grid_x=gx,grid_y=gy,round=round)
 if value=='fail':row.update(status='failed',error='incorrect output at 3')
 else:row.update(status='ok',correctness='passed',guards='passed',kernel_ms=value,query_wall_ms=value+1,
  cpu_setup_reference_ms=1.0,iterations=1,warmup_launches=1,max_abs_error=0.0,max_scaled_error=0.0,
  input_elements=a+b,output_elements=out,device_bytes=4*(a+b+out+32),
  correctness_scope='full_cpu_reference' if out<=4096 else '64_sampled_cpu_reference',
  checked_outputs=out if out<=4096 else 64,timestamp_unix_ns=123456)
 print(json.dumps(row),flush=True)
'''

class Pipeline(unittest.TestCase):
    def make_run(self, folder, timings=None):
        root=Path(folder)/'run';root.mkdir()
        shutil.copytree(SOURCE,root/'source',ignore=shutil.ignore_patterns('__pycache__','runs'))
        (root/'benchmark').write_text(FAKE);(root/'benchmark').chmod(0o755)
        (root/'fake_device.json').write_text(json.dumps(DEVICE))
        (root/'fake_timings.json').write_text(json.dumps(timings or {}))
        return root

    def run_collection(self, root, mode='smoke', *options):
        args=collect.parser().parse_args([str(root),mode,'0',*options])
        with patch.object(collect.shutil,'which',return_value=None),contextlib.redirect_stdout(io.StringIO()):
            return collect.collect(args)

    def test_provenance_and_corruption(self):
        self.assertEqual(len(collect.validate_sources(SOURCE)['sources']),12)
        with tempfile.TemporaryDirectory() as d:
            root=self.make_run(d);(root/'source/kernels/add.cu').write_text('corrupt')
            with self.assertRaises(ValueError):collect.validate_sources(root/'source')

    def test_all_shapes_have_exact_original_actions(self):
        for name in KINDS:
            for shape in [SMOKE[name],*LADDERS[name]]:
                aa=actions(name,*shape,DEVICE)
                self.assertEqual([(a['block_x'],a['block_y']) for a in aa],[(b,1) for b in THREAD_COUNTS])
                work=counts(name,*shape)[2] if name in ELEMENTWISE else shape[0]
                self.assertTrue(all(a['grid_x']*a['block_x']>=work and a['grid_y']==1 for a in aa))
                self.assertTrue(all((a['grid_x']-1)*a['block_x']<work for a in aa))

    def test_missing_1024_action_refused(self):
        device=dict(DEVICE,kernels=dict(DEVICE['kernels'],nbody=dict(max_threads_per_block=512)))
        with self.assertRaisesRegex(ValueError,'All 16'):actions('nbody',259,1,1,device)

    def test_memory_and_bounds(self):
        self.assertEqual(device_bytes('vector_add',16384,1,1),3*2**30+128)
        self.assertEqual(counts('periodic_tile',32768,1,1),(221,0,1073741824))
        self.assertEqual(counts('segmented_sum',16,1,16777216),(268435456,0,16))
        for name in TENSOR_OUTPUT:
            self.assertEqual(counts(name,8,1,16777216),(134217728,0,134217728))
            self.assertEqual(device_bytes(name,8,1,16777216),2**30+128)
        self.assertEqual(counts('accumulate',16,1,16777216),(268435456,0,16))
        for shape in [('vector_add',32769,1,1),('mean',16384,1,16777216),('gemm',17,17,17)]:
            with self.assertRaises(ValueError):counts(*shape)

    def test_smoke_exports_192_observations(self):
        with tempfile.TemporaryDirectory() as d:
            root=self.make_run(d);self.assertEqual(self.run_collection(root),0)
            summary=json.loads((root/'heavy_summary.json').read_text())
            self.assertEqual(summary['contexts'],12);self.assertEqual(summary['planned_trials'],192)
            for file,fields in [('trials.csv',legacy_format.FIELDS),('measurements.csv',legacy_analyze.MEASUREMENT_FIELDS),('action_table.csv',legacy_analyze.ACTION_FIELDS)]:
                with (root/file).open() as f:self.assertEqual(csv.DictReader(f).fieldnames,list(fields))
            self.assertEqual(legacy_analyze.analyze(root)['status'],'COMPLETE')
            with (root/'heavy_trials.csv').open() as f:
                rows=list(csv.DictReader(f))
            for name in (*TENSOR_OUTPUT,'accumulate'):
                jobs=[r for r in rows if r['kernel_id']==name]
                self.assertEqual(len(jobs),16)
                self.assertTrue(all(int(r['input_elements'])==35*67 for r in jobs))
                self.assertTrue(all(int(r['output_elements'])==(35*67 if name in TENSOR_OUTPUT else 35) for r in jobs))

    def test_single_launch_and_seed_aliases(self):
        with tempfile.TemporaryDirectory() as d:
            root=self.make_run(d)
            self.assertEqual(self.run_collection(root,'smoke','--kernels','vector_add,segmented_sum','--seeds','2'),0)
            self.assertEqual(legacy_analyze.analyze(root)['duplicate_observed_keys'],0)
            with (root/'trials.csv').open() as f:
                for r in csv.DictReader(f):
                    self.assertEqual(float(r['batch_ms']),float(r['kernel_ms']))
                    self.assertEqual(int(r['iterations']),1)

    def test_fast_family_retained_and_targets_honest(self):
        with tempfile.TemporaryDirectory() as d:
            root=self.make_run(d,dict(vector_add=.3))
            with patch.dict(collect.LADDERS,vector_add=[(240,1,1),(1024,1,1)]):
                self.assertEqual(self.run_collection(root,'full','--kernels','vector_add','--repeats','1'),0)
            report=json.loads((root/'duration_targets.json').read_text())[0]
            self.assertTrue(report['fast_control_only']);self.assertFalse(report['reached_ms']['5'])
            self.assertFalse(report['reached_0_5_to_2_seconds'])

    def test_seconds_mode_requires_every_requested_family(self):
        with tempfile.TemporaryDirectory() as d:
            root=self.make_run(d,dict(vector_add=.3,nbody=1000.))
            with patch.dict(collect.LADDERS,vector_add=[(240,1,1)],nbody=[(8192,1,1)]):
                self.assertEqual(self.run_collection(root,'seconds','--kernels','vector_add,nbody','--repeats','1'),1)
            summary=json.loads((root/'heavy_summary.json').read_text())
            self.assertEqual(summary['kernels_without_selected_shapes'],['vector_add'])
            self.assertEqual(summary['seconds_target_unreached'],['vector_add'])

    def test_failed_accuracy_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root=self.make_run(d,dict(variance='fail'))
            with patch.dict(collect.LADDERS,variance=[(256,1,1024)]):
                self.assertEqual(self.run_collection(root,'full','--kernels','variance'),1)
            probe=json.loads((root/'calibration.json').read_text())[0]
            self.assertEqual(probe['error'],'incorrect output at 3')
            self.assertNotIn('kernel_ms',probe)

    def test_seconds_qualification_uses_final_measurements(self):
        with tempfile.TemporaryDirectory() as d:
            root=self.make_run(d,dict(nbody=dict(probe=1000.,sweep=2100.)))
            self.assertEqual(self.run_collection(root,'seconds','--kernels','nbody','--repeats','1'),1)
            summary=json.loads((root/'heavy_summary.json').read_text())
            self.assertEqual(summary['failed_trials'],0)
            self.assertEqual(summary['seconds_target_unreached'],['nbody'])

    def test_memory_limited_shape_not_launched(self):
        with tempfile.TemporaryDirectory() as d:
            root=self.make_run(d,dict(vector_add=.3))
            with patch.dict(collect.LADDERS,vector_add=[(240,1,1),(32768,1,1)]):
                self.assertEqual(self.run_collection(root,'quick','--kernels','vector_add','--repeats','1'),0)
            probes=json.loads((root/'calibration.json').read_text())
            self.assertEqual(probes[-1]['status'],'memory_limit')

    def test_invalid_and_missing_actions_remain_visible(self):
        with tempfile.TemporaryDirectory() as d:
            root=self.make_run(d);self.run_collection(root,'smoke','--kernels','nbody')
            with (root/'heavy_trials.csv').open() as f:rows=list(csv.DictReader(f));fields=list(rows[0])
            rows[0]['status']='failed';rows[0]['kernel_ms']='';rows.pop()
            with (root/'heavy_trials.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(rows)
            summary=analyze(root);self.assertEqual(summary['status'],'INCOMPLETE');self.assertEqual(summary['missing_trials'],1)
            episode=json.loads((root/'replay.jsonl').read_text())
            self.assertEqual(sum(a['median_ms'] is None for a in episode['actions']),2)

    def test_calibration_only_and_above_range(self):
        with tempfile.TemporaryDirectory() as d:
            root=self.make_run(d,dict(nbody=2500.))
            self.assertEqual(self.run_collection(root,'calibrate','--kernels','nbody'),0)
            self.assertTrue((root/'CALIBRATED').exists())
            self.assertEqual(json.loads((root/'selected_shapes.json').read_text()),[])
            with (root/'trials.csv').open() as f:self.assertEqual(list(csv.DictReader(f)),[])

if __name__=='__main__':unittest.main()
