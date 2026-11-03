"""Offline checks of admission, raw accounting, and strongest-control pairing."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

path = Path(__file__).resolve().parents[1] / 'scripts/qualify_results.py'
spec = importlib.util.spec_from_file_location('qualification', path)
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


def samples():
    return {'schema': 'cgs.serial-telemetry.v1', 'scheduler_job_id': 'job', 'uuid': 'GPU-a', 'clean': True, 'samples': [
        {'phase': phase, 'sampled_at_epoch_ns': i, 'devices': [{'uuid': 'GPU-a', 'utilization_percent': 0}, {'uuid': 'GPU-b', 'utilization_percent': 0}], 'processes': [], 'issues': []}
        for i, phase in enumerate(('before', 'during', 'after'), 1)]}


def raw():
    return {'schema':'cgs.campaign.v1', 'status':'ok', 'accuracy_accepted':True, 'validation_scope':'first-and-last-complete-steps', 'backend':'direct', 'case':'case', 'gate':'ste', 'body':'product', 'precision':'fp16', 'condition_update':'none', 'tile':32, 'stages':1, 'workload_sha256':'a'*64, 'device_uuid':'GPU-a', 'input':'synthetic', 'seed':1, 'reuse_horizon':16, 'parameter_update':True, 'nominal_input_change_percent':1, 'applied_input_change_percent':0, 'cold_device_preparation_includes_host_condition_maps':True, 'replay_input_preparation_ms':1., 'host_layout_preparation_ms':2., 'cold_device_preparation_ms':3., 'cold_execution_preparation_ms':6., 'recurrent_complete_horizon_ms':[5.], 'output_recovery_ms':[1.]}


class QualificationTests(unittest.TestCase):
    def test_serial_guard_fails_closed(self):
        q.telemetry(samples(), 'job', 'GPU-a')
        for change in ('no_during', 'other_busy', 'foreign_process', 'issue', 'missing_device', 'target_busy'):
            data = samples()
            s = data['samples']
            if change == 'no_during': s.pop(1)
            if change == 'other_busy': s[1]['devices'][1]['utilization_percent'] = 1
            if change == 'foreign_process': s[1]['processes'] = [{'uuid':'GPU-a', 'owned':False}]
            if change == 'issue': s[1]['issues'] = ['foreign process']
            if change == 'missing_device': s[1]['devices'].pop()
            if change == 'target_busy': s[1]['devices'][0]['utilization_percent'] = 30
            with self.subTest(change=change), self.assertRaises((ValueError, TypeError)):
                q.telemetry(data, 'job', 'GPU-a')

    def test_component_accounting_and_correctness(self):
        x = raw()
        response = {'status':'ok', 'request_id':'r', 'case_id':'case', 'observed_workload_sha256':'a'*64, 'samples':[12000.]}
        row = {'candidate_id':'direct', 'case_id':'case', 'samples_json':'[12000.0]'}
        case = {'input': {'campaign_workload_sha256':'a'*64}, 'parameters':{k:x[k] for k in ('gate','body','precision','condition_update','tile','stages')}}
        self.assertEqual(q.campaign(x,response,row,case,'GPU-a'), 12000.)
        for key, bad in [('cold_execution_preparation_ms',7.), ('accuracy_accepted',False), ('workload_sha256','b'*64), ('condition_update','rebuild'), ('output_recovery_ms',[2.])]:
            modified = copy.deepcopy(x); modified[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError): q.campaign(modified,response,row,case,'GPU-a')

    def test_scheduler_ids_are_distinct_and_exclusive_cleanup_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = {'job_id':'job', 'attempt_id':'autobench-id', 'receipt':'receipt.json', 'reserved_seconds':30, 'expected_gpu_resource_id':'gpu:host:GPU-a', 'gpu_resource_id':'gpu:host:GPU-a'}
            receipt = {'id':'job', 'state':'succeeded', 'cleanup_verified':True, 'attempt':{'attempt_id':'scheduler-id', 'cleanup_verified':True, 'created_at':'2026-11-02T00:00:00Z','finished_at':'2026-11-02T00:00:10Z', 'execution_context':{'allocations':[{'resource_id':'gpu:host:GPU-a','mode':'exclusive'}]}}}
            (root/'receipt.json').write_text(json.dumps(receipt))
            self.assertEqual(q.job_receipt(root,row,True)[2], 'GPU-a')
            for key in ('cleanup', 'mode', 'job'):
                changed=copy.deepcopy(receipt)
                if key == 'cleanup': changed['cleanup_verified']=False
                if key == 'mode': changed['attempt']['execution_context']['allocations'][0]['mode']='shared'
                if key == 'job': changed['id']='other'
                (root/'receipt.json').write_text(json.dumps(changed))
                with self.subTest(key=key), self.assertRaises(ValueError): q.job_receipt(root,row,True)

    def test_strongest_control_and_missing_arm(self):
        phases = {}
        for repeat in range(6):
            for index, (arm, us) in enumerate(([('scalar',100),('bitmap',95),('direct',90),('cohort-direct',80)] if repeat % 2 else [('cohort-direct',80),('direct',90),('bitmap',95),('scalar',100)])):
                obj=raw(); obj['backend']=arm
                phases[('run','supplemental',repeat,arm)]={'index':repeat*4+index,'us':us,'raw':obj}
        packet={'job':'job','uuid':'GPU-a','phases':phases,'config':{'candidates':['cohort-direct']}}
        result=q.comparisons([packet])[0]
        self.assertEqual(result['strongest_control'],'direct')
        self.assertEqual(result['decision'],'paired threshold met; study-wide recommendation pending sanitizer coverage')
        del phases[('run','supplemental',0,'bitmap')]
        self.assertIn('incomplete',q.comparisons([packet])[0]['decision'])
        phases[('run','supplemental',0,'bitmap')]={'index':9,'us':95,'raw':dict(raw(), seed=2)}
        self.assertIn('unmatched',q.comparisons([packet])[0]['decision'])

    def test_v3_matched_staging_ablation_is_separate(self):
        phases={}
        for n in range(6):
            order=['cohort-direct','cohort-direct-two-tile','cohort-shared'] if n%2 else ['cohort-shared','cohort-direct-two-tile','cohort-direct']
            for pos,arm in enumerate(order):
                phases[('run','supplemental',n,arm)]={'index':n*3+pos,'us':80 if arm == 'cohort-shared' else 100,'raw':dict(raw(),backend=arm)}
        packet={'job':'job','uuid':'GPU-a','phases':phases,'config':{'incumbent':'scalar','candidates':['cohort-direct','cohort-direct-two-tile','cohort-shared']}}
        result=q.staging_ablations([packet])[0]
        self.assertEqual(result['control'],'cohort-direct-two-tile')
        self.assertEqual(result['candidate'],'cohort-shared')
        self.assertEqual(result['calculation']['decision'],'calculator-threshold-met')
        self.assertIn('does not replace',result['interpretation'])
        # Conventional controls are absent: an ablation alone cannot promote.
        self.assertTrue(all('incomplete' in r['decision'] for r in q.comparisons([packet])))
        del phases[('run','supplemental',0,'cohort-direct-two-tile')]
        self.assertIn('incomplete',q.staging_ablations([packet])[0]['decision'])
        phases[('run','supplemental',0,'cohort-direct-two-tile')]={'index':0,'us':100,'raw':dict(raw(),backend='cohort-direct-two-tile',stages=2)}
        self.assertIn('unmatched',q.staging_ablations([packet])[0]['decision'])
        packet['config']['candidates'].remove('cohort-direct-two-tile')
        self.assertEqual(q.staging_ablations([packet]),[])

    def test_v4_masked_control_and_matched_ownership_are_separate(self):
        phases={}; arms=['scalar','bitmap','direct','masked-direct-two-tile','cohort-direct-two-tile','cohort-shared']
        for n in range(6):
            order=arms if n%2 else list(reversed(arms))
            for pos,arm in enumerate(order):
                values={'scalar':100,'bitmap':95,'direct':90,'masked-direct-two-tile':80,'cohort-direct-two-tile':70,'cohort-shared':72}
                phases[('run','supplemental',n,arm)]={'index':n*len(order)+pos,'us':values[arm],'raw':dict(raw(),backend=arm)}
        packet={'job':'job','uuid':'GPU-a','phases':phases,'matrix_schema':'cgs.measurement-matrix.v4','config':{'incumbent':'scalar','candidates':arms[1:]}}
        reports={r['candidate']:r for r in q.comparisons([packet])}
        self.assertEqual(reports['cohort-direct-two-tile']['strongest_control'],'masked-direct-two-tile')
        self.assertNotIn('masked-direct-two-tile',reports['masked-direct-two-tile']['required_controls'])
        self.assertEqual(reports['masked-direct-two-tile']['strongest_control'],'direct')
        ownership=q.masked_ownership_ablations([packet])[0]
        self.assertEqual(ownership['control'],'masked-direct-two-tile');self.assertEqual(ownership['candidate'],'cohort-direct-two-tile')
        del phases[('run','supplemental',0,'masked-direct-two-tile')]
        self.assertIn('incomplete',q.comparisons([packet])[0]['decision'])
        self.assertIn('incomplete',q.masked_ownership_ablations([packet])[0]['decision'])
        packet['matrix_schema']='cgs.measurement-matrix.v3'
        self.assertEqual(q.masked_ownership_ablations([packet]),[])

    def test_frozen_v2_v3_selection_rejects_mixing(self):
        root=Path(__file__).resolve().parents[3]
        contract=root/'planning/chromatin-geometry-study-2026-11-02/contract-freeze-v3.json'
        matrix=root/'bench/chromatin_geometry/docs/measurement-matrix-v3.json'
        q.validate_selection(contract,matrix)
        with self.assertRaisesRegex(ValueError,'mixed'):
            q.validate_selection(contract,root/'bench/chromatin_geometry/docs/measurement-matrix-v2.json')

    def test_same_packet_update_ablation_and_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); update=root/'kda/update-ablation';update.mkdir(parents=True)
            frozen=root/'frozen.json';frozen.write_text(json.dumps({'workloads':[{'id':'case','seed':1,'reuse_horizon':16,'update_policy':'update_every_step','requested_leaf_change_percent':1}]}))
            matrix=root/'matrix.json';matrix.write_text(json.dumps({'workloads':['case'],'scenarios':[{'workload':0,'input':'synthetic','gate':'supplied','body':'product','precision':'fp16','updates':['incremental','rebuild']}]}))
            expected={'id':'case','input':{'campaign_workload_sha256':'a'*64},'parameters':{'gate':'supplied','body':'product','precision':'fp16','condition_update':'incremental','tile':32,'stages':1}}
            config={'supplemental_updates':['rebuild','incremental']};metrics={'scheduler_job_id':'job','archive_sha256':'c'*64,'binary_sha256':'b'*64}; declared=['direct']
            reference=dict(raw(),backend='direct',gate='supplied',condition_update='incremental',applied_input_change_percent=1,numerical_contract={'multiply':'fp32'})
            phases={('run','exploration',0,'direct'):{'raw':reference}}
            rounds=[]
            for n in range(6):
                order=['direct@rebuild','direct@incremental'] if n%2 else ['direct@incremental','direct@rebuild']; records=[]
                for pos,arm_id in enumerate(order):
                    mode=arm_id.split('@')[1]; data=dict(reference,condition_update=mode); path=update/f'round-{n:02d}-{arm_id}.json';path.write_text(json.dumps(data))
                    records.append({'arm_id':arm_id,'backend':'direct','condition_update':mode,'update_mode':mode,'position':pos,'path':path.name,'status':'pass','raw_sha256':q.sha(path),'complete_us':12000.,'workload_sha256':'a'*64})
                rounds.append({'round_id':f'round-{n:02d}','order':order,'records':records})
            summary={'schema':'cgs.update-ablation.v1','scheduler_job_id':'job','device_uuid':'GPU-a','source_archive_sha256':'c'*64,'binary_sha256':'b'*64,'update_modes':['rebuild','incremental'],'status':'pass','arms':['direct@rebuild','direct@incremental'],'rounds':rounds}
            (update/'summary.json').write_text(json.dumps(summary))
            result=q.update_ablation_records(root,root,config,metrics,expected,'GPU-a',frozen,matrix,declared,phases,'run')
            report=q.update_ablations([{'job':'job','uuid':'GPU-a','updates':result}])[0]
            self.assertEqual(len(report['pairs']),6);self.assertEqual(report['control'],'rebuild');self.assertEqual(report['candidate'],'incremental')
            self.assertIn('separate from specialization',report['interpretation'])
            result['records'][(0,'direct','incremental')]['raw']['applied_input_change_percent']=0
            self.assertIn('unmatched',q.update_ablations([{'job':'job','uuid':'GPU-a','updates':result}])[0]['decision'])
            summary['rounds'][1]['order']=summary['rounds'][0]['order'];(update/'summary.json').write_text(json.dumps(summary))
            with self.assertRaises(ValueError): q.update_ablation_records(root,root,config,metrics,expected,'GPU-a',frozen,matrix,declared,phases,'run')
            expected['parameters']['gate']='ste'
            with self.assertRaisesRegex(ValueError,'not declared'):
                q.update_ablation_records(root,root,config,metrics,expected,'GPU-a',frozen,matrix,declared,phases,'run')

    def test_runtime_snapshot_and_command_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'runtime-wrapper.py').write_text('wrapper bytes')
            for name in ('experiment.json','contract-freeze.json','interference-evidence.json'):
                (root/name).write_text(name)
            wrapper_hash=q.sha(root/'runtime-wrapper.py')
            manifest={'scripts/run_packet.py':wrapper_hash,'.autobench/data/closure.tar.gz':'b'*64,'.autobench/data/frozen-binary':'d'*64, **{name:q.sha(root/name) for name in ('experiment.json','contract-freeze.json','interference-evidence.json')}}
            snapshot_id=q.hashlib.sha256('\n'.join(f'{key}\0{manifest[key]}' for key in sorted(manifest)).encode()).hexdigest()
            snapshot='/snapshot/'+snapshot_id
            (root/'snapshot-manifest.json').write_text(json.dumps({'schema':'cgs.reconstructed-snapshot.v1','snapshot_id':snapshot_id,'files':manifest}))
            command={'working_directory':snapshot, 'result_directory':'/scratch/auto', 'argv':['python3',snapshot+'/scripts/run_packet.py','ste-test','--archive-sha256','b'*64,'--suite','kda','--uuid','GPU-a','--config',snapshot+'/experiment.json','--output','/scratch/auto']}
            (root/'registered-command.json').write_text(json.dumps(command))
            (root/'registration.toml').write_text('[['+'packet]]\nkill_seconds=900\nargv=["python3", "{snapshot}/scripts/run_packet.py"]\n[[packet.resources.scheduler]]\nkind="gpu"\nmode="exclusive"\n[packet.resources.scheduler.selectors]\nresource_id="gpu:host:GPU-a"\n')
            identity={'schema':'cgs.runtime-wrapper.v1','job_id':'job','attempt_id':'auto','snapshot_id':snapshot_id,'snapshot_inputs':manifest,'wrapper_sha256':wrapper_hash,'packet_manifest_sha256':q.sha(root/'registered-command.json'),'snapshot_manifest_sha256':q.sha(root/'snapshot-manifest.json')}
            (root/'runtime-wrapper-identity.json').write_text(json.dumps(identity))
            row={'job_id':'job','attempt_id':'auto','gpu_resource_id':'gpu:host:GPU-a'}
            config={'source_archive_sha256':'b'*64,'binary_sha256':'d'*64}
            self.assertEqual(q.runtime_wrapper(root,row,config,wrapper_hash)['snapshot_id'],snapshot_id)
            changed=dict(manifest); changed['extra.txt']='f'*64
            (root/'snapshot-manifest.json').write_text(json.dumps({'schema':'cgs.reconstructed-snapshot.v1','snapshot_id':snapshot_id,'files':changed}))
            identity['snapshot_manifest_sha256']=q.sha(root/'snapshot-manifest.json')
            (root/'runtime-wrapper-identity.json').write_text(json.dumps(identity))
            with self.assertRaisesRegex(ValueError,'fingerprint mismatch'):
                q.runtime_wrapper(root,row,config,wrapper_hash)
            (root/'snapshot-manifest.json').write_text(json.dumps({'schema':'cgs.reconstructed-snapshot.v1','snapshot_id':snapshot_id,'files':manifest}))
            identity['snapshot_manifest_sha256']=q.sha(root/'snapshot-manifest.json')
            (root/'runtime-wrapper-identity.json').write_text(json.dumps(identity))
            (root/'runtime-wrapper.py').write_text('different wrapper')
            with self.assertRaisesRegex(ValueError,'wrapper identity'):
                q.runtime_wrapper(root,row,config,wrapper_hash)

    def test_v3_sanitizer_adapter_and_backend_binding(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'runtime-wrapper.py').write_text('fake adapter')
            adapter=q.REVIEWED_V3_SANITIZER_ADAPTER_SHA256
            files={'scripts/sanitize_chain.py':adapter,'.autobench/data/closure.tar.gz':'a'*64,'.autobench/data/frozen-binary':'b'*64,'.autobench/data/frozen-build.json':'c'*64}
            snapshot_id=q.hashlib.sha256('\n'.join(f'{k}\0{files[k]}' for k in sorted(files)).encode()).hexdigest();wd='/snapshot/'+snapshot_id
            (root/'snapshot-manifest.json').write_text(json.dumps({'schema':'cgs.reconstructed-snapshot.v1','snapshot_id':snapshot_id,'files':files}))
            command={'working_directory':wd,'result_directory':'/scratch/auto','argv':['python3',wd+'/scripts/sanitize_chain.py','run','--sanitizer','memcheck','--uuid','GPU-a','--backend','cohort-direct-two-tile']}
            (root/'registered-command.json').write_text(json.dumps(command))
            identity={'schema':'cgs.runtime-wrapper.v1','job_id':'job','attempt_id':'auto','snapshot_id':snapshot_id,'snapshot_inputs':files,'wrapper_sha256':adapter,'packet_manifest_sha256':q.sha(root/'registered-command.json'),'snapshot_manifest_sha256':q.sha(root/'snapshot-manifest.json')}
            (root/'runtime-wrapper-identity.json').write_text(json.dumps(identity))
            row={'job_id':'job','attempt_id':'auto'};metric={'archive_sha256':'a'*64,'binary_sha256':'b'*64,'sanitizer':'memcheck','uuid':'GPU-a'}; real_sha=q.sha
            with patch.object(q,'sha',side_effect=lambda p:adapter if Path(p)==root/'runtime-wrapper.py' else real_sha(p)):
                self.assertEqual(q.sanitizer_snapshot(root,row,metric,'c'*64,adapter,'cohort-direct-two-tile')['adapter_sha256'],adapter)
                with self.assertRaisesRegex(ValueError,'command mismatch'):
                    q.sanitizer_snapshot(root,row,metric,'c'*64,adapter,'cohort-shared')
                with self.assertRaisesRegex(ValueError,'identity mismatch'):
                    q.sanitizer_snapshot(root,row,metric,'c'*64,'f'*64,'cohort-direct-two-tile')

    def test_v2_retrospective_child_binding_and_negative_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'kda/supplemental').mkdir(parents=True)
            output=root/'kda/supplemental/round-00-direct.json';output.write_text(json.dumps(raw()))
            (root/'registered-command.json').write_text(json.dumps({'result_directory':'/original/scratch'}))
            expected={'id':'case','input':{'campaign_workload_sha256':'a'*64},'parameters':{'gate':'ste','body':'product','precision':'fp16','condition_update':'none','tile':32,'stages':1}}
            argv=['/build/cgs_study','--mode','campaign','--case','case','--backend','direct','--gate','ste','--body','product','--precision','fp16','--condition-update','none','--tile','32','--stages','1','--output','/original/scratch/kda/supplemental/round-00-direct.json']
            start={'event':'start','invocation_id':'i','pid':42,'device_uuid':'GPU-a','binary_sha256':'b'*64,'backend':'direct','argv':argv,'spawn_lower_ns':1500000000,'spawn_upper_ns':1600000000}
            end={'event':'end','invocation_id':'i','pid':42,'reap_upper_ns':1800000000,'returncode':0,'output_path':argv[-1],'output_sha256':q.sha(output)}
            history=root/'native-child-lifetimes.jsonl'
            def write_history(): history.write_text(json.dumps(start)+'\n'+json.dumps(end)+'\n')
            write_history()
            def sample(phase,t,util=0):
                return {'phase':phase,'sampled_at_epoch_ns':t,'gpu_query_start_ns':t,'gpu_query_end_ns':t+1,'devices':[{'uuid':'GPU-a','utilization_percent':util},{'uuid':'GPU-b','utilization_percent':0}],'processes':[],'issues':[]}
            before=sample('before',1000000000); during=sample('during',2000000000,80); after=sample('after',3000000000)
            during.update(requires_lifetime_attribution=True,attributed_invocation_ids=['i'])
            telemetry={'schema':'cgs.serial-telemetry.v2','scheduler_job_id':'job','uuid':'GPU-a','clean':True,'retrospective_window_bound_ns':1000000000,'lifetime_issues':[],'lifetime_sha256':q.sha(history),'samples':[before,during,after]}
            metrics={'binary_sha256':'b'*64,'binary_path':'/build/cgs_study'};config={'cases':[expected]}
            q.telemetry_v2(telemetry,'job','GPU-a',root,metrics,config,0,10)
            for change in ('future','outside','wrong_uuid','wrong_binary','wrong_pid','wrong_output','foreign','non_target','bound'):
                a,b,t=copy.deepcopy(start),copy.deepcopy(end),copy.deepcopy(telemetry)
                if change=='future': a['spawn_lower_ns']=2100000000; a['spawn_upper_ns']=2200000000; b['reap_upper_ns']=2300000000
                if change=='outside': a['spawn_lower_ns']=1; a['spawn_upper_ns']=2; b['reap_upper_ns']=3
                if change=='wrong_uuid': a['device_uuid']='GPU-other'
                if change=='wrong_binary': a['binary_sha256']='c'*64
                if change=='wrong_pid': b['pid']=99
                if change=='wrong_output': b['output_sha256']='0'*64
                if change=='foreign': t['samples'][1]['processes']=[{'uuid':'GPU-a','pid':999,'owned':False}]
                if change=='non_target': t['samples'][1]['devices'][1]['utilization_percent']=1
                if change=='bound': t['retrospective_window_bound_ns']=2000000000
                history.write_text(json.dumps(a)+'\n'+json.dumps(b)+'\n');t['lifetime_sha256']=q.sha(history)
                with self.subTest(change=change), self.assertRaises(ValueError): q.telemetry_v2(t,'job','GPU-a',root,metrics,config,0,10)

    def test_v3_current_process_query_is_bracketed_by_exact_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'kda/supplemental').mkdir(parents=True)
            out=root/'kda/supplemental/round-00-direct.json';out.write_text(json.dumps(raw()))
            (root/'registered-command.json').write_text(json.dumps({'result_directory':'/original/scratch'}))
            expected={'id':'case','input':{'campaign_workload_sha256':'a'*64},'parameters':{'gate':'ste','body':'product','precision':'fp16','condition_update':'none','tile':32,'stages':1}}
            argv=['/build/cgs_study','--mode','campaign','--case','case','--backend','direct','--gate','ste','--body','product','--precision','fp16','--condition-update','none','--tile','32','--stages','1','--output','/original/scratch/kda/supplemental/round-00-direct.json']
            start={'event':'start','invocation_id':'i','pid':42,'proc_start_ticks':321,'device_uuid':'GPU-a','binary_sha256':'b'*64,'backend':'direct','argv':argv,'spawn_lower_ns':1500000000,'spawn_upper_ns':1600000000}
            end={'event':'end','invocation_id':'i','pid':42,'proc_start_ticks':321,'reap_lower_ns':2500000000,'reap_upper_ns':2600000000,'returncode':0,'output_path':argv[-1],'output_sha256':q.sha(out)}
            history=root/'native-child-lifetimes.jsonl'
            def sample(phase,t,util=0):return {'phase':phase,'sampled_at_epoch_ns':t,'gpu_query_start_ns':t,'gpu_query_end_ns':t+1,'process_query_start_ns':t+2,'process_query_end_ns':t+3,'devices':[{'uuid':'GPU-a','utilization_percent':util},{'uuid':'GPU-b','utilization_percent':0}],'processes':[],'issues':[]}
            before=sample('before',1000000000); during=sample('during',2000000000,80); after=sample('after',3000000000)
            during['processes']=[{'uuid':'GPU-a','pid':42,'owned':True,'invocation_id':'i','proc_start_ticks':321}]
            telemetry={'schema':'cgs.serial-telemetry.v3','scheduler_job_id':'job','uuid':'GPU-a','clean':True,'retrospective_window_bound_ns':1000000000,'lifetime_issues':[],'samples':[before,during,after]}
            metrics={'binary_sha256':'b'*64,'binary_path':'/build/cgs_study'}; config={'cases':[expected]}
            def run(a,b,t):
                history.write_text(json.dumps(a)+'\n'+json.dumps(b)+'\n');t['lifetime_sha256']=q.sha(history)
                return q.telemetry_v2(t,'job','GPU-a',root,metrics,config,0,10,require_process_proof=True)
            self.assertTrue(run(start,end,telemetry)['process_query_identity_verified'])
            for change in ('legacy_schema','missing_query','late_query','pid_reuse','wrong_invocation','wrong_end_tick','missing_reap_lower','future_spawn','foreign'):
                a,b,t=copy.deepcopy(start),copy.deepcopy(end),copy.deepcopy(telemetry); process=t['samples'][1]['processes'][0]
                if change=='legacy_schema':t['schema']='cgs.serial-telemetry.v2'
                if change=='missing_query':del t['samples'][1]['process_query_end_ns']
                if change=='late_query':t['samples'][1]['process_query_end_ns']=2700000000
                if change=='pid_reuse':process['proc_start_ticks']=322
                if change=='wrong_invocation':process['invocation_id']='other'
                if change=='wrong_end_tick':b['proc_start_ticks']=322
                if change=='missing_reap_lower':del b['reap_lower_ns']
                if change=='future_spawn':a['spawn_lower_ns']=2100000000;a['spawn_upper_ns']=2200000000
                if change=='foreign':process['owned']=False
                with self.subTest(change=change),self.assertRaises(ValueError):run(a,b,t)

    def test_compiler_signature_binds_cuda_flags_and_tools(self):
        build={'command':['cmake','-S','/source','-B','/build','-DCMAKE_CUDA_ARCHITECTURES=70','-DCMAKE_CUDA_FLAGS=-lineinfo'], 'toolchain':{'tools':{'nvcc':{'sha256':'a'*64}},'linked_file_sha256':{'/lib/x':'b'*64},'host_pkg_config_versions':{}}}
        expected=q.compiler_signature(build)
        changed=copy.deepcopy(build);changed['command'][-1]='-DCMAKE_CUDA_FLAGS=-fast-math'
        self.assertNotEqual(expected,q.compiler_signature(changed))
        changed=copy.deepcopy(build);changed['toolchain']['tools']['nvcc']['sha256']='c'*64
        self.assertNotEqual(expected,q.compiler_signature(changed))
        changed=copy.deepcopy(build);changed['command'].remove('-DCMAKE_CUDA_ARCHITECTURES=70')
        with self.assertRaisesRegex(ValueError,'compiler flags'):
            q.compiler_signature(changed)

    def test_triage_adapter_binds_snapshot_receipt_and_raw_arm(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            study=Path(tmp); root=study/'runs/packets/job';(root/'triage').mkdir(parents=True)
            row={'job_id':'job','attempt_id':'auto','packet_evidence':'runs/packets/job','build_evidence':'build.json','receipt':'receipt.json','gpu_resource_id':'gpu:host:GPU-a'}
            metrics={'schema':'cgs.packet.v1','suite':'triage','scheduler_job_id':'job','scheduler_attempt_id':'sched','returncode':0,'uuid':'GPU-a','archive_sha256':'c'*64,'binary_sha256':'b'*64}
            (study/'build.json').write_text(json.dumps({'schema':'cgs.build.v1','suite':'triage-binary','target':'cgs_study','archive_sha256':'c'*64,'binary_sha256':'b'*64}))
            (study/'receipt.json').write_text(json.dumps({'attempt':{'attempt_id':'sched'}}));(root/'result.json').write_text(json.dumps({'scheduler_job_id':'job','scheduler_attempt_id':'sched','status':'succeeded','returncode':0}))
            case={'case':'case','backend':'direct','gate':'ste','body':'product','precision':'fp16','condition_update':'none','tile':32,'stages':1,'match_group':'g'}
            config={'incumbent':'direct','candidates':[],'cases':[{'id':'case','input':{'campaign_workload_sha256':'a'*64},'parameters':{k:case[k] for k in ('gate','body','precision','condition_update','tile','stages')}}]}
            (root/'experiment.json').write_text(json.dumps({'schema':'cgs.triage.v1','runs':[case]})); triage_raw=raw();(root/'triage/run-00.json').write_text(json.dumps(triage_raw))
            summary={'schema':'cgs.triage-result.v1','status':'pass','device_uuid':'GPU-a','records':[{'run':0,'validation':'pass','backend':'direct','match_group':'g','workload_sha256':'a'*64}]};(root/'triage/summary.json').write_text(json.dumps(summary))
            wrapper=q.REVIEWED_TRIAGE_ADAPTER_SHA256;(root/'runtime-wrapper.py').write_text('fake reviewed adapter')
            files={'scripts/triage_binary.py':wrapper,'.autobench/data/closure.tar.gz':'c'*64,'.autobench/data/frozen-binary':'b'*64,'.autobench/data/frozen-build.json':'d'*64,'experiment.json':q.sha(root/'experiment.json')}
            snapshot=q.hashlib.sha256('\n'.join(f'{k}\0{files[k]}' for k in sorted(files)).encode()).hexdigest();wd='/snapshot/'+snapshot
            (root/'snapshot-manifest.json').write_text(json.dumps({'schema':'cgs.reconstructed-snapshot.v1','snapshot_id':snapshot,'files':files}))
            cmd={'working_directory':wd,'result_directory':'/scratch/auto','argv':['python3',wd+'/scripts/triage_binary.py','run','--config',wd+'/experiment.json','--uuid','GPU-a','--output','/scratch/auto/triage-packet']};(root/'registered-command.json').write_text(json.dumps(cmd))
            ident={'schema':'cgs.runtime-wrapper.v1','snapshot_id':snapshot,'snapshot_inputs':files,'job_id':'job','attempt_id':'auto','wrapper_sha256':wrapper,'packet_manifest_sha256':q.sha(root/'registered-command.json'),'snapshot_manifest_sha256':q.sha(root/'snapshot-manifest.json')};(root/'runtime-wrapper-identity.json').write_text(json.dumps(ident))
            nested=root/'triage-packet';(nested/'triage').mkdir(parents=True)
            for name,content in [('metrics.json',metrics),('experiment.json',{'schema':'cgs.triage.v1','runs':[case]}),('triage/summary.json',summary),('triage/run-00.json',triage_raw)]:
                (root/name).write_text(json.dumps(content));(nested/name).write_text(json.dumps(content))
            names=['metrics.json','experiment.json','triage/summary.json','triage/run-00.json']
            (root/'triage-aliases.json').write_text(json.dumps({'schema':'cgs.triage-aliases.v1','files':[{'original':'triage-packet/'+name,'alias':name,'sha256':q.sha(root/name)} for name in names]}))
            (root/'registration.toml').write_text('[[packet]]\nkill_seconds=600\nargv=["python3", "{snapshot}/scripts/triage_binary.py"]\n[[packet.resources.scheduler]]\nkind="gpu"\nmode="exclusive"\n[packet.resources.scheduler.selectors]\nresource_id="gpu:host:GPU-a"\n')
            original={'target':'cgs_study','archive_sha256':'c'*64,'binary_sha256':'b'*64}; real_sha=q.sha
            with patch.object(q,'sha',side_effect=lambda p:wrapper if Path(p)==root/'runtime-wrapper.py' else real_sha(p)), patch.object(q,'original_build',return_value=(original,'d'*64)):
                result=q.verify_triage_adapter(study,row,metrics,summary,config)
                self.assertEqual(result['triage_arm_records'][0]['backend'],'direct')
                bad=dict(triage_raw,device_uuid='GPU-other');(root/'triage/run-00.json').write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError,'triage alias differs'):
                    q.verify_triage_adapter(study,row,metrics,summary,config)

    def test_unreserved_audit_exact_window_and_windowless_cancellation(self):
        with tempfile.TemporaryDirectory() as tmp:
            study=Path(tmp); job='aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa';folder=study/'runs/failures'/job;folder.mkdir(parents=True)
            row={'job_id':job,'attempt_id':'autobench','packet_evidence':'runs/failures/'+job,'kind':'unreserved_auto_submission','state':'settled','terminal_state':'cancelled','cleanup_verified':True,'reserved_seconds':60,'charged_seconds':10.0}
            attempt={'attempt_id':'scheduler','cleanup_verified':True,'created_at':'2026-11-02T00:00:00Z','finished_at':'2026-11-02T00:00:10Z','execution_context':{'allocations':[{'resource_id':'gpu:host:GPU-a','mode':'exclusive'}]}}
            receipt={'id':job,'job_id':job,'state':'cancelled','cleanup_verified':True,'attempt_id':'scheduler','attempt':attempt,'spec':{'argv':['python','/packets/autobench.json']}}
            path=folder/'receipt.json';path.write_text(json.dumps(receipt))
            windows,audits=q.ledger_windows(study,{'reservations':[row]},return_audits=True)
            self.assertEqual(windows[job][2],'GPU-a');self.assertIn('unreserved',audits[0]['classification'])
            for field in ('cleanup','allocation','charge','attempt'):
                r=copy.deepcopy(receipt); candidate=copy.deepcopy(row)
                if field=='cleanup': r['cleanup_verified']=False
                if field=='allocation': r['attempt']['execution_context']['allocations'][0]['mode']='shared'
                if field=='charge': candidate['charged_seconds']=9.0
                if field=='attempt': r['attempt_id']='other'
                path.write_text(json.dumps(r))
                with self.subTest(field=field),self.assertRaises(ValueError):q.unreserved_audit(study,candidate)
            cancelled={'id':job,'job_id':job,'state':'cancelled','cleanup_verified':True,'attempt_id':None,'lease_id':None,'lease_binding':None,'spec':{'argv':['python','/packets/autobench.json']}}
            row['charged_seconds']=0;path.write_text(json.dumps(cancelled));window,audit=q.unreserved_audit(study,row)
            self.assertIsNone(window);self.assertIn('no GPU window inferred',audit['window_status'])
            for field,value in [('attempt_id','scheduler'),('lease_id','lease'),('lease_binding',{'x':1})]:
                modified=dict(cancelled,**{field:value});path.write_text(json.dumps(modified))
                with self.subTest(field=field),self.assertRaises(ValueError):q.unreserved_audit(study,row)
            path.write_text(json.dumps(cancelled));row['charged_seconds']=1
            with self.assertRaises(ValueError):q.unreserved_audit(study,row)

    def test_reserved_windowless_cancellation_requires_no_attempt_or_lease(self):
        with tempfile.TemporaryDirectory() as tmp:
            study=Path(tmp);row={'job_id':'job','attempt_id':'auto','receipt':'receipt.json','terminal_state':'cancelled','worker_status':'not_started','charged_seconds':0}
            receipt={'id':'job','job_id':'job','state':'cancelled','cleanup_verified':True,'attempt_id':None,'lease_id':None,'lease_binding':None,'spec':{'argv':['python','/packets/auto.json']}}
            path=study/'receipt.json';path.write_text(json.dumps(receipt))
            self.assertIn('no GPU window inferred',q.reserved_windowless_cancellation(study,row)['classification'])
            for key,value in [('attempt_id','scheduler'),('lease_id','lease'),('lease_binding',{'x':1}),('cleanup_verified',False)]:
                path.write_text(json.dumps(dict(receipt,**{key:value})))
                with self.subTest(key=key),self.assertRaises(ValueError):q.reserved_windowless_cancellation(study,row)
            path.write_text(json.dumps(receipt));row['charged_seconds']=1
            with self.assertRaises(ValueError):q.reserved_windowless_cancellation(study,row)

    def test_unsettled_ledger_fails_closed(self):
        with self.assertRaisesRegex(ValueError,'unsettled'):
            q.ledger_windows(Path('/tmp'), {'reservations':[{'state':'unknown'}]})


class PacketJoinTests(unittest.TestCase):
    """Construct all retained joins in a temporary tree; hash actual JSON bytes."""
    def test_full_packet_and_tampered_supplement(self):
        from unittest.mock import patch
        import csv
        import datetime
        with tempfile.TemporaryDirectory() as tmp:
            study = Path(tmp)
            def write(path, data):
                path = study / path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data))
                return path
            frozen = write('frozen.json', {'format':'cgs.contract-freeze.v1', 'workloads':[{'id':'case','seed':1,'reuse_horizon':16,'update_policy':'update_every_step','requested_leaf_change_percent':1}]})
            matrix = write('matrix.json', {'workloads':['case'], 'scenarios':[{'workload':0,'input':'synthetic','gate':'ste','body':'product','precision':'fp16','updates':['none']}], 'capacity_rejection':'fixture'})
            expected = {'id':'case','input':{'campaign_workload_sha256':'a'*64},'parameters':{'gate':'ste','body':'product','precision':'fp16','condition_update':'none','tile':32,'stages':1}}
            arms = ['scalar','bitmap','direct','cohort-direct']
            config = {'schema':'cgs.kda-config.v1','source_archive_sha256':'b'*64,'binary_sha256':'','cases':[expected], 'incumbent':'scalar','candidates':arms[1:],'contract_freeze_sha256':q.sha(frozen)}
            binary = study/'runs/builds'/'placeholder'; binary.parent.mkdir(parents=True); binary.write_bytes(b'binary')
            binary_hash=q.sha(binary); binary.unlink(); binary=study/'runs/builds'/binary_hash/'binary'; binary.parent.mkdir(); binary.write_bytes(b'binary')
            config['binary_sha256']=binary_hash
            job='job'; attempt='scheduler-attempt'; triage_job='triage'
            def receipt(name, attempt_id, start, end):
                return {'id':name,'state':'succeeded','cleanup_verified':True,'attempt':{'attempt_id':attempt_id,'cleanup_verified':True,'created_at':start,'finished_at':end,'execution_context':{'allocations':[{'resource_id':'gpu:host:GPU-a','mode':'exclusive'}]}}}
            start='2026-11-02T00:00:00Z'; end='2026-11-02T00:00:10Z'
            receipt_path=write('runs/receipts/job.json', receipt(job,attempt,start,end))
            triage_receipt=write('runs/receipts/triage.json',receipt(triage_job,'triage-attempt','2026-11-01T00:00:00Z','2026-11-01T00:00:10Z'))
            row={'job_id':job,'attempt_id':'autobench-id','state':'settled','serialized_confirmation':True,'worker_status':'succeeded','reserved_seconds':30,'expected_gpu_resource_id':'gpu:host:GPU-a','gpu_resource_id':'gpu:host:GPU-a','receipt':'runs/receipts/job.json','packet_evidence':'runs/packets/job','build_evidence':'build.json'}
            triage_row=dict(row,job_id=triage_job,receipt='runs/receipts/triage.json',serialized_confirmation=False)
            write('runs/campaign-budget.json', {'reservations':[row,triage_row]})
            write('build.json',{'schema':'cgs.build.v1','archive_sha256':'b'*64,'binary_sha256':binary_hash})
            tmetrics=write('runs/packets/triage/metrics.json',{'success':True,'archive_sha256':'b'*64,'binary_sha256':binary_hash,'uuid':'GPU-a'})
            write('runs/packets/triage/triage/summary.json',{'status':'pass'})
            config['triage']={'job_id':triage_job,'receipt_sha256':q.sha(triage_receipt),'packet_metrics_sha256':q.sha(tmetrics)}
            fallback=write('runs/packets/job/interference-evidence.json',{'schema':'cgs.serialized-fallback.v1','device_uuid':'GPU-a','source_archive_sha256':'b'*64,'binary_sha256':binary_hash,'workload_sha256':'a'*64,'triage_job_id':'triage','controller_serialization_required':True,'all_device_telemetry_required':True})
            config['interference_attestation']={'sha256':q.sha(fallback)}
            write('runs/packets/job/experiment.json',config); write('runs/packets/job/kda/experiment.json',config)
            write('runs/packets/job/contract-freeze.json',json.loads(frozen.read_text()))
            write('runs/packets/job/snapshot-manifest.json',{'files':{'.autobench/data/frozen-build.json':q.sha(study/'build.json')}})
            write('runs/packets/job/metrics.json',{'schema':'cgs.packet.v1','scheduler_job_id':job,'scheduler_attempt_id':attempt,'suite':'kda','uuid':'GPU-a','archive_sha256':'b'*64,'binary_sha256':binary_hash,'success':True,'serial_telemetry_clean':True,'returncode':0})
            write('runs/packets/job/result.json',{'scheduler_job_id':job,'scheduler_attempt_id':attempt,'status':'succeeded','returncode':0})
            write('runs/packets/job/kda/metrics.json',{'schema':'cgs.kda-packet.v1','scheduler_job_id':job,'uuid':'GPU-a','exit_code':0,'engine_result':{'status':'complete'}})
            guard=samples(); guard['scheduler_job_id']=job
            epoch=int(datetime.datetime.fromisoformat(start.replace('Z','+00:00')).timestamp())*10**9
            for i, sample in enumerate(guard['samples']): sample['sampled_at_epoch_ns']=epoch+(i+1)*10**9
            write('runs/packets/job/serial-telemetry.json',guard)
            workspace=study/'runs/packets/job/kda/kda-workspace'
            rows=[]
            environment={'device_uuid':'GPU-a'}
            case=dict(expected,workload_sha256=q.digest_json(expected))
            for index,arm in enumerate(arms):
                base=Path('runs/packets/job/kda/kda-workspace/runs/kda-experiments/attempts/run/stages')/arm
                x=dict(raw(),backend=arm)
                valid=base/'validation'
                write(valid/'requests/reference-0.json',dict(x,backend='scalar'));write(valid/'requests/candidate-0.json',x)
                write(valid/'responses/v.json',{'status':'ok','cases':[{'status':'pass','observed_workload_sha256':'a'*64}]})
                stage=base/'exploration/round-000/casehash'
                request={'request_id':'req','round':0,'phase':'exploration','candidate':{'id':arm},'case':case,'environment':environment}
                write(stage/'requests/req.json',request);write(stage/'requests/campaign.json',x)
                response={'status':'ok','request_id':'req','case_id':'case','workload_sha256':case['workload_sha256'],'observed_workload_sha256':'a'*64,'samples':[12000.]}
                result=write(stage/'responses/req.json',response)
                rows.append({'run_id':'run','candidate_id':arm,'phase':'exploration','case_id':'case','round':'0','metric':'complete_operational_horizon','unit':'us','direction':'minimize','samples_json':'[12000.0]','workload_sha256':case['workload_sha256'],'environment_fingerprint':q.digest_json(environment),'result_sha256':q.sha(result)})
            workspace.mkdir(parents=True,exist_ok=True)
            with (workspace/'benchmark.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            rounds=[]
            for n in range(6):
                order=arms if n%2==0 else list(reversed(arms)); records=[]
                for position,arm in enumerate(order):
                    data=dict(raw(),backend=arm)
                    if arm=='cohort-direct': data['recurrent_complete_horizon_ms']=[3.]
                    name=f'round-{n:02d}-{arm}.json'; path=write(Path('runs/packets/job/kda/supplemental')/name,data)
                    records.append({'backend':arm,'position':position,'path':name,'status':'pass','raw_sha256':q.sha(path),'complete_us':1000*(data['cold_execution_preparation_ms']+data['recurrent_complete_horizon_ms'][0]+data['output_recovery_ms'][0]),'workload_sha256':'a'*64})
                rounds.append({'round_id':f'round-{n:02d}','order':order,'records':records})
            write('runs/packets/job/kda/supplemental/summary.json',{'schema':'cgs.supplemental.v1','scheduler_job_id':job,'device_uuid':'GPU-a','source_archive_sha256':'b'*64,'binary_sha256':binary_hash,'arms':arms,'status':'pass','rounds':rounds})
            with patch.object(q,'verify_archive',return_value={'archive_sha256':'b'*64, 'wrapper_sha256':'c'*64}), patch.object(q,'runtime_wrapper',return_value={'runtime_wrapper_sha256':'c'*64}), patch.object(q,'telemetry_v2',return_value={'lifetime_sha256':'e'*64}):
                result=q.packet(study,row,{'job':(q.stamp(start),q.stamp(end),'GPU-a')},frozen,matrix,{})
                self.assertEqual(q.comparisons([result])[0]['decision'],'paired threshold met; study-wide recommendation pending sanitizer coverage')
                path=study/'runs/packets/job/kda/supplemental/round-00-direct.json'
                data=json.loads(path.read_text());data['output_recovery_ms']=[100];path.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError,'raw hash mismatch'):
                    q.packet(study,row,{'job':(q.stamp(start),q.stamp(end),'GPU-a')},frozen,matrix,{})


if __name__ == '__main__': unittest.main()
