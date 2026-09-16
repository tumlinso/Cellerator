#!/usr/bin/env python3
"""Disposable structural, provenance and failure tests; never a native/GPU pass."""
from __future__ import annotations
import copy
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
sys.dont_write_bytecode=True
import common
import bootstrap
import capture_worktrees
import run_gate
import validate_package


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def package_copy(self):
        p=self.root/'package';shutil.copytree(common.PACKAGE,p);return p
    def change_json(self,p,name,fn):
        target=p/name; data=common.load(target);fn(data);target.write_text(json.dumps(data))
    def test_structure(self):
        self.assertEqual(validate_package.validate(common.PACKAGE,manifest=False)['records'],9)
    def test_cycle(self):
        with self.assertRaises(ValueError):validate_package.acyclic({'a':{'b'},'b':{'a'}})
    def test_unknown_dependency(self):
        with self.assertRaises(ValueError):validate_package.acyclic({'a':{'b'}})
    def test_drop_legacy_obligation(self):
        p=self.package_copy();self.change_json(p,'machine/legacy-disposition.json',lambda d:d[0].update(requirements=[]))
        with self.assertRaises(ValueError):validate_package.validate(p,manifest=False)
    def test_old_task_cannot_be_retired_as_done(self):
        p=self.package_copy();self.change_json(p,'machine/legacy-disposition.json',lambda d:next(x for x in d if x['observed_status']=='done').update(disposition='superseded'))
        with self.assertRaises(ValueError):validate_package.validate(p,manifest=False)
    def test_microtask_provider_field_forbidden(self):
        p=self.package_copy();self.change_json(p,'machine/native-plan.json',lambda d:d['tasks'][0].update(model='expensive'))
        with self.assertRaises(ValueError):validate_package.validate(p,manifest=False)
    def test_integrator_cannot_wait_for_all_at_claim_time(self):
        p=self.package_copy();data=common.load(p/'machine/native-plan.json');o=common.load(p/'machine/outcomes.json')
        final=next(t for t in data['tasks'] if t['id'].endswith(('-QUALIFY','-ACCEPT')))
        self.assertLess(len(final['depends_on']),len(o[final['id']]['completion_requires']))
    def test_coordinator_not_aggregate(self):
        plan=common.load(common.PACKAGE/'machine/native-plan.json')
        root=plan['runs'][0]['root_task_id'];controller=next(t for t in plan['tasks'] if t['id'].endswith('-CONTROL'))
        self.assertNotEqual(controller['kind'],'epic');self.assertNotIn(root,[x['task_id'] for x in controller['depends_on']])
    def test_exact_compaction_not_optional(self):
        req=next(r for r in common.load(common.PACKAGE/'machine/requirements.json') if r['id']=='CE-SUPPORT')
        self.assertEqual(req['classification'],'required')
    def test_manifest_tamper(self):
        p=self.root/'sealed';p.mkdir();(p/'a').write_text('one');(p/'MANIFEST.sha256').write_text(common.sha(p/'a')+'  a\n')
        common.verify_manifest(p);(p/'a').write_text('two')
        with self.assertRaises(ValueError):common.verify_manifest(p)
    def test_manifest_extra_file(self):
        p=self.root/'sealed';p.mkdir();(p/'a').write_text('one');(p/'MANIFEST.sha256').write_text(common.sha(p/'a')+'  a\n');(p/'b').write_text('extra')
        with self.assertRaises(ValueError):common.verify_manifest(p)
    def test_manifest_symlink(self):
        p=self.root/'sealed';p.mkdir();(p/'a').symlink_to('/etc/hostname');(p/'MANIFEST.sha256').write_text(common.sha(p/'a')+'  a\n')
        with self.assertRaises(ValueError):common.verify_manifest(p)
    def test_external_evidence(self):
        with self.assertRaises(ValueError):common.external(self.root/'repo'/'bad',[self.root/'repo'])
    def test_exclusive_receipt_write(self):
        path=self.root/'receipt';common.write_new(path,{'a':1})
        with self.assertRaises(FileExistsError):common.write_new(path,{'a':2})
        self.assertEqual(common.load(path),{'a':1})
    def inventory(self,props=None,command=None):
        return {'tests':[{'name':'real','command':command or ['/bin/true'],'properties':props or []}]}
    def test_empty_inventory(self):
        with self.assertRaises(ValueError):run_gate.inventory_ok(self.inventory(),[])
    def test_missing_test(self):
        with self.assertRaises(ValueError):run_gate.inventory_ok(self.inventory(),['absent'])
    def test_disabled_test(self):
        with self.assertRaises(ValueError):run_gate.inventory_ok(self.inventory([{'name':'DISABLED','value':True}]),['real'])
    def test_skip_as_success(self):
        with self.assertRaises(ValueError):run_gate.inventory_ok(self.inventory([{'name':'SKIP_RETURN_CODE','value':77}]),['real'])
    def test_nested_lock_runner(self):
        with self.assertRaises(ValueError):run_gate.inventory_ok(self.inventory(command=['python3','/x/run_gpu_gate.py']),['real'])
    def test_valid_inventory(self):self.assertEqual(set(run_gate.inventory_ok(self.inventory(),['real'])),{'real'})
    def test_junit_success(self):
        f=self.root/'j.xml';f.write_text('<testsuite><testcase name="real" status="run"/></testsuite>');run_gate.junit_ok(f,['real'])
    def test_junit_skipped(self):
        f=self.root/'j.xml';f.write_text('<testsuite><testcase name="real"><skipped/></testcase></testsuite>')
        with self.assertRaises(ValueError):run_gate.junit_ok(f,['real'])
    def test_junit_duplicate(self):
        f=self.root/'j.xml';f.write_text('<testsuite><testcase name="real"/><testcase name="real"/></testsuite>')
        with self.assertRaises(ValueError):run_gate.junit_ok(f,['real'])
    def test_junit_unexecuted(self):
        f=self.root/'j.xml';f.write_text('<testsuite><testcase name="real" status="notrun"/></testsuite>')
        with self.assertRaises(ValueError):run_gate.junit_ok(f,['real'])
    def test_unrelated_binding_edit_does_not_invalidate(self):
        a={'tasks':{'A':{'x':1},'B':{'x':2}},'shared':'s'};b=copy.deepcopy(a);b['tasks']['B']['x']=3
        self.assertEqual(run_gate.selected_binding_digest(a,'A'),run_gate.selected_binding_digest(b,'A'))
    def test_relevant_binding_edit_invalidates(self):
        a={'tasks':{'A':{'x':1}},'shared':'s'};b=copy.deepcopy(a);b['tasks']['A']['x']=2
        self.assertNotEqual(run_gate.selected_binding_digest(a,'A'),run_gate.selected_binding_digest(b,'A'))
    def test_done_without_successful_disposition(self):self.assertFalse(common.successful({'status':'done','result':None}))
    def test_blocked_completion(self):
        s=types.SimpleNamespace(todo_tables={'tasks':[{'id':'CE-X','status':'in_progress','result':None}]},todo_revision=1)
        with self.assertRaises(ValueError):common.completed_prerequisites({'completion_requires':['CE-X']},{'cellerator':s})
    def git_repo(self):
        r=self.root/'repo';r.mkdir();subprocess.run(['git','init','-q',str(r)],check=True)
        for k,v in [('user.name','Fixture'),('user.email','fixture@example.invalid')]:subprocess.run(['git','-C',str(r),'config',k,v],check=True)
        (r/'source.cc').write_text('original\n');subprocess.run(['git','-C',str(r),'add','.'],check=True);subprocess.run(['git','-C',str(r),'commit','-qm','fixture'],check=True);return r
    def test_source_dirty_rejected(self):
        r=self.git_repo();(r/'source.cc').write_text('changed\n')
        with self.assertRaises(ValueError):common.clean_source(r)
    def test_generated_projection_exclusion_is_bounded(self):
        r=self.git_repo();(r/'todos.md').write_text('generated\n');self.assertEqual(common.clean_source(r)['excluded_generated_projection_changes'],['todos.md'])
        (r/'evil.cu').write_text('uncommitted')
        with self.assertRaises(ValueError):common.clean_source(r)
    def test_git_worktree_catalog_handles_spaces(self):
        r=self.git_repo();w=self.root/'work tree';subprocess.run(['git','-C',str(r),'worktree','add','-q','-b','codex/ce-nf1-fixture',str(w)],check=True)
        self.assertIn(str(w),[x['worktree'] for x in capture_worktrees.catalog(r)])
    def test_dirty_capture_bytes_and_symlink(self):
        r=self.git_repo();(r/'source.cc').write_bytes(b'edited\x00bytes');(r/'reporter').symlink_to('/not/read')
        self.assertEqual(set(capture_worktrees.dirty_files(r)),{'source.cc','reporter'})
        self.assertEqual(capture_worktrees.material(r/'reporter'),('symlink',b'/not/read'))
    def test_native_gate_binding_argv_is_exact(self):
        p=common.load(common.PACKAGE/'machine/native-plan.json')
        self.assertTrue(all(g['argv'][-2:]==['--task',t['id']] for t in p['tasks'] for g in t.get('gates',[])))
    def test_unconfigured_binding_stops_before_host_or_device_access(self):
        meta=common.load(common.PACKAGE/'machine/package.json');path=self.root/'binding.json'
        common.write_new(path,{'pair_id':meta['pair_id'],'project':meta['project'],'tasks':{}})
        task=meta['baseline']['prefix']+'-ADOPT'
        with patch.object(run_gate,'verify_manifest'),patch.object(run_gate,'host_context',side_effect=AssertionError('host touched')),patch('sys.argv',['run_gate.py','--task',task,'--bindings',str(path)]):
            with self.assertRaisesRegex(ValueError,'unconfigured'):run_gate.main()
    def import_fixture(self):
        old={'id':'OLD','kind':'task','status':'done','result':'validated','title':'kept'}
        task={'id':'NEW','kind':'task','title':'new','objective':'real outcome','gates':[{'id':'G','argv':['actual','--binding','x']}]}
        plan={'tasks':[task],'runs':[{'id':'R','root_task_id':'NEW','lanes':[{'id':'L','role':'coordinator','tasks':['NEW'],'workspace':{'mode':'read_shared'}}]}]}
        snapshot=types.SimpleNamespace(project_uuid='p',todo_revision=2,todo_tables={
            'tasks':[old,{**task,'status':'planned'}], 'workflow_runs':[{'id':'R','root_task_id':'NEW'}],
            'workflow_lanes':[{'id':'L','run_id':'R','role':'coordinator','workspace_mode':'read_shared'}],
            'workflow_lane_tasks':[{'lane_id':'L','task_id':'NEW','position':0,'state':'queued'}],
            'gates':[{'id':'G','task_id':'NEW','config_json':json.dumps({'argv':['actual','--binding','x']})}]})
        return {'proposal':{'proposed_change':plan},'preserved_task_rows':{'OLD':common.sha_bytes(common.canonical(old))}},snapshot
    def test_import_verifies_actual_records(self):
        review,snapshot=self.import_fixture();self.assertEqual(bootstrap.verify_import(review,snapshot)['status'],'verified')
    def test_import_detects_old_record_mutation(self):
        review,snapshot=self.import_fixture();snapshot.todo_tables['tasks'][0]['title']='changed'
        with self.assertRaises(ValueError):bootstrap.verify_import(review,snapshot)
    def test_import_detects_wrong_gate_binding(self):
        review,snapshot=self.import_fixture();snapshot.todo_tables['gates'][0]['config_json']='{}'
        with self.assertRaises(ValueError):bootstrap.verify_import(review,snapshot)
    def test_import_detects_dispatched_lane(self):
        review,snapshot=self.import_fixture();snapshot.todo_tables['workflow_lane_tasks'][0]['state']='active'
        with self.assertRaises(ValueError):bootstrap.verify_import(review,snapshot)
    def test_apply_uses_saved_proposal_once_and_preserves_unknown_outcome(self):
        meta=common.load(common.PACKAGE/'machine/package.json');rp=self.root/'review.json'
        review={'project':meta['project'],'pair_id':meta['pair_id'],'expires_at':(dt.datetime.now(dt.timezone.utc)+dt.timedelta(hours=1)).isoformat(),
                'packages':{},'runtime':{},'roots':{'cellerator':str(self.root/'ce'),'glasshelix':str(self.root/'gh')},'proposal':{'sentinel':'exact reviewed bytes'}}
        common.write_new(rp,review);calls=[]
        def fail(cfg,project,proposal):calls.append(proposal);raise RuntimeError('unknown outcome')
        module=types.ModuleType('project_control.mutation');module.apply_proposal=fail
        with patch.dict(sys.modules,{'project_control.mutation':module}),patch.object(bootstrap,'host_context',return_value=(None,{p:Path(r) for p,r in review['roots'].items()},{})),patch.object(bootstrap,'package_pair',return_value={}),patch.object(bootstrap,'runtime_stamp',return_value={}):
            with self.assertRaises(RuntimeError):bootstrap.apply(rp,common.sha(rp))
            with self.assertRaises(FileExistsError):bootstrap.apply(rp,common.sha(rp))
        self.assertEqual(calls,[review['proposal']]);self.assertTrue(Path(str(rp)+'.uncertain.json').exists())
    def test_apply_wrong_confirmation_never_calls_mutation(self):
        rp=self.root/'review.json';common.write_new(rp,{})
        module=types.ModuleType('project_control.mutation');module.apply_proposal=lambda *a:(_ for _ in ()).throw(AssertionError('called'))
        with patch.dict(sys.modules,{'project_control.mutation':module}):
            with self.assertRaises(ValueError):bootstrap.apply(rp,'wrong')
    def test_expired_review_rejected(self):
        meta=common.load(common.PACKAGE/'machine/package.json');rp=self.root/'review.json'
        common.write_new(rp,{'project':meta['project'],'pair_id':meta['pair_id'],'expires_at':'2000-01-01T00:00:00+00:00'})
        module=types.ModuleType('project_control.mutation');module.apply_proposal=lambda *a:None
        with patch.dict(sys.modules,{'project_control.mutation':module}):
            with self.assertRaises(ValueError):bootstrap.apply(rp,common.sha(rp))

if __name__=='__main__':unittest.main(verbosity=2)
