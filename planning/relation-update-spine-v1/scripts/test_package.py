#!/usr/bin/env python3
"""Adversarial tests for package/approval gates. All native-service doubles are local.
No test contacts Todo authority, launches agents, runs CUDA, or applies a real plan.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import copy,hashlib,io,json,subprocess,sys,tempfile,time,unittest,shutil,shutil,shutil
sys.dont_write_bytecode=True
from compile_plan import assemble,PACKAGE,PLAN
from validate_package import validate_graph,validate,safe_rel
import todo_bootstrap as boot
import native_bridge as bridge
import run_gate as gate
import run_sanitizer as sanitizer
MASTER=json.loads((PACKAGE/'machine/proposed_todos.json').read_text())
PLAN_DATA=assemble(MASTER)
def receipt():
    h=hashlib.sha256((json.dumps(PLAN_DATA,sort_keys=True,separators=(',',':'),ensure_ascii=False)+'\n').encode()).hexdigest()
    c={'workspace_id':'cellerator','project_uuid':boot.UUID,'todo_revision':6877,'workflow_revision':6877,
       'todo_semantic_authority_fingerprint':'same','workflow_authority_fingerprint':'same',
       'repository_commits':{'cellerator':boot.BASE},'observed_at':'2026-09-06T14:59:06Z','provider_skew':{},
       'worktrees':{'main':{'head':boot.BASE,'working_tree_fingerprint':'clean'}},'run_id':'compat-v2'}
    return {'status':'validated','valid':True,'project_uuid':boot.UUID,'revision':6877,'plan_digest':h,
            'would_add':sorted(t['id'] for t in PLAN_DATA['tasks']),'would_modify':[],'warnings':[],
            'current_observation_preconditions':c,'runtime_identity':{'test_only':True}}
class GraphTests(unittest.TestCase):
    def bad(self,change):
        m=copy.deepcopy(MASTER);change(m)
        with self.assertRaises((ValueError,KeyError)):validate_graph(m)
    def test_valid_graph(self):self.assertEqual(validate_graph(MASTER)['leaf_tasks'],34)
    def test_all_projections(self):self.assertEqual(validate(PACKAGE,integrity=False)['todo_records'],35)
    def test_authority_flag(self):self.bad(lambda m:m.update(authority_to_apply=True))
    def test_wrong_schema(self):self.bad(lambda m:m.update(native_schema_version=2))
    def test_duplicate_task(self):self.bad(lambda m:m['tasks'].append(copy.deepcopy(m['tasks'][0])))
    def test_unknown_dependency(self):self.bad(lambda m:m['tasks'][0]['native_record'].update(depends_on=[{'type':'task','task_id':'CE-RU1-MISSING'}]))
    def test_self_dependency(self):self.bad(lambda m:m['tasks'][0]['native_record'].update(depends_on=[{'type':'task','task_id':'CE-RU1-I01'}]))
    def test_dependency_cycle(self):self.bad(lambda m:m['tasks'][0]['native_record'].update(depends_on=[{'type':'task','task_id':'CE-RU1-C01'}]))
    def test_queue_cycle(self):self.bad(lambda m:m['run']['lanes'][1]['tasks'].reverse())
    def test_duplicate_lane_assignment(self):self.bad(lambda m:m['run']['lanes'][2]['tasks'].append('CE-RU1-C01'))
    def test_missing_lane_assignment(self):self.bad(lambda m:m['run']['lanes'][2]['tasks'].pop())
    def test_unordered_scope_overlap(self):
        def change(m):
            a=next(t for t in m['tasks'] if t['id']=='CE-RU1-F01')
            a['write_scope'].append('src/compute/operation/prepared_relation.cu')
            a['native_record']['scope']['exclusive_paths']=a['write_scope']
        self.bad(change)
    def test_missing_checkpoint(self):self.bad(lambda m:m['barriers'][0]['requirements'][0].update(id='NO-SUCH-CHECKPOINT'))
    def test_precompleted_task(self):self.bad(lambda m:m['tasks'][0]['native_record'].update(status='completed'))
    def test_active_run(self):self.bad(lambda m:m['run'].update(status='active'))
    def test_path_escape(self):
        for s in ['../x','/tmp/x','a/../b','a//b','a\\b','./a']:
            with self.subTest(s=s),self.assertRaises(ValueError):safe_rel(s)
class ApprovalTests(unittest.TestCase):
    def test_valid_native_receipt(self):self.assertEqual(boot.check_native(receipt(),PLAN_DATA)['todo_revision'],6877)
    def test_wrong_uuid(self):
        r=receipt();r['project_uuid']='other'
        with self.assertRaises(ValueError):boot.check_native(r,PLAN_DATA)
    def test_collision(self):
        r=receipt();r['would_modify']=['CE-RU1-C01']
        with self.assertRaises(ValueError):boot.check_native(r,PLAN_DATA)
    def test_partial_reapply(self):
        r=receipt();r['would_add'].pop()
        with self.assertRaises(ValueError):boot.check_native(r,PLAN_DATA)
    def test_native_warning(self):
        r=receipt();r['warnings']=['needs review']
        with self.assertRaises(ValueError):boot.check_native(r,PLAN_DATA)
    def test_wrong_digest(self):
        r=receipt();r['plan_digest']='wrong'
        with self.assertRaises(ValueError):boot.check_native(r,PLAN_DATA)
    def test_review_expiry(self):
        old={'kind':'ce-ru1-reviewed-preview-v1','created_unix':0,'source':{},'native_response':receipt()}
        with self.assertRaises(ValueError):boot.check_preview(old,{},receipt(),PLAN_DATA,now=3601)
    def test_reviewed_revision_changes(self):
        old={'kind':'ce-ru1-reviewed-preview-v1','created_unix':100,'source':{},'native_response':receipt()}
        r=receipt();r['revision']+=1;r['current_observation_preconditions']['todo_revision']+=1;r['current_observation_preconditions']['workflow_revision']+=1
        with self.assertRaises(ValueError):boot.check_preview(old,{},r,PLAN_DATA,now=101)
    def test_unrelated_worktree_change(self):
        old={'kind':'ce-ru1-reviewed-preview-v1','created_unix':100,'source':{},'native_response':receipt()}
        r=receipt();r['current_observation_preconditions']['worktrees']['main']['head']='changed'
        with self.assertRaises(ValueError):boot.check_preview(old,{},r,PLAN_DATA,now=101)
    def test_runtime_drift(self):
        old={'kind':'ce-ru1-reviewed-preview-v1','created_unix':100,'source':{},'native_response':receipt()}
        r=receipt();r['runtime_identity']={'different':True}
        with self.assertRaises(ValueError):boot.check_preview(old,{},r,PLAN_DATA,now=101)
    def test_only_timestamps_may_advance(self):
        old={'kind':'ce-ru1-reviewed-preview-v1','created_unix':100,'source':{},'native_response':receipt()}
        r=receipt();r['current_observation_preconditions']['observed_at']='later'
        boot.check_preview(old,{},r,PLAN_DATA,now=101)
    def test_apply_requires_authorization_before_import(self):
        with patch.object(sys,'argv',['native_bridge.py','apply']),patch.object(bridge,'runtime') as r:
            with self.assertRaises(ValueError):bridge.main()
            r.assert_not_called()
    def test_apply_uses_reviewed_preconditions(self):
        # A pure local test double, not an applied Todo or a native acceptance test.
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);plan=d/'plan.json';approved=d/'preview.json'
            plan.write_text(json.dumps(PLAN_DATA));old={'kind':'ce-ru1-reviewed-preview-v1','native_response':receipt()};approved.write_text(json.dumps(old))
            observed=[]
            models=SimpleNamespace(ObservationPreconditions=SimpleNamespace(model_validate=lambda x:SimpleNamespace(**x)),
                                   ProposalEnvelope=SimpleNamespace(create=lambda **k:k))
            def fake_apply(config,project,proposal):observed.append(proposal);return {'status':'TEST_DOUBLE_ONLY'}
            mutation=SimpleNamespace(plan_digest=lambda p:receipt()['plan_digest'],apply_proposal=fake_apply)
            rt=(mutation,models,SimpleNamespace(load_config=lambda:None),receipt()['runtime_identity'])
            with patch.object(bridge,'runtime',return_value=rt),patch.object(sys,'argv',['native_bridge.py','apply','--plan',str(plan),'--approved-preview',str(approved),'--confirm',bridge.CONFIRM]),patch('sys.stdout',new=io.StringIO()):bridge.main()
            self.assertEqual(len(observed),1);self.assertEqual(observed[0]['observation_preconditions'].todo_revision,6877)
            self.assertEqual(observed[0]['proposed_change'],PLAN_DATA)
class GateTests(unittest.TestCase):
    def test_missing_ctest(self):
        with self.assertRaises(ValueError):gate.select_inventory({'tests':[]},['required'])
    def test_disabled_or_skip_test(self):
        for prop in [{'name':'DISABLED','value':True},{'name':'SKIP_RETURN_CODE','value':77},{'name':'SKIP_REGULAR_EXPRESSION','value':'skip'}]:
            with self.subTest(prop=prop),self.assertRaises(ValueError):gate.select_inventory({'tests':[{'name':'x','command':['binary'],'properties':[prop]}]},['x'])
    def test_junit_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'junit.xml';p.write_text('<testsuite><testcase name="x"><skipped/></testcase></testsuite>')
            with self.assertRaises(ValueError):gate.check_junit(p,['x'])
    def test_junit_no_test(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'junit.xml';p.write_text('<testsuite/>')
            with self.assertRaises(ValueError):gate.check_junit(p,['x'])
    def test_sanitizer_exception_cannot_be_pass(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);matrix=json.loads((PACKAGE/'machine/acceptance_matrix.json').read_text())
            for t in matrix['tests']:
                if t['group'] in ['gpu','demo']:(d/t['binary']).touch()
            output=d/'receipt.json';n=[0]
            def fake_run(cmd,**kw):
                n[0]+=1
                if n[0]>1:raise subprocess.TimeoutExpired(cmd,1)
                Path(cmd[cmd.index('--log-file')+1]).write_text('ERROR SUMMARY: 0 errors\n')
                return SimpleNamespace(returncode=0,stdout='TEST DOUBLE ONLY',stderr='')
            with patch.object(sys,'argv',['run_sanitizer.py','--binary-dir',str(d),'--receipt',str(output)]),patch.object(sanitizer.shutil,'which',return_value='/not-a-real-sanitizer'),patch.object(sanitizer.subprocess,'run',side_effect=fake_run),patch('sys.stdout',new=io.StringIO()):
                with self.assertRaises(subprocess.TimeoutExpired):sanitizer.main()
            self.assertFalse(json.loads(output.read_text())['passed'])
class IntegrityTests(unittest.TestCase):
    def mutate(self,fn):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'overlay';shutil.copytree(PACKAGE.parents[1],root)
            package=root/'planning/relation-update-spine-v1';fn(root,package)
            with self.assertRaises((ValueError,OSError)):validate(package,root)
    def test_missing_package_file(self):self.mutate(lambda r,p:(p/'01_SCOPE_AND_DECISIONS.md').unlink())
    def test_changed_demo(self):self.mutate(lambda r,p:(r/'examples/relation_update_spine_v1/regulatory_learning.cc').write_text('tampered'))
    def test_extra_package_file(self):self.mutate(lambda r,p:(p/'unlisted.txt').write_text('unlisted'))
if __name__=='__main__':unittest.main(verbosity=2)
