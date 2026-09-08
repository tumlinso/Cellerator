#!/usr/bin/env python3
"""Offline negative tests for catalog, native projection and evidence guards. Never touches Todo authority."""
from pathlib import Path
import copy,hashlib,json,shutil,subprocess,tempfile,unittest,sys,xml.etree.ElementTree as ET
sys.dont_write_bytecode=True
from compile_plan import PACKAGE,load,assemble
import validate_package as vp
from validate_package import graph,ancestors,canonical,demand
from run_gate import select_inventory,check_junit
import todo_bootstrap as bootstrap
M=load(PACKAGE/'machine/proposed_todos.json')

class PlanTests(unittest.TestCase):
    def test_01_actual_plan(self):graph(M)
    def test_02_native_projection(self):self.assertEqual(assemble(M),load(PACKAGE/'machine'/M['native_plan_file']))
    def test_03_duplicate_id(self):
        m=copy.deepcopy(M);m['tasks'].append(m['tasks'][0])
        with self.assertRaisesRegex(ValueError,'duplicate task'):graph(m)
    def test_04_foreign_dependency(self):
        m=copy.deepcopy(M);m['tasks'][0]['native_record']['depends_on']=[{'type':'task','task_id':'FOREIGN-TASK'}]
        with self.assertRaises(ValueError):graph(m)
    def test_05_self_cycle(self):
        m=copy.deepcopy(M);t=m['tasks'][0];t['native_record']['depends_on']=[{'type':'task','task_id':t['id']}]
        with self.assertRaises(ValueError):graph(m)
    def test_06_queue_cycle(self):
        m=copy.deepcopy(M);lane=m['run']['lanes'][1];a,b=lane['tasks'][:2]
        t=next(x for x in m['tasks'] if x['id']==a);t['native_record']['depends_on']=[{'type':'task','task_id':b}]
        with self.assertRaisesRegex(ValueError,'cycle'):graph(m)
    def test_07_duplicate_lane_owner(self):
        m=copy.deepcopy(M);m['run']['lanes'][2]['tasks'].append(m['tasks'][0]['id'])
        with self.assertRaises(ValueError):graph(m)
    def test_08_foreign_lane_parent(self):
        m=copy.deepcopy(M);m['run']['lanes'][1]['parent_lane_id']='OTHER-PROJECT-LANE'
        with self.assertRaises(ValueError):graph(m)
    def test_09_unordered_scope_conflict(self):
        m=copy.deepcopy(M);_,parents,_,_,_=graph(m);anc=ancestors(parents)
        pair=next((a,b) for i,a in enumerate(m['tasks']) for b in m['tasks'][i+1:] if a['id'] not in anc[b['id']] and b['id'] not in anc[a['id']])
        for t in pair:t['write_scope']=['collision'];t['native_record']['scope']['exclusive_paths']=['collision']
        with self.assertRaisesRegex(ValueError,'Unordered write conflict'):graph(m)
    def test_10_premature_done(self):
        m=copy.deepcopy(M);m['tasks'][0]['native_record']['status']='done'
        with self.assertRaises(ValueError):graph(m)
    def test_11_missing_checkpoint(self):
        m=copy.deepcopy(M);m['barriers'][0]['requirements'][0]['id']='UNKNOWN'
        with self.assertRaises(ValueError):graph(m)
    def test_12_cross_closure_cycle(self):
        # Independent of lane graph: the minimal final-receipt chain must not wait on itself.
        p={'CE-ready':set(),'GH-consumer':{'CE-ready'},'CE-final':{'GH-consumer'},'GH-final':{'CE-final'}}
        ancestors(p);p['CE-ready'].add('GH-final')
        with self.assertRaisesRegex(ValueError,'cycle'):ancestors(p)
    def test_13_safe_path(self):
        for p in ['/tmp/x','../x','x/../y','a\\b','a/./b','a//b']:
            with self.assertRaises(ValueError):vp.safe_rel(p)
    def test_14_schema_two_refused(self):
        m=copy.deepcopy(M);m['native_schema_version']=2
        with self.assertRaises(ValueError):assemble(m)

class EvidenceTests(unittest.TestCase):
    def inventory(self,properties=None):return {'tests':[{'name':'real','command':['/bin/true'],'properties':properties or []}]}
    def test_15_missing_test(self):
        with self.assertRaisesRegex(ValueError,'missing'):select_inventory(self.inventory(),['missing'])
    def test_16_disabled_test(self):
        with self.assertRaisesRegex(ValueError,'disabled'):select_inventory(self.inventory([{'name':'DISABLED','value':True}]),['real'])
    def test_17_skip_code_forbidden(self):
        with self.assertRaisesRegex(ValueError,'skip-as-success'):select_inventory(self.inventory([{'name':'SKIP_RETURN_CODE','value':77}]),['real'])
    def test_18_skip_regex_forbidden(self):
        with self.assertRaises(ValueError):select_inventory(self.inventory([{'name':'SKIP_REGULAR_EXPRESSION','value':['skip']}]),['real'])
    def test_19_junit_skip(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x.xml';p.write_text('<testsuite><testcase name="real"><skipped/></testcase></testsuite>')
            with self.assertRaises(ValueError):check_junit(p,['real'])
    def test_20_junit_missing(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x.xml';p.write_text('<testsuite/>')
            with self.assertRaises(ValueError):check_junit(p,['real'])
    def test_21_junit_pass(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x.xml';p.write_text('<testsuite><testcase name="real" status="run"/></testsuite>');check_junit(p,['real'])
    def response(self):
        p=assemble(M);c={'project_uuid':M['project_uuid'],'todo_revision':10,'workflow_revision':10,'workflow_authority_fingerprint':'s','todo_semantic_authority_fingerprint':'s'}
        return {'valid':True,'status':'validated','project_uuid':M['project_uuid'],'revision':10,'would_add':[t['id'] for t in p['tasks']],'would_modify':[],'warnings':[],'plan_digest':hashlib.sha256(canonical(p)).hexdigest(),'current_observation_preconditions':c,'runtime_identity':{'mock':'offline test double only'}}
    def test_22_valid_native_shape_double(self):bootstrap.check_native(self.response(),assemble(M),M['project_uuid'])
    def test_23_native_collision(self):
        v=self.response();v['would_modify']=['old-task']
        with self.assertRaises(ValueError):bootstrap.check_native(v,assemble(M),M['project_uuid'])
    def test_24_native_wrong_uuid(self):
        v=self.response();v['project_uuid']='another'
        with self.assertRaises(ValueError):bootstrap.check_native(v,assemble(M),M['project_uuid'])
    def test_25_native_digest(self):
        v=self.response();v['plan_digest']='0'*64
        with self.assertRaises(ValueError):bootstrap.check_native(v,assemble(M),M['project_uuid'])
    def test_26_native_warning(self):
        v=self.response();v['warnings']=['requires review']
        with self.assertRaises(ValueError):bootstrap.check_native(v,assemble(M),M['project_uuid'])
    def test_27_reviewed_preview_expired(self):
        old={'kind':'nf1-reviewed-preview-v1','created_unix':1}
        with self.assertRaisesRegex(ValueError,'expired'):bootstrap.check_preview(old,{},{},self.response(),assemble(M),M['project_uuid'],now=4002)
    def test_28_reviewed_source_drift(self):
        old={'kind':'nf1-reviewed-preview-v1','created_unix':1,'source':{'head':'old'},'peer_source':{}}
        with self.assertRaisesRegex(ValueError,'source or peer'):bootstrap.check_preview(old,{'head':'new'},{},self.response(),assemble(M),M['project_uuid'],now=2)
    def test_29_receipt_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'receipt.json';p.write_text('{}')
            with self.assertRaises(ValueError):bootstrap.outside(p,[])
    def test_30_receipt_not_in_source(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):bootstrap.outside(Path(td)/'new.json',[td])
    def test_31_no_tests_inventory(self):
        with self.assertRaises(ValueError):select_inventory({'tests':[]},['real'])
    def test_32_junit_failure(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x.xml';p.write_text('<testsuite><testcase name="real"><failure/></testcase></testsuite>')
            with self.assertRaises(ValueError):check_junit(p,['real'])
    def test_33_wrong_manual_confirmation_no_runtime_import(self):
        p=subprocess.run([sys.executable,'-B',str(PACKAGE/'scripts/native_bridge.py'),'apply','--confirm','WRONG'],text=True,capture_output=True)
        self.assertNotEqual(p.returncode,0);self.assertIn('Explicit manual apply confirmation',p.stderr)
    def test_34_pending_cross_receipt_refused(self):
        import verify_receipt
        edge=load(PACKAGE/'machine/cross_repository_contracts.json')['edges'][0]
        with self.assertRaisesRegex(ValueError,'qualified capability'):verify_receipt.verify(edge,{'kind':'nf1-capability-receipt-v1','status':'pending'},{},Path('/not-used'))
    def test_35_early_validation_has_no_native_import(self):
        text=(PACKAGE/'scripts/validate_package.py').read_text()
        self.assertNotIn('import project_control',text);self.assertNotIn('apply_proposal(',text)


class IntegrityTests(unittest.TestCase):
    def copied(self,td):
        repo=Path(td)/'repo';destination=repo/M['package_path'];destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(PACKAGE,destination)
        for entry in load(PACKAGE/'machine/delivery_files.json')['files']:
            src=PACKAGE.parents[1]/entry['path'];dst=repo/entry['path'];dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
        return destination
    def test_36_delivered_integrity(self):vp.validate(PACKAGE)
    def test_37_manifest_tampering(self):
        with tempfile.TemporaryDirectory() as td:
            p=self.copied(td);f=p/'README.md';f.write_text(f.read_text()+'tamper')
            with self.assertRaisesRegex(ValueError,'integrity failure'):vp.validate(p)
    def test_38_extra_package_file(self):
        with tempfile.TemporaryDirectory() as td:
            p=self.copied(td);(p/'unexpected.py').write_text('pass')
            with self.assertRaisesRegex(ValueError,'membership'):vp.validate(p)
    def test_39_delivery_tampering(self):
        with tempfile.TemporaryDirectory() as td:
            p=self.copied(td);f=p.parents[1]/load(p/'machine/delivery_files.json')['files'][0]['path'];f.write_text(f.read_text()+'tamper')
            with self.assertRaisesRegex(ValueError,'delivery mismatch'):vp.validate(p)
    def test_40_extra_example_file(self):
        with tempfile.TemporaryDirectory() as td:
            p=self.copied(td);d=load(p/'machine/delivery_files.json');f=p.parents[1]/d['roots'][0]/'unlisted.cc';f.write_text('unlisted')
            with self.assertRaisesRegex(ValueError,'delivery membership'):vp.validate(p)

if __name__=='__main__':unittest.main(verbosity=2)
