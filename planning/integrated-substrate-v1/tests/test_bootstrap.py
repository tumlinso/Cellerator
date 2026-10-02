"""Contract/parser/graph unit tests. Fixtures are NOT live acceptance evidence."""
import copy,hashlib,json,pathlib,shutil,subprocess,sys,tempfile,unittest
from datetime import datetime,timezone,timedelta
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import gates
import validate_package
import stage

class GateContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();base=pathlib.Path(self.tmp.name)
        self.roots={pr:base/pr for pr in gates.PROJECTS}
        self.heads={pr:(str(i+1)*40) for i,pr in enumerate(gates.PROJECTS)}
        self.evidence=[]
        for pr,root in self.roots.items():
            root.mkdir();(root/'test-only.log').write_text('PARSER UNIT FIXTURE. NOT REAL NATIVE OR AUTHORITY EVIDENCE.\n')
            self.evidence.append({'project':pr,'path':'test-only.log','sha256':hashlib.sha256((root/'test-only.log').read_bytes()).hexdigest()})
        rows=[]
        for row in gates.load(ROOT/'machine/run-watchlist.json'):
            rows.append({**row,'disposition':'completed','reason':'parser unit fixture only'})
        self.review={'schema_version':1,'record_kind':'is1_activation_review','approved':True,'reviewed_by':'UNIT_TEST_ONLY',
          'captured_at':datetime.now(timezone.utc).isoformat(),'heads':self.heads,'inventory_complete':True,
          'claims_and_pending_patches_reconciled':True,'predecessor_scope_reviewed':True,
          'run_inventory':rows,'authority_evidence':self.evidence}
        self.link=gates.load(ROOT/'machine/cross-authority.json')[1]
        names=self.link['required']
        ce=self.evidence[0]
        self.provider={'schema_version':1,'record_kind':'is1_provider_receipt','project':'cellerator',
         'task_id':'CE-IS1-INTEGRATE','source_commit':self.heads['cellerator'],'reviewed_by':'UNIT_TEST_ONLY','reviewed_at':'unit fixture',
         'capabilities':[{'name':n,'status':'implemented','call':'fixture','shapes':['fixture'],'numerical_policy':'fixture'} for n in names],
         'build_artifacts':[ce],
         'checks':[{'kind':k,'execution':'real','status':'passed','exit_code':0,'argv':['NOT_EXECUTED_UNIT_FIXTURE'],
             'source_commits':{'cellerator':self.heads['cellerator']},'backend':'cuda_sm70',**ce}
             for k in ('native_correctness','installed_consumer','derivatives','framework_consumer')]}
        self.state=lambda pr,task:'done'
    def tearDown(self):self.tmp.cleanup()
    def activation(self,r=None,state=None):
        gates.validate_activation(r or self.review,{'max_review_age_seconds':600},self.heads,state or self.state,self.roots)
    def provider_check(self,r=None,state=None):
        gates.validate_provider(r or self.provider,self.link,self.heads,state or self.state,self.roots)
    def test_parser_contract_valid_fixture(self):
        self.activation();self.provider_check()
    def test_unapproved_template_is_not_activation(self):
        r=copy.deepcopy(self.review);r['approved']=False
        with self.assertRaises(ValueError):self.activation(r)
    def test_unfinished_known_predecessor_blocks(self):
        with self.assertRaises(ValueError):self.activation(state=lambda pr,t:'planned' if t=='CE-MOON-000' else 'done')
    def test_unknown_or_omitted_run_blocks(self):
        r=copy.deepcopy(self.review);r['run_inventory'].pop()
        with self.assertRaises(ValueError):self.activation(r)
        r=copy.deepcopy(self.review);r['run_inventory'][0]['disposition']='unclassified'
        with self.assertRaises(ValueError):self.activation(r)
    def test_newly_discovered_predecessor_is_checked(self):
        r=copy.deepcopy(self.review);r['run_inventory'].append({'project':'baseplane','run_id':'NEW','root_task_id':'BP-LATE','disposition':'predecessor','reason':'newly discovered'})
        with self.assertRaises(ValueError):self.activation(r,state=lambda p,t:'planned' if t=='BP-LATE' else 'done')
    def test_permission_locked_run_not_activated(self):
        r=copy.deepcopy(self.review)
        for row in r['run_inventory']:
            if row['root_task_id']=='CE-AMP-00':row.update(disposition='permission_locked',preservation_owner='CE-IS1-ADOPT',trigger='separate explicit human permission')
        self.activation(r,state=lambda p,t:'planned' if t=='CE-AMP-00' else 'done')
    def test_stale_or_source_changed_activation(self):
        r=copy.deepcopy(self.review);r['captured_at']=(datetime.now(timezone.utc)-timedelta(hours=2)).isoformat()
        with self.assertRaises(ValueError):self.activation(r)
        r=copy.deepcopy(self.review);r['heads']['baseplane']='a'*40
        with self.assertRaises(ValueError):self.activation(r)
    def test_evidence_corruption_and_escape_rejected(self):
        (self.roots['cellerator']/'test-only.log').write_text('changed')
        with self.assertRaises(ValueError):self.activation()
        with self.assertRaises(ValueError):gates.safe_file(self.roots['baseplane'],'../cellerator/test-only.log')
        (self.roots['baseplane']/'linked.log').symlink_to(self.roots['baseplane']/'test-only.log')
        with self.assertRaises(ValueError):gates.safe_file(self.roots['baseplane'],'linked.log')
    def test_provider_task_or_source_change_blocks(self):
        with self.assertRaises(ValueError):self.provider_check(state=lambda p,t:'planned')
        r=copy.deepcopy(self.provider);r['source_commit']='0'*40
        with self.assertRaises(ValueError):self.provider_check(r)
    def test_mock_reference_or_missing_cuda_cannot_qualify(self):
        r=copy.deepcopy(self.provider);r['checks'][0]['execution']='reference'
        with self.assertRaises(ValueError):self.provider_check(r)
        r=copy.deepcopy(self.provider);r['checks'][0]['backend']='host_fixture'
        with self.assertRaises(ValueError):self.provider_check(r)
    def test_missing_derivative_capability_or_check(self):
        r=copy.deepcopy(self.provider);r['capabilities']=[c for c in r['capabilities'] if c['name']!='jvp']
        with self.assertRaises(ValueError):self.provider_check(r)
        r=copy.deepcopy(self.provider);r['checks']=[c for c in r['checks'] if c['kind']!='derivatives']
        with self.assertRaises(ValueError):self.provider_check(r)
    def test_provider_rejects_stale_sibling_check(self):
        r=copy.deepcopy(self.provider);r['checks'][0]['source_commits']['glasshelix']='a'*40
        with self.assertRaises(ValueError):self.provider_check(r)

class PackageTests(unittest.TestCase):
    def test_package_dag_and_preservation(self):
        r=validate_package.validate();self.assertEqual(r['outcomes'],45);self.assertEqual(r['maximum_ready_wave'],17)
    def mutate_copy(self,fn):
        with tempfile.TemporaryDirectory() as tmp:
            dst=pathlib.Path(tmp)/'pkg';shutil.copytree(ROOT,dst,ignore=shutil.ignore_patterns('__pycache__'))
            fn(dst)
            with self.assertRaises(ValueError):validate_package.validate(dst)
    def test_cycle_rejected(self):
        def mutate(dst):
            p=dst/'machine/cellerator.todo-plan.json';x=json.loads(p.read_text())
            next(t for t in x['tasks'] if t['id']=='CE-IS1-STATE')['depends_on'].append({'type':'task','task_id':'CE-IS1-CLOSE'})
            p.write_text(json.dumps(x))
        self.mutate_copy(mutate)
    def test_unordered_write_collision_rejected(self):
        def mutate(dst):
            p=dst/'machine/cellerator.todo-plan.json';x=json.loads(p.read_text())
            next(t for t in x['tasks'] if t['id']=='CE-IS1-PACK')['scope']['exclusive_paths'].append('src/state')
            p.write_text(json.dumps(x))
        self.mutate_copy(mutate)
    def test_lost_prior_card_rejected(self):
        def mutate(dst):
            p=dst/'machine/preservation.json';x=json.loads(p.read_text());x['records']=[r for r in x['records'] if r['id']!='BP:E48'];p.write_text(json.dumps(x))
        self.mutate_copy(mutate)
    def test_isolated_patch_cannot_skip_integration(self):
        def mutate(dst):
            p=dst/'machine/cellerator.todo-plan.json';x=json.loads(p.read_text())
            next(t for t in x['tasks'] if t['id']=='CE-IS1-DIFF')['depends_on'].append({'type':'task','task_id':'CE-IS1-STATE'})
            p.write_text(json.dumps(x))
        self.mutate_copy(mutate)
    def test_staging_dry_run_apply_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo=pathlib.Path(tmp)/'repo';repo.mkdir()
            subprocess.run(['git','init','-q',str(repo)],check=True)
            (repo/'README.md').write_text('temporary stage test repository')
            subprocess.run(['git','-C',str(repo),'add','README.md'],check=True)
            subprocess.run(['git','-C',str(repo),'-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','fixture'],check=True)
            r=stage.stage(ROOT,repo,'cellerator',False);self.assertFalse(pathlib.Path(r['destination']).exists())
            r=stage.stage(ROOT,repo,'cellerator',True);dest=pathlib.Path(r['destination'])
            self.assertTrue((dest/'START_CONTROLLER.md').is_file());validate_package.verify_manifest(dest)
            self.assertFalse((repo/'.todo-orchestrator').exists())
            with self.assertRaises(FileExistsError):stage.stage(ROOT,repo,'cellerator',True)

if __name__=='__main__':unittest.main()
