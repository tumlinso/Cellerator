import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('controller_activation', Path(__file__).with_name('controller_activation.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class ControllerReviewTests(unittest.TestCase):
    def fixture(self):
        claim = {'id':'claim', 'task_id':'CE-IS1-ADOPT', 'session_id':'root-session', 'state':'active', 'session_state':'active'}
        dispatch = {'dispatch_id':'dispatch', 'claim_id':'claim', 'session_id':'root-session', 'task_id':'CE-IS1-ADOPT'}
        state = {'tasks':[{'id':'CE-IS1-ADOPT', 'effective_state':'active', 'active_claim':{'id':'claim'}, 'scopes':[{'mode':'exclusive','path':'include'}]}]}
        workflow = {'local_children':[], 'first_class_agents':[], 'runs':[{'id':'CE-IS1-RUN-V1', 'root_task_id':'CE-IS1-000', 'lanes':[{'id':'CE-IS1-ADOPT-L', 'dispatch':dispatch, 'queue':[{'task_id':'CE-IS1-ADOPT','state':'active'}]}]}]}
        plan = {'tasks':[{'id':'CE-IS1-ADOPT','scope':{'exclusive_paths':['include']}}]}
        return state, workflow, {'cellerator':'CE-IS1-ADOPT'}, {'cellerator':'root-session'}, [claim], plan

    def test_claim_requires_unique_public_dispatch(self):
        state, workflow, *_ = self.fixture()
        state['read_authority_fingerprint'] = workflow['read_authority_fingerprint'] = 'same'
        state['tasks'][0]['active_claim']['state'] = 'active'
        claims = m.read_claims(state, workflow)
        self.assertEqual(claims[0]['session_id'], 'root-session')
        workflow['runs'][0]['lanes'][0]['dispatch'] = None
        with self.assertRaisesRegex(ValueError, 'missing or duplicate dispatch'):
            m.read_claims(state, workflow)

    def test_exact_binding_passes(self):
        self.assertEqual(len(m.validate_current('cellerator', *self.fixture())),1)

    def test_different_claim_rejected(self):
        args = self.fixture()
        args[1]['runs'][0]['lanes'][0]['dispatch']['claim_id'] = 'different'
        with self.assertRaisesRegex(ValueError,'claim binding'):
            m.validate_current('cellerator',*args)

    def test_unrelated_claim_rejected(self):
        args = self.fixture()
        args[4].append({'task_id':'CE-AMP-00','id':'other'})
        with self.assertRaisesRegex(ValueError,'controller claims'):
            m.validate_current('cellerator',*args)

    def test_unrelated_agent_rejected(self):
        args = self.fixture()
        args[1]['first_class_agents'] = [{'dispatch_id':'other'}]
        with self.assertRaisesRegex(ValueError,'observable agent'):
            m.validate_current('cellerator',*args)

    def test_changed_scope_rejected(self):
        args = self.fixture()
        args[0]['tasks'][0]['scopes'].append({'mode':'exclusive','path':'src'})
        with self.assertRaisesRegex(ValueError,'scope changed'):
            m.validate_current('cellerator',*args)

    def test_wrong_session_rejected(self):
        args = self.fixture()
        args[3]['cellerator'] = 'foreign-session'
        with self.assertRaisesRegex(ValueError,'session mismatch'):
            m.validate_current('cellerator',*args)

    def test_write_failure_invalidates_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            projects = {p:Path(directory)/p for p in m.base.PROJECTS}
            invpath = Path(directory)/'inventory.json'
            invpath.write_text('[]')
            reviewpath = projects['cellerator']/m.base.PACKAGE/'results/activation-review.json'
            original_write = m.base.write_private
            def fail_evidence(path,data):
                if path.name == 'authority-review.json':
                    raise OSError('simulated evidence write failure')
                original_write(path,data)
            with patch.object(m.base,'PROJECTS',projects), patch.object(m.base,'write_private',fail_evidence):
                original_write(reviewpath,m.base.encode({'approved':True}))
                with self.assertRaisesRegex(OSError,'simulated'):
                    m.base.publish_activation({'cellerator':{}},[],{},invpath,'root',lambda:None)
                self.assertFalse(json.loads(reviewpath.read_text())['approved'])

if __name__ == '__main__':
    unittest.main()
