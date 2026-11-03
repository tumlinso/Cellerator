"""Host-only adversarial checks of the cross-arm triage envelope."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
SCRIPT = STUDY / 'scripts/triage.py'


class TriageTests(unittest.TestCase):
    def test_requires_grant_context_and_rejects_unpaired_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'config.json'
            config.write_text(json.dumps({'schema': 'cgs.triage.v1',
                                          'runs': [{'match_group': 'one'}]}))
            cmd = [sys.executable, str(SCRIPT), '--binary', '/bin/true',
                   '--config', str(config), '--output', str(root / 'out')]
            env = dict(os.environ)
            env.pop('AUTOBENCH_SCHEDULER_JOB_ID', None)
            result = subprocess.run(cmd, text=True, capture_output=True, env=env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('requires an admitted', result.stderr)
            env.update(AUTOBENCH_SCHEDULER_JOB_ID='test-only', CUDA_VISIBLE_DEVICES='GPU-test')
            result = subprocess.run(cmd, text=True, capture_output=True, env=env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('at least two matched arms', result.stderr)


if __name__ == '__main__':
    unittest.main()
