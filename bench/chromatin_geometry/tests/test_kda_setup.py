"""CPU-only KDA registration and preparation against a temporary fixture."""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
KDA = Path('/home/tumlinson/kda')


class KdaSetupTests(unittest.TestCase):
    def test_prepares_candidate_identity_without_gpu_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'Cellerator'
            study = source / 'bench/chromatin_geometry'
            for directory in ('scripts', 'backends'):
                (study / directory).mkdir(parents=True, exist_ok=True)
            shutil.copy2(STUDY / 'scripts/kda_driver.py', study / 'scripts/kda_driver.py')
            for name in ('consumers.cu', 'conditions.cu', 'grouped_linear_library.cu'):
                shutil.copy2(STUDY / 'backends' / name, study / 'backends' / name)
            for name in ('workloads.cc', 'workloads.hh', 'campaign_runner.cu'):
                shutil.copy2(STUDY / name, study / name)
            shutil.copytree(STUDY / 'reference', study / 'reference')
            (root / 'SOURCE-MANIFEST.json').write_text('{}\n')
            binary = root / 'synthetic-binary'
            shutil.copy2('/bin/true', binary)
            evidence = root / 'interference-evidence.json'
            evidence.write_text(json.dumps({'schema': 'fixture-only'}) + '\n')
            contract = root / 'contract-freeze.json'
            contract.write_text(json.dumps({'format': 'fixture-only'}) + '\n')
            config = root / 'config.json'
            config.write_text(json.dumps({'schema': 'cgs.kda-config.v1', 'task_id': 'test-only',
                                          'incumbent': 'scalar', 'candidates': ['direct'],
                                          'interference_attestation': {'sha256': hashlib.sha256(evidence.read_bytes()).hexdigest()},
                                          'contract_freeze_sha256': hashlib.sha256(contract.read_bytes()).hexdigest(),
                                          'cases': [{'id': 'test-only-case', 'input': {
                                              'campaign_workload_sha256': 'a' * 64},
                                                     'parameters': {'gate': 'ste', 'body': 'product',
                                                                    'precision': 'fp16', 'condition_update': 'none',
                                                                    'tile': 32, 'stages': 1}}]}))
            proc = subprocess.run([sys.executable, str(STUDY / 'scripts/kda_setup.py'),
                                   '--source', str(source), '--kda-root', str(KDA),
                                   '--binary', str(binary), '--workspace', str(root / 'workspace'),
                                   '--config', str(config)], text=True, capture_output=True, timeout=60)
            self.assertEqual(proc.returncode, 0, proc.stdout + '\n' + proc.stderr)
            self.assertTrue((root / 'workspace/engine/candidates.jsonl').is_file())
            summary = json.loads(proc.stdout.strip().splitlines()[-1])
            self.assertEqual(summary['candidate_ids'], ['scalar', 'direct'])
            self.assertIn('preliminary', summary['selection_status'])


if __name__ == '__main__':
    unittest.main()
