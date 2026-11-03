"""Host-only check that failed study packets remain visible beside passes."""
import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/export_correctness.py'
spec = importlib.util.spec_from_file_location('cgs_export', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ExportTests(unittest.TestCase):
    def test_failed_packet_does_not_hide_passing_packet(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            module.STUDY = root
            dirs = [root / 'runs/packets', root / 'runs/receipts', root / 'runs/builds', root / 'runs/closures']
            for d in dirs: d.mkdir(parents=True)
            binary = b'fixture-binary'
            archive = b'fixture-archive'
            binary_sha, archive_sha = hashlib.sha256(binary).hexdigest(), hashlib.sha256(archive).hexdigest()
            build = root / 'runs/builds' / binary_sha
            build.mkdir()
            (build / 'binary').write_bytes(binary)
            receipt_data = json.dumps({'binary_sha256': binary_sha, 'archive_sha256': archive_sha}).encode()
            (build / (hashlib.sha256(receipt_data).hexdigest() + '.json')).write_bytes(receipt_data)
            (root / 'runs/closures' / (archive_sha + '.tar.gz')).write_bytes(archive)
            rows = []
            for job_id, ok in [('pass', True), ('fail', False)]:
                packet = root / 'runs/packets' / job_id
                packet.mkdir()
                metrics = {'schema': 'cgs.packet.v1', 'scheduler_job_id': job_id,
                           'scheduler_attempt_id': job_id + '-attempt', 'uuid': 'GPU-test',
                           'archive_sha256': archive_sha, 'binary_sha256': binary_sha,
                           'sanitizer': 'none', 'success': ok, 'returncode': 0 if ok else 1}
                (packet / 'metrics.json').write_text(json.dumps(metrics))
                (packet / 'result.json').write_text(json.dumps({'status': 'succeeded' if ok else 'failed'}))
                (packet / 'test.stdout').write_text('')
                (packet / 'test.stderr').write_text('' if ok else 'fixture failure')
                (root / 'runs/receipts' / (job_id + '.json')).write_text(json.dumps({
                    'id': job_id, 'cleanup_verified': True, 'state': 'succeeded',
                    'attempt': {'attempt_id': job_id + '-attempt'}}))
                rows.append({'job_id': job_id, 'state': 'settled', 'charged_seconds': 1,
                             'gpu_resource_id': 'gpu:test:GPU-test', 'receipt': f'runs/receipts/{job_id}.json'})
            (root / 'runs/campaign-budget.json').write_text(json.dumps({'budget_seconds': 14400, 'reservations': rows}))
            old_argv = sys.argv
            try:
                sys.argv = ['export_correctness', '--output', str(root / 'index.json')]
                with contextlib.redirect_stdout(io.StringIO()): module.main()
            finally:
                sys.argv = old_argv
            index = json.loads((root / 'index.json').read_text())
            self.assertEqual([r['scheduler_job_id'] for r in index['records']], ['pass'])
            self.assertEqual([r['scheduler_job_id'] for r in index['failures']], ['fail'])


if __name__ == '__main__':
    unittest.main()
