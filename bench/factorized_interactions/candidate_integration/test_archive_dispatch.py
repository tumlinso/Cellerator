"""Exercise archive validation and both build dispatches without CUDA."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

HELPER = Path(__file__).resolve().parents[1] / 'build_from_archive.py'
spec = importlib.util.spec_from_file_location('factorized_archive_dispatch', HELPER)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class ArchiveDispatchTests(unittest.TestCase):
    def test_default_and_native_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            content = b'# source marker\n'
            name = 'bench/factorized_interactions/candidate_integration/build_candidate_runner.py'
            archive = root / 'source.tar'
            with tarfile.open(archive, 'w') as tar:
                member = tarfile.TarInfo(name)
                member.size = len(content)
                tar.addfile(member, io.BytesIO(content))
            sha = lambda data: hashlib.sha256(data).hexdigest()
            manifest = {'schema': 'cellerator.factorized_interactions.archived_source_closure.v1',
                'archive_path': archive.name, 'archive_bytes': archive.stat().st_size,
                'archive_sha256': sha(archive.read_bytes()),
                'build_stager_path': HELPER.name, 'build_stager_sha256': sha(HELPER.read_bytes()),
                'file_count': 1, 'files': [{'path': name, 'bytes': len(content), 'sha256': sha(content)}],
                'source_closure_sha256': 'fixture'}
            manifest_path = root / 'manifest.json'
            manifest_path.write_text(json.dumps(manifest))
            for native in (False, True):
                build = root / ('native' if native else 'baseline')
                args = ['build_from_archive.py', '--archive', str(archive), '--manifest', str(manifest_path),
                        '--build', str(build), '--parallel', '2', '--configure-only']
                if native: args.append('--native-only')
                with patch.object(sys, 'argv', args), patch.object(module.subprocess, 'run') as run:
                    self.assertEqual(module.main(), 0)
                command = run.call_args.args[0]
                expected = ('candidate_integration/build_candidate_runner.py' if native else 'build_runner.py')
                self.assertTrue(command[1].endswith(expected), command)
                self.assertEqual(command[command.index('--build') + 1], str(build / 'native_candidate' if native else build))
                self.assertIn('--configure-only', command)
                self.assertEqual(run.call_args.kwargs['cwd'], build / 'source')
                self.assertTrue(run.call_args.kwargs['check'])

if __name__ == '__main__':
    unittest.main()
