import argparse
import hashlib
import importlib.util
import io
import json
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

script = Path(__file__).resolve().parents[1] / 'scripts/triage_binary.py'
spec = importlib.util.spec_from_file_location('cgs_triage_binary', script)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
sys.path.insert(0, str(script.parent))
register_spec = importlib.util.spec_from_file_location('cgs_register_triage_binary', script.with_name('register_triage_binary.py'))
register_adapter = importlib.util.module_from_spec(register_spec)
register_spec.loader.exec_module(register_adapter)


def write_archive(path: Path, files: dict[str, bytes]) -> str:
    manifest = {'files': {name: {'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                          for name, data in files.items()}}
    all_files = {**files, 'SOURCE-MANIFEST.json': json.dumps(manifest).encode()}
    with tarfile.open(path, 'w:gz') as tar:
        for name, data in all_files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    return adapter.sha256(path)


class TriageBinaryTests(unittest.TestCase):
    def test_registered_packet_uses_child_output_directory(self):
        argv = register_adapter.triage_packet_argv('GPU-123')
        self.assertEqual(argv[-2:], ['--output', '{output}/triage-packet'])
        self.assertEqual(argv[argv.index('--uuid') + 1], 'GPU-123')

    def test_extract_checks_manifest_and_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'source.tar.gz'
            digest = write_archive(archive, {'Cellerator/bench/chromatin_geometry/scripts/triage.py': b'pass\n'})
            extracted = root / 'source'
            manifest = adapter.extract_archive(archive, extracted, digest)
            self.assertIn('Cellerator/bench/chromatin_geometry/scripts/triage.py', manifest['files'])
            self.assertEqual((extracted / 'Cellerator/bench/chromatin_geometry/scripts/triage.py').read_bytes(), b'pass\n')
            with self.assertRaises(adapter.AdapterError):
                adapter.extract_archive(archive, extracted, digest)
            with self.assertRaises(adapter.AdapterError):
                adapter.extract_archive(archive, root / 'wrong', '0' * 64)

    def test_extract_rejects_unsafe_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'unsafe.tar.gz'
            files = {'../outside': b'bad'}
            manifest = {'files': {'../outside': {'size': 3, 'sha256': hashlib.sha256(b'bad').hexdigest()}}}
            with tarfile.open(archive, 'w:gz') as tar:
                for name, data in {**files, 'SOURCE-MANIFEST.json': json.dumps(manifest).encode()}.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
            with self.assertRaises(adapter.AdapterError):
                adapter.extract_archive(archive, root / 'out', adapter.sha256(archive))
            self.assertFalse((root / 'outside').exists())

    def test_build_copies_exact_binary_and_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'source.tar.gz'
            archive_sha = write_archive(archive, {'Cellerator/bench/chromatin_geometry/scripts/triage.py': b'pass\n'})
            binary = root / 'source-binary'
            binary.write_bytes(b'qualified binary bytes')
            receipt = root / 'original-build.json'
            original = {'schema': 'cgs.build.v1', 'target': 'cgs_study', 'archive_sha256': archive_sha,
                        'binary_sha256': adapter.sha256(binary)}
            receipt.write_text(json.dumps(original))
            args = argparse.Namespace(archive=archive, archive_sha256=archive_sha, binary=binary,
                                      receipt=receipt, build=root / 'build')
            result = adapter.build(args)
            self.assertEqual(result['binary_sha256'], original['binary_sha256'])
            self.assertEqual(adapter.sha256(root / 'build/cgs_study'), original['binary_sha256'])
            self.assertTrue((root / 'build/cgs_study').stat().st_mode & 0o100)
            self.assertEqual(adapter.validate_build(root / 'build')[0]['archive_sha256'], archive_sha)

    def test_build_rejects_binary_receipt_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'source.tar.gz'
            archive_sha = write_archive(archive, {'Cellerator/bench/chromatin_geometry/scripts/triage.py': b'pass\n'})
            binary = root / 'binary'
            binary.write_bytes(b'bytes')
            receipt = root / 'receipt.json'
            receipt.write_text(json.dumps({'schema': 'cgs.build.v1', 'target': 'cgs_study',
                                           'archive_sha256': archive_sha, 'binary_sha256': 'f' * 64}))
            args = argparse.Namespace(archive=archive, archive_sha256=archive_sha, binary=binary,
                                      receipt=receipt, build=root / 'build')
            with self.assertRaises(adapter.AdapterError):
                adapter.build(args)

    def test_run_requires_scheduler_owned_execution(self):
        with patch.dict(os.environ, {}, clear=True):
            args = argparse.Namespace(uuid='GPU-test', build=Path('/missing'),
                                      config=Path('/missing'), output=Path('/missing-output'))
            with self.assertRaises(adapter.AdapterError):
                adapter.run(args)

    def test_run_emits_standard_triage_metrics_for_scheduler_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build = root / 'build'
            build.mkdir()
            (build / 'source/Cellerator/bench/chromatin_geometry/scripts').mkdir(parents=True)
            (build / 'source/replay').mkdir()
            (build / 'source/Cellerator/bench/chromatin_geometry/scripts/triage.py').write_text('pass\n')
            binary = build / 'cgs_study'
            binary.write_bytes(b'qualified')
            binary.chmod(0o744)
            original = {'schema':'cgs.build.v1','target':'cgs_study','archive_sha256':'a'*64,
                        'binary_sha256':adapter.sha256(binary)}
            original_path = build / 'original-build.json'
            original_path.write_text(json.dumps(original))
            receipt = {**original,'suite':'triage-binary','binary_relative_path':'cgs_study',
                       'original_build_receipt_sha256':adapter.sha256(original_path)}
            (build / 'build.json').write_text(json.dumps(receipt))
            config = root / 'config.json'
            config.write_text(json.dumps({'schema':'cgs.triage.v1','runs':[{},{}]}))
            output = root / 'output'
            args = argparse.Namespace(uuid='GPU-test', build=build, config=config, output=output)

            def fake_run(argv, **kwargs):
                out = Path(argv[argv.index('--output') + 1])
                out.mkdir()
                (out / 'summary.json').write_text(json.dumps(
                    {'schema':'cgs.triage-result.v1','status':'pass','device_uuid':'GPU-test','records':[{},{}]}))
                return argparse.Namespace(returncode=0, stdout='triage pass', stderr='')

            with patch.dict(os.environ, {'AUTOBENCH_SCHEDULER_JOB_ID':'job-1',
                                         'AUTOBENCH_SCHEDULER_ATTEMPT_ID':'attempt-1',
                                         'CUDA_VISIBLE_DEVICES':'GPU-test'}), \
                 patch.object(adapter, 'quiescence', return_value=[{'uuid':'GPU-test','utilization_percent':0}]), \
                 patch.object(adapter.subprocess, 'check_output', return_value='GPU-test, Tesla V100, 1'), \
                 patch.object(adapter.subprocess, 'run', side_effect=fake_run):
                result = adapter.run(args)

            self.assertTrue(result['success'])
            self.assertEqual(result['schema'], 'cgs.packet.v1')
            self.assertEqual(result['suite'], 'triage')
            self.assertEqual(result['archive_sha256'], 'a'*64)
            self.assertEqual(result['binary_sha256'], adapter.sha256(binary))
            self.assertTrue((output / 'experiment.json').is_file())
            self.assertTrue((output / 'metrics.json').is_file())


if __name__ == '__main__':
    unittest.main()
