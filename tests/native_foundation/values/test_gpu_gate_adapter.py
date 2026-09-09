import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('run_gpu_gate', HERE / 'run_gpu_gate.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
child_spec = importlib.util.spec_from_file_location('run_leased_test', HERE / 'run_leased_test.py')
child = importlib.util.module_from_spec(child_spec)
child_spec.loader.exec_module(child)

class AdapterAdmissionTests(unittest.TestCase):
    def test_junit_requires_actual_complete_non_skipped_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'result.xml'
            passed = '<testsuite tests="1" skipped="0"><testcase name="ru1_test" status="run"/></testsuite>'
            path.write_text(passed)
            child.validate_regression_junit(path, ['ru1_test'])
            for report in (
                passed.replace('status="run"', 'status="notrun"'),
                passed.replace('/>', '><skipped/></testcase>'),
                passed.replace('skipped="0"', 'skipped="1"'),
                passed.replace('/>', '><failure/></testcase>'),
                passed.replace('ru1_test', 'ru1_other'),
                '<testsuite tests="0"/>',
            ):
                with self.subTest(report=report), self.assertRaises(ValueError):
                    path.write_text(report)
                    child.validate_regression_junit(path, ['ru1_test'])

    def test_required_regressions_cannot_be_silently_omitted(self):
        result = subprocess.run([sys.executable, '-B', str(HERE / 'run_leased_test.py'),
            '--executable', '/absent', '--sanitizer', '/absent', '--require-regressions'],
            text=True, capture_output=True)
        self.assertNotEqual(0, result.returncode)
        self.assertIn('retained RU1 regressions must be configured', result.stderr)

    def test_supplemental_failure_cannot_pass_aggregate_evidence(self):
        # Entire external process boundary is mocked: this tests evidence plumbing,
        # makes no native reservation, and executes no device command.
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            evidence = base / 'evidence'
            evidence.mkdir()
            scripts = base / 'scripts'
            scripts.mkdir()
            (base / 'machine').mkdir()
            (base / 'machine/acceptance_matrix.json').write_text(json.dumps(
                {'gate_groups': {'test': {'requires_gpu': False}}}))
            script = scripts / 'gate.py'
            verifier = scripts / 'verifier.py'
            script.write_text('# mocked gate\n')
            verifier.write_text('# mocked verifier\n')
            lease = base / 'lease.json'
            lease.write_text(json.dumps({'format': 'CUDA-FOREGROUND-LEASE/1', 'state': 'active',
                'project_root': str(Path.cwd()), 'resource_ids': ['accelerator:GPU-test']}))
            bindings = base / 'bindings.json'
            bindings.write_text(json.dumps({'evidence_dir': str(evidence), 'build_dir': str(base),
                                           'supplemental_ctest_names': ['memcheck']}))
            child = evidence / 'leased-executable-test.json'
            def process(argv, **kwargs):
                if argv[0] == 'nvidia-smi':
                    return subprocess.CompletedProcess(argv, 0, '0, GPU-test\n', '')
                if str(verifier) in argv:
                    return subprocess.CompletedProcess(argv, 0, '{}', '')
                if str(script) in argv:
                    derived = Path(argv[argv.index('--bindings') + 1])
                    child.write_text(json.dumps({'bindings_sha256': hashlib.sha256(derived.read_bytes()).hexdigest(),
                        'executable_sha256': 'actual-binary-fixture', 'sanitizer_sha256': 'sanitizer-fixture', 'passed': True}))
                    Path(argv[argv.index('--receipt') + 1]).write_text(json.dumps({'passed': True, 'source_commit': 'fixture'}))
                    return subprocess.CompletedProcess(argv, 0)
                self.assertEqual('ctest', argv[0])
                return subprocess.CompletedProcess(argv, 8, 'supplemental failed', '')
            with mock.patch.dict(os.environ, {'TODO_GPU_LEASE_RECEIPT': str(lease),
                    'CUDA_VISIBLE_DEVICES': '0', 'NF1_EXECUTION_BINDINGS': str(bindings)}), \
                 mock.patch.object(sys, 'argv', ['adapter', '--gate-script', str(script), '--verifier',
                    str(verifier), '--group', 'test', '--gpu-uuid', 'GPU-test']), \
                 mock.patch.object(adapter, 'SHARED_LOCK', str(base / 'lock')), \
                 mock.patch.object(adapter.subprocess, 'run', side_effect=process):
                self.assertNotEqual(0, adapter.main())
            sidecar = json.loads(next(evidence.glob('lease-evidence-*.json')).read_text())
            self.assertTrue(sidecar['required_group_passed'])
            self.assertFalse(sidecar['passed'])
            self.assertNotEqual(0, sidecar['returncode'])
            self.assertEqual(8, sidecar['supplemental_tests'][0]['returncode'])
            linked = sidecar['child_executable_evidence']
            self.assertEqual(1, len(linked))
            self.assertEqual(hashlib.sha256(child.read_bytes()).hexdigest(), linked[0]['sha256'])
            self.assertEqual('actual-binary-fixture', linked[0]['executable_sha256'])

    def test_missing_lease_fails_before_any_device_probe(self):
        env = dict(os.environ)
        env.pop('TODO_GPU_LEASE_RECEIPT', None)
        result = subprocess.run([sys.executable, '-B', str(HERE / 'run_gpu_gate.py'),
            '--gate-script', '/absent', '--verifier', '/absent', '--group', 'v01',
            '--gpu-uuid', 'GPU-test'], env=env, text=True, capture_output=True)
        self.assertNotEqual(0, result.returncode)
        self.assertIn('native TODO_GPU_LEASE_RECEIPT is required', result.stderr)

    def test_receipt_and_visibility_policy_rejects_mismatches(self):
        # Synthetic parser fixtures do not assert a live native owner or reserve hardware.
        receipt = {'format': 'CUDA-FOREGROUND-LEASE/1', 'state': 'active',
                   'resource_ids': ['accelerator:GPU-a']}
        for visible in ('', '1', '0,0', 'GPU-b', '0,1'):
            with self.subTest(visible=visible), self.assertRaises(ValueError):
                adapter.validate_device_binding(receipt, ['GPU-a'], visible, {'0': 'GPU-a', '1': 'GPU-b'})
        with self.assertRaises(ValueError):
            adapter.validate_device_binding(receipt, ['GPU-b'], '1', {'1': 'GPU-b'})
        with self.assertRaises(ValueError):
            adapter.validate_device_binding({**receipt, 'state': 'released'}, ['GPU-a'], '0', {'0': 'GPU-a'})
        with self.assertRaises(ValueError):
            adapter.validate_device_binding(receipt, [], '0', {'0': 'GPU-a'})
        adapter.validate_device_binding(receipt, ['GPU-a'], '0', {'0': 'GPU-a'})

if __name__ == '__main__':
    unittest.main()
