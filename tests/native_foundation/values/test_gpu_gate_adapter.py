import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('run_gpu_gate', HERE / 'run_gpu_gate.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)

class AdapterAdmissionTests(unittest.TestCase):
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
