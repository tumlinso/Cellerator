"""CPU-only tests for the complete operational timing boundary."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/kda_driver.py'
spec = importlib.util.spec_from_file_location('cgs_kda_driver_test', SCRIPT)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class BoundaryTests(unittest.TestCase):
    def test_complete_cost_counts_each_operational_phase_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'candidate'
            source.mkdir()
            (source / 'backend.json').write_text('{"backend":"direct"}\n')
            request = {'request_id': 'test', 'candidate': {'source_dir': str(source), 'build': {'directory': str(root)}},
                       'case': {'id': 'fixture', 'workload_sha256': 'a' * 64},
                       'timing': {'warmup': 1, 'repetitions': 1, 'scope': 'fixture',
                                  'metric': 'horizon', 'unit': 'us'}}
            raw = {'replay_input_preparation_ms': 1., 'host_layout_preparation_ms': 2.,
                   'cold_device_preparation_ms': 3., 'cold_execution_preparation_ms': 6.,
                   'cold_device_preparation_includes_host_condition_maps': True,
                   'recurrent_complete_horizon_ms': [7.], 'output_recovery_ms': [8.],
                   'workload_sha256': 'b' * 64}
            with patch.object(driver, 'run_one', return_value=raw):
                driver.benchmark(str(root / 'request.json'), request, str(root / 'result.json'))
            result = json.loads((root / 'result.json').read_text())
            self.assertEqual(result['samples'], [21000.])
            raw['cold_execution_preparation_ms'] = 5.
            with patch.object(driver, 'run_one', return_value=raw):
                with self.assertRaisesRegex(RuntimeError, 'omits layout'):
                    driver.benchmark(str(root / 'request.json'), request, str(root / 'result.json'))


if __name__ == '__main__':
    unittest.main()
