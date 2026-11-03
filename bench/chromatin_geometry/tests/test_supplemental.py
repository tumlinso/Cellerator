import importlib.util
import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


path = Path(__file__).resolve().parents[1] / 'scripts/kda_evaluate.py'
spec = importlib.util.spec_from_file_location('cgs_supplemental_test', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SupplementalTest(unittest.TestCase):
    def test_all_arms_have_balanced_order_and_raw_records(self):
        def run_one(binary, arm, case, warmups, repeats, output):
            data = {'cold_execution_preparation_ms': 3, 'replay_input_preparation_ms': 1,
                    'host_layout_preparation_ms': 1, 'cold_device_preparation_ms': 1,
                    'recurrent_complete_horizon_ms': [2], 'output_recovery_ms': [1],
                    'workload_sha256': 'a' * 64}
            Path(output).write_text(json.dumps(data))
            return data
        driver = SimpleNamespace(run_one=run_one)
        fake_spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda x: None))
        cfg = {'incumbent': 'scalar', 'candidates': ['bitmap', 'direct', 'cohort-shared'],
               'source_archive_sha256': 'b' * 64, 'binary_sha256': 'c' * 64}
        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(module.os.environ, {'AUTOBENCH_SCHEDULER_JOB_ID': 'job', 'CUDA_VISIBLE_DEVICES': 'GPU-a'}), \
             patch.object(module.importlib.util, 'spec_from_file_location', return_value=fake_spec), \
             patch.object(module.importlib.util, 'module_from_spec', return_value=driver):
            result = module.supplemental_matrix(Path('binary'), {'cases': [{'parameters': {'condition_update': 'none'}}]}, cfg, Path(tmp) / 'out')
            self.assertEqual(result['status'], 'pass')
            self.assertEqual(len(result['rounds']), 6)
            for left, right in itertools.combinations(result['arms'], 2):
                self.assertEqual(sum(row['order'].index(left) < row['order'].index(right) for row in result['rounds']), 3)
            for row in result['rounds']:
                self.assertEqual(set(row['order']), set(result['arms']))
                self.assertTrue(all((Path(tmp) / 'out' / r['path']).is_file() for r in row['records']))
            update = module.supplemental_matrix(Path('binary'), {'cases': [{'parameters': {'condition_update': 'incremental'}}]},
                                                cfg, Path(tmp) / 'updates', ['rebuild', 'incremental'])
            self.assertEqual(update['schema'], 'cgs.update-ablation.v1')
            self.assertEqual(len(update['arms']), 2 * len(result['arms']))
            for backend in result['arms']:
                left, right = backend + '@rebuild', backend + '@incremental'
                self.assertEqual(sum(row['order'].index(left) < row['order'].index(right) for row in update['rounds']), 3)


if __name__ == '__main__':
    unittest.main()
