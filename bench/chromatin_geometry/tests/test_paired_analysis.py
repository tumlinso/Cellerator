"""CPU-only checks for matched-pair and promotion thresholds."""
import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/analyze_pairs.py'
spec = importlib.util.spec_from_file_location('cgs_pairs', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def dataset():
    return {'schema': 'cgs.paired-trials.v1', 'validation_passed': True,
            'interference_check_passed': True, 'measurement_boundary': 'complete synthetic fixture',
            'strongest_matched_control': 'control', 'pairs': [
                {'case': 'case1', 'repeat': i, 'device_uuid': 'GPU-example',
                 'order': 'control-first' if i % 2 else 'candidate-first',
                 'control_us': 100.0, 'candidate_us': 90.0} for i in range(6)]}


class PairedAnalysisTests(unittest.TestCase):
    def test_promote_only_when_practical_and_interval_pass(self):
        case = module.analyze(dataset(), bootstrap=200)['cases'][0]
        self.assertEqual(case['decision'], 'calculator-threshold-met')
        self.assertGreater(case['paired_bootstrap_95_percent_interval'][0], 1.0)
        d = dataset()
        for pair in d['pairs']:
            pair['candidate_us'] = 96.0
        self.assertEqual(module.analyze(d, bootstrap=200)['cases'][0]['decision'], 'not established')
        for latency, expected in [(95.1, 'not established'), (95.0, 'not established'),
                                  (94.9, 'calculator-threshold-met')]:
            for pair in d['pairs']:
                pair['candidate_us'] = latency
            self.assertEqual(module.analyze(d, bootstrap=200)['cases'][0]['decision'], expected)

    def test_unbalanced_order_is_inconclusive(self):
        d = dataset()
        for pair in d['pairs']:
            pair['order'] = 'control-first'
        self.assertIn('inconclusive', module.analyze(d, bootstrap=200)['cases'][0]['decision'])

    def test_cross_device_or_missing_validation_fails_closed(self):
        d = dataset()
        d['pairs'][0]['candidate_device_uuid'] = 'GPU-other'
        with self.assertRaisesRegex(ValueError, 'same physical GPU'):
            module.analyze(d, bootstrap=200)
        d = dataset()
        d['validation_passed'] = False
        with self.assertRaisesRegex(ValueError, 'prerequisites'):
            module.analyze(d, bootstrap=200)


if __name__ == '__main__':
    unittest.main()
