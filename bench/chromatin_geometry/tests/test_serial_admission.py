import importlib.util
from pathlib import Path
import unittest


path = Path(__file__).resolve().parents[1] / 'scripts/campaign.py'
spec = importlib.util.spec_from_file_location('cgs_campaign_serial', path)
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


class SerialAdmissionTest(unittest.TestCase):
    def test_confirmation_requires_all_gpu_jobs_settled(self):
        for state in ('reserving', 'unknown', 'submitted'):
            with self.subTest(state=state), self.assertRaises(RuntimeError):
                campaign.validate_serial_admission({'reservations': [{'state': state}]}, True)
        campaign.validate_serial_admission({'reservations': [{'state': 'settled'}, {'state': 'cpu_build'}]}, True)

    def test_open_confirmation_blocks_all_other_submissions(self):
        for state in ('reserving', 'unknown', 'submitted'):
            row = {'state': state, 'serialized_confirmation': True}
            for incoming in (False, True):
                with self.subTest(state=state, incoming=incoming), self.assertRaises(RuntimeError):
                    campaign.validate_serial_admission({'reservations': [row]}, incoming)
        campaign.validate_serial_admission({'reservations': [{'state': 'settled', 'serialized_confirmation': True}]}, False)


if __name__ == '__main__':
    unittest.main()
