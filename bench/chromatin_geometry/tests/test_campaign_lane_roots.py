import importlib.util
import tempfile
import unittest
from pathlib import Path

path = Path(__file__).resolve().parents[1] / 'scripts/campaign.py'
spec = importlib.util.spec_from_file_location('cgs_lane_roots', path)
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)
freezer_spec = importlib.util.spec_from_file_location('cgs_lane_freezer', path.with_name('freeze_source.py'))
freezer = importlib.util.module_from_spec(freezer_spec)
freezer_spec.loader.exec_module(freezer)


class LaneRootTests(unittest.TestCase):
    def test_lane_root_stays_inside_study_even_through_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            study = parent / 'study'
            study.mkdir()
            (study / 'escape').symlink_to(parent, target_is_directory=True)
            old = campaign.STUDY
            campaign.STUDY = study
            try:
                self.assertEqual(campaign.validated_lane_root(study / 'new-lanes'), study / 'new-lanes')
                for bad in (study, parent, study / 'escape' / 'elsewhere'):
                    with self.subTest(path=bad), self.assertRaises(RuntimeError):
                        campaign.validated_lane_root(bad)
            finally:
                campaign.STUDY = old

    def test_lane_identity_and_legacy_rows_fail_closed(self):
        old = campaign.LANES
        campaign.LANES = Path('/study/new-lanes')
        try:
            self.assertTrue(campaign.same_lane({'lane': 0}, 0))
            self.assertTrue(campaign.same_lane({'lane': 0, 'lane_root': '/study/new-lanes'}, 0))
            self.assertFalse(campaign.same_lane({'lane': 0, 'lane_root': '/study/other-lanes'}, 0))
            self.assertFalse(campaign.same_lane({'lane': 1}, 0))
        finally:
            campaign.LANES = old

    def test_serialized_confirmation_spans_roots(self):
        ledger = {'reservations': [{'lane': 0, 'lane_root': '/study/older', 'state': 'submitted'}]}
        with self.assertRaisesRegex(RuntimeError, 'all study GPU reservations'):
            campaign.validate_serial_admission(ledger, True)

    def test_new_lane_state_excluded_from_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'study/.campaign-lanes-20261103').mkdir(parents=True)
            (root / 'study/.campaign-lanes-20261103/state').write_text('runtime')
            (root / 'study/input').write_text('source')
            files = {}
            freezer.collect(root, 'study', 'Source', files)
            self.assertEqual(set(files), {'Source/study/input'})


if __name__ == '__main__':
    unittest.main()
