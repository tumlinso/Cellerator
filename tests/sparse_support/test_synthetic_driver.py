import unittest

from examples.sparse_support.train_synthetic import (
    clipped_event_windows, event_fixture_slice, pair_support, shuffled_support,
)
from cellerator.torch.sparse_support import make_next_event_fixture


class SyntheticDriverTest(unittest.TestCase):
    def test_proposed_support_is_seeded_sparse_and_shuffled_control_changes_routes(self):
        first = pair_support(31)
        self.assertEqual(first, pair_support(31))
        self.assertNotEqual(first, pair_support(32))
        self.assertEqual(len(first.edge_ids), 16 * 6)
        self.assertLess(len(first.edge_ids), 0.8 * 16 * 8)
        shuffled = shuffled_support(first, 9)
        self.assertEqual(shuffled.edge_ids, first.edge_ids)
        self.assertEqual(shuffled.sources, first.sources)
        self.assertEqual(len(shuffled.destinations), len(first.destinations))
        self.assertNotEqual(shuffled.destinations, first.destinations)

    def test_event_count_and_time_bounds_are_independent_and_prefix_clipped(self):
        self.assertEqual(clipped_event_windows(11, (6, 11), (20, 38), 31),
                         (6, 11, 20, 31))
        # A timestamp prerequisite can exclude an event that lies inside the
        # ordinal window; an early prefix can also have no eligible history.
        self.assertIsNone(clipped_event_windows(7, (6, 11), (20, 38), 17))

    def test_event_stream_chunk_retains_absolute_windows_for_checkpoint_restore(self):
        fixture = make_next_event_fixture(seed=7, cells=2, steps=9, entities=24)
        suffix = event_fixture_slice(fixture, 4, 9)
        self.assertEqual(tuple(suffix["events"].shape), (2, 5))
        self.assertTrue((suffix["count_windows"] == fixture["count_windows"]).all())
        self.assertTrue((suffix["time_windows"] == fixture["time_windows"]).all())
        self.assertEqual(suffix["ring_capacity"], 9)


if __name__ == "__main__":
    unittest.main()
