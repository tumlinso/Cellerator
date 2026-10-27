import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench" / "sparse_support"))

from scaled_fixtures import make_scaled_fixture, profile, write_raw_bundle  # noqa: E402


class ScaledSparseFixtureTest(unittest.TestCase):
    def test_small_fixture_is_deterministic_and_query_csr_matches_oracle(self):
        a = profile("small", seed=17)
        b = profile("small", seed=17)
        self.assertEqual(a.metadata["topology_sha256"], b.metadata["topology_sha256"])
        for key in a.arrays:
            self.assertTrue(np.array_equal(a[key], b[key]), key)
        self.assertIn("empty", a.metadata["query_case_names"])
        for row in range(a["query_case_size"].size):
            qlo, qhi = a["query_indptr"][row:row + 2]
            terms = set(a["query_entities"][qlo:qhi].tolist())
            expected = sorted({int(edge_id)
                               for term in terms
                               for edge_id in a["source_edge_ids"][
                                   a["source_edge_indptr"][term]:
                                   a["source_edge_indptr"][term + 1]]})
            elo, ehi = a["query_edge_indptr"][row:row + 2]
            self.assertEqual(a["query_edge_ids"][elo:ehi].tolist(), expected)
        # Source postings contain each stable ID once and remain sorted.
        for entity in range(a.metadata["entities"]):
            lo, hi = a["source_edge_indptr"][entity:entity + 2]
            row = a["source_edge_ids"][lo:hi]
            self.assertTrue(np.all(row[1:] > row[:-1]))

    def test_shuffle_changes_physical_layout_only(self):
        identity = make_scaled_fixture(edges=1024, entities=128, cells=8,
                                       modules=8, gates=8, events_per_cell=16,
                                       cohort_size=4, seed=91, layout="identity",
                                       query_sizes=(0, 1, 8, 32))
        shuffled = make_scaled_fixture(edges=1024, entities=128, cells=8,
                                       modules=8, gates=8, events_per_cell=16,
                                       cohort_size=4, seed=91, layout="shuffled",
                                       query_sizes=(0, 1, 8, 32))
        self.assertEqual(identity.metadata["topology_sha256"],
                         shuffled.metadata["topology_sha256"])
        layout_specific = {"physical_to_logical_ordinal", "physical_to_logical",
                           "logical_to_physical", "layout_tag",
                           "physical_source_entity_id", "physical_participant_entity_id"}
        for key in identity.arrays.keys() - layout_specific:
            self.assertTrue(np.array_equal(identity[key], shuffled[key]), key)
        self.assertFalse(np.array_equal(identity["physical_to_logical"],
                                         shuffled["physical_to_logical"]))
        np.testing.assert_array_equal(
            shuffled["physical_source_entity_id"],
            shuffled["source_ids"][shuffled["physical_to_logical"]])

    def test_events_gates_cohorts_and_ranked_identity_are_explicit(self):
        fixture = profile("small", seed=33)
        arrays = fixture.arrays
        self.assertEqual(arrays["event_offsets"][-1], arrays["event_edge_id"].size)
        self.assertEqual(arrays["event_timestamps"].size, arrays["event_ordinals"].size)
        self.assertTrue(np.any(np.unique(arrays["event_edge_id"], return_counts=True)[1] > 1))
        for cell in range(fixture.metadata["cells"]):
            count_lo, count_hi = arrays["count_window_bounds"][cell]
            time_lo, time_hi = arrays["time_window_bounds"][cell]
            event_lo, event_hi = arrays["event_offsets"][cell:cell + 2]
            ordinals = arrays["event_ordinals"][event_lo:event_hi]
            timestamps = arrays["event_timestamps"][event_lo:event_hi]
            count_selected = ((ordinals >= count_lo) & (ordinals < count_hi))
            time_selected = ((timestamps >= time_lo) & (timestamps < time_hi))
            np.testing.assert_array_equal(count_selected, time_selected)
            row = cell * 2 + 1
            wlo, whi = arrays["window_event_indptr"][row:row + 2]
            np.testing.assert_array_equal(
                arrays["window_event_indices"][wlo:whi],
                np.arange(event_lo + count_lo, event_lo + count_hi))
        # Members of a cohort reuse exactly the same query support but have
        # distinct per-cell state values.
        for cohort in np.unique(arrays["cell_cohort_id"]):
            cells = np.flatnonzero(arrays["cell_cohort_id"] == cohort)
            if cells.size > 1:
                first = cells[0]
                lo, hi = arrays["query_offsets"][first:first + 2]
                for cell in cells[1:]:
                    lo2, hi2 = arrays["query_offsets"][cell:cell + 2]
                    np.testing.assert_array_equal(arrays["query_terms"][lo:hi],
                                                  arrays["query_terms"][lo2:hi2])
                    vlo, vhi = first * 24, (first + 1) * 24
                    vlo2, vhi2 = cell * 24, (cell + 1) * 24
                    self.assertFalse(np.array_equal(arrays["cell_selected_values"][vlo:vhi],
                                                    arrays["cell_selected_values"][vlo2:vhi2]))
        self.assertEqual(arrays["rank_offsets"][-1], arrays["rank_entity_ids"].size)
        self.assertEqual(arrays["edge_condition_indptr"][-1],
                         arrays["edge_condition_predicate_id"].size)
        self.assertEqual(arrays["edge_condition_predicate_id"].size,
                         fixture.metadata["edges"] * 2)
        self.assertEqual(arrays["source_degree_histogram_counts"].sum(),
                         fixture.metadata["entities"])
        for group in np.unique(arrays["gate_change_group_id"]):
            rows = np.flatnonzero(arrays["gate_change_group_id"] == group)
            self.assertEqual(np.unique(arrays["gate_change_predicate_id"][rows]).size, 1)
            self.assertEqual(np.unique(arrays["gate_change_timestamp"][rows]).size, 1)
        self.assertTrue(fixture.metadata["cell_state_values_independent"])

    def test_raw_bundle_hashes_and_roundtrips_arrays(self):
        fixture = profile("small", seed=5)
        with tempfile.TemporaryDirectory() as temp:
            manifest_path = write_raw_bundle(fixture, temp)
            manifest = json.loads(Path(manifest_path).read_text())
            for name, descriptor in manifest["arrays"].items():
                raw = (Path(temp) / descriptor["file"]).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), descriptor["sha256"])
                loaded = np.frombuffer(raw, dtype=np.dtype(descriptor["dtype"]))
                loaded = loaded.reshape(descriptor["shape"])
                np.testing.assert_array_equal(loaded, fixture[name])

    def test_large_profiles_have_target_shapes_without_dense_batch_edge_arrays(self):
        ranked = profile("large-ranked", seed=7)
        self.assertEqual((ranked.metadata["edges"], ranked.metadata["entities"],
                          ranked.metadata["cells"]), (262_144, 4_096, 32))
        self.assertNotIn("cell_edge_state", ranked.arrays)
        events = profile("large-events", seed=8)
        self.assertEqual((events.metadata["edges"], events.metadata["entities"],
                          events.metadata["cells"]), (1_280_000, 20_000, 8))
        self.assertFalse(any(value.shape == (events.metadata["cells"],
                                             events.metadata["edges"])
                             for value in events.arrays.values()))
        self.assertEqual(events.metadata["event_occurrences"],
                         events.metadata["cells"] * events.metadata["events_per_cell"])
        self.assertEqual(events.metadata["array_bytes"],
                         sum(value.nbytes for value in events.arrays.values()))


if __name__ == "__main__":
    unittest.main()
