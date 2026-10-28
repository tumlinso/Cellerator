"""CPU input-contract and raw-bundle compatibility tests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from bench.sparse_support.evaluation.inputs import (
    CapacityExceeded, CapacityLimits, load_raw_bundle, prepare_events,
    prepare_ranked, write_raw_bundle,
)


ROOT = Path(__file__).resolve().parents[2]


def _existing_loader():
    path = ROOT / "bench/sparse_support/packed_training_benchmark.py"
    spec = importlib.util.spec_from_file_location("packed_training_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_raw_bundle


class EvaluationInputTests(unittest.TestCase):
    def topology(self):
        return dict(edge_ids=[90, 10, 70, 30], source_entity_ids=[8, 8, 9, 10],
                    destination_entity_ids=[100, 101, 100, 102])

    def test_events_preserve_occurrences_input_order_and_ordinal(self):
        result = prepare_events(
            **self.topology(), cell_ids=[5, 5, 5, 2], entity_ids=[8, 8, 9, 44],
            occurrence_ids=[501, 502, 503, 204], event_ordinals=[7, 3, 3, -9],
            timestamps=[10.0, 30.0, 20.0, 5.0], position_windows={5: (1, 3)},
            timestamp_windows={5: (15.0, 31.0)}, provenance={"dataset": "test-input"},
        )
        arrays = result.arrays
        # Grouped per cell, but original event sequence is preserved in-cell.
        row = np.flatnonzero(arrays["query_cell_ids"] == 5)
        self.assertEqual(arrays["query_record_ids"][row].tolist(), [501, 502, 503])
        self.assertEqual(arrays["event_ordinals"][row].tolist(), [7, 3, 3])
        self.assertEqual(arrays["event_timestamps"][row].tolist(), [10.0, 30.0, 20.0])
        self.assertEqual(arrays["query_sequence_positions"][row].tolist(), [0, 1, 2])
        self.assertEqual(arrays["query_window_eligible"][row].tolist(), [0, 1, 1])
        self.assertEqual(result.metadata["provenance"], {"dataset": "test-input"})

    def test_rank_is_not_temporal_and_repeats_are_retained(self):
        result = prepare_ranked(
            **self.topology(), cell_ids=[9, 9, 9], gene_ids=[8, 8, 999],
            rank_positions=[3, 0, 2], token_ids=[1001, 1002, 1003],
        )
        self.assertEqual(result.arrays["ranked_entity_ids"].tolist(), [8, 8, 999])
        self.assertEqual(result.arrays["rank_positions"].tolist(), [3, 0, 2])
        self.assertEqual(result.arrays["query_sequence_positions"].tolist(), [0, 1, 2])
        with self.assertRaisesRegex(TypeError, "position_windows"):
            # The ranked API has no window argument by design; rank is not time.
            prepare_ranked(**self.topology(), cell_ids=[9], gene_ids=[8],
                           rank_positions=[0], position_windows={9: (0, 1)})

    def test_conditional_eligibility_is_sparse_and_conjunctive(self):
        result = prepare_events(
            **self.topology(), cell_ids=[4], entity_ids=[8], occurrence_ids=[400],
            event_ordinals=[99], conditions={77: [500, 501], 78: [700]},
            edge_conditions={10: [77], 90: [77, 78]},
            active_conditions={4: [77]},
        )
        self.assertEqual(result.arrays["candidate_edge_ids"].tolist(), [10, 90])
        self.assertEqual(result.arrays["condition_eligible_edge_ids"].tolist(), [10])

    def test_unsorted_duplicate_ids_and_capacities_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "token_ids must be unique"):
            prepare_ranked(**self.topology(), cell_ids=[1, 1], gene_ids=[8, 9],
                           rank_positions=[0, 1], token_ids=[9, 9])
        with self.assertRaises(CapacityExceeded):
            prepare_events(**self.topology(), cell_ids=[1, 1], entity_ids=[8, 8],
                            occurrence_ids=[1, 2], event_ordinals=[0, 1],
                            capacities=CapacityLimits(candidate_items_per_view=1))
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            CapacityLimits(edges=1.5)
        with self.assertRaisesRegex(ValueError, "integer half-open"):
            prepare_events(**self.topology(), cell_ids=[1], entity_ids=[8],
                            occurrence_ids=[1], event_ordinals=[0],
                            position_windows={1: (0.5, 1)})

    def test_large_logical_shape_does_not_create_dense_cell_edge_matrix(self):
        edge_count = 10_000
        result = prepare_events(
            edge_ids=np.arange(100_000, 100_000 + edge_count, dtype=np.uint64),
            source_entity_ids=np.arange(edge_count, dtype=np.int64) * 3,
            destination_entity_ids=np.arange(edge_count, dtype=np.int64) * 5 + 1,
            cell_ids=[0, 127], entity_ids=[0, 3 * (edge_count - 1)],
            occurrence_ids=[1, 2], event_ordinals=[4, 1],
            cell_id_universe=np.arange(128),
        )
        self.assertFalse(result.metadata["materialized_dense_cell_edge_matrix"])
        self.assertEqual(result.arrays["query_cell_offsets"].shape, (129,))
        self.assertEqual(result.arrays["candidate_edge_ids"].size, 2)
        self.assertNotIn("cell_edge_matrix", result.arrays)

    def test_bundle_roundtrip_with_current_harness_loader(self):
        result = prepare_ranked(
            **self.topology(), cell_ids=[3, 3], gene_ids=[8, 9], rank_positions=[1, 0],
            token_ids=[300, 301], targets=np.asarray([1, 0], dtype=np.int8),
            target_kind="classification", provenance={"source": "already-tokenized"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            write_raw_bundle(result, tmp)
            expected, manifest, _ = load_raw_bundle(tmp)
            old_load = _existing_loader()
            selected = ("edge_ids", "ranked_entity_ids", "rank_positions", "targets")
            actual, old_manifest, _ = old_load(Path(tmp), selected)
            self.assertEqual(manifest["schema"], "cellerator.scaled_sparse_fixture_bundle.v1")
            self.assertEqual(old_manifest["schema"], manifest["schema"])
            for name in selected:
                np.testing.assert_array_equal(actual[name], expected[name])

    def test_empty_input_and_zero_stable_id_are_valid(self):
        result = prepare_events(
            edge_ids=[0], source_entity_ids=[-50], destination_entity_ids=[2**40],
            cell_ids=[], entity_ids=[], occurrence_ids=[], event_ordinals=[],
            cell_id_universe=[-1, 99],
        )
        self.assertEqual(result.arrays["edge_ids"].tolist(), [0])
        self.assertEqual(result.arrays["query_cell_offsets"].tolist(), [0, 0, 0])
        self.assertEqual(result.summary["query_records"], 0)

    def test_native_adapter_ids_are_positive_and_reversible(self):
        result = prepare_events(
            edge_ids=[0, -9], source_entity_ids=[0, -20],
            destination_entity_ids=[-20, 0], cell_ids=[-3], entity_ids=[-20],
            occurrence_ids=[-4], event_ordinals=[-100],
        )
        arrays = result.arrays
        self.assertEqual(arrays["edge_ids"].tolist(), [-9, 0])
        self.assertEqual(arrays["native_edge_ids"].tolist(), [1, 2])
        np.testing.assert_array_equal(
            arrays["native_candidate_edge_ids"],
            arrays["native_edge_ids"][arrays["candidate_edge_ordinals"]],
        )
        self.assertTrue(np.all(arrays["native_edge_source_entity_ids"] > 0))
        self.assertEqual(arrays["query_entity_ids"].tolist(), [-20])
        native_entity = int(arrays["native_query_entity_ids"][0])
        self.assertEqual(arrays["native_entity_external_ids"][native_entity - 1], -20)
        self.assertEqual(arrays["query_cell_ids"].tolist(), [-3])
        self.assertEqual(arrays["native_cell_external_ids"].tolist(), [-3])
        self.assertEqual(arrays["event_ordinals"].tolist(), [-100])
        self.assertEqual(arrays["native_event_ordinals"].tolist(), [0])
        self.assertEqual(arrays["event_occurrence_ids"].tolist(), [-4])
        self.assertGreater(int(arrays["native_event_ids"][0]), 0)

    def test_mapping_keys_are_not_lossily_coerced(self):
        for key in (1.5, True):
            with self.subTest(key=key), self.assertRaises(ValueError):
                prepare_events(
                    **self.topology(), cell_ids=[1], entity_ids=[8],
                    occurrence_ids=[1], event_ordinals=[0],
                    active_conditions={key: [7]}, conditions={7: [8]},
                )

    def test_catalog_paths_are_present_and_every_family_is_retained(self):
        from bench.sparse_support.evaluation import mechanism_catalog

        catalog = mechanism_catalog()
        self.assertIn("Retain every family", catalog["selection_policy"])
        self.assertGreaterEqual(len(catalog["families"]), 8)
        family_by_id = {item["id"]: item for item in catalog["families"]}
        self.assertTrue({"counted_flat_conditions", "shared_condition_pair_cache",
                         "parallel_workset_compaction", "prepared_bitmap_retrieval"}
                        .issubset(family_by_id))
        self.assertIn("native specialized", family_by_id[
            "ordered_posting_concatenation"]["status"])
        self.assertIn("v9_corrected/REPORT.md", family_by_id[
            "learned_latent_gates"]["evidence"][0])
        for family in catalog["families"]:
            for field in ("sources", "evidence"):
                for relative in family[field]:
                    self.assertTrue((ROOT / relative).exists(), relative)

    def test_cli_builds_bundle_from_npz_and_json(self):
        from bench.sparse_support.evaluation.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez(root / "input.npz", edge_ids=np.asarray([0, 8], dtype=np.uint64),
                     source_entity_ids=np.asarray([4, 9]),
                     destination_entity_ids=np.asarray([20, 21]),
                     cell_ids=np.asarray([2, 2]), gene_ids=np.asarray([4, 9]),
                     rank_positions=np.asarray([7, 1]))
            (root / "spec.json").write_text(json.dumps({"provenance": {"source": "cli-test"}}))
            self.assertEqual(main(["prepare-ranked", "--input-npz", str(root / "input.npz"),
                                   "--spec-json", str(root / "spec.json"),
                                   "--output", str(root / "bundle")]), 0)
            arrays, manifest, _ = load_raw_bundle(root / "bundle")
            self.assertEqual(manifest["metadata"]["evaluation"]["provenance"],
                             {"source": "cli-test"})
            self.assertEqual(arrays["rank_positions"].tolist(), [7, 1])


if __name__ == "__main__":
    unittest.main()
