"""Optional _torch bindings for the Torch-free sparse support host contracts."""

from __future__ import annotations

import numpy as np
import importlib
import unittest


class NativeIndexingBindingsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.native = importlib.import_module("cellerator.torch._torch")
        except ImportError as error:
            raise unittest.SkipTest(f"optional _torch extension is unavailable: {error}")

    @classmethod
    def support_handle(cls):
        return cls.native.NativeSupportHandle(
            np.array([10, 20], dtype=np.int64),
            np.array([0, 2, 4], dtype=np.int64),
            np.array([0, 1, 0, 1], dtype=np.int32),
            np.array([1, 2, 1, 3], dtype=np.int64),
            7, 99, 1, 8,
        )

    def test_native_support_publication_and_reverse_index(self):
        handle = self.support_handle()
        index = self.native.NativeReverseIndex(handle)
        result = index.query_any(np.array([0], np.int32), np.array([1], np.int64), 2)
        self.assertEqual(result["logical_edge_ids"].tolist(), [10, 20])
        with self.assertRaises(OverflowError):
            index.query_any(np.array([0], np.int32), np.array([1], np.int64), 1)

        handle.stage_promotion(30, np.array([0, 1], np.int32),
                               np.array([4, 5], np.int64), 11, 4, 0.75)
        handle.stage_revocation(10)
        handle.pin_forward_tape(1)
        with self.assertRaises(RuntimeError):
            handle.publish_pending(2)
        handle.retire_forward_tape(1)
        published = handle.publish_pending_snapshot(2)
        self.assertEqual(published["structure_epoch"], 2)
        snapshot = handle.snapshot()
        self.assertEqual(snapshot["base"]["edge_ids"].tolist(), [20])
        self.assertEqual(snapshot["delta"]["edge_ids"].tolist(), [30])
        self.assertEqual(snapshot["revoked_edge_ids"].tolist(), [10])
        self.assertEqual(snapshot["promotion_evidence_round_ids"].tolist(), [11])
        with self.assertRaisesRegex(ValueError, "stale"):
            index.query_any(np.array([0], np.int32), np.array([1], np.int64), 2)

        handle.stage_promotion(40, np.array([0, 1], np.int32),
                               np.array([6, 7], np.int64), 12, 5, 0.9)
        handle.stage_base_promotion(30)
        handle.publish_pending_snapshot(3)
        snapshot = handle.snapshot()
        self.assertEqual(snapshot["base"]["edge_ids"].tolist(), [20, 30])
        self.assertEqual(snapshot["delta"]["edge_ids"].tolist(), [40])

    def test_cold_snapshot_restore_preserves_partition_tombstones_and_evidence(self):
        source = self.support_handle()
        source.stage_promotion(30, np.array([0, 1], dtype=np.int32),
                               np.array([4, 5], dtype=np.int64), 11, 4, 0.75)
        source.stage_promotion(40, np.array([0, 1], dtype=np.int32),
                               np.array([6, 7], dtype=np.int64), 12, 5, 0.9)
        source.publish_pending_snapshot(2)
        source.stage_base_promotion(30)
        source.stage_revocation(40)
        source_snapshot = source.publish_pending_snapshot(3)
        self.assertEqual(source_snapshot["base"]["edge_ids"].tolist(), [10, 20, 30])
        self.assertEqual(source_snapshot["delta"]["edge_ids"].tolist(), [])
        self.assertEqual(source_snapshot["revoked_edge_ids"].tolist(), [40])

        target = self.support_handle()
        target.pin_forward_tape(1)
        with self.assertRaisesRegex(RuntimeError, "status"):
            target.restore_snapshot(source_snapshot)
        target.retire_forward_tape(1)
        target.restore_snapshot(source_snapshot)
        restored = target.snapshot()
        self.assertEqual(target.structure_epoch, 3)
        self.assertEqual(restored["base"]["edge_ids"].tolist(), [10, 20, 30])
        self.assertEqual(restored["delta"]["edge_ids"].tolist(), [])
        self.assertEqual(restored["revoked_edge_ids"].tolist(), [40])
        self.assertEqual(restored["promotion_evidence_edge_ids"].tolist(), [30, 40])
        self.assertEqual(restored["promotion_evidence_round_ids"].tolist(), [11, 12])
        with self.assertRaisesRegex(RuntimeError, "status"):
            target.stage_promotion(40, np.array([0, 1], dtype=np.int32),
                                   np.array([8, 9], dtype=np.int64))

        conflict = self.native.NativeSupportHandle(
            np.array([10, 20], dtype=np.int64),
            np.array([0, 2, 4], dtype=np.int64),
            np.array([0, 1, 0, 1], dtype=np.int32),
            np.array([1, 2, 4, 3], dtype=np.int64),
            7, 99, 1, 8)
        with self.assertRaisesRegex(RuntimeError, "status"):
            conflict.restore_snapshot(source_snapshot)
        self.assertEqual(conflict.structure_epoch, 1)

    def test_native_temporal_window_keeps_occurrence_identity(self):
        ring = self.native.NativeTemporalOccurrenceRing(1, 7, 3)
        ring.append(100, 0, None)
        ring.append(101, 1, 10)
        ring.append(102, 2, 20)
        result = ring.query_window(1, None, 10, None)
        self.assertEqual(result["occurrence_ids"].tolist(), [101, 102])
        self.assertEqual(result["ordinals"].tolist(), [1, 2])

        with self.assertRaises(OverflowError):
            ring.append(103, (1 << 64) - 1, 30)
        self.assertEqual(ring.query_window()["occurrence_ids"].tolist(), [100, 101, 102])

    def test_native_condition_worksets_and_bounded_proposals(self):
        opcodes = np.array([0, 0, 1], dtype=np.uint8)
        expected = np.ones(3, dtype=np.uint8)
        first_child = np.zeros(3, dtype=np.uint32)
        child_count = np.array([0, 0, 2], dtype=np.uint32)
        predicate_index = np.array([0, 1, 0], dtype=np.uint32)
        binding_ordinals = np.array([0, 2], dtype=np.uint64)
        binding_roots = np.array([2, 0], dtype=np.uint32)
        predicates = np.array([[1, 1], [0, 1]], dtype=np.uint8)
        primal_offsets = np.array([0, 3, 6], dtype=np.int64)
        primal_ids = np.array([20, 30, 40, 20, 30, 40], dtype=np.int64)
        primal_ordinals = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
        derivative_offsets = np.array([0, 1, 2], dtype=np.int64)
        derivative_ids = np.array([40, 20], dtype=np.int64)
        derivative_ordinals = np.array([2, 0], dtype=np.int64)
        work = self.native.build_native_condition_worksets(
            2, 3, 9, opcodes, expected, first_child, child_count, predicate_index,
            binding_ordinals, binding_roots, predicates, primal_offsets, primal_ids,
            primal_ordinals, derivative_offsets, derivative_ids, derivative_ordinals,
            6, 6,
        )
        self.assertEqual(work["condition_bitmap_words"].tolist(), [7, 2])
        self.assertEqual(work["primal_offsets"].tolist(), [0, 3, 4])
        self.assertEqual(work["derivative_offsets"].tolist(), [0, 3, 4])
        self.assertEqual(work["structure_epoch"], 2)
        self.assertEqual(work["state_generation"], 9)

        proposals = self.native.propose_native_support(
            np.array([1, 2, 3], dtype=np.int64),
            np.array([9], dtype=np.int64),
            np.array([2, 3], dtype=np.int64),
            np.array([[1, 2]], dtype=np.int64),
            2, 123, 4, 20,
            np.array([9], dtype=np.int64),
            np.array([[2, 3, -1]], dtype=np.int64),
        )
        self.assertLessEqual(proposals["attempts"], 20)
        self.assertEqual(
            proposals["participant_ids"].shape,
            (len(proposals["mechanism_ids"]), 3),
        )
        self.assertTrue(proposals["missed_candidate_count_exact"])
        self.assertFalse(proposals["candidate_space_upper_bound_saturated"])

    def test_native_rotating_derivative_sample_preserves_admitted(self):
        result = self.native.build_native_derivative_sample(
            np.array([0, 1, 1], dtype=np.int64),
            np.array([2], dtype=np.int64),
            np.array([1], dtype=np.int64),
            np.array([0, 3, 5], dtype=np.int64),
            np.array([1, 2, 3, 6, 7], dtype=np.int64),
            np.array([0, 1, 2, 5, 6], dtype=np.int64),
            np.array([1, 0], dtype=np.int64),
            4, 123, 9, 17,
        )
        self.assertEqual(result["derivative_offsets"].tolist(), [0, 2, 2])
        row_ids = result["derivative_edge_ids"].tolist()
        self.assertIn(2, row_ids)
        self.assertEqual(len(set(row_ids)), len(row_ids))
        self.assertEqual(result["eligible_dormant_counts"].tolist(), [2, 2])
        self.assertEqual(result["selected_dormant_counts"].tolist(), [1, 0])
        self.assertEqual(result["remaining_dormant_counts"].tolist(), [1, 2])
        self.assertEqual(result["structure_epoch"], 9)
        self.assertEqual(result["state_generation"], 17)
        self.assertEqual(result["sampling_policy"], "biased_cyclic_v1")
