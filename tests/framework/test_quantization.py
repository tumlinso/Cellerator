"""Focused checks for explicit compact quantization of state values."""
import unittest

import numpy as np

from cellerator.state.quantization import quantize_state


class QuantizationTests(unittest.TestCase):
    def test_binary_state_thresholds_and_identity_order(self):
        values = np.asarray([[0.0, 8.0, 2.0], [3.0, 1.0, 9.0]], dtype=np.float32)
        state = quantize_state(values, bits=1, feature_ids=("gA", "gB", "peak"),
                               thresholds={"gA": 1.0, "gB": 4.0, "peak": 5.0},
                               binary_levels=(0.0, 1.0))
        self.assertEqual(state.packed_codes.nbytes, 1)
        np.testing.assert_array_equal(state.dequantize(), [[0, 1, 0], [1, 0, 1]])
        np.testing.assert_array_equal(state.dequantize(feature_ids=("peak", "gA", "gB")),
                                      [[0, 0, 1], [1, 1, 0]])
        self.assertEqual(state.diagnostics.maximum_absolute_error, 8.0)

    def test_uniform_code_pack_round_trip_and_error_diagnostics(self):
        rng = np.random.default_rng(11)
        for bits in (1, 2, 4, 8):
            values = rng.uniform(-3.0, 4.0, size=(5, 7)).astype(np.float32)
            state = quantize_state(values, bits=bits, feature_ids=tuple(f"g{i}" for i in range(7)))
            decoded = state.dequantize()
            bound = (values.max(axis=0) - values.min(axis=0)).max() / ((1 << bits) - 1) + 1e-6
            self.assertLessEqual(float(np.max(np.abs(decoded - values))), bound)
            self.assertAlmostEqual(state.diagnostics.root_mean_square_error,
                                   float(np.sqrt(np.mean((decoded - values) ** 2))), places=6)
            self.assertEqual(state.packed_codes.nbytes, (values.size * bits + 7) // 8)

    def test_integer_source_decodes_to_real_values_and_accepts_array_ranges(self):
        values = np.asarray([[0, 100], [20, 200]], dtype=np.int32)
        state = quantize_state(values, bits=2, feature_ids=("a", "b"),
                               value_range=np.asarray([[0, 100], [20, 200]]))
        decoded = state.dequantize()
        self.assertEqual(decoded.dtype, np.dtype(np.float64))
        np.testing.assert_allclose(decoded, [[0, 100], [20, 200]])

    def test_codebooks_follow_identity_through_feature_reordering(self):
        original = np.asarray([[0.8, 8.0], [0.1, 1.7]], dtype=np.float32)
        books = {"gene-b": (0.0, 2.0, 10.0), "gene-a": (0.0, 1.0)}
        state = quantize_state(original, bits=2, feature_ids=("gene-a", "gene-b"),
                               codebooks=books)
        reordered_values = original[:, [1, 0]]
        reordered = quantize_state(reordered_values, bits=2,
                                   feature_ids=("gene-b", "gene-a"), codebooks=books)
        np.testing.assert_array_equal(state.dequantize(feature_ids=("gene-b", "gene-a")),
                                      reordered.dequantize())
        self.assertLessEqual(state.diagnostics.maximum_absolute_error, 2.0)

    def test_sparse_incremental_update_uses_feature_identity_and_leading_index(self):
        values = np.asarray([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]], dtype=np.float32)
        state = quantize_state(values, bits=4, feature_ids=("a", "b", "c"))
        updated = state.update_at("a", 1.0, leading_index=(1,))
        decoded = updated.dequantize()
        self.assertLessEqual(abs(float(decoded[1, 0]) - 1.0), 1.0 / 15.0 + 1e-6)
        self.assertEqual(int(updated._codes()[1, 0]), state.code("a", 1.0))
        self.assertEqual(decoded[0, 0], state.dequantize()[0, 0])
        self.assertFalse(updated.diagnostics.error_is_current)
        batch = state.update_many([((0,), "c", 1.0), ((1,), "a", 0.0)])
        self.assertLessEqual(abs(float(batch.dequantize()[0, 2]) - 1.0), 1.0 / 15.0 + 1e-6)
        self.assertEqual(batch.dequantize()[1, 0], 0.0)

    def test_invalid_modes_ranges_and_identity_sets_fail_clearly(self):
        values = np.ones((2, 2), dtype=np.float32)
        with self.assertRaises(ValueError):
            quantize_state(values, bits=3)
        one_bit = quantize_state(values, bits=1)
        self.assertEqual(one_bit.bits, 1)
        with self.assertRaises(ValueError):
            quantize_state(values, feature_ids=("a", "a"))
        with self.assertRaises(ValueError):
            quantize_state(values, bits=2, value_range=([1.0, 0.0], [0.0, 1.0]))
        with self.assertRaises(ValueError):
            quantize_state(values, bits=1, feature_ids=("a", "b"),
                           thresholds={"a": 0.5})


if __name__ == "__main__":
    unittest.main()
