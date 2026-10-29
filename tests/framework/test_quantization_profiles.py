"""Reusable per-feature calibration and contextual precision overrides."""
import unittest
import tempfile
from pathlib import Path

import numpy as np

from cellerator.state.quantization_profiles import QuantizationProfile


class QuantizationProfileTests(unittest.TestCase):
    def test_quantile_profiles_share_codebook_arrays_across_reordered_cells(self):
        calibration = np.asarray([[0, 0], [1, 2], [2, 4], [3, 6]], dtype=np.float32)
        profile = QuantizationProfile.calibrate(
            calibration, ("gene-a", "gene-b"), levels=3, bits=2,
            namespace="expression", species="Mus_musculus",
            measurement_id="rna_counts_v1", transform_id="log1p")
        first = profile.encode(np.asarray([[0.8, 5.0]], dtype=np.float32),
                               feature_ids=("gene-a", "gene-b"))
        second = profile.encode(np.asarray([[5.0, 0.8]], dtype=np.float32),
                                feature_ids=("gene-b", "gene-a"))
        self.assertIs(first.codebooks[0], profile.codebooks[0])
        self.assertIs(first.codebooks[1], profile.codebooks[1])
        self.assertEqual(first.profile_id, second.profile_id)
        np.testing.assert_array_equal(first.dequantize(feature_ids=("gene-b", "gene-a")),
                                      second.dequantize())
        self.assertEqual(profile.storage_bytes_for((100, 2), state_count=4),
                         profile.codebook_bytes + 4 * ((200 * profile.bits + 7) // 8))

    def test_identity_compatibility_checks_species_measurement_and_values(self):
        base = QuantizationProfile.from_codebooks(
            ("a", "b"), {"a": (0, 1), "b": (0, 2)}, bits=2,
            namespace="rna", species="human", measurement_id="counts", transform_id="log1p")
        reordered = QuantizationProfile.from_codebooks(
            ("b", "a"), {"a": (0, 1), "b": (0, 2)}, bits=2,
            namespace="rna", species="human", measurement_id="counts", transform_id="log1p")
        other_species = QuantizationProfile.from_codebooks(
            ("a", "b"), {"a": (0, 1), "b": (0, 2)}, bits=2,
            namespace="rna", species="mouse", measurement_id="counts", transform_id="log1p")
        other_transform = QuantizationProfile.from_codebooks(
            ("a", "b"), {"a": (0, 1), "b": (0, 2)}, bits=2,
            namespace="rna", species="human", measurement_id="counts", transform_id="sqrt")
        self.assertTrue(base.compatible_with(reordered))
        self.assertFalse(base.compatible_with(other_species))
        self.assertFalse(base.compatible_with(other_transform))
        selected = base.select(("b",))
        self.assertIs(selected.codebooks[0], base.codebooks[1])
        different_shapes = QuantizationProfile.from_codebooks(
            ("a", "b"), ((0.0, 1.0, 2.0), (0.0,)), bits=2,
            namespace="same", species="human")
        same_bytes_different_shapes = QuantizationProfile.from_codebooks(
            ("a", "b"), ((0.0,), (1.0, 2.0)), bits=2,
            namespace="same", species="human")
        self.assertNotEqual(different_shapes.profile_id, same_bytes_different_shapes.profile_id)

    def test_lloyd_calibration_is_deterministic_and_quantile_is_available(self):
        data = np.asarray([[0.0], [0.1], [0.2], [3.0], [3.2], [3.4]], dtype=np.float64)
        lloyd = QuantizationProfile.calibrate(data, ("signal",), method="lloyd",
                                              levels=2, bits=2, iterations=10)
        again = QuantizationProfile.calibrate(data, ("signal",), method="lloyd",
                                              levels=2, bits=2, iterations=10)
        quantile = QuantizationProfile.calibrate(data, ("signal",), method="quantile",
                                                 levels=2, bits=2)
        np.testing.assert_array_equal(lloyd.codebooks[0], again.codebooks[0])
        self.assertEqual(len(lloyd.codebooks[0]), 2)
        self.assertEqual(len(quantile.codebooks), 1)

    def test_profile_persistence_round_trips_ids_and_codebooks_without_pickle(self):
        profile = QuantizationProfile.from_codebooks(
            ("gene-a", (7, 2)), ((0.0, 1.0), (0.0, 4.0, 8.0)),
            namespace="expression", species="human", measurement_id="counts",
            transform_id="sqrt", bits=2)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.npz"
            profile.save(str(path))
            restored = QuantizationProfile.load(str(path))
        self.assertEqual(restored.profile_id, profile.profile_id)
        self.assertTrue(restored.compatible_with(profile))
        custom = QuantizationProfile.from_codebooks((123,), ((0.0, 1.0),), bits=1)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "custom.npz"
            custom.save(str(path), identity_encoder=lambda identity: str(identity))
            restored_custom = QuantizationProfile.load(str(path), identity_decoder=int)
        self.assertEqual(restored_custom.feature_ids, (123,))

    def test_contextual_exact_values_are_overrides_and_survive_updates(self):
        profile = QuantizationProfile.from_codebooks(
            ("gene-a", "gene-b"), ((0.0, 1.0), (0.0, 10.0)), bits=2,
            namespace="expression", species="human", measurement_id="rna",
            transform_id="normalized")
        values = np.asarray([[0.2, 5.5], [0.8, 9.0]], dtype=np.float32)
        mask = np.asarray([[True, False], [False, True]])
        state = profile.encode(values, exact_mask=mask, override_dtype=np.float64)
        decoded = state.dequantize()
        self.assertEqual(decoded[0, 0], values[0, 0])
        self.assertEqual(decoded[1, 1], values[1, 1])
        self.assertEqual(state.exact_overrides[0], float(values[0, 0]))
        self.assertEqual(state.diagnostics.override_count, 2)
        self.assertEqual(state.diagnostics.override_value_bytes, 16)
        self.assertEqual(state.diagnostics.shared_profile_bytes, profile.codebook_bytes)
        self.assertTrue(state.diagnostics.error_is_current)
        updated = state.update_at("gene-a", 0.35, leading_index=(0,))
        self.assertEqual(updated.dequantize()[0, 0], 0.35)
        self.assertEqual(updated.profile_id, profile.profile_id)
        self.assertFalse(updated.diagnostics.error_is_current)
        packed_update = updated.update_at("gene-a", 0.0, leading_index=(0,), exact=False)
        self.assertNotIn(0, packed_update.exact_overrides)
        self.assertIn(3, packed_update.exact_overrides)

    def test_profile_rejects_unknown_reordered_features_and_incompatible_values(self):
        profile = QuantizationProfile.from_codebooks(("a", "b"), ((0, 1), (0, 2)), bits=2)
        with self.assertRaises(ValueError):
            profile.encode(np.asarray([[0.0, 1.0]]), feature_ids=("b", "missing"))
        with self.assertRaises(ValueError):
            profile.encode(np.asarray([[0.0]]), feature_ids=("a", "a"))
        with self.assertRaises(ValueError):
            QuantizationProfile.calibrate(np.asarray([[np.nan]]), ("a",))


if __name__ == "__main__":
    unittest.main()
