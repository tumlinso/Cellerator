"""Finite-probe calibration for state-dependent exact quantization overrides."""
import unittest

import numpy as np

from cellerator.experimental.response_precision import (
    ResponseProbe,
    calibrate_response_precision,
)
from cellerator.state.quantization_profiles import QuantizationProfile


class ResponsePrecisionTests(unittest.TestCase):
    def setUp(self):
        self.profile = QuantizationProfile.from_codebooks(
            ("gate", "buffer", "idle"),
            ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
            namespace="toy-expression", species="example-organism",
            measurement_id="normalized-expression", transform_id="identity",
            bits=1)

    @staticmethod
    def threshold_response(state, probe):
        shift = np.asarray(probe.inputs)
        return ((state[:, 0] + shift) >= 0.5).astype(np.float32)

    def test_exact_mask_is_selected_for_context_sensitive_cell_coordinate(self):
        values = np.asarray([[0.49, 0.20, 0.40],
                             [0.49, 0.80, 0.60]], dtype=np.float32)
        source = values.copy()
        probes = (ResponseProbe("local-pulse", inputs=np.asarray([0.05, 0.0]),
                                horizon=2.0),)

        result = calibrate_response_precision(
            values, self.profile, probes, self.threshold_response,
            response_tolerance=0.0, max_overrides=1)

        self.assertTrue(result.met_tolerance)
        self.assertEqual(result.stop_reason, "tolerance_met")
        np.testing.assert_array_equal(result.exact_mask,
                                      np.asarray([[True, False, False],
                                                  [False, False, False]]))
        self.assertEqual(result.selected_coordinates[0].leading_index, (0,))
        self.assertEqual(result.selected_coordinates[0].feature_id, "gate")
        self.assertEqual(result.comparisons[0].initial_error, 1.0)
        self.assertEqual(result.comparisons[0].final_error, 0.0)
        self.assertEqual(result.state.diagnostics.override_count, 1)
        self.assertEqual(result.state.diagnostics.override_value_bytes, values.dtype.itemsize)
        self.assertEqual(result.candidate_evaluations, 6)
        np.testing.assert_array_equal(values, source)

    def test_responses_see_isolated_state_copies(self):
        values = np.asarray([[0.49, 0.2, 0.4]], dtype=np.float32)
        source = values.copy()

        def mutating_response(state, probe):
            state[:, 1] = -123.0
            return self.threshold_response(state, probe)

        result = calibrate_response_precision(
            values, self.profile,
            (ResponseProbe("pulse", inputs=np.asarray([0.05])),),
            mutating_response, response_tolerance=0.0, max_overrides=1)

        self.assertTrue(result.met_tolerance)
        np.testing.assert_array_equal(values, source)

    def test_budget_exhaustion_is_reported_without_false_tolerance_claim(self):
        values = np.asarray([[0.49, 0.2, 0.4]], dtype=np.float32)
        result = calibrate_response_precision(
            values, self.profile,
            (ResponseProbe("pulse", inputs=np.asarray([0.05])),),
            self.threshold_response, response_tolerance=0.0, max_overrides=0)

        self.assertFalse(result.met_tolerance)
        self.assertEqual(result.stop_reason, "override_count_budget")
        self.assertEqual(result.state.diagnostics.override_count, 0)
        self.assertGreater(result.maximum_probe_error, 0.0)

    def test_candidate_search_must_be_explicitly_bounded(self):
        values = np.ones((3, 3), dtype=np.float32) * 0.49
        with self.assertRaisesRegex(ValueError, "exceed the evaluation cap"):
            calibrate_response_precision(
                values, self.profile,
                (ResponseProbe("pulse", inputs=np.zeros(3)),),
                self.threshold_response, max_candidate_evaluations=4)

    def test_no_response_gain_does_not_spend_exact_overrides(self):
        values = np.asarray([[0.49, 0.2, 0.4]], dtype=np.float32)
        result = calibrate_response_precision(
            values, self.profile, (ResponseProbe("constant"),),
            lambda state, probe: np.asarray([1.0]),
            response_tolerance=0.0, max_overrides=2)

        self.assertTrue(result.met_tolerance)
        self.assertEqual(result.stop_reason, "tolerance_met")
        self.assertEqual(result.selected_coordinates, ())
        self.assertEqual(result.state.diagnostics.override_count, 0)


if __name__ == "__main__":
    unittest.main()
