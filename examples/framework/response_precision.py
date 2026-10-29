"""A toy threshold response drives one cell-specific exact-value override.

The numbers are a synthetic demonstration of the calibration mechanism, not
claims about a real gene's thresholds or expression distribution.
"""
import numpy as np

from cellerator.experimental.response_precision import (
    ResponseProbe,
    calibrate_response_precision,
)
from cellerator.state.quantization_profiles import QuantizationProfile


def threshold_observation(state: np.ndarray, probe: ResponseProbe) -> np.ndarray:
    """A toy regulatory output: is the first feature above a signal threshold?"""
    signal = np.asarray(probe.inputs, dtype=np.float32)
    return ((state[:, 0] + signal) >= 0.5).astype(np.float32)


def main() -> None:
    # One reusable two-level codebook per feature, shared across both cells.
    profile = QuantizationProfile.from_codebooks(
        ("threshold-regulator", "buffer", "housekeeping"),
        ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        namespace="synthetic-example", bits=1)
    state = np.asarray([[0.49, 0.20, 0.40],
                        [0.49, 0.80, 0.60]], dtype=np.float32)

    # The same regulatory value has different response sensitivity under two
    # cell contexts: a small local signal reaches the threshold only in cell 0.
    probes = (
        ResponseProbe("local-pulse", inputs=np.asarray([0.05, 0.0]), horizon=2.0),
        ResponseProbe("no-pulse", inputs=np.asarray([0.0, 0.0]), horizon=2.0),
    )
    result = calibrate_response_precision(
        state, profile, probes, threshold_observation,
        response_tolerance=0.0,
        max_overrides=1,
        max_override_bytes=state.dtype.itemsize,
    )

    print("exact override mask:\n", result.exact_mask.astype(int))
    print("selected state coordinates:", result.selected_coordinates)
    for comparison in result.comparisons:
        print(f"{comparison.probe_name}: {comparison.initial_error:.1f} -> "
              f"{comparison.final_error:.1f}")
    print("override bytes:", result.state.diagnostics.override_value_bytes)
    print("finite probes met tolerance:", result.met_tolerance)


if __name__ == "__main__":
    main()
