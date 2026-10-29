"""Probe-conditioned exact overrides for reusable state quantization profiles.

This is a small experimental calibration tool, not an optimizer and not a
claim that a finite probe set preserves every possible model response. It
measures a caller-defined response of a state under a finite set of probes,
then spends a bounded exact-value budget on state coordinates that improve
those measured responses.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import numbers
from typing import Any

import numpy as np

from cellerator.state.quantization import QuantizedState
from cellerator.state.quantization_profiles import QuantizationProfile


@dataclass(frozen=True)
class ResponseProbe:
    """One caller-defined future input and horizon to test during calibration.

    ``inputs`` is deliberately opaque. The response callback decides how it
    affects a model, including continuous time, discrete steps, interventions,
    or coupled-system context.
    """

    name: str
    inputs: Any = None
    horizon: float = 1.0
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("probe name must be a non-empty string")
        try:
            horizon = float(self.horizon)
            weight = float(self.weight)
        except (TypeError, ValueError) as exc:
            raise ValueError("probe horizon and weight must be finite numbers") from exc
        if not np.isfinite(horizon) or horizon < 0:
            raise ValueError("probe horizon must be finite and nonnegative")
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("probe weight must be finite and positive")
        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(self, "weight", weight)


@dataclass(frozen=True)
class ResponseComparison:
    """Measured response error for one probe before and after exact overrides."""

    probe_name: str
    initial_error: float
    final_error: float


@dataclass(frozen=True)
class ExactCoordinate:
    """A selected coordinate in the input state's leading dimensions and feature axis."""

    leading_index: tuple[int, ...]
    feature_index: int
    feature_id: Any


@dataclass(frozen=True)
class ResponseSelectionStep:
    """One greedy exact restoration and its observed per-probe error changes."""

    coordinate: ExactCoordinate
    errors_before: tuple[float, ...]
    errors_after: tuple[float, ...]
    override_count: int
    override_value_bytes: int


@dataclass(frozen=True)
class ResponsePrecisionResult:
    """Quantized payload, exact mask, finite-probe evidence, and consumed budgets."""

    state: QuantizedState
    exact_mask: np.ndarray
    selected_coordinates: tuple[ExactCoordinate, ...]
    comparisons: tuple[ResponseComparison, ...]
    selection_trace: tuple[ResponseSelectionStep, ...]
    candidates_considered: int
    candidate_evaluations: int
    max_candidate_evaluations: int
    max_overrides: int
    max_override_bytes: int | None
    response_tolerance: float
    met_tolerance: bool
    stop_reason: str
    weighted_mean_error: float
    maximum_probe_error: float

    def decode(self, *, dtype: Any | None = None) -> np.ndarray:
        """Reconstruct the selected state using the profile's recorded feature order."""
        return self.state.dequantize(dtype=dtype)


def _default_error(reference: Any, approximate: Any) -> float:
    """Maximum absolute elementwise difference for numeric observations."""
    expected = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(approximate, dtype=np.float64)
    if expected.shape != actual.shape:
        raise ValueError("response outputs must have the same shape")
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("response outputs must be finite")
    return float(np.max(np.abs(expected - actual), initial=0.0))


def _probe_name(probe: Any, index: int) -> str:
    name = getattr(probe, "name", None)
    if name is None:
        return f"probe-{index}"
    if not isinstance(name, str) or not name:
        raise ValueError("each probe name must be a non-empty string")
    return name


def _probe_weight(probe: Any) -> float:
    weight = getattr(probe, "weight", 1.0)
    try:
        value = float(weight)
    except (TypeError, ValueError) as exc:
        raise ValueError("probe weights must be finite positive numbers") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError("probe weights must be finite positive numbers")
    return value


def calibrate_response_precision(
    values: Any,
    profile: QuantizationProfile,
    probes: Sequence[Any],
    response: Callable[[np.ndarray, Any], Any],
    *,
    feature_ids: Sequence[Any] | None = None,
    candidate_mask: Any | None = None,
    max_candidate_evaluations: int = 128,
    max_overrides: int = 16,
    max_override_bytes: int | None = None,
    response_tolerance: float = 0.0,
    error_metric: Callable[[Any, Any], float] | None = None,
    override_dtype: Any | None = None,
) -> ResponsePrecisionResult:
    """Select exact state values that improve responses to supplied probes.

    ``response(state_copy, probe)`` returns the observable to compare. It may
    advance an arbitrary model, but must not rely on mutation of the supplied
    state: every invocation receives a fresh array copy. The original ``values``
    are never passed to the callback. This leaves model semantics and state
    ownership with the caller while allowing finite, model-specific experiments.
    A callback wrapping a stateful ``Cell`` or ``Experiment`` should construct
    a transient instance from ``state_copy`` on each call; it must not advance
    a caller-owned live cell or mutate shared model parameters.

    Candidate coordinates default to every state value. Large states should
    pass a boolean ``candidate_mask`` and keep the candidate count within the
    explicit evaluation cap. Candidates are evaluated greedily by reduction
    in weighted mean response error. Selection stops when every probe is under
    ``response_tolerance``, a budget is exhausted, or no tested restoration
    improves the measured weighted error.

    The reported guarantee is limited to these probes, this state, this
    response callback, and its error metric. It does not establish sufficiency
    for untested inputs, other cells, or later model versions.
    """
    source = np.asarray(values)
    if source.ndim < 1 or source.shape[-1] == 0:
        raise ValueError("values must have a non-empty final feature axis")
    if source.dtype.kind not in "fiu" or not np.isfinite(source).all():
        raise ValueError("values must contain finite real numeric data")
    if not isinstance(profile, QuantizationProfile):
        raise TypeError("profile must be a QuantizationProfile")
    ids = profile.feature_ids if feature_ids is None else tuple(feature_ids)
    try:
        unique_ids = set(ids)
    except TypeError as exc:
        raise TypeError("feature identities must be hashable") from exc
    if len(ids) != source.shape[-1] or len(unique_ids) != len(ids):
        raise ValueError("feature_ids must be unique and match the values final axis")
    if not unique_ids.issubset(profile.feature_ids):
        raise ValueError("feature_ids must be members of the quantization profile")
    if not callable(response):
        raise TypeError("response must be callable")
    probes = tuple(probes)
    if not probes:
        raise ValueError("at least one response probe is required")
    if (isinstance(max_candidate_evaluations, bool)
            or not isinstance(max_candidate_evaluations, numbers.Integral)
            or max_candidate_evaluations < 0):
        raise ValueError("max_candidate_evaluations must be nonnegative")
    if (isinstance(max_overrides, bool) or not isinstance(max_overrides, numbers.Integral)
            or max_overrides < 0):
        raise ValueError("max_overrides must be nonnegative")
    if max_override_bytes is not None and (
            isinstance(max_override_bytes, bool)
            or not isinstance(max_override_bytes, numbers.Integral)
            or max_override_bytes < 0):
        raise ValueError("max_override_bytes must be nonnegative or None")
    max_candidate_evaluations = int(max_candidate_evaluations)
    max_overrides = int(max_overrides)
    if max_override_bytes is not None:
        max_override_bytes = int(max_override_bytes)
    tolerance = float(response_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("response_tolerance must be finite and nonnegative")
    metric = _default_error if error_metric is None else error_metric
    if not callable(metric):
        raise TypeError("error_metric must be callable")

    names = tuple(_probe_name(probe, index) for index, probe in enumerate(probes))
    if len(set(names)) != len(names):
        raise ValueError("probe names must be unique")
    weights = np.asarray([_probe_weight(probe) for probe in probes], dtype=np.float64)
    weights /= weights.sum()

    if candidate_mask is None:
        candidates = np.arange(source.size, dtype=np.int64)
    else:
        mask = np.asarray(candidate_mask, dtype=np.bool_)
        if mask.shape != source.shape:
            raise ValueError("candidate_mask must have the same shape as values")
        candidates = np.flatnonzero(mask.reshape(-1))
    if candidates.size > max_candidate_evaluations:
        raise ValueError(
            f"{candidates.size} candidate coordinates exceed the evaluation cap "
            f"({max_candidate_evaluations}); pass a smaller candidate_mask or raise the cap")

    # Cache the full-state reference responses once. Every callback receives a
    # fresh state copy; this fixed reference is safe to reuse across trials.
    references: list[Any] = []
    for probe in probes:
        references.append(response(np.array(source, copy=True), probe))

    def compare(decoded: np.ndarray) -> tuple[float, ...]:
        errors = []
        for reference, probe in zip(references, probes):
            observed = response(np.array(decoded, copy=True), probe)
            try:
                error = float(metric(reference, observed))
            except (TypeError, ValueError) as exc:
                raise ValueError("error_metric must return a finite nonnegative scalar") from exc
            if not np.isfinite(error) or error < 0:
                raise ValueError("error_metric must return a finite nonnegative scalar")
            errors.append(error)
        return tuple(errors)

    override_value_dtype = np.dtype(source.dtype if override_dtype is None else override_dtype)
    if override_value_dtype.kind not in "fiu":
        raise TypeError("override_dtype must be a real numeric dtype")
    if not np.can_cast(source.dtype, override_value_dtype, casting="safe"):
        raise ValueError("override_dtype must safely preserve the source value dtype")
    value_bytes = override_value_dtype.itemsize
    exact_mask = np.zeros(source.shape, dtype=np.bool_)
    baseline = profile.encode(source, feature_ids=ids, override_dtype=override_dtype)
    current_errors = compare(baseline.dequantize())
    initial_errors = current_errors
    current_mask = exact_mask
    selected: list[ExactCoordinate] = []
    trace: list[ResponseSelectionStep] = []
    candidate_evaluations = 0

    def weighted_error(errors: tuple[float, ...]) -> float:
        return float(np.dot(weights, np.asarray(errors, dtype=np.float64)))

    stop_reason = "tolerance_met" if max(current_errors, default=0.0) <= tolerance else "no_improving_override"
    if stop_reason != "tolerance_met":
        while len(selected) < max_overrides and candidate_evaluations < max_candidate_evaluations:
            if max(current_errors, default=0.0) <= tolerance:
                stop_reason = "tolerance_met"
                break
            if max_override_bytes is not None and (len(selected) + 1) * value_bytes > max_override_bytes:
                stop_reason = "override_byte_budget"
                break

            best: tuple[float, float, int, np.ndarray, tuple[float, ...]] | None = None
            current_objective = weighted_error(current_errors)
            for flat_index in candidates:
                flat_index = int(flat_index)
                if current_mask.reshape(-1)[flat_index]:
                    continue
                trial_mask = current_mask.copy()
                trial_mask.reshape(-1)[flat_index] = True
                trial_state = profile.encode(source, feature_ids=ids,
                                             exact_mask=trial_mask,
                                             override_dtype=override_dtype)
                candidate_evaluations += 1
                trial_errors = compare(trial_state.dequantize())
                objective = weighted_error(trial_errors)
                gain = current_objective - objective
                worst_gain = max(current_errors) - max(trial_errors)
                # Deterministic tie-breaking: objective gain, worst-probe gain,
                # then source coordinate order.
                candidate = (gain, worst_gain, -flat_index, trial_mask, trial_errors)
                if best is None or candidate[:3] > best[:3]:
                    best = candidate
                if candidate_evaluations >= max_candidate_evaluations:
                    break

            if best is None or best[0] <= 0:
                stop_reason = "no_improving_override"
                break
            _, _, neg_flat_index, next_mask, next_errors = best
            flat_index = -neg_flat_index
            coordinate = tuple(int(i) for i in np.unravel_index(flat_index, source.shape))
            feature_index = coordinate[-1]
            leading_index = coordinate[:-1]
            current_mask = next_mask
            selected.append(ExactCoordinate(leading_index, feature_index, ids[feature_index]))
            trace.append(ResponseSelectionStep(
                selected[-1], current_errors, next_errors, len(selected), len(selected) * value_bytes))
            current_errors = next_errors
        else:
            if max(current_errors, default=0.0) <= tolerance:
                stop_reason = "tolerance_met"
            elif len(selected) >= max_overrides:
                stop_reason = "override_count_budget"
            elif candidate_evaluations >= max_candidate_evaluations:
                stop_reason = "candidate_evaluation_budget"

    final_state = profile.encode(source, feature_ids=ids,
                                 exact_mask=current_mask,
                                 override_dtype=override_dtype)
    final_errors = compare(final_state.dequantize())
    met_tolerance = max(final_errors, default=0.0) <= tolerance
    if met_tolerance:
        stop_reason = "tolerance_met"
    current_mask.flags.writeable = False
    comparisons = tuple(ResponseComparison(name, before, after)
                        for name, before, after in zip(names, initial_errors, final_errors))
    return ResponsePrecisionResult(
        state=final_state,
        exact_mask=current_mask,
        selected_coordinates=tuple(selected),
        comparisons=comparisons,
        selection_trace=tuple(trace),
        candidates_considered=int(candidates.size),
        candidate_evaluations=candidate_evaluations,
        max_candidate_evaluations=max_candidate_evaluations,
        max_overrides=max_overrides,
        max_override_bytes=max_override_bytes,
        response_tolerance=tolerance,
        met_tolerance=met_tolerance,
        stop_reason=stop_reason,
        weighted_mean_error=weighted_error(final_errors),
        maximum_probe_error=max(final_errors, default=0.0),
    )


__all__ = [
    "ExactCoordinate",
    "ResponseComparison",
    "ResponsePrecisionResult",
    "ResponseProbe",
    "ResponseSelectionStep",
    "calibrate_response_precision",
]
