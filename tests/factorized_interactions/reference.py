"""Independent scalar oracle for the factorized interaction law.

This deliberately uses semantic Python records and float64 scalar loops.  It
does not traverse or emulate packet storage, CUDA thread ownership, or the
candidate's forward/backward algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Mechanism:
    """One ordered interaction with canonical parameter-bank identities."""

    target: int
    participants: tuple[int, ...]
    context_ids: tuple[int, ...]
    gamma_ids: tuple[int, ...]
    bias_id: int
    alpha_id: int


@dataclass(frozen=True)
class Problem:
    """Inputs use public logical axes, before any physical packet encoding.

    * ``x[occurrence, channel, cell]``
    * ``z[context, cell]``
    * ``alpha[alpha_row, channel]``
    * ``bias[bias_id]`` and ``gamma[gamma_id]``
    * ``target`` indexes the first dimension of the returned output.

    Array dtypes describe the stored values.  The oracle reads those exact
    quantized values into float64 and evaluates the law there.
    """

    x: np.ndarray
    z: np.ndarray
    alpha: np.ndarray
    bias: np.ndarray
    gamma: np.ndarray
    mechanisms: tuple[Mechanism, ...]
    target_count: int


@dataclass
class VJP:
    dx: np.ndarray
    dz: np.ndarray
    dalpha: np.ndarray
    dbias: np.ndarray
    dgamma: np.ndarray


@dataclass
class VJPScale:
    """Sum of absolute per-interaction contributions for rounding bounds."""

    dx: np.ndarray
    dz: np.ndarray
    dalpha: np.ndarray
    dbias: np.ndarray
    dgamma: np.ndarray


@dataclass
class Evaluation:
    # Unrounded float64 sum and requested storage output are both retained.
    y_float64: np.ndarray
    y: np.ndarray
    sum_abs_y_contributions: np.ndarray
    vjp: Optional[VJP] = None
    vjp_scale: Optional[VJPScale] = None


def _as64(array: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(array)
    if value.dtype.kind != "f":
        raise TypeError(f"{name} must have a floating dtype")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value.astype(np.float64, copy=False)


def validate_problem(problem: Problem) -> None:
    x = _as64(problem.x, "x")
    z = _as64(problem.z, "z")
    alpha = _as64(problem.alpha, "alpha")
    bias = _as64(problem.bias, "bias")
    gamma = _as64(problem.gamma, "gamma")
    if x.ndim != 3:
        raise ValueError("x must have axes [occurrence, channel, cell]")
    if z.ndim != 2:
        raise ValueError("z must have axes [context, cell]")
    if alpha.ndim != 2 or alpha.shape[1] != x.shape[1]:
        raise ValueError("alpha must have axes [row, channel] matching x")
    if bias.ndim != 1 or gamma.ndim != 1:
        raise ValueError("bias and gamma must be one-dimensional parameter banks")
    if z.shape[1] != x.shape[2]:
        raise ValueError("z and x must have the same cell extent")
    if not isinstance(problem.target_count, int) or problem.target_count < 0:
        raise ValueError("target_count must be a nonnegative integer")
    if not isinstance(problem.mechanisms, tuple):
        raise TypeError("mechanisms must be a tuple")
    for index, mechanism in enumerate(problem.mechanisms):
        if not isinstance(mechanism, Mechanism):
            raise TypeError(f"mechanisms[{index}] is not a Mechanism")
        arity = len(mechanism.participants)
        if arity not in (2, 3, 4):
            raise ValueError(f"mechanisms[{index}] has unsupported arity {arity}")
        if len(mechanism.context_ids) != len(mechanism.gamma_ids):
            raise ValueError(f"mechanisms[{index}] context/gamma term counts differ")
        if not (0 <= mechanism.target < problem.target_count):
            raise IndexError(f"mechanisms[{index}] target is out of range")
        if not (0 <= mechanism.bias_id < bias.size):
            raise IndexError(f"mechanisms[{index}] bias_id is out of range")
        if not (0 <= mechanism.alpha_id < alpha.shape[0]):
            raise IndexError(f"mechanisms[{index}] alpha_id is out of range")
        for occurrence in mechanism.participants:
            if not (0 <= occurrence < x.shape[0]):
                raise IndexError(f"mechanisms[{index}] participant is out of range")
        for context_id, gamma_id in zip(mechanism.context_ids, mechanism.gamma_ids):
            if not (0 <= context_id < z.shape[0]):
                raise IndexError(f"mechanisms[{index}] context_id is out of range")
            if not (0 <= gamma_id < gamma.size):
                raise IndexError(f"mechanisms[{index}] gamma_id is out of range")


def _sigmoid(value: float) -> float:
    """Overflow-safe scalar sigmoid in float64."""
    if value >= 0.0:
        tail = np.exp(-value)
        return float(1.0 / (1.0 + tail))
    head = np.exp(value)
    return float(head / (1.0 + head))


def _gate(problem: Problem, mechanism: Mechanism, z64: np.ndarray,
          bias64: np.ndarray, gamma64: np.ndarray, cell: int) -> tuple[float, float]:
    preactivation = float(bias64[mechanism.bias_id])
    for context_id, gamma_id in zip(mechanism.context_ids, mechanism.gamma_ids):
        preactivation += float(gamma64[gamma_id]) * float(z64[context_id, cell])
    gate = _sigmoid(preactivation)
    return gate, preactivation


def evaluate(problem: Problem, dy: Optional[np.ndarray] = None,
             output_dtype: np.dtype | type = np.float16) -> Evaluation:
    """Evaluate the law and optionally its full vector-Jacobian product.

    ``y_float64`` is the sum before output storage conversion.  ``dy`` is the
    native FP32 cotangent by contract; its supplied values are accumulated in
    float64 by this oracle.  Gradients are returned in float64 so callers can
    apply the frozen FP32 error model without losing reference precision.
    """
    validate_problem(problem)
    x = _as64(problem.x, "x")
    z = _as64(problem.z, "z")
    alpha = _as64(problem.alpha, "alpha")
    bias = _as64(problem.bias, "bias")
    gamma = _as64(problem.gamma, "gamma")
    occurrences, channels, cells = x.shape
    y64 = np.zeros((problem.target_count, channels, cells), dtype=np.float64)
    y_scale = np.zeros_like(y64)

    for mechanism in problem.mechanisms:
        for cell in range(cells):
            gate, _ = _gate(problem, mechanism, z, bias, gamma, cell)
            for channel in range(channels):
                product = 1.0
                for occurrence in mechanism.participants:
                    product *= float(x[occurrence, channel, cell])
                contribution = (gate * float(alpha[mechanism.alpha_id, channel])
                                * product)
                index = (mechanism.target, channel, cell)
                y64[index] += contribution
                y_scale[index] += abs(contribution)

    if dy is None:
        return Evaluation(y64, y64.astype(output_dtype), y_scale)

    upstream = _as64(dy, "dy")
    if upstream.shape != y64.shape:
        raise ValueError(f"dy shape {upstream.shape} does not match output {y64.shape}")

    dx = np.zeros_like(x)
    dz = np.zeros_like(z)
    dalpha = np.zeros_like(alpha)
    dbias = np.zeros_like(bias)
    dgamma = np.zeros_like(gamma)
    scale_x = np.zeros_like(x)
    scale_z = np.zeros_like(z)
    scale_alpha = np.zeros_like(alpha)
    scale_bias = np.zeros_like(bias)
    scale_gamma = np.zeros_like(gamma)

    # The role-excluding product is recomputed directly.  In particular, no
    # derivative divides by a participant value, so zeros and repeated roles
    # have their ordinary finite product derivatives.
    for mechanism in problem.mechanisms:
        for cell in range(cells):
            gate, _ = _gate(problem, mechanism, z, bias, gamma, cell)
            gate_adjoint = 0.0
            for channel in range(channels):
                cotangent = float(upstream[mechanism.target, channel, cell])
                product = 1.0
                for occurrence in mechanism.participants:
                    product *= float(x[occurrence, channel, cell])
                alpha_value = float(alpha[mechanism.alpha_id, channel])
                alpha_contribution = cotangent * gate * product
                dalpha[mechanism.alpha_id, channel] += alpha_contribution
                scale_alpha[mechanism.alpha_id, channel] += abs(alpha_contribution)
                gate_contribution = cotangent * alpha_value * product
                gate_adjoint += gate_contribution
                for role, occurrence in enumerate(mechanism.participants):
                    role_product = 1.0
                    for other_role, other_occurrence in enumerate(mechanism.participants):
                        if role != other_role:
                            role_product *= float(x[other_occurrence, channel, cell])
                    input_contribution = (cotangent * gate * alpha_value
                                          * role_product)
                    dx[occurrence, channel, cell] += input_contribution
                    scale_x[occurrence, channel, cell] += abs(input_contribution)

            sigmoid_adjoint = gate_adjoint * gate * (1.0 - gate)
            dbias[mechanism.bias_id] += sigmoid_adjoint
            scale_bias[mechanism.bias_id] += abs(sigmoid_adjoint)
            for context_id, gamma_id in zip(mechanism.context_ids,
                                            mechanism.gamma_ids):
                gamma_contribution = sigmoid_adjoint * float(z[context_id, cell])
                context_contribution = sigmoid_adjoint * float(gamma[gamma_id])
                dgamma[gamma_id] += gamma_contribution
                dz[context_id, cell] += context_contribution
                scale_gamma[gamma_id] += abs(gamma_contribution)
                scale_z[context_id, cell] += abs(context_contribution)

    vjp = VJP(dx, dz, dalpha, dbias, dgamma)
    scale = VJPScale(scale_x, scale_z, scale_alpha, scale_bias, scale_gamma)
    return Evaluation(y64, y64.astype(output_dtype), y_scale, vjp, scale)


def contract_allowance(sum_abs_contributions: np.ndarray) -> np.ndarray:
    """Root-frozen FP32 absolute error allowance, applied elementwise."""
    scale = np.asarray(sum_abs_contributions, dtype=np.float64)
    if not np.isfinite(scale).all() or np.any(scale < 0.0):
        raise ValueError("contribution scales must be finite and nonnegative")
    return 2.0e-6 + 2.0e-5 * scale


def fp16_ulp(value: np.ndarray) -> np.ndarray:
    """Largest adjacent FP16 spacing at each correctly rounded value."""
    value = np.asarray(value, dtype=np.float16)
    positive = np.nextafter(value, np.full_like(value, np.inf))
    negative = np.nextafter(value, np.full_like(value, -np.inf))
    return np.maximum(np.abs(positive.astype(np.float64) - value.astype(np.float64)),
                       np.abs(value.astype(np.float64) - negative.astype(np.float64)))


def assert_fp32_contract(actual: np.ndarray, reference: np.ndarray,
                         sum_abs_contributions: np.ndarray, label: str) -> None:
    actual64 = np.asarray(actual, dtype=np.float64)
    reference64 = np.asarray(reference, dtype=np.float64)
    scales = np.asarray(sum_abs_contributions, dtype=np.float64)
    if actual64.shape != reference64.shape or scales.shape != reference64.shape:
        raise AssertionError(f"{label}: shape mismatch")
    error = np.abs(actual64 - reference64)
    allowance = contract_allowance(scales)
    if not np.isfinite(actual64).all() or np.any(error > allowance):
        worst = int(np.argmax(error - allowance)) if error.size else 0
        detail = "empty tensor" if error.size == 0 else (
            f"index={np.unravel_index(worst, error.shape)}, "
            f"actual={actual64.flat[worst]:.9g}, "
            f"reference={reference64.flat[worst]:.9g}, "
            f"error={error.flat[worst]:.3g}, allowance={allowance.flat[worst]:.3g}")
        raise AssertionError(f"{label}: outside FP32 contract ({detail})")


def assert_fp16_contract(actual: np.ndarray, rounded_reference: np.ndarray,
                         sum_abs_contributions: np.ndarray, label: str) -> None:
    actual64 = np.asarray(actual, dtype=np.float64)
    reference16 = np.asarray(rounded_reference, dtype=np.float16)
    reference64 = reference16.astype(np.float64)
    scales = np.asarray(sum_abs_contributions, dtype=np.float64)
    if actual64.shape != reference64.shape or scales.shape != reference64.shape:
        raise AssertionError(f"{label}: shape mismatch")
    error = np.abs(actual64 - reference64)
    allowance = fp16_ulp(reference16) + contract_allowance(scales)
    if not np.isfinite(actual64).all() or np.any(error > allowance):
        worst = int(np.argmax(error - allowance)) if error.size else 0
        detail = "empty tensor" if error.size == 0 else (
            f"index={np.unravel_index(worst, error.shape)}, "
            f"actual={actual64.flat[worst]:.9g}, "
            f"reference={reference64.flat[worst]:.9g}, "
            f"error={error.flat[worst]:.3g}, allowance={allowance.flat[worst]:.3g}")
        raise AssertionError(f"{label}: outside FP16 output contract ({detail})")
