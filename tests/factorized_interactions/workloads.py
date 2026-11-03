"""Deterministic fixed workload corpus generator.

The integer counter-based generator is specified here rather than delegated to
NumPy's versioned RNG implementation, so a workload name and seed identify the
same values across environments.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from reference import Mechanism, Problem


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "bench" / "factorized_interactions" / "workloads.json"
MASK64 = (1 << 64) - 1
SEED = int("6a09e667f3bcc909", 16)
GOLDEN = 0x9E3779B97F4A7C15


def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _splitmix64(value: int) -> int:
    value = (value + GOLDEN) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def _uniform(seed: int, counter: int) -> float:
    bits = _splitmix64((seed + counter * GOLDEN) & MASK64)
    return (bits >> 11) * (1.0 / (1 << 53))


def _values(shape: tuple[int, ...], seed_offset: int,
            low: float, high: float, dtype: np.dtype | type) -> np.ndarray:
    count = int(np.prod(shape, dtype=np.int64))
    counters = np.arange(count, dtype=np.uint64)
    state = (np.uint64((SEED + seed_offset) & MASK64)
             + counters * np.uint64(GOLDEN))
    state += np.uint64(GOLDEN)
    state = (state ^ (state >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    state = (state ^ (state >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    bits = state ^ (state >> np.uint64(31))
    unit = (bits >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))
    values = low + (high - low) * unit
    return values.reshape(shape).astype(dtype)


def workload_spec(workload_id: str) -> dict:
    matches = [case for case in manifest()["cases"] if case["id"] == workload_id]
    if len(matches) != 1:
        raise KeyError(f"unknown workload {workload_id!r}")
    return matches[0]


def _fixed_problem(workload_id: str, activation_dtype: np.dtype | type,
                   parameter_dtype: np.dtype | type) -> tuple[Problem, np.ndarray]:
    if workload_id == "correctness_ordered_tied_zero":
        x = np.array([
            [[0.0, 0.25], [0.5, -0.5], [0.25, -0.75]],
            [[0.5, -0.25], [-0.5, 0.75], [1.0, -1.0]],
            [[-0.75, 0.5], [0.0, 0.25], [-0.25, 0.5]],
            [[0.25, -0.5], [0.75, 0.25], [-1.0, 0.5]],
            [[0.0, -0.5], [0.5, 1.0], [0.75, 0.0]],
        ], dtype=activation_dtype)
        z = np.array([[0.25, -0.5], [1.0, 0.0], [-1.0, 0.5]],
                     dtype=activation_dtype)
        alpha = np.array([[0.5, -0.75, 0.25], [-0.25, 0.5, 1.0],
                          [1.0, -1.0, 0.75]], dtype=parameter_dtype)
        bias = np.array([0.2, -0.4, 3.5], dtype=parameter_dtype)
        gamma = np.array([0.5, -0.25, 0.75], dtype=parameter_dtype)
        mechanisms = (
            Mechanism(0, (0, 1), (0, 1), (0, 1), 0, 0),
            Mechanism(0, (1, 1, 2), (1, 1), (1, 1), 0, 0),
            Mechanism(1, (0, 0, 3, 4), (2, 0), (2, 0), 1, 1),
            Mechanism(0, (4, 3, 2, 0), (), (), 2, 2),
            Mechanism(1, (0, 1, 4), (0, 0), (0, 0), 0, 0),
        )
        problem = Problem(x, z, alpha, bias, gamma, mechanisms, 2)
        dy = np.array([
            [[1.0, -0.75], [0.5, 0.25], [-0.5, 1.0]],
            [[-0.25, 0.75], [1.0, -1.0], [0.5, -0.5]],
        ], dtype=np.float32)
        return problem, dy

    if workload_id == "correctness_saturated_gates":
        x = np.ones((2, 1, 1), dtype=activation_dtype)
        z = np.ones((32, 1), dtype=activation_dtype)
        alpha = np.ones((2, 1), dtype=parameter_dtype)
        bias = np.array([4.0, -4.0], dtype=parameter_dtype)
        gamma = np.concatenate((np.ones(32), -np.ones(32))).astype(parameter_dtype)
        terms = tuple(range(32))
        mechanisms = (
            Mechanism(0, (0, 1), terms, tuple(range(32)), 0, 0),
            Mechanism(0, (1, 0), terms, tuple(range(32, 64)), 1, 1),
        )
        problem = Problem(x, z, alpha, bias, gamma, mechanisms, 1)
        dy = np.ones((1, 1, 1), dtype=np.float32)
        return problem, dy

    if workload_id == "correctness_ordered_role_context_variants":
        x = np.array([[[0.5]], [[0.25]], [[-0.75]]], dtype=activation_dtype)
        z = np.array([[0.5], [-0.25]], dtype=activation_dtype)
        alpha = np.array([[0.8]], dtype=parameter_dtype)
        bias = np.array([0.2], dtype=parameter_dtype)
        gamma = np.array([0.5, -0.4], dtype=parameter_dtype)
        mechanisms = (
            Mechanism(0, (0, 0, 1, 2), (0, 1), (0, 1), 0, 0),
            Mechanism(0, (0, 1, 0, 2), (1, 0), (1, 0), 0, 0),
        )
        return Problem(x, z, alpha, bias, gamma, mechanisms, 1), \
            np.ones((1, 1, 1), dtype=np.float32)

    if workload_id == "correctness_fma_product_residual_cancellation":
        if np.dtype(activation_dtype) != np.dtype(np.float32) or \
                np.dtype(parameter_dtype) != np.dtype(np.float32):
            raise ValueError("the FMA product-residual fixture requires FP32 storage")
        # These FP32 values are exact integer multiples of one ULP at 1.0.
        # The rounded products cancel, while the exact FP64 products have a
        # small positive residual. Keep ordered pairs repeated to expose a
        # reduction that compensates sum addition but loses product residuals.
        unit = 2.0 ** -23
        a = np.float32(1.0 + 500 * unit)
        b = np.float32(1.0 + 4500 * unit)
        c = np.float32(-(1.0 + 2500 * unit))
        d = np.float32(1.0 + 2499 * unit)
        pair_count = 4096
        problem = Problem(
            np.ones((2, 1, 1), dtype=np.float32),
            np.array([[b], [d]], dtype=np.float32),
            np.ones((1, 1), dtype=np.float32),
            np.zeros((1,), dtype=np.float32),
            np.array([a, c], dtype=np.float32),
            (Mechanism(0, (0, 1), (0, 1) * pair_count,
                       (0, 1) * pair_count, 0, 0),),
            1,
        )
        return problem, np.ones((1, 1, 1), dtype=np.float32)

    if workload_id in ("correctness_context_cancellation_middle",
                       "correctness_context_cancellation_tail"):
        term_count = 8193
        z = np.empty((term_count, 1), dtype=activation_dtype)
        z[:4096, 0] = 1.0
        z[4096, 0] = 0.0002
        z[4097:, 0] = -1.0
        if workload_id.endswith("_middle"):
            context_ids = tuple(range(term_count))
        else:
            context_ids = tuple(range(4096)) + tuple(range(4097, term_count)) + (4096,)
        gamma_ids = (0,) * term_count
        problem = Problem(
            np.ones((2, 1, 1), dtype=activation_dtype), z,
            np.ones((1, 1), dtype=parameter_dtype),
            np.zeros((1,), dtype=parameter_dtype),
            np.ones((1,), dtype=parameter_dtype),
            (Mechanism(0, (0, 1), context_ids, gamma_ids, 0, 0),), 1,
        )
        return problem, np.ones((1, 1, 1), dtype=np.float32)

    high_fanin_ids = {
        "correctness_high_fanin_100k",
        "correctness_packet_high_fanin_fp32_100k",
        "correctness_packet_high_fanin_1m",
    }
    if workload_id in high_fanin_ids:
        mechanism_count = 1_000_000 if workload_id.endswith("_1m") else 100_000
        alpha_value = (0.0757218450307846
                       if workload_id == "correctness_packet_high_fanin_fp32_100k"
                       else 0.0002)
        repeated = Mechanism(0, (0, 1), (), (), 0, 0)
        problem = Problem(
            np.ones((2, 1, 1), dtype=activation_dtype),
            np.zeros((0, 1), dtype=activation_dtype),
            np.array([[alpha_value]], dtype=parameter_dtype),
            np.zeros((1,), dtype=parameter_dtype),
            np.zeros((0,), dtype=parameter_dtype),
            (repeated,) * mechanism_count, 1,
        )
        return problem, np.ones((1, 1, 1), dtype=np.float32)

    if workload_id == "correctness_empty_mechanisms":
        problem = Problem(
            np.zeros((0, 3, 2), dtype=activation_dtype),
            np.zeros((0, 2), dtype=activation_dtype),
            np.zeros((0, 3), dtype=parameter_dtype),
            np.zeros((0,), dtype=parameter_dtype),
            np.zeros((0,), dtype=parameter_dtype),
            (), 2,
        )
        return problem, np.ones((2, 3, 2), dtype=np.float32)

    if workload_id == "correctness_empty_cells":
        problem = Problem(
            np.ones((2, 3, 0), dtype=activation_dtype),
            np.ones((1, 0), dtype=activation_dtype),
            np.ones((1, 3), dtype=parameter_dtype),
            np.zeros((1,), dtype=parameter_dtype),
            np.ones((1,), dtype=parameter_dtype),
            (Mechanism(0, (0, 1), (0,), (0,), 0, 0),), 1,
        )
        return problem, np.zeros((1, 3, 0), dtype=np.float32)

    if workload_id == "correctness_empty_channels":
        problem = Problem(
            np.zeros((2, 0, 2), dtype=activation_dtype),
            np.ones((1, 2), dtype=activation_dtype),
            np.zeros((1, 0), dtype=parameter_dtype),
            np.zeros((1,), dtype=parameter_dtype),
            np.ones((1,), dtype=parameter_dtype),
            (Mechanism(0, (0, 1), (0,), (0,), 0, 0),), 1,
        )
        return problem, np.zeros((1, 0, 2), dtype=np.float32)

    raise KeyError(f"unknown fixed correctness case {workload_id!r}")


def make_workload(workload_id: str, activation_dtype: np.dtype | type = np.float16,
                  parameter_dtype: np.dtype | type | None = None
                  ) -> tuple[Problem, np.ndarray]:
    """Return a deterministic semantic problem and FP32 cotangent."""
    if parameter_dtype is None:
        parameter_dtype = activation_dtype
    correctness_ids = {case["id"] for case in manifest().get("correctness_cases", [])}
    if workload_id in correctness_ids:
        return _fixed_problem(workload_id, activation_dtype, parameter_dtype)
    spec = workload_spec(workload_id)
    cells = int(spec["cells"])
    channels = int(spec["channels"])
    mechanisms_count = int(spec["mechanisms"])
    arity = int(spec["arity"])
    unique_participants = int(spec["unique_participants"])
    context_terms = int(spec["context_terms"])
    unique_contexts = int(spec["unique_contexts"])
    output_fanin = int(spec["output_fanin"])
    tied = bool(spec["tied_parameters"])
    if not (0 < unique_participants <= mechanisms_count * arity):
        raise ValueError(f"{workload_id}: invalid unique participant count")
    if not (0 < output_fanin <= mechanisms_count
            and mechanisms_count % output_fanin == 0):
        raise ValueError(f"{workload_id}: output_fanin must divide mechanism count")
    if context_terms and unique_contexts <= 0:
        raise ValueError(f"{workload_id}: context terms require context values")

    occurrences = unique_participants
    target_count = mechanisms_count // output_fanin
    gamma_count = unique_contexts if tied else mechanisms_count * context_terms
    bias_count = 1 if tied else mechanisms_count
    alpha_count = 1 if tied else mechanisms_count

    x = _values((occurrences, channels, cells), 0x10000000,
                -0.95, 0.95, activation_dtype)
    z = _values((unique_contexts, cells), 0x20000000,
                -1.0, 1.0, activation_dtype)
    alpha = _values((alpha_count, channels), 0x30000000,
                    -1.0, 1.0, parameter_dtype)
    bias = _values((bias_count,), 0x40000000,
                   -0.5, 0.5, parameter_dtype)
    gamma = _values((gamma_count,), 0x50000000,
                    -0.5, 0.5, parameter_dtype)
    dy = _values((target_count, channels, cells), 0x60000000,
                 -1.0, 1.0, np.float32)

    mechanisms = []
    role_slots = mechanisms_count * arity
    participant_stride = 37
    while np.gcd(participant_stride, unique_participants) != 1:
        participant_stride += 2
    participant_offset = 11 % unique_participants
    for mechanism_id in range(mechanisms_count):
        participants = tuple(
            ((mechanism_id * arity + role) * participant_stride
             + participant_offset) % unique_participants
            for role in range(arity)
        )
        context_ids = tuple(
            ((mechanism_id * max(1, context_terms) + term) % unique_contexts)
            for term in range(context_terms)
        )
        if tied:
            bias_id = 0
            alpha_id = 0
            gamma_ids = tuple(context_id for context_id in context_ids)
        else:
            bias_id = mechanism_id
            alpha_id = mechanism_id
            gamma_ids = tuple(mechanism_id * context_terms + term
                              for term in range(context_terms))
        mechanisms.append(Mechanism(
            target=mechanism_id // output_fanin,
            participants=participants,
            context_ids=context_ids,
            gamma_ids=gamma_ids,
            bias_id=bias_id,
            alpha_id=alpha_id,
        ))
    if len(mechanisms) * arity != role_slots:
        raise AssertionError("internal workload role-count inconsistency")
    return Problem(x, z, alpha, bias, gamma, tuple(mechanisms), target_count), dy


def workload_ids() -> tuple[str, ...]:
    data = manifest()
    return (tuple(case["id"] for case in data["cases"])
            + tuple(case["id"] for case in data.get("correctness_cases", [])))


def performance_workload_ids() -> tuple[str, ...]:
    return tuple(case["id"] for case in manifest()["cases"])


def correctness_workload_ids() -> tuple[str, ...]:
    return tuple(case["id"] for case in manifest().get("correctness_cases", []))
