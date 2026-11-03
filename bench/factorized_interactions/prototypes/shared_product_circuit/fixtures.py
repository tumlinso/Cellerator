"""Frozen CPU-generated cases for the shared-product circuit experiment.

The wire encoding and independent mathematical oracle remain the existing
factorized-interaction helpers.  This module fixes this prototype's topology,
case values, and identities without changing that shared support.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Iterable
import zipfile

import numpy as np


HERE = Path(__file__).resolve().parent
TEST_SUPPORT = HERE.parents[3] / "tests" / "factorized_interactions"
if str(TEST_SUPPORT) not in sys.path:
    sys.path.insert(0, str(TEST_SUPPORT))

from case_io import read_case, write_case  # noqa: E402
from reference import Mechanism, Problem, evaluate, validate_problem  # noqa: E402
import workloads  # noqa: E402


SCHEMA = "cellerator.shared_product_circuit.fixtures.v1"
CHANNELS = 32
CORRECTNESS_CELLS = 64
TIMED_CELLS = 1024
X_OCCURRENCES = 10
CONTEXTS = 2
ALPHA_ROWS = 4
BIAS_ROWS = 3
GAMMA_ROWS = 4
ORDINARY_SEEDS = {
    "ordinary_seed_0": 0x0123456789ABCDEF,
    "ordinary_seed_1": 0xFEDCBA9876543210,
}
OUTPUT_DTYPE = np.dtype(np.float32)
ACTIVATION_DTYPE = np.dtype(np.float16)
PARAMETER_DTYPE = np.dtype(np.float16)


def _mechanisms(*, repeated_tail: bool = False) -> tuple[Mechanism, ...]:
    records = []
    for mechanism_id in range(8):
        tail = 0 if repeated_tail and mechanism_id == 0 else 2 + mechanism_id
        records.append(Mechanism(
            target=0,
            participants=(0, 1, tail),
            context_ids=(0, 1),
            gamma_ids=(mechanism_id % GAMMA_ROWS,
                       (mechanism_id + 1) % GAMMA_ROWS),
            bias_id=mechanism_id % BIAS_ROWS,
            alpha_id=mechanism_id % ALPHA_ROWS,
        ))
    return tuple(records)


def structure_sha256(problem: Problem) -> str:
    """Hash extents and canonical mechanism records, excluding runtime values."""
    description = {
        "axes": {
            "x": list(problem.x.shape), "z": list(problem.z.shape),
            "alpha": list(problem.alpha.shape), "bias": list(problem.bias.shape),
            "gamma": list(problem.gamma.shape), "y": [
                problem.target_count, problem.x.shape[1], problem.x.shape[2]],
        },
        "mechanisms": [
            {
                "target": item.target,
                "participants": list(item.participants),
                "context_ids": list(item.context_ids),
                "gamma_ids": list(item.gamma_ids),
                "bias_id": item.bias_id,
                "alpha_id": item.alpha_id,
            }
            for item in problem.mechanisms
        ],
        "schema": "cellerator.shared_product_circuit.structure.v1",
    }
    encoded = json.dumps(description, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _generated_case(seed: int, cells: int) -> tuple[Problem, np.ndarray]:
    """Use the shared, versioned splitmix64-indexed generator at fixed offsets."""
    values = workloads._values  # The existing helper is the specified generator.
    x = values((X_OCCURRENCES, CHANNELS, cells), seed + 0x10000000,
               -0.95, 0.95, ACTIVATION_DTYPE)
    z = values((CONTEXTS, cells), seed + 0x20000000,
               -1.0, 1.0, ACTIVATION_DTYPE)
    alpha = values((ALPHA_ROWS, CHANNELS), seed + 0x30000000,
                   -1.0, 1.0, PARAMETER_DTYPE)
    bias = values((BIAS_ROWS,), seed + 0x40000000,
                  -0.5, 0.5, PARAMETER_DTYPE)
    gamma = values((GAMMA_ROWS,), seed + 0x50000000,
                   -0.5, 0.5, PARAMETER_DTYPE)
    dy = values((1, CHANNELS, cells), seed + 0x60000000,
                -1.0, 1.0, np.float32)
    problem = Problem(x, z, alpha, bias, gamma, _mechanisms(), 1)
    validate_problem(problem)
    return problem, dy


def _copy_problem(problem: Problem, *, x: np.ndarray | None = None,
                  z: np.ndarray | None = None, alpha: np.ndarray | None = None,
                  bias: np.ndarray | None = None, gamma: np.ndarray | None = None,
                  mechanisms: tuple[Mechanism, ...] | None = None) -> Problem:
    copied = Problem(
        x=np.array(problem.x if x is None else x, copy=True),
        z=np.array(problem.z if z is None else z, copy=True),
        alpha=np.array(problem.alpha if alpha is None else alpha, copy=True),
        bias=np.array(problem.bias if bias is None else bias, copy=True),
        gamma=np.array(problem.gamma if gamma is None else gamma, copy=True),
        mechanisms=problem.mechanisms if mechanisms is None else mechanisms,
        target_count=problem.target_count,
    )
    validate_problem(copied)
    return copied


def _ordinary_cases() -> dict[str, tuple[Problem, np.ndarray]]:
    cases = {
        name: _generated_case(seed, CORRECTNESS_CELLS)
        for name, seed in ORDINARY_SEEDS.items()
    }
    cases["timed_seed_0"] = _generated_case(
        ORDINARY_SEEDS["ordinary_seed_0"], TIMED_CELLS)
    return cases


def build_cases() -> dict[str, tuple[Problem, np.ndarray]]:
    """Build all frozen cases using only FP16 inputs and FP32 output/cotangent."""
    cases = _ordinary_cases()
    base, base_dy = cases["ordinary_seed_0"]

    zeros_x = np.array(base.x, copy=True)
    zeros_x[0, 0, 0] = np.float16(0.0)
    zeros_x[1, 1, 1] = np.float16(0.0)
    cases["zero_participants"] = (_copy_problem(base, x=zeros_x), base_dy.copy())

    # Each of eight equal-magnitude mechanism contributions alternates sign.
    # The output, dx, dz, and dbias totals cancel while their absolute
    # contribution scales remain nonzero, making the accepted bound observable.
    cancellation_x = np.full_like(base.x, np.float16(0.5))
    cancellation_z = np.zeros_like(base.z)
    cancellation_alpha = np.tile(
        np.asarray((1.0, -1.0, 1.0, -1.0), dtype=np.float16)[:, None],
        (1, CHANNELS))
    cancellation_bias = np.zeros_like(base.bias)
    cancellation_gamma = np.zeros_like(base.gamma)
    cancellation_dy = np.ones_like(base_dy, dtype=np.float32)
    cases["signed_cancellation"] = (
        _copy_problem(base, x=cancellation_x, z=cancellation_z,
                      alpha=cancellation_alpha, bias=cancellation_bias,
                      gamma=cancellation_gamma), cancellation_dy)

    saturated_z = np.ones_like(base.z, dtype=np.float16)
    saturated_gamma = np.full_like(base.gamma, np.float16(4.0))
    saturated_bias = np.asarray((11.0, -19.0, 11.0), dtype=np.float16)
    cases["saturated_gates"] = (
        _copy_problem(base, z=saturated_z, bias=saturated_bias,
                      gamma=saturated_gamma), base_dy.copy())

    repeated = list(base.mechanisms)
    first = repeated[0]
    repeated[0] = Mechanism(
        target=first.target, participants=(0, 1, 0),
        context_ids=first.context_ids, gamma_ids=first.gamma_ids,
        bias_id=first.bias_id, alpha_id=first.alpha_id)
    repeated_tail_problem = _copy_problem(base, mechanisms=tuple(repeated))
    cases["repeated_tail_role"] = (repeated_tail_problem, base_dy.copy())

    cases["runtime_refresh_before"] = (base, base_dy.copy())
    refreshed_alpha = (base.alpha.astype(np.float32) * 0.75 + 0.125).astype(np.float16)
    refreshed_bias = (base.bias.astype(np.float32) - 0.25).astype(np.float16)
    refreshed_gamma = (base.gamma.astype(np.float32) * -0.5 + 0.375).astype(np.float16)
    cases["runtime_refresh_after"] = (
        _copy_problem(base, alpha=refreshed_alpha, bias=refreshed_bias,
                      gamma=refreshed_gamma), base_dy.copy())
    return cases


def _array_identity(problem: Problem, dy: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (problem.x, problem.z, problem.alpha, problem.bias,
                  problem.gamma, dy):
        canonical = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<")))
        digest.update(str(canonical.dtype).encode("ascii"))
        digest.update(json.dumps(list(canonical.shape), separators=(",", ":")).encode("ascii"))
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def _evaluation_arrays(evaluation) -> dict[str, np.ndarray]:
    arrays = {"y": evaluation.y_float64,
              "scale_y": evaluation.sum_abs_y_contributions}
    for name in ("dx", "dz", "dalpha", "dbias", "dgamma"):
        arrays[name] = getattr(evaluation.vjp, name)
        arrays[f"scale_{name}"] = getattr(evaluation.vjp_scale, name)
    return arrays


def _deterministic_npy(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.ascontiguousarray(array),
                              version=(1, 0), allow_pickle=False)
    return stream.getvalue()


def _write_oracle_cache(path: Path,
                        values: dict[str, np.ndarray]) -> None:
    """Write a reproducible, uncompressed NPZ with fixed ZIP timestamps."""
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(values):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, _deterministic_npy(values[name]))


def freeze_cases(output_dir: Path) -> dict:
    """Write an immutable CFINT001 corpus and deterministic manifest.

    The destination must be new or empty, so a rerun cannot silently replace
    evidence from another attempt.
    """
    output_dir = Path(output_dir)
    if output_dir.is_symlink():
        raise ValueError("fixture output directory cannot be a symlink")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"fixture output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cases()
    rows = []
    for name, (problem, dy) in cases.items():
        case_path = output_dir / f"{name}.case"
        identity = write_case(case_path, problem, dy, OUTPUT_DTYPE)
        loaded_problem, loaded_dy, loaded_output = read_case(case_path)
        if loaded_output != OUTPUT_DTYPE or _array_identity(loaded_problem, loaded_dy) != \
                _array_identity(problem, dy):
            raise RuntimeError(f"CFINT001 round trip changed case {name}")
        relative_path = case_path.name
        rows.append({
            "id": name,
            "path": relative_path,
            "bytes": identity["bytes"],
            "sha256": identity["sha256"],
            "array_sha256": _array_identity(problem, dy),
            "structure_sha256": structure_sha256(problem),
            "channels": problem.x.shape[1],
            "cells": problem.x.shape[2],
            "mechanisms": len(problem.mechanisms),
            "repeated_tail": problem.mechanisms[0].participants == (0, 1, 0),
            "role": "timed_input" if name == "timed_seed_0" else "correctness_input",
        })
    oracle_arrays: dict[str, np.ndarray] = {}
    for case_id, (problem, dy) in cases.items():
        evaluation = evaluate(problem, dy=dy, output_dtype=np.float32)
        for tensor_name, array in _evaluation_arrays(evaluation).items():
            oracle_arrays[f"{case_id}/{tensor_name}"] = np.asarray(array, dtype=np.float64)
    oracle_path = output_dir / "oracle_cache.npz"
    _write_oracle_cache(oracle_path, oracle_arrays)
    oracle_digest = hashlib.sha256(oracle_path.read_bytes()).hexdigest()
    manifest = {
        "schema": SCHEMA,
        "law": "y = sum_m(sigmoid(bias[m%3] + gamma[m%4]*z[0] + gamma[(m+1)%4]*z[1]) * alpha[m%4] * x[0]*x[1]*x[2+m])",
        "dimensions": {
            "mechanisms": 8, "channels": CHANNELS,
            "input_occurrences": X_OCCURRENCES, "contexts": CONTEXTS,
            "alpha_rows": ALPHA_ROWS, "bias_rows": BIAS_ROWS,
            "gamma_rows": GAMMA_ROWS, "target_count": 1,
            "correctness_cells": CORRECTNESS_CELLS, "timed_cells": TIMED_CELLS,
        },
        "dtypes": {
            "x_z_alpha_bias_gamma": "float16",
            "dy_y_dx_dz_dalpha_dbias_dgamma": "float32",
            "arithmetic_and_reductions": "float32",
        },
        "generator": {
            "implementation": "tests/factorized_interactions/workloads.py::_values",
            "algorithm": "splitmix64-indexed-v1",
            "seed_root_hex": f"0x{workloads.SEED:016x}",
            "ordinary_seed_offsets": {
                name: f"0x{offset:016x}" for name, offset in ORDINARY_SEEDS.items()},
            "field_salts_hex": {
                "x": "0x10000000", "z": "0x20000000",
                "alpha": "0x30000000", "bias": "0x40000000",
                "gamma": "0x50000000", "dy": "0x60000000",
            },
        },
        "gradient_contract": {
            "overwrite": "one oracle VJP",
            "accumulate_seed": 0.125,
            "repeat_vjp": "second VJP uses independent destinations and scratch",
        },
        "fp32_error_contract": {
            "absolute_floor": 2.0e-6,
            "relative_to_sum_abs_reference_contributions": 2.0e-5,
        },
        "oracle_cache": {
            "path": oracle_path.name,
            "sha256": oracle_digest,
            "reference": "tests/factorized_interactions/reference.py::evaluate",
            "stored_values": "FP64 oracle tensors and per-contribution scales",
        },
        "cases": rows,
    }
    encoded = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
    (output_dir / "manifest.json").write_bytes(encoded)
    return {"path": str(output_dir / "manifest.json"),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "cases": rows}


def verify_frozen_cases(manifest_path: Path) -> dict[str, tuple[Problem, np.ndarray]]:
    """Read and verify every frozen wire case against its manifest identity."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unknown shared-product fixture manifest schema")
    root = manifest_path.parent.resolve(strict=True)
    cases = {}
    for item in manifest.get("cases", []):
        case_id = item.get("id")
        relative = Path(item.get("path", ""))
        if not case_id or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("fixture manifest has an unsafe or empty case path")
        source_path = root / relative
        if source_path.is_symlink():
            raise ValueError("fixture case must be a regular file under its manifest")
        path = source_path.resolve(strict=True)
        if path.parent != root:
            raise ValueError("fixture case must be a regular file under its manifest")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != item.get("sha256") or len(raw) != item.get("bytes"):
            raise ValueError(f"fixture case identity mismatch: {case_id}")
        problem, dy, output_dtype = read_case(path)
        if output_dtype != OUTPUT_DTYPE:
            raise ValueError(f"fixture output dtype differs from FP32: {case_id}")
        if structure_sha256(problem) != item.get("structure_sha256"):
            raise ValueError(f"fixture structure identity mismatch: {case_id}")
        if _array_identity(problem, dy) != item.get("array_sha256"):
            raise ValueError(f"fixture array identity mismatch: {case_id}")
        cases[case_id] = (problem, dy)
    if len(cases) != len(manifest.get("cases", [])):
        raise ValueError("fixture manifest contains duplicate case ids")
    required = {
        "ordinary_seed_0", "ordinary_seed_1", "timed_seed_0",
        "zero_participants", "signed_cancellation", "saturated_gates",
        "repeated_tail_role", "runtime_refresh_before", "runtime_refresh_after",
    }
    if set(cases) != required:
        raise ValueError("fixture corpus does not match the frozen case set")
    return cases


def load_oracle_cache(manifest_path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Load precomputed independent expectations without evaluating on the GPU host."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unknown shared-product fixture manifest schema")
    entry = manifest.get("oracle_cache", {})
    relative = Path(entry.get("path", ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("oracle cache path is unsafe")
    path = manifest_path.parent / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError("oracle cache must be a regular file")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry.get("sha256"):
        raise ValueError("oracle cache SHA256 differs from the frozen manifest")
    result: dict[str, dict[str, np.ndarray]] = {}
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            if "/" not in key:
                raise ValueError("oracle cache key lacks a case namespace")
            case_id, tensor_name = key.split("/", 1)
            result.setdefault(case_id, {})[tensor_name] = archive[key].copy()
    if set(result) != {item["id"] for item in manifest["cases"]}:
        raise ValueError("oracle cache cases differ from the frozen manifest")
    expected_tensors = {
        "y", "scale_y", "dx", "dz", "dalpha", "dbias", "dgamma",
        "scale_dx", "scale_dz", "scale_dalpha", "scale_dbias", "scale_dgamma",
    }
    for case_id, arrays in result.items():
        if set(arrays) != expected_tensors:
            raise ValueError(f"oracle cache tensors differ for {case_id}")
        for name, array in arrays.items():
            if array.dtype != np.float64 or not np.isfinite(array).all():
                raise ValueError(f"oracle cache contains invalid values: {case_id}/{name}")
    return result


def runtime_structure_pair(cases: dict[str, tuple[Problem, np.ndarray]]) -> tuple[str, str]:
    """Return the before/after structure identities for the refresh check."""
    before = structure_sha256(cases["runtime_refresh_before"][0])
    after = structure_sha256(cases["runtime_refresh_after"][0])
    return before, after


def relative_case_paths(manifest_path: Path) -> Iterable[tuple[str, Path]]:
    """Yield verified case paths in manifest order for the orchestration driver."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unknown shared-product fixture manifest schema")
    for item in manifest["cases"]:
        yield item["id"], manifest_path.parent / item["path"]
