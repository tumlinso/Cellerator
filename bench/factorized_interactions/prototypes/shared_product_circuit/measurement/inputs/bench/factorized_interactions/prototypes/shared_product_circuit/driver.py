#!/usr/bin/env python3
"""Source-bound validation and paired-measurement driver for this prototype.

This file only orchestrates the private runner.  It neither submits Scheduler
work nor treats CPU fixtures as CUDA results.  Independent FP64 reference
outputs and contribution scales are produced by ``freeze`` before admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
TEST_SUPPORT = HERE.parents[3] / "tests" / "factorized_interactions"
if str(TEST_SUPPORT) not in sys.path:
    sys.path.insert(0, str(TEST_SUPPORT))

from case_io import read_result  # noqa: E402
from reference import assert_fp32_contract  # noqa: E402
from fixtures import (  # noqa: E402
    SCHEMA as FIXTURE_SCHEMA,
    freeze_cases,
    load_oracle_cache,
    structure_sha256,
    verify_frozen_cases,
)


DRIVER_SCHEMA = "cellerator.shared_product_circuit.driver.v1"
RUNNER_SCHEMA = "cellerator.shared_product_circuit.runner.v1"
PAIRED_SCHEMA = "cellerator.shared_product_circuit.paired.v1"
BUILD_SCHEMA = "cellerator.shared_product_circuit.build.v1"
MAX_SCRATCH_BYTES = 128 * 1024 * 1024
MAX_ATTEMPT_SECONDS = 170
GRADIENT_SEED = 0.125
REGISTERED_SANITIZER_PATH = Path(
    "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer")
EXPECTED_SANITIZER_VERSION = "2025.2.1.0"
CUDA_UUID_RE = re.compile(r"^GPU-[A-Za-z0-9-]+$")
PROTOTYPE_CANDIDATES = ("unfactored", "circuit")
ALL_CANDIDATES = ("unfactored", "circuit", "native_direct")
EXPECTED_BACKENDS = {
    "unfactored": {"literal_unfactored"},
    "circuit": {"shared_product_circuit"},
    "native_direct": {"direct", "stable"},
}
GRADIENTS = ("dx", "dz", "dalpha", "dbias", "dgamma")
MEMORY_FIELDS = {
    "persistent_layout", "resident_values_and_cotangent", "output",
    "gradient_destinations", "tape", "forward_scratch", "backward_scratch",
    "repeat_gradient_and_scratch",
}
SETUP_FIELDS = {
    "host_preparation", "device_layout_upload", "arm_allocation_and_seed",
    "shared_value_allocation", "shared_activation_and_cotangent_upload",
    "shared_parameter_upload",
}


class DriverError(RuntimeError):
    """A source, protocol, execution, or acceptance identity failed closed."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_regular(path: Path) -> bytes:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise DriverError(f"expected a regular non-symlink file: {path}")
    return path.read_bytes()


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    encoded = _read_regular(path)
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DriverError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DriverError(f"JSON root must be an object: {path}")
    return value, _sha256_bytes(encoded)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode("ascii")
    return _sha256_bytes(encoded)


def _python_source_identity() -> dict[str, Any]:
    paths = (
        HERE / "driver.py", HERE / "fixtures.py", HERE / "test_driver.py",
        TEST_SUPPORT / "reference.py", TEST_SUPPORT / "case_io.py",
        TEST_SUPPORT / "workloads.py",
    )
    files = []
    for path in paths:
        raw = _read_regular(path)
        files.append({"path": path.relative_to(REPO_ROOT).as_posix(),
                      "bytes": len(raw), "sha256": _sha256_bytes(raw)})
    files.sort(key=lambda row: row["path"])
    return {"schema": "cellerator.shared_product_circuit.python_sources.v1",
            "files": files, "source_sha256": _canonical_sha256(files)}


def _verify_build_manifest(manifest_path: Path, runner_path: Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path).absolute()
    runner_path = Path(runner_path).absolute()
    manifest, manifest_sha = _read_json(manifest_path)
    if manifest.get("schema") != BUILD_SCHEMA:
        raise DriverError("build manifest schema is not the private shared-product v1 schema")
    source_root_value = manifest.get("source_root")
    if not isinstance(source_root_value, str) or not Path(source_root_value).is_absolute():
        raise DriverError("build manifest source_root must be absolute")
    source_root = Path(source_root_value)
    if source_root.is_symlink() or not source_root.is_dir():
        raise DriverError("build manifest source_root must be a real directory")
    source_files = manifest.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise DriverError("build manifest must name its nonempty compiled source closure")
    if any(not isinstance(row, dict) for row in source_files):
        raise DriverError("build manifest source entry is not an object")
    if source_files != sorted(source_files, key=lambda row: row.get("path", "")):
        raise DriverError("build manifest source_files are not sorted by path")
    seen: set[str] = set()
    for row in source_files:
        relative_value = row.get("path")
        if not isinstance(relative_value, str):
            raise DriverError("build manifest source entry lacks a relative path")
        relative = Path(relative_value)
        if relative.is_absolute() or ".." in relative.parts or relative_value in seen:
            raise DriverError("build manifest contains an unsafe or duplicate source path")
        seen.add(relative_value)
        source_path = source_root / relative
        if source_path.is_symlink() or not source_path.is_file():
            raise DriverError(f"build source is not a regular file: {relative_value}")
        raw = source_path.read_bytes()
        if len(raw) != row.get("bytes") or _sha256_bytes(raw) != row.get("sha256"):
            raise DriverError(f"compiled source identity changed: {relative_value}")
    source_closure_sha = _canonical_sha256(source_files)
    if source_closure_sha != manifest.get("source_closure_sha256"):
        raise DriverError("source closure aggregate differs from its listed file identities")
    if runner_path.is_symlink() or not runner_path.is_file():
        raise DriverError("runner must be a regular copied executable")
    if not os.access(runner_path, os.X_OK):
        raise DriverError("runner lacks its executable bit in the scheduled worker")
    runner_sha = _sha256_file(runner_path)
    if runner_sha != manifest.get("runner_binary_sha256"):
        raise DriverError("runner bytes differ from the private build manifest")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "build_schema": manifest["schema"],
        "source_root": str(source_root.resolve(strict=True)),
        "source_file_count": len(source_files),
        "source_closure_sha256": source_closure_sha,
        "runner_path": str(runner_path),
        "runner_binary_sha256": runner_sha,
        "architecture": manifest.get("architecture"),
        "fast_math": manifest.get("fast_math"),
    }


def _same_build_identity(left: dict, right: dict) -> bool:
    """Compare content/toolchain identities, allowing a copied runner path."""
    keys = ("manifest_sha256", "build_schema", "source_closure_sha256",
            "source_file_count", "runner_binary_sha256", "architecture", "fast_math")
    return all(left.get(key) == right.get(key) for key in keys)


def _require_build_stable(before: dict[str, Any], manifest_path: Path,
                          runner_path: Path) -> dict[str, Any]:
    after = _verify_build_manifest(manifest_path, runner_path)
    if not _same_build_identity(after, before):
        raise DriverError("source, build manifest, or runner changed during runner execution")
    return after


def _attempt_context(expected_uuid: str) -> dict[str, str]:
    if not CUDA_UUID_RE.fullmatch(expected_uuid):
        raise DriverError("expected UUID must be an explicit CUDA GPU UUID")
    job = os.environ.get("AUTOBENCH_SCHEDULER_JOB_ID", "").strip()
    attempt = os.environ.get("AUTOBENCH_SCHEDULER_ATTEMPT_ID", "").strip()
    device_job = os.environ.get("DEVICE_SCHEDULER_JOB_ID", "").strip()
    device_attempt = os.environ.get("DEVICE_SCHEDULER_ATTEMPT_ID", "").strip()
    if not job or not attempt:
        raise DriverError("validation and measurement require an admitted Scheduler job and attempt")
    if device_job and device_job != job:
        raise DriverError("Autobench and Device Scheduler job identifiers disagree")
    if device_attempt and device_attempt != attempt:
        raise DriverError("Autobench and Device Scheduler attempt identifiers disagree")
    return {"scheduler_job_id": job, "scheduler_attempt_id": attempt,
            "device_uuid": expected_uuid}


def _fixture_set(manifest_path: Path) -> tuple[dict, dict, dict[str, tuple], dict]:
    manifest, manifest_sha = _read_json(manifest_path)
    if manifest.get("schema") != FIXTURE_SCHEMA:
        raise DriverError("unknown fixture manifest schema")
    cases = verify_frozen_cases(manifest_path)
    oracle = load_oracle_cache(manifest_path)
    if set(cases) != set(oracle):
        raise DriverError("fixture and independent oracle cache case identities differ")
    return manifest, manifest_sha, cases, oracle


def _case_record(manifest: dict, case_id: str) -> dict:
    matches = [row for row in manifest["cases"] if row.get("id") == case_id]
    if len(matches) != 1:
        raise DriverError(f"fixture manifest has no unique case {case_id}")
    return matches[0]


def _verify_case_file(manifest_path: Path, case: dict) -> Path:
    root = Path(manifest_path).absolute().parent
    relative = Path(case["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise DriverError("case path escapes its frozen fixture directory")
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise DriverError(f"case is not a regular frozen input: {case['id']}")
    raw = path.read_bytes()
    if len(raw) != case["bytes"] or _sha256_bytes(raw) != case["sha256"]:
        raise DriverError(f"case wire bytes differ from manifest: {case['id']}")
    return path


def _validate_result_tensors(actual: dict[str, np.ndarray], oracle: dict[str, np.ndarray],
                             *, gradient_mode: str, repeat_vjp: bool) -> dict[str, dict]:
    if gradient_mode not in ("overwrite", "accumulate"):
        raise DriverError("unsupported gradient mode")
    if gradient_mode == "accumulate" and repeat_vjp is False:
        raise DriverError("accumulation case must exercise a repeated VJP")
    expected_names = {"y", *GRADIENTS}
    if not expected_names.issubset(actual):
        raise DriverError("CFOUT001 result omitted one or more of the six required tensors")
    if repeat_vjp and not {f"{name}_repeat" for name in GRADIENTS}.issubset(actual):
        raise DriverError("CFOUT001 result omitted the repeated VJP tensor set")
    evidence = {}
    for name in ("y", *GRADIENTS):
        expected = oracle[name]
        scale_key = "scale_y" if name == "y" else f"scale_{name}"
        scale = oracle[scale_key]
        if name != "y" and gradient_mode == "accumulate":
            expected = expected + GRADIENT_SEED
        candidate = np.asarray(actual[name])
        if candidate.dtype != np.dtype(np.float32):
            raise DriverError(f"{name} must be returned in FP32")
        try:
            assert_fp32_contract(candidate, expected, scale, name)
            evidence[name] = _tensor_error_evidence(candidate, expected, scale)
        except AssertionError as exc:
            raise DriverError(str(exc)) from exc
        if name != "y" and repeat_vjp:
            repeated = np.asarray(actual[f"{name}_repeat"])
            if repeated.dtype != np.dtype(np.float32):
                raise DriverError(f"{name}_repeat must be returned in FP32")
            try:
                assert_fp32_contract(repeated, expected, scale, f"{name} repeated VJP")
                evidence[f"{name}_repeat"] = _tensor_error_evidence(
                    repeated, expected, scale)
            except AssertionError as exc:
                raise DriverError(str(exc)) from exc
    return evidence


def _tensor_error_evidence(actual: np.ndarray, reference: np.ndarray,
                           scale: np.ndarray) -> dict:
    actual64 = np.asarray(actual, dtype=np.float64)
    reference64 = np.asarray(reference, dtype=np.float64)
    scales = np.asarray(scale, dtype=np.float64)
    allowance = 2.0e-6 + 2.0e-5 * scales
    errors = np.abs(actual64 - reference64)
    if not errors.size:
        return {"max_abs_error": 0.0, "max_allowed_error": 0.0,
                "max_error_to_allowance": 0.0, "worst_index": None}
    flat = int(np.argmax(errors / allowance))
    index = np.unravel_index(flat, errors.shape)
    return {
        "max_abs_error": float(errors.flat[flat]),
        "max_allowed_error": float(allowance.flat[flat]),
        "max_error_to_allowance": float((errors / allowance).flat[flat]),
        "worst_index": [int(value) for value in index],
        "worst_sum_abs_reference_contributions": float(scales.flat[flat]),
    }


def _memory_evidence(memory: Any, *, repeat_vjp: bool) -> dict:
    if not isinstance(memory, dict) or set(memory) != MEMORY_FIELDS or any(
            type(value) is not int or value < 0 for value in memory.values()):
        raise DriverError("runner memory breakdown is missing or invalid")
    repeat_extra = memory["repeat_gradient_and_scratch"]
    if repeat_vjp:
        if repeat_extra < memory["gradient_destinations"]:
            raise DriverError("repeat-VJP allocation breakdown omits gradient destinations")
        repeat_scratch = repeat_extra - memory["gradient_destinations"]
        if repeat_scratch != memory["backward_scratch"]:
            raise DriverError("repeat-VJP workspace differs from the primary backward scratch")
    else:
        if repeat_extra != 0:
            raise DriverError("runner reports repeat-VJP allocation when none was requested")
        repeat_scratch = 0
    scratch_bytes = (memory["tape"] + memory["forward_scratch"]
                     + memory["backward_scratch"] + repeat_scratch)
    return {"memory_bytes": dict(memory), "scratch_bytes": scratch_bytes,
            "total_allocation_bytes": sum(memory.values())}


def _setup_evidence(setup: Any) -> dict:
    if not isinstance(setup, dict) or set(setup) != SETUP_FIELDS or any(
            not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
            for value in setup.values()):
        raise DriverError("runner preparation/upload/allocation breakdown is missing or invalid")
    return dict(setup)


def _validate_metadata(metadata: dict, *, candidate: str, gradient_mode: str,
                       problem, case: dict, context: dict,
                       repeat_vjp: bool) -> dict:
    if metadata.get("schema") != RUNNER_SCHEMA:
        raise DriverError("native runner returned an unknown metadata schema")
    expected = {
        "requested_candidate": candidate,
        "gradient_mode": gradient_mode,
        "device_uuid": context["device_uuid"],
        "scheduler_job_id": context["scheduler_job_id"],
        "scheduler_attempt_id": context["scheduler_attempt_id"],
        "cells": problem.x.shape[2],
        "channels": problem.x.shape[1],
        "mechanisms": len(problem.mechanisms),
        "terms": sum(len(item.context_ids) for item in problem.mechanisms),
        "repeated_tail": case["repeated_tail"],
        "repeat_vjp": repeat_vjp,
        "gradient_seed": GRADIENT_SEED if gradient_mode == "accumulate" else 0.0,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise DriverError(f"runner metadata {key} differs from the frozen run")
    if metadata.get("selected_backend") not in EXPECTED_BACKENDS[candidate]:
        raise DriverError("runner selected a backend inconsistent with the candidate identity")
    memory = _memory_evidence(metadata.get("memory_bytes"), repeat_vjp=repeat_vjp)
    if memory["scratch_bytes"] > MAX_SCRATCH_BYTES:
        raise DriverError("shared scratch allocation exceeds 128 MiB")
    setup = _setup_evidence(metadata.get("setup_ms"))
    return {key: metadata[key] for key in (*expected.keys(), "selected_backend")} | {
        **memory, "setup_ms": setup,
    }


def _run_child(command: list[str], *, cwd: Path, log_path: Path,
               deadline: float) -> dict:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise DriverError("the per-attempt wall budget expired before the next runner command")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        with log_path.open("xb") as log:
            completed = subprocess.run(command, cwd=cwd, check=False,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       timeout=remaining)
    except subprocess.TimeoutExpired as exc:
        raise DriverError(f"runner command exceeded its remaining attempt budget: {command[0]}") from exc
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        raise DriverError(f"runner command exited {completed.returncode}; see {log_path}")
    return {"argv": command, "elapsed_seconds": elapsed,
            "log_path": str(log_path), "log_sha256": _sha256_file(log_path)}


def _run_one(*, candidate: str, case_id: str, manifest_path: Path,
             manifest: dict, cases: dict, oracle_cache: dict,
             runner_path: Path, build_manifest: Path, build_identity: dict,
             context: dict, output_dir: Path, gradient_mode: str,
             repeat_vjp: bool, deadline: float,
             sanitizer_bin: Path | None = None, sanitizer_tool: str | None = None
             ) -> dict:
    case_record = _case_record(manifest, case_id)
    input_path = _verify_case_file(manifest_path, case_record)
    problem, _dy = cases[case_id]
    if structure_sha256(problem) != case_record["structure_sha256"]:
        raise DriverError(f"frozen structure changed before candidate run: {case_id}")
    output_root = output_dir / case_id / candidate / gradient_mode
    if sanitizer_tool:
        output_root = output_root / sanitizer_tool
    output_root.mkdir(parents=True, exist_ok=True)
    result_path = output_root / "result.cfout"
    metadata_path = Path(str(result_path) + ".metadata.json")
    log_path = output_root / "runner.log"
    if any(path.exists() for path in (result_path, metadata_path, log_path)):
        raise DriverError(f"refusing to overwrite existing run artifacts: {output_root}")
    command = [str(runner_path), "--input", str(input_path),
               "--output", str(result_path), "--candidate", candidate,
               "--gradient-mode", gradient_mode,
               "--expected-uuid", context["device_uuid"]]
    if gradient_mode == "accumulate":
        command.extend(("--seed-gradient", str(GRADIENT_SEED)))
    if repeat_vjp:
        command.append("--repeat-vjp")
    if sanitizer_tool:
        if sanitizer_bin is None:
            raise DriverError("sanitizer binary is required for sanitizer runs")
        command = [str(sanitizer_bin), "--tool", sanitizer_tool,
                   "--error-exitcode", "99"]
        if sanitizer_tool == "memcheck":
            command.extend(("--leak-check", "full"))
        command.extend((str(runner_path), "--input", str(input_path),
                        "--output", str(result_path), "--candidate", candidate,
                        "--gradient-mode", gradient_mode,
                        "--expected-uuid", context["device_uuid"]))
        if gradient_mode == "accumulate":
            command.extend(("--seed-gradient", str(GRADIENT_SEED)))
        if repeat_vjp:
            command.append("--repeat-vjp")
    before = _verify_build_manifest(build_manifest, runner_path)
    if not _same_build_identity(before, build_identity):
        raise DriverError("build identity changed before runner invocation")
    child = _run_child(command, cwd=output_root, log_path=log_path, deadline=deadline)
    after = _require_build_stable(build_identity, build_manifest, runner_path)
    _verify_case_file(manifest_path, case_record)
    metadata, metadata_sha = _read_json(metadata_path)
    identity = _validate_metadata(metadata, candidate=candidate,
                                  gradient_mode=gradient_mode,
                                  problem=problem, case=case_record, context=context,
                                  repeat_vjp=repeat_vjp)
    actual = read_result(result_path, problem, np.float32)
    tensor_evidence = _validate_result_tensors(
        actual, oracle_cache[case_id], gradient_mode=gradient_mode,
        repeat_vjp=repeat_vjp)
    return {
        "case_id": case_id,
        "case_sha256": case_record["sha256"],
        "case_sha256_before": case_record["sha256"],
        "case_sha256_after": case_record["sha256"],
        "structure_sha256": case_record["structure_sha256"],
        "candidate": candidate,
        "gradient_mode": gradient_mode,
        "repeat_vjp": repeat_vjp,
        "sanitizer_tool": sanitizer_tool,
        "runner_identity": identity,
        "build_identity_before": before,
        "build_identity_after": after,
        "runner_metadata_path": str(metadata_path),
        "runner_metadata_sha256": metadata_sha,
        "result_path": str(result_path),
        "result_sha256": _sha256_file(result_path),
        "contract_evidence": tensor_evidence,
        "process": child,
    }


def _validation_summary(*, manifest_path: Path, manifest: dict,
                        manifest_sha: str, build_identity: dict,
                        python_sources: dict, python_sources_after: dict,
                        context: dict, runs: list[dict], sanitizer_sha: str,
                        sanitizer_path: Path, sanitizer_version: str,
                        artifact_dir: Path) -> dict:
    summary = {
        "schema": DRIVER_SCHEMA,
        "kind": "gpu_validation",
        "gpu_executed": True,
        "validated": True,
        "timed": False,
        "fixture_schema": FIXTURE_SCHEMA,
        "fixture_manifest_path": str(Path(manifest_path).absolute()),
        "fixture_manifest_sha256": manifest_sha,
        "oracle_cache_sha256": manifest["oracle_cache"]["sha256"],
        "build_identity": build_identity,
        "python_source_identity_before": python_sources,
        "python_source_identity_after": python_sources_after,
        "scheduler_context": context,
        "sanitizer_binary_path": str(sanitizer_path),
        "sanitizer_binary_sha256": sanitizer_sha,
        "sanitizer_version": sanitizer_version,
        "runs": runs,
        "scratch_limit_bytes": MAX_SCRATCH_BYTES,
        "artifact_dir": str(artifact_dir.absolute()),
    }
    return summary


def run_validation(*, manifest_path: Path, runner_path: Path,
                   build_manifest: Path, sanitizer_bin: Path,
                   expected_uuid: str, artifact_dir: Path,
                   wall_budget_seconds: int = 165) -> dict:
    if not (1 <= wall_budget_seconds <= MAX_ATTEMPT_SECONDS):
        raise DriverError(f"wall budget must be between 1 and {MAX_ATTEMPT_SECONDS}s")
    started = time.monotonic()
    deadline = started + wall_budget_seconds
    python_sources_before = _python_source_identity()
    context = _attempt_context(expected_uuid)
    manifest, manifest_sha, cases, oracle = _fixture_set(manifest_path)
    build_identity = _verify_build_manifest(build_manifest, runner_path)
    sanitizer_bin = Path(sanitizer_bin).absolute()
    if sanitizer_bin.is_symlink() or not sanitizer_bin.is_file():
        raise DriverError("compute-sanitizer must be a regular executable")
    if not os.access(sanitizer_bin, os.X_OK):
        raise DriverError("compute-sanitizer lacks an executable bit")
    if sanitizer_bin.resolve(strict=True) != REGISTERED_SANITIZER_PATH.resolve(strict=True):
        raise DriverError("compute-sanitizer path differs from the registered host tool")
    try:
        version_result = subprocess.run([str(sanitizer_bin), "--version"],
                                        check=False, capture_output=True, text=True,
                                        timeout=10)
    except subprocess.SubprocessError as exc:
        raise DriverError(f"could not identify compute-sanitizer: {exc}") from exc
    sanitizer_version = (version_result.stdout + version_result.stderr).strip()
    if version_result.returncode != 0 or EXPECTED_SANITIZER_VERSION not in sanitizer_version:
        raise DriverError("compute-sanitizer version differs from the registered host tool")
    sanitizer_sha_before = _sha256_file(sanitizer_bin)
    artifact_dir = Path(artifact_dir).absolute()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = artifact_dir / "candidates"
    runs = []

    # All correctness-only fixtures are checked on both prototype arms. The
    # timed-size case separately checks the production extent before pairing.
    correctness_ids = [
        "ordinary_seed_0", "ordinary_seed_1", "zero_participants",
        "signed_cancellation", "saturated_gates", "repeated_tail_role",
        "runtime_refresh_before", "runtime_refresh_after",
    ]
    before_sha, after_sha = (
        _case_record(manifest, "runtime_refresh_before")["structure_sha256"],
        _case_record(manifest, "runtime_refresh_after")["structure_sha256"],
    )
    if before_sha != after_sha:
        raise DriverError("runtime refresh fixtures do not preserve the prepared structure")
    for case_id in (*correctness_ids, "timed_seed_0"):
        for candidate in PROTOTYPE_CANDIDATES:
            runs.append(_run_one(
                candidate=candidate, case_id=case_id, manifest_path=manifest_path,
                manifest=manifest, cases=cases, oracle_cache=oracle,
                runner_path=runner_path, build_manifest=build_manifest,
                build_identity=build_identity, context=context,
                output_dir=candidates_dir, gradient_mode="overwrite",
                repeat_vjp=True, deadline=deadline))

    # The sentinel detects overwrite/accumulate confusion; repeated outputs
    # must independently match one seeded accumulation, not compound it.
    for candidate in PROTOTYPE_CANDIDATES:
        runs.append(_run_one(
            candidate=candidate, case_id="repeated_tail_role",
            manifest_path=manifest_path, manifest=manifest, cases=cases,
            oracle_cache=oracle, runner_path=runner_path,
            build_manifest=build_manifest, build_identity=build_identity,
            context=context, output_dir=candidates_dir,
            gradient_mode="accumulate", repeat_vjp=True, deadline=deadline))

    # Targeted sanitizers run on a repeat-role case that stresses multiple
    # writes to the same canonical input while retaining the private topology.
    sanitizer_case = "repeated_tail_role"
    sanitizer_case_hash = _sha256_file(sanitizer_bin)
    if sanitizer_case_hash != sanitizer_sha_before:
        raise DriverError("compute-sanitizer changed before the sanitizer matrix")
    for tool in ("memcheck", "racecheck", "initcheck", "synccheck"):
        for candidate in PROTOTYPE_CANDIDATES:
            runs.append(_run_one(
                candidate=candidate, case_id=sanitizer_case,
                manifest_path=manifest_path, manifest=manifest, cases=cases,
                oracle_cache=oracle, runner_path=runner_path,
                build_manifest=build_manifest, build_identity=build_identity,
                context=context, output_dir=candidates_dir,
                gradient_mode="overwrite", repeat_vjp=False,
                deadline=deadline, sanitizer_bin=sanitizer_bin,
                sanitizer_tool=tool))
    if _sha256_file(sanitizer_bin) != sanitizer_sha_before:
        raise DriverError("compute-sanitizer changed during validation")
    _require_build_stable(build_identity, build_manifest, runner_path)
    python_sources_after = _python_source_identity()
    if python_sources_after != python_sources_before:
        raise DriverError("Python fixture, oracle, test, or driver sources changed during validation")
    if _sha256_file(manifest_path) != manifest_sha:
        raise DriverError("frozen fixture manifest bytes changed during validation")
    oracle_path = Path(manifest_path).parent / manifest["oracle_cache"]["path"]
    if _sha256_file(oracle_path) != manifest["oracle_cache"]["sha256"]:
        raise DriverError("precomputed CPU oracle cache changed during validation")
    summary = _validation_summary(
        manifest_path=manifest_path, manifest=manifest, manifest_sha=manifest_sha,
        build_identity=build_identity, python_sources=python_sources_before,
        python_sources_after=python_sources_after,
        context=context, runs=runs,
        sanitizer_sha=sanitizer_sha_before,
        sanitizer_path=sanitizer_bin.resolve(strict=True),
        sanitizer_version=sanitizer_version, artifact_dir=artifact_dir)
    summary["elapsed_seconds"] = time.monotonic() - started
    summary_path = artifact_dir / "validation_summary.json"
    if summary_path.exists():
        raise DriverError(f"refusing to overwrite validation summary: {summary_path}")
    _write_json_exclusive(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    summary["summary_sha256"] = _sha256_file(summary_path)
    return summary


def _verify_validation_summary(path: Path, *, fixture_sha: str,
                               build_identity: dict, python_sources: dict) -> dict:
    summary, _summary_sha = _read_json(path)
    if summary.get("schema") != DRIVER_SCHEMA or summary.get("kind") != "gpu_validation" \
            or summary.get("gpu_executed") is not True \
            or summary.get("validated") is not True or summary.get("timed") is not False:
        raise DriverError("measurement requires a successful real GPU validation summary")
    if summary.get("fixture_manifest_sha256") != fixture_sha:
        raise DriverError("validation summary names different frozen fixture bytes")
    if not _same_build_identity(summary.get("build_identity", {}), build_identity):
        raise DriverError("validation and measurement build/source/binary identities differ")
    if summary.get("python_source_identity_before") != python_sources \
            or summary.get("python_source_identity_after") != python_sources:
        raise DriverError("validation and measurement Python source identities differ")
    runs = summary.get("runs")
    if not isinstance(runs, list) or not runs:
        raise DriverError("validation summary lacks its per-run evidence")
    required_sanitizers = {
        (tool, candidate)
        for tool in ("memcheck", "racecheck", "initcheck", "synccheck")
        for candidate in PROTOTYPE_CANDIDATES
    }
    seen = {(row.get("sanitizer_tool"), row.get("candidate"))
            for row in runs if row.get("sanitizer_tool")}
    if seen != required_sanitizers:
        raise DriverError("validation summary does not cover all four sanitizers on both arms")
    correctness_ids = {
        "ordinary_seed_0", "ordinary_seed_1", "timed_seed_0",
        "zero_participants", "signed_cancellation", "saturated_gates",
        "repeated_tail_role", "runtime_refresh_before", "runtime_refresh_after",
    }
    expected_primary = {
        (case_id, candidate, "overwrite", True)
        for case_id in correctness_ids for candidate in PROTOTYPE_CANDIDATES
    }
    seen_primary = {(row.get("case_id"), row.get("candidate"),
                     row.get("gradient_mode"), row.get("repeat_vjp"))
                    for row in runs if not row.get("sanitizer_tool")}
    if not expected_primary.issubset(seen_primary):
        raise DriverError("validation summary lacks a required overwrite/repeated-VJP case")
    expected_accumulation = {
        ("repeated_tail_role", candidate, "accumulate", True)
        for candidate in PROTOTYPE_CANDIDATES
    }
    if not expected_accumulation.issubset(seen_primary):
        raise DriverError("validation summary lacks overwrite/accumulate coverage on both arms")
    return summary


def _validate_metrics(metrics: dict, *, context: dict, timed_case: dict) -> dict:
    if metrics.get("schema") != PAIRED_SCHEMA:
        raise DriverError("paired runner returned an unknown measurement schema")
    expected_top = {"device_uuid": context["device_uuid"],
                    "scheduler_job_id": context["scheduler_job_id"],
                    "scheduler_attempt_id": context["scheduler_attempt_id"],
                    "cells": timed_case["cells"],
                    "channels": timed_case["channels"],
                    "mechanisms": timed_case["mechanisms"],
                    "terms": 16,
                    "gradient_mode": "overwrite"}
    for key, value in expected_top.items():
        if metrics.get(key) != value:
            raise DriverError(f"paired metrics {key} differs from the admitted experiment")
    if metrics.get("warmup_iterations_per_arm") != 10 \
            or metrics.get("block_count") != 7 \
            or metrics.get("iterations_per_block") != 30:
        raise DriverError("paired metrics do not report the frozen 10/7x30 schedule")
    rows = metrics.get("measurements")
    if not isinstance(rows, list) or len(rows) != 21:
        raise DriverError("paired metrics must retain all 21 candidate block samples")
    by_block: dict[int, list[dict]] = {}
    for row in rows:
        block = row.get("block")
        candidate = row.get("candidate")
        position = row.get("position")
        if not isinstance(block, int) or not 0 <= block < 7:
            raise DriverError("paired timing row has an invalid block number")
        if candidate not in ALL_CANDIDATES or not isinstance(position, int) \
                or position not in (0, 1, 2):
            raise DriverError("paired timing row has an invalid candidate or position")
        if row.get("iterations") != 30:
            raise DriverError("paired timing row does not contain 30 full forward/VJP steps")
        elapsed = row.get("host_block_ms")
        mean = row.get("host_mean_ms")
        if not isinstance(elapsed, (float, int)) or not math.isfinite(elapsed) or elapsed <= 0:
            raise DriverError("paired timing row has a nonpositive or invalid host block time")
        if not isinstance(mean, (float, int)) or not math.isfinite(mean) \
                or not math.isclose(float(mean), float(elapsed) / 30.0,
                                    rel_tol=1e-9, abs_tol=1e-12):
            raise DriverError("paired timing mean differs from its raw host block sample")
        if row.get("selected_backend") not in EXPECTED_BACKENDS[candidate]:
            raise DriverError("paired timing row selected an unexpected backend")
        by_block.setdefault(block, []).append(row)
    previous_pair_order = None
    for block in range(7):
        block_rows = by_block.get(block, [])
        if len(block_rows) != 3 or {row["candidate"] for row in block_rows} != set(ALL_CANDIDATES) \
                or {row["position"] for row in block_rows} != {0, 1, 2}:
            raise DriverError(f"paired block {block} is incomplete or duplicated")
        block_rows = sorted(block_rows, key=lambda row: row["position"])
        native_position = next(row["position"] for row in block_rows
                               if row["candidate"] == "native_direct")
        if native_position != block % 3:
            raise DriverError("native_direct position did not rotate through all three slots")
        pair_order = tuple(row["candidate"] for row in block_rows
                           if row["candidate"] != "native_direct")
        if previous_pair_order is not None and pair_order == previous_pair_order:
            raise DriverError("unfactored/circuit relative order did not alternate")
        previous_pair_order = pair_order
    actual_order = metrics.get("actual_order")
    expected_order = [
        [row["candidate"] for row in sorted(by_block[block], key=lambda row: row["position"])]
        for block in range(7)
    ]
    if actual_order != expected_order:
        raise DriverError("paired metrics actual_order differs from raw per-block samples")
    diagnostics = metrics.get("diagnostics")
    if not isinstance(diagnostics, list) or {row.get("candidate") for row in diagnostics} != set(ALL_CANDIDATES):
        raise DriverError("paired metrics lack separate event diagnostics for all three arms")
    for row in diagnostics:
        candidate = row["candidate"]
        if row.get("selected_backend") not in EXPECTED_BACKENDS[candidate]:
            raise DriverError("event diagnostics report an unexpected candidate backend")
        for key in ("forward_ms", "backward_ms", "full_ms"):
            samples = row.get(key)
            if not isinstance(samples, list) or len(samples) != 5 or any(
                    not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
                    for value in samples):
                raise DriverError("event diagnostics must retain five positive samples per phase")
    arms = metrics.get("arms")
    if not isinstance(arms, list) or {row.get("candidate") for row in arms} != set(ALL_CANDIDATES):
        raise DriverError("paired metrics lack memory/setup records for all three arms")
    normalized_arms = []
    shared_values_bytes = None
    private_allocation_bytes = 0
    process_scratch_bytes = 0
    for row in arms:
        candidate = row["candidate"]
        if row.get("selected_backend") not in EXPECTED_BACKENDS[candidate]:
            raise DriverError("paired arm setup/backend/memory identity is invalid")
        memory = _memory_evidence(row.get("memory_bytes"), repeat_vjp=False)
        if memory["memory_bytes"]["repeat_gradient_and_scratch"] != 0:
            raise DriverError("paired timing unexpectedly allocated a repeated VJP owner")
        setup = _setup_evidence(row.get("setup_ms"))
        common_values = memory["memory_bytes"]["resident_values_and_cotangent"]
        if shared_values_bytes is None:
            shared_values_bytes = common_values
        elif shared_values_bytes != common_values:
            raise DriverError("paired arms disagree about the shared resident value allocation")
        process_scratch_bytes += memory["scratch_bytes"]
        private_allocation_bytes += memory["total_allocation_bytes"] - common_values
        normalized_arms.append({
            "candidate": candidate,
            "selected_backend": row["selected_backend"],
            **memory,
            "setup_ms": setup,
        })
    if process_scratch_bytes > MAX_SCRATCH_BYTES:
        raise DriverError("simultaneously resident paired-arm scratch exceeds 128 MiB")
    return {"warmup_iterations_per_arm": 10, "block_count": 7,
            "iterations_per_block": 30, "raw_measurements": rows,
            "diagnostics": diagnostics,
            "arms": normalized_arms, "actual_order": actual_order,
            "memory_summary": {
                "shared_values_counted_once_bytes": shared_values_bytes,
                "all_arm_private_allocations_bytes": private_allocation_bytes,
                "peak_device_allocated_bytes": shared_values_bytes + private_allocation_bytes,
                "simultaneous_scratch_bytes": process_scratch_bytes,
            }}


def run_measurement(*, manifest_path: Path, runner_path: Path,
                    build_manifest: Path, validation_summary_path: Path,
                    expected_uuid: str, artifact_dir: Path,
                    wall_budget_seconds: int = 165) -> dict:
    if not (1 <= wall_budget_seconds <= MAX_ATTEMPT_SECONDS):
        raise DriverError(f"wall budget must be between 1 and {MAX_ATTEMPT_SECONDS}s")
    started = time.monotonic()
    deadline = started + wall_budget_seconds
    python_sources_before = _python_source_identity()
    context = _attempt_context(expected_uuid)
    manifest, manifest_sha, cases, oracle = _fixture_set(manifest_path)
    build_identity = _verify_build_manifest(build_manifest, runner_path)
    validation = _verify_validation_summary(
        validation_summary_path, fixture_sha=manifest_sha,
        build_identity=build_identity, python_sources=python_sources_before)
    timed_record = _case_record(manifest, "timed_seed_0")
    input_path = _verify_case_file(manifest_path, timed_record)
    problem, _dy = cases["timed_seed_0"]
    artifact_dir = Path(artifact_dir).absolute()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    prefix = artifact_dir / "paired"
    metrics_path = artifact_dir / "paired.metrics.json"
    log_path = artifact_dir / "paired.runner.log"
    outputs = [Path(f"{prefix}.{candidate}.cfout") for candidate in ALL_CANDIDATES]
    preexisting = [path for path in (*outputs, metrics_path, log_path)
                   if path.exists() or Path(str(path) + ".metadata.json").exists()]
    if preexisting:
        raise DriverError(f"refusing to overwrite measurement artifacts: {preexisting[0]}")
    command = [str(Path(runner_path).absolute()), "--input", str(input_path),
               "--output", str(prefix), "--candidate", "paired",
               "--metrics", str(metrics_path), "--expected-uuid", expected_uuid]
    build_before = _verify_build_manifest(build_manifest, runner_path)
    if not _same_build_identity(build_before, build_identity):
        raise DriverError("build identity changed before paired measurement")
    process = _run_child(command, cwd=artifact_dir, log_path=log_path, deadline=deadline)
    build_after = _require_build_stable(build_identity, build_manifest, runner_path)
    python_sources_after = _python_source_identity()
    if python_sources_after != python_sources_before:
        raise DriverError("Python fixture, oracle, test, or driver sources changed during measurement")
    if _sha256_file(manifest_path) != manifest_sha:
        raise DriverError("frozen fixture manifest bytes changed during measurement")
    oracle_path = Path(manifest_path).parent / manifest["oracle_cache"]["path"]
    if _sha256_file(oracle_path) != manifest["oracle_cache"]["sha256"]:
        raise DriverError("precomputed CPU oracle cache changed during measurement")
    _verify_case_file(manifest_path, timed_record)
    metrics, metrics_sha = _read_json(metrics_path)
    metric_evidence = _validate_metrics(metrics, context=context,
                                        timed_case=timed_record)
    output_evidence = []
    for candidate, result_path in zip(ALL_CANDIDATES, outputs):
        metadata_path = Path(str(result_path) + ".metadata.json")
        metadata, metadata_sha = _read_json(metadata_path)
        identity = _validate_metadata(metadata, candidate=candidate,
                                      gradient_mode="overwrite", problem=problem,
                                      case=timed_record, context=context,
                                      repeat_vjp=False)
        actual = read_result(result_path, problem, np.float32)
        tensors = _validate_result_tensors(
            actual, oracle["timed_seed_0"], gradient_mode="overwrite",
            repeat_vjp=False)
        output_evidence.append({
            "candidate": candidate,
            "selected_backend": identity["selected_backend"],
            "result_path": str(result_path),
            "result_sha256": _sha256_file(result_path),
            "metadata_path": str(metadata_path),
            "metadata_sha256": metadata_sha,
            "contract_evidence": tensors,
        })
    fixture_case_sha_after = _sha256_file(input_path)
    if fixture_case_sha_after != timed_record["sha256"]:
        raise DriverError("timed case bytes changed during paired measurement")
    summary = {
        "schema": DRIVER_SCHEMA,
        "kind": "paired_measurement",
        "gpu_executed": True,
        "validated": True,
        "timed": True,
        "fixture_manifest_path": str(Path(manifest_path).absolute()),
        "fixture_manifest_sha256": manifest_sha,
        "case_id": "timed_seed_0",
        "case_sha256": timed_record["sha256"],
        "case_sha256_before": timed_record["sha256"],
        "case_sha256_after": fixture_case_sha_after,
        "structure_sha256": timed_record["structure_sha256"],
        "build_identity": build_identity,
        "python_source_identity_before": python_sources_before,
        "python_source_identity_after": python_sources_after,
        "build_identity_before": build_before,
        "build_identity_after": build_after,
        "validation_summary_path": str(Path(validation_summary_path).absolute()),
        "validation_summary_sha256": _sha256_file(validation_summary_path),
        "validation_scheduler_context": validation["scheduler_context"],
        "scheduler_context": context,
        "metrics_path": str(metrics_path),
        "metrics_sha256": metrics_sha,
        "measurements": metric_evidence,
        "candidate_outputs": output_evidence,
        "process": process,
        "elapsed_seconds": time.monotonic() - started,
    }
    summary_path = artifact_dir / "measurement_summary.json"
    if summary_path.exists():
        raise DriverError(f"refusing to overwrite measurement summary: {summary_path}")
    _write_json_exclusive(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    summary["summary_sha256"] = _sha256_file(summary_path)
    return summary


def _write_json_exclusive(path: Path, value: dict) -> None:
    encoded = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with Path(path).open("xb") as stream:
        stream.write(encoded)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--case-dir", type=Path, required=True)
    for name in ("validate", "measure"):
        child = subparsers.add_parser(name)
        child.add_argument("--case-manifest", type=Path, required=True)
        child.add_argument("--runner", type=Path, required=True)
        child.add_argument("--build-manifest", type=Path, required=True)
        child.add_argument("--expected-uuid", required=True)
        child.add_argument("--artifact-dir", type=Path, required=True)
        child.add_argument("--wall-budget-seconds", type=int, default=165)
    validate_parser = subparsers.choices["validate"]
    validate_parser.add_argument("--sanitizer-bin", type=Path, required=True)
    measure_parser = subparsers.choices["measure"]
    measure_parser.add_argument("--validation-summary", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            result = freeze_cases(args.case_dir)
            summary = {"schema": DRIVER_SCHEMA, "kind": "frozen_inputs",
                       "gpu_executed": False, "timed": False, **result}
        elif args.command == "validate":
            summary = run_validation(
                manifest_path=args.case_manifest, runner_path=args.runner,
                build_manifest=args.build_manifest, sanitizer_bin=args.sanitizer_bin,
                expected_uuid=args.expected_uuid, artifact_dir=args.artifact_dir,
                wall_budget_seconds=args.wall_budget_seconds)
        else:
            summary = run_measurement(
                manifest_path=args.case_manifest, runner_path=args.runner,
                build_manifest=args.build_manifest,
                validation_summary_path=args.validation_summary,
                expected_uuid=args.expected_uuid, artifact_dir=args.artifact_dir,
                wall_budget_seconds=args.wall_budget_seconds)
    except (DriverError, OSError, ValueError, AssertionError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"shared-product driver: {exc}\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
