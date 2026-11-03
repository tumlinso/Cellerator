#!/usr/bin/env python3
"""Collect one bounded, source-bound factorized-interaction timing partition.

The script runs inside an already-admitted Autobench attempt. It never asks a
Scheduler for a device. Each invocation covers one fixed workload and one
phase (seven paired blocks), retains validated raw timing/evidence summaries,
and delegates wire creation, numerical validation, sampled scratch accounting,
and success-only wire cleanup to ``bounded_validation``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import sys
from typing import Any


SCHEMA = "cellerator.factorized_interactions.timing_partition.v1"
QUALIFICATION_SCHEMA = "cellerator.factorized_interactions.recorded_qualification.v1"
HARNESS_PROVENANCE_SCHEMA = "cellerator.factorized_interactions.timing_harness_provenance.v1"
HARNESS_COMPONENT_KEYS = frozenset({
    "timing_collector_sha256", "bounded_validation_sha256",
    "recorded_qualification_sha256", "recorded_receipts_sha256",
    "sanitizer_wrapper_sha256",
})
EXPECTED_SOURCE_CLOSURE_SHA256 = (
    "1f8e34ce89594fc3d8fb3d63c46600091c24f3016d6a9ddfd9dc3630eb08543c"
)
EXPECTED_VALIDATOR_SHA256 = (
    "466b495ac56fb0df9c248649f7307d871de681200a4a90de43457c0c6447de31"
)
PRIMARY_METRIC = "warm_forward_vjp_batch_host_wall_mean"
ITERATIONS = 30
PAIRED_BLOCKS = 7
EXPECTED_PERFORMANCE_WORKLOADS = 13
MAX_ATTEMPT_BYTES = 128 * 1024 * 1024
DTYPES = {"activation": "fp16", "parameter": "fp16", "output": "fp16"}
GRADIENT_MODE = "overwrite"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")
_STABLE_CONTEXT_FIELDS = (
    "scheduler_job_id", "scheduler_attempt_id", "device_uuid", "cgroup", "unit_name",
)


class TimingPartitionError(RuntimeError):
    """The partition did not complete under the fixed workload contract."""


class PartitionRunFailure(TimingPartitionError):
    """A run failed after preserving only the observations completed so far."""

    def __init__(self, message: str, partial_report: dict[str, Any]):
        super().__init__(message)
        self.partial_report = partial_report


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path_value: str | Path, label: str) -> Path:
    path = Path(path_value)
    if path.is_symlink() or not path.is_file():
        raise TimingPartitionError(f"{label} must be a regular non-symlink file: {path}")
    return path.resolve(strict=True)


def _reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    if any(component.is_symlink() for component in (absolute, *absolute.parents)):
        raise TimingPartitionError(f"path cannot contain a symlink: {path}")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TimingPartitionError(f"cannot read {label} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise TimingPartitionError(f"{label} JSON must contain an object")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    """Write a durable report without replacing existing task evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise TimingPartitionError(f"refusing to overwrite timing evidence: {path}") from exc
    _fsync_directory(path.parent)


def _validate_context(context: Mapping[str, Any], expected_uuid: str,
                      label: str) -> None:
    if context.get("status") != "observed":
        raise TimingPartitionError(f"{label} context is not a live observed process record")
    if context.get("device_uuid") != expected_uuid:
        raise TimingPartitionError(f"{label} context does not match the requested GPU UUID")
    for key in _STABLE_CONTEXT_FIELDS:
        if not isinstance(context.get(key), str) or not context[key]:
            raise TimingPartitionError(f"{label} context omitted {key}")
    if context.get("scheduler_job_id") != os.environ.get("AUTOBENCH_SCHEDULER_JOB_ID", "").strip():
        raise TimingPartitionError(f"{label} context does not match the inherited Autobench job")
    if context.get("scheduler_attempt_id") != os.environ.get(
            "AUTOBENCH_SCHEDULER_ATTEMPT_ID", "").strip():
        raise TimingPartitionError(f"{label} context does not match the inherited Autobench attempt")


def _same_context(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return all(first.get(key) == second.get(key) for key in _STABLE_CONTEXT_FIELDS) and \
        first.get("context_sha256", first.get("context_path")) == \
        second.get("context_sha256", second.get("context_path"))


def _identity_from_build(build: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "source_closure_sha256": build.get("source_closure_sha256"),
        "candidate_runner_sha256": build.get("runner_sha256"),
        "candidate_manifest_sha256": build.get("build_manifest_sha256"),
        "library_sha256": build.get("library_sha256"),
        "validator_sha256": build.get("validator_sha256"),
    }
    if any(not isinstance(value, str) or not _SHA256.fullmatch(value)
           for value in required.values()):
        raise TimingPartitionError("verified build identity omitted a required binary/source digest")
    if required["source_closure_sha256"] != EXPECTED_SOURCE_CLOSURE_SHA256:
        raise TimingPartitionError("native build does not use the accepted immutable source closure")
    if required["validator_sha256"] != EXPECTED_VALIDATOR_SHA256:
        raise TimingPartitionError("native build does not use the accepted fixed validator")
    candidate_manifest = _regular_file(build.get("build_manifest_path", ""),
                                       "candidate build manifest")
    if sha256_file(candidate_manifest) != required["candidate_manifest_sha256"]:
        raise TimingPartitionError("candidate build manifest differs from the verified build record")
    candidate_runner = _regular_file(build.get("runner_path", ""), "candidate runner")
    if sha256_file(candidate_runner) != required["candidate_runner_sha256"]:
        raise TimingPartitionError("candidate runner differs from the verified build record")
    manifest_data = _read_json(candidate_manifest, "candidate build manifest")
    matches: list[tuple[Path, dict[str, Any]]] = []
    for ancestor in candidate_manifest.parents:
        for name in ("factorized_interactions_baseline_build_manifest.json",
                     "factorized_interactions_build_manifest.json"):
            possible = ancestor / name
            if possible == candidate_manifest or possible.is_symlink() or not possible.is_file():
                continue
            try:
                data = _read_json(possible, "base build manifest")
                runner_entry = data.get("binaries", {}).get("runner")
                library_entry = data.get("binaries", {}).get("factorized_interaction_cuda")
                if not isinstance(runner_entry, dict) or not isinstance(library_entry, dict):
                    continue
                if data.get("schema") != "cellerator.factorized_interactions.build.v1" or \
                        data.get("architecture") != "sm_70" or data.get("fast_math") is not False or \
                        data.get("source_closure_sha256") != required["source_closure_sha256"] or \
                        data.get("source_files") != manifest_data.get("source_files") or \
                        library_entry.get("sha256") != required["library_sha256"]:
                    continue
                base_runner = _checked_manifest_binary(possible, runner_entry, "base runner")
                base_library = _checked_manifest_binary(possible, library_entry, "base library")
                if base_runner.stat().st_size != runner_entry.get("bytes") or \
                        sha256_file(base_runner) != runner_entry.get("sha256") or \
                        base_library.stat().st_size != library_entry.get("bytes") or \
                        sha256_file(base_library) != required["library_sha256"]:
                    continue
                base_candidate = data.get("candidate_runner")
                if not isinstance(base_candidate, dict):
                    continue
                candidate_link_manifest = (possible.parent /
                                           base_candidate.get("manifest_path", "")).resolve(strict=True)
                candidate_link_runner = (possible.parent /
                                         base_candidate.get("binary_path", "")).resolve(strict=True)
                if candidate_link_manifest != candidate_manifest or \
                        candidate_link_runner != candidate_runner or \
                        base_candidate.get("manifest_sha256") != required["candidate_manifest_sha256"] or \
                        base_candidate.get("binary_sha256") != required["candidate_runner_sha256"]:
                    continue
                matches.append((possible.resolve(strict=True), data))
            except (OSError, TimingPartitionError, json.JSONDecodeError, ValueError):
                continue
    if len(matches) != 1:
        raise TimingPartitionError("could not uniquely bind candidate build to its source-matched base runner")
    base_manifest, base_data = matches[0]
    required["candidate_manifest_path"] = str(candidate_manifest)
    required["baseline_runner_sha256"] = base_data["binaries"]["runner"]["sha256"]
    required["baseline_manifest_sha256"] = sha256_file(base_manifest)
    required["baseline_manifest_path"] = str(base_manifest)
    return required


def _checked_manifest_binary(manifest_path: Path, entry: Mapping[str, Any],
                             label: str) -> Path:
    relative = entry.get("path")
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise TimingPartitionError(f"{label} manifest path is invalid")
    binary = manifest_path.parent / relative
    if binary.is_symlink() or not binary.is_file():
        raise TimingPartitionError(f"{label} is missing or nonregular")
    resolved = binary.resolve(strict=True)
    if not resolved.is_relative_to(manifest_path.parent.resolve(strict=True)):
        raise TimingPartitionError(f"{label} escapes its immutable build directory")
    return resolved


def validate_qualification_bundle(
    bundle_path: Path,
    *,
    source: Path,
    build_identity: Mapping[str, Any],
    expected_uuid: str,
    bundle_validator: Callable[[Path, Path, Path, Path, str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Delegate full report/receipt validation to the independent CPU verifier."""
    path = _regular_file(bundle_path, "qualification bundle")
    try:
        verified = bundle_validator(
            path,
            source,
            Path(build_identity["candidate_manifest_path"]),
            Path(build_identity["baseline_manifest_path"]),
            expected_uuid,
        )
    except Exception as exc:
        raise TimingPartitionError(f"qualified prerequisites were rejected: {exc}") from exc
    if not isinstance(verified, Mapping):
        raise TimingPartitionError("qualification verifier did not return verified evidence")
    if verified.get("schema") != QUALIFICATION_SCHEMA or \
            verified.get("status") != "validated_pending_independent_review":
        raise TimingPartitionError("qualification verifier did not accept the complete prerequisite bundle")
    bundle = _read_json(path, "qualification bundle")
    if bundle.get("schema") != QUALIFICATION_SCHEMA or \
            bundle.get("status") != "validated_pending_independent_review":
        raise TimingPartitionError("qualification bundle is not the accepted complete pre-timing bundle")
    if bundle.get("device_uuid") != expected_uuid:
        raise TimingPartitionError("qualification bundle is for a different GPU UUID")
    for key in ("schema", "status", "device_uuid", "build", "coverage", "reports",
                "evidence_root", "sanitizer_wrapper_path", "sanitizer_wrapper_sha256",
                "sanitizer_tool_identity"):
        if verified.get(key) != bundle.get(key):
            raise TimingPartitionError(f"qualification verifier and bundle differ for {key}")
    if bundle.get("evidence_root") not in (None, "."):
        raise TimingPartitionError("qualification bundle evidence_root must be its own directory")
    bundle_build = bundle.get("build")
    if not isinstance(bundle_build, dict):
        raise TimingPartitionError("qualification bundle omitted exact build identities")
    for key, value in (
            ("source_closure_sha256", build_identity["source_closure_sha256"]),
            ("candidate_runner_sha256", build_identity["candidate_runner_sha256"]),
            ("candidate_manifest_sha256", build_identity["candidate_manifest_sha256"]),
            ("baseline_runner_sha256", build_identity["baseline_runner_sha256"]),
            ("baseline_manifest_sha256", build_identity["baseline_manifest_sha256"]),
            ("library_sha256", build_identity["library_sha256"]),
            ("validator_sha256", build_identity["validator_sha256"])):
        if bundle_build.get(key) != value:
            raise TimingPartitionError(f"qualification bundle build identity mismatch for {key}")
    reports = bundle.get("reports")
    coverage = bundle.get("coverage")
    if not isinstance(reports, list) or not isinstance(coverage, dict):
        raise TimingPartitionError("qualification verifier omitted gate references or coverage")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema": bundle["schema"],
        "status": bundle["status"],
        "device_uuid": expected_uuid,
        "build": dict(bundle_build),
        "coverage": dict(coverage),
        "verification": {
            "validator": "recorded_qualification.validate_existing_bundle",
            "prerequisite_validator_passed": True,
            "schema": verified.get("schema"),
            "status": verified.get("status"),
        },
        "evidence_root": bundle.get("evidence_root"),
        "sanitizer_wrapper_path": bundle.get("sanitizer_wrapper_path"),
        "sanitizer_wrapper_sha256": bundle.get("sanitizer_wrapper_sha256"),
        "sanitizer_tool_identity": bundle.get("sanitizer_tool_identity"),
        "reports": list(reports),
    }


def pair_order(phase: str, round_index: int) -> tuple[list[str], int]:
    """Return incumbent/challenger AB/BA and descriptive-arm rotation."""
    if phase not in {"exploration", "confirmation"}:
        raise TimingPartitionError("phase must be exploration or confirmation")
    if isinstance(round_index, bool) or not isinstance(round_index, int) or \
            not 0 <= round_index < PAIRED_BLOCKS:
        raise TimingPartitionError("round index must identify one of seven fixed phase blocks")
    phase_offset = 0 if phase == "exploration" else PAIRED_BLOCKS
    global_index = phase_offset + round_index
    paired = (["strongest_direct", "packet"] if global_index % 2 == 0
              else ["packet", "strongest_direct"])
    descriptive_position = global_index % 3
    return paired, descriptive_position


def _finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or \
            not math.isfinite(float(value)) or float(value) <= 0.0:
        raise TimingPartitionError(f"{label} must be a finite positive number")
    return float(value)


def _validate_measurement(
    evidence: Mapping[str, Any],
    *,
    workload: str,
    mode: str,
    route: str,
    expected_identity: tuple[str, str, str],
    expected_uuid: str,
    context: Mapping[str, Any],
    build_identity: Mapping[str, Any],
) -> dict[str, Any]:
    summary = evidence.get("summary")
    if not isinstance(summary, dict) or \
            summary.get("schema") != "cellerator.factorized_interactions.validation.v1" or \
            summary.get("validated") is not True or summary.get("timed") is not True:
        raise TimingPartitionError(f"{mode}/{workload} did not pass the fixed timed validator")
    expected_summary = {
        "workload": workload,
        "route": route,
        "activation_storage": DTYPES["activation"],
        "parameter_storage": DTYPES["parameter"],
        "output_storage": DTYPES["output"],
        "gradient_mode": GRADIENT_MODE,
        "scheduler_job_id": context["scheduler_job_id"],
    }
    if any(summary.get(key) != value for key, value in expected_summary.items()):
        raise TimingPartitionError(f"{mode}/{workload} validator summary does not match request")
    native = summary.get("native_execution")
    metrics = summary.get("metrics")
    provenance = summary.get("build_provenance")
    if not isinstance(native, dict) or not isinstance(metrics, dict) or not isinstance(provenance, dict):
        raise TimingPartitionError(f"{mode}/{workload} omitted dispatch, metric, or build identity")
    expected_native = {
        "candidate": mode,
        "requested_route": route,
        "selected_backend": expected_identity[0],
        "actual_implementation": expected_identity[1],
        "structural_layout": expected_identity[2],
        "device_uuid": expected_uuid,
        "scheduler_job_id": context["scheduler_job_id"],
    }
    if any(native.get(key) != value for key, value in expected_native.items()):
        raise TimingPartitionError(f"{mode}/{workload} selected an unexpected native implementation")
    if any(metrics.get(key) != value for key, value in (
            ("device_uuid", expected_uuid),
            ("scheduler_job_id", context["scheduler_job_id"]),
            ("iterations", ITERATIONS),
            ("selected_backend", expected_identity[0]))):
        raise TimingPartitionError(f"{mode}/{workload} metrics use a different run identity")
    if metrics.get("schema") != "cellerator.factorized_interactions.metrics.v1":
        raise TimingPartitionError(f"{mode}/{workload} returned an unknown metrics schema")
    if any(provenance.get(key) != value for key, value in (
            ("source_closure_sha256", build_identity["source_closure_sha256"]),
            ("runner_binary_sha256", build_identity["candidate_runner_sha256"]),
            ("build_manifest_sha256", build_identity["candidate_manifest_sha256"]))):
        raise TimingPartitionError(f"{mode}/{workload} run is not bound to the qualified build")
    if summary.get("input_case_sha256") != evidence.get("input_case_sha256"):
        raise TimingPartitionError(f"{mode}/{workload} input digest differs from its wire record")
    output_digest = evidence.get("output_result_sha256")
    if not isinstance(output_digest, str) or not _SHA256.fullmatch(output_digest):
        raise TimingPartitionError(f"{mode}/{workload} omitted the output result digest")
    inventory = evidence.get("wire_inventory")
    if not isinstance(inventory, list) or len(inventory) < 2:
        raise TimingPartitionError(f"{mode}/{workload} omitted input/output wire inventory")
    if evidence.get("success_wires_removed") is not True:
        raise TimingPartitionError(f"{mode}/{workload} left successful wire files behind")
    peak = evidence.get("peak_scratch_bytes")
    if isinstance(peak, bool) or not isinstance(peak, int) or peak < 0 or peak >= MAX_ATTEMPT_BYTES:
        raise TimingPartitionError(f"{mode}/{workload} exceeded the hard attempt scratch cap")
    if evidence.get("validator_sha256") != build_identity["validator_sha256"]:
        raise TimingPartitionError(f"{mode}/{workload} used a different fixed validator")
    peak_method = evidence.get("peak_measurement")
    if not isinstance(peak_method, str) or \
            not peak_method.startswith("50ms sampled whole packet tree"):
        raise TimingPartitionError(f"{mode}/{workload} omitted the declared 50 ms scratch sampler evidence")
    measured_context = evidence.get("context")
    if not isinstance(measured_context, dict) or not _same_context(context, measured_context):
        raise TimingPartitionError(f"{mode}/{workload} observed a different outer execution context")
    contract = summary.get("contract_evidence")
    if not isinstance(contract, dict) or not all(
            field in contract for field in ("y", "dx", "dz", "dalpha", "dbias", "dgamma")):
        raise TimingPartitionError(f"{mode}/{workload} omitted output or one of the five VJP checks")
    phases = metrics.get("phase_ms")
    if not isinstance(phases, dict):
        raise TimingPartitionError(f"{mode}/{workload} omitted phase metrics")
    primary_ms = _finite_positive(phases.get(PRIMARY_METRIC), f"{mode} primary metric")
    raw = metrics.get("samples_ms")
    if not isinstance(raw, dict):
        raise TimingPartitionError(f"{mode}/{workload} omitted raw timing vectors")
    for phase_name in ("forward", "backward", "forward_vjp", "parameter_refresh"):
        values = raw.get(phase_name)
        if not isinstance(values, list) or len(values) != ITERATIONS or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(float(value)) or float(value) < 0.0 for value in values):
            raise TimingPartitionError(
                f"{mode}/{workload} raw {phase_name} samples must contain {ITERATIONS} finite values")
    return {
        "candidate_argument": mode,
        "requested_route": route,
        "selected_backend": expected_identity[0],
        "actual_implementation": expected_identity[1],
        "structural_layout": expected_identity[2],
        "binary_identity": {
            "source_closure_sha256": build_identity["source_closure_sha256"],
            "runner_binary_sha256": build_identity["candidate_runner_sha256"],
            "build_manifest_sha256": build_identity["candidate_manifest_sha256"],
            "library_sha256": build_identity["library_sha256"],
            "validator_sha256": build_identity["validator_sha256"],
        },
        "execution_context": {
            key: context[key] for key in
            ("scheduler_job_id", "scheduler_attempt_id", "device_uuid", "cgroup", "unit_name")
        },
        "primary_ms": primary_ms,
        "metrics": metrics,
        "validated_summary": summary,
        "input_case_sha256": evidence["input_case_sha256"],
        "output_result_sha256": output_digest,
        "wire_inventory": inventory,
        "helper_evidence_path": evidence.get("evidence_relative_path"),
        "helper_evidence_sha256": sha256_file(Path(evidence["evidence_path"])),
        "helper_stdout_path": evidence.get("stdout_relative_path"),
        "helper_stdout_sha256": evidence.get("stdout_sha256"),
        "helper_stderr_path": evidence.get("stderr_relative_path"),
        "helper_stderr_sha256": evidence.get("stderr_sha256"),
        "peak_scratch_bytes": peak,
        "peak_measurement": evidence.get("peak_measurement"),
        "success_wires_removed": True,
        "validator_sha256": evidence["validator_sha256"],
        "case_id": evidence.get("case_id"),
    }


def _report_base(*, source: Path, workload: str, phase: str,
                 expected_uuid: str, build_identity: Mapping[str, Any],
                 context: Mapping[str, Any], qualification: Mapping[str, Any],
                 facts: Mapping[str, Any], artifact_dir: Path,
                 packet_root: Path,
                 harness_before: Mapping[str, str]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "running",
        "workload": workload,
        "phase": phase,
        "phase_offset": 0 if phase == "exploration" else PAIRED_BLOCKS,
        "paired_blocks": PAIRED_BLOCKS,
        "arms_per_block": 3,
        "iterations_per_arm": ITERATIONS,
        "primary_metric": PRIMARY_METRIC,
        "timing_scope": (
            "uninstrumented caller wall time for 30 validated forward/VJP calls plus final stream sync, "
            "divided by 30; includes native guard/dispatch, tape, output and gradient reductions; "
            "excludes preparation, upload/conversion and optimizer work"
        ),
        "activation_storage": DTYPES["activation"],
        "parameter_storage": DTYPES["parameter"],
        "output_storage": DTYPES["output"],
        "gradient_mode": GRADIENT_MODE,
        "source_path": str(source),
        "build": dict(build_identity),
        "expected_device_uuid": expected_uuid,
        "outer_execution_context": {
            "status": "observed_inside_active_attempt",
            **dict(context),
            "scheduler_receipt_status": "pending_external_terminal_receipt_and_cleanup",
        },
        "qualification": dict(qualification),
        "harness_provenance": {
            "schema": HARNESS_PROVENANCE_SCHEMA,
            "source_closure_sha256": build_identity["source_closure_sha256"],
            "validator_sha256": build_identity["validator_sha256"],
            "before": dict(harness_before),
            "after": None,
            "unchanged": None,
        },
        "direct_control_facts": dict(facts),
        "artifact_root": artifact_dir.relative_to(packet_root).as_posix(),
        "attempt_scratch_limit_bytes": MAX_ATTEMPT_BYTES,
        "projected_complete_attempt_bytes": 0,
        "attempt_peak_bound_bytes": 0,
        "scratch_measurement": {
            "method": "shared validator samples the managed output tree every 50 ms and adds post-write evidence allowance",
            "authority": "Autobench attempt monitor enforces the hard 128 MiB cap",
            "measurement_overhead": "external host sampler runs equally for every arm and does not add timing events",
            "timing_note": "50 ms sampling can miss shorter transient peaks; the sampler can add shared CPU scheduling noise, and all per-arm raw timing vectors are retained",
        },
        "max_observed_peak_scratch_bytes": 0,
        "blocks": [],
    }


def _tree_bytes(root: Path) -> int:
    total = 0
    for current, dirs, files in os.walk(root, followlinks=False):
        base = Path(current)
        if any((base / name).is_symlink() for name in dirs):
            raise TimingPartitionError("attempt output contains a symlink directory")
        for name in files:
            path = base / name
            if path.is_symlink() or not path.is_file():
                raise TimingPartitionError("attempt output contains a nonregular file")
            total += path.stat().st_size
    return total


def _report_projection_fits(packet_root: Path, report: Mapping[str, Any]) -> bool:
    encoded = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    return _tree_bytes(packet_root) + len(encoded) < MAX_ATTEMPT_BYTES


def _qualification_evidence_file(qualification: Mapping[str, Any], value: Any,
                                 label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise TimingPartitionError(f"qualification {label} path is invalid")
    bundle_path = _regular_file(qualification["path"], "qualification bundle")
    if qualification.get("evidence_root") == ".":
        relative = PurePosixPath(value)
        if relative.is_absolute() or any(part in {".", ".."} for part in relative.parts):
            raise TimingPartitionError(f"portable qualification {label} path escapes its bundle")
        path = bundle_path.parent.joinpath(*relative.parts)
        _reject_symlink_components(path)
        return _regular_file(path, label)
    path = Path(value)
    if not path.is_absolute():
        raise TimingPartitionError(f"legacy qualification {label} path must be absolute")
    return _regular_file(path, label)


def _harness_component_paths(bundle_path: Path) -> dict[str, Path]:
    bounded_dir = Path(__file__).resolve().parent
    paths = {
        "timing_collector_sha256": Path(__file__).resolve(strict=True),
        "bounded_validation_sha256": bounded_dir / "bounded_validation.py",
        "recorded_qualification_sha256": bounded_dir / "recorded_qualification.py",
        "recorded_receipts_sha256": bounded_dir / "recorded_receipts.py",
    }
    path = _regular_file(bundle_path, "qualification bundle")
    raw_bundle = _read_json(path, "qualification bundle")
    wrapper_path = raw_bundle.get("sanitizer_wrapper_path")
    if not isinstance(wrapper_path, str):
        raise TimingPartitionError("qualification bundle omitted the fixed sanitizer wrapper identity")
    paths["sanitizer_wrapper_sha256"] = _qualification_evidence_file(
        {"path": str(path), "evidence_root": raw_bundle.get("evidence_root")},
        wrapper_path, "sanitizer wrapper",
    )
    return {key: _regular_file(value, key) for key, value in paths.items()}


def _harness_snapshot(paths: Mapping[str, Path]) -> dict[str, str]:
    if set(paths) != HARNESS_COMPONENT_KEYS:
        raise TimingPartitionError("timing harness identity map has missing or unknown components")
    return {key: sha256_file(_regular_file(path, key)) for key, path in paths.items()}


def _verify_qualification_snapshot(qualification: Mapping[str, Any]) -> None:
    bundle_path = _regular_file(qualification["path"], "qualification bundle")
    if sha256_file(bundle_path) != qualification["sha256"]:
        raise TimingPartitionError("qualification bundle changed during timing")
    for item in qualification["reports"]:
        if not isinstance(item, Mapping) or not isinstance(item.get("kind"), str):
            raise TimingPartitionError("qualification bundle contains a malformed report reference")
        for path_key, hash_key, label in (
                ("report_path", "report_sha256", "qualification report"),
                ("receipt_path", "receipt_sha256", "Scheduler receipt"),
                ("pointer_path", "pointer_sha256", "Autobench pointer")):
            path = _qualification_evidence_file(qualification, item.get(path_key), label)
            if sha256_file(path) != item[hash_key]:
                raise TimingPartitionError(f"{item['kind']} {label} changed during timing")
    wrapper_path = qualification.get("sanitizer_wrapper_path")
    wrapper_hash = qualification.get("sanitizer_wrapper_sha256")
    if wrapper_path is not None or wrapper_hash is not None:
        path = _qualification_evidence_file(qualification, wrapper_path, "sanitizer wrapper")
        if sha256_file(path) != wrapper_hash:
            raise TimingPartitionError("qualification sanitizer wrapper changed during timing")


def collect_partition(
    *,
    source: Path,
    runner: Path,
    build_manifest: Path,
    workload: str,
    phase: str,
    expected_uuid: str,
    qualification_bundle: Path,
    artifact_dir: Path,
    output: Path,
    verify_build: Callable[[Path, Path, Path], Mapping[str, Any]],
    context_reader: Callable[[Path, str], Mapping[str, Any]],
    run_validation: Callable[..., Mapping[str, Any]],
    bundle_validator: Callable[[Path, Path, Path, Path, str], Mapping[str, Any]],
    facts: Mapping[str, Any],
    performance_ids: Sequence[str],
    expected_mode_identity: Callable[[str, Mapping[str, Any]], tuple[str, str, str]],
    harness_paths: Mapping[str, Path] | None = None,
    harness_before: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Execute and durably report one seven-block timing partition.

    The passed ``run_validation`` callback must be the separately owned
    ``bounded_validation.run_validation`` implementation. This function only
    orchestrates route order and checks its returned source/numeric evidence.
    """
    source = source.resolve(strict=True)
    if not source.is_dir():
        raise TimingPartitionError("--source must name the immutable native source tree")
    if phase not in {"exploration", "confirmation"}:
        raise TimingPartitionError("phase must be exploration or confirmation")
    if not _UUID.fullmatch(expected_uuid):
        raise TimingPartitionError("expected UUID must be one explicit NVIDIA GPU UUID")
    if workload not in performance_ids:
        raise TimingPartitionError("workload must be one of the fixed performance workload IDs")
    facts = dict(facts)
    if not isinstance(facts, dict) or facts.get("workload") != workload:
        raise TimingPartitionError("fixed direct-control facts do not match the requested workload")

    harness_paths = dict(harness_paths or _harness_component_paths(qualification_bundle))
    first_snapshot = _harness_snapshot(harness_paths)
    if harness_before is None:
        harness_before = first_snapshot
    else:
        harness_before = dict(harness_before)
        if set(harness_before) != HARNESS_COMPONENT_KEYS or first_snapshot != harness_before:
            raise TimingPartitionError("timing harness changed after its initial preflight snapshot")
    built = dict(verify_build(source, runner, build_manifest))
    identity = _identity_from_build(built)
    context_before = dict(context_reader(source, expected_uuid))
    _validate_context(context_before, expected_uuid, "initial")
    qualification = validate_qualification_bundle(
        qualification_bundle,
        source=source,
        build_identity=identity,
        expected_uuid=expected_uuid,
        bundle_validator=bundle_validator,
    )
    if qualification.get("sanitizer_wrapper_sha256") != harness_before["sanitizer_wrapper_sha256"]:
        raise TimingPartitionError("sanitizer wrapper changed between initial hashing and qualification validation")

    source_validator = _regular_file(
        source / "bench" / "factorized_interactions" / "validate_runner.py",
        "fixed validator",
    )
    if sha256_file(source_validator) != EXPECTED_VALIDATOR_SHA256:
        raise TimingPartitionError("immutable source validator bytes differ from the trusted fixed validator")
    artifact_dir = artifact_dir.absolute()
    output = output.absolute()
    _reject_symlink_components(artifact_dir)
    _reject_symlink_components(output)
    packet_root = output.parent.resolve()
    if not artifact_dir.is_relative_to(packet_root) or not output.is_relative_to(packet_root):
        raise TimingPartitionError("artifact and output paths must stay under this Autobench attempt output")
    if output.exists() or output.is_symlink():
        raise TimingPartitionError("refusing to overwrite an existing timing report")
    if artifact_dir.exists() or artifact_dir.is_symlink():
        raise TimingPartitionError("refusing to reuse existing per-partition artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=False)
    cache_dir = artifact_dir / "reference-cache"

    report = _report_base(
        source=source, workload=workload, phase=phase,
        expected_uuid=expected_uuid, build_identity=identity,
        context=context_before, qualification=qualification,
        facts=facts, artifact_dir=artifact_dir, packet_root=packet_root,
        harness_before=harness_before,
    )
    direct_mode = facts.get("strongest_direct_mode")
    if direct_mode not in {"minimal_direct", "direct_unique_output"}:
        raise TimingPartitionError("fixed direct-control facts selected an unsupported direct implementation")
    modes = {
        "strongest_direct": direct_mode,
        "packet": "packet",
        "native_direct": "native_direct",
    }
    expected = {arm: expected_mode_identity(mode, facts) for arm, mode in modes.items()}
    partition_input_digest: str | None = None
    try:
        for round_index in range(PAIRED_BLOCKS):
            paired_order, descriptive_position = pair_order(phase, round_index)
            execution_order = list(paired_order)
            execution_order.insert(descriptive_position, "native_direct")
            block = {
                "round_index": round_index,
                "global_round_index": (0 if phase == "exploration" else PAIRED_BLOCKS) + round_index,
                "paired_order": paired_order,
                "execution_order": execution_order,
                "descriptive_position": descriptive_position,
                "arms": {},
            }
            report["blocks"].append(block)
            for position, arm in enumerate(execution_order):
                context_now = dict(context_reader(source, expected_uuid))
                if not _same_context(context_before, context_now):
                    raise TimingPartitionError("outer Scheduler context changed during the partition")
                mode = modes[arm]
                route = "packet" if mode == "packet" else "direct"
                case_id = f"{phase}-block-{round_index:02d}-position-{position}-{arm}"
                evidence = run_validation(
                    source=source,
                    build=built,
                    expected_uuid=expected_uuid,
                    context=context_before,
                    artifact_root=artifact_dir,
                    packet_root=packet_root,
                    cache_dir=cache_dir,
                    case_id=case_id,
                    workload=workload,
                    mode=mode,
                    activation=DTYPES["activation"],
                    parameter=DTYPES["parameter"],
                    output_dtype=DTYPES["output"],
                    iterations=ITERATIONS,
                    repeat_vjp=False,
                    expected_identity=expected[arm],
                    facts=facts,
                )
                observation = _validate_measurement(
                    evidence,
                    workload=workload,
                    mode=mode,
                    route=route,
                    expected_identity=expected[arm],
                    expected_uuid=expected_uuid,
                    context=context_before,
                    build_identity=identity,
                )
                if partition_input_digest is None:
                    partition_input_digest = observation["input_case_sha256"]
                    report["input_case_sha256"] = partition_input_digest
                elif observation["input_case_sha256"] != partition_input_digest:
                    raise TimingPartitionError("fixed workload input bytes changed within the timing partition")
                if block["arms"]:
                    prior_input = next(iter(block["arms"].values()))["input_case_sha256"]
                    if observation["input_case_sha256"] != prior_input:
                        raise TimingPartitionError("the three paired arms received different fixed input bytes")
                block["arms"][arm] = observation
                report["max_observed_peak_scratch_bytes"] = max(
                    report["max_observed_peak_scratch_bytes"],
                    observation["peak_scratch_bytes"],
                )
            if list(block["arms"]) != execution_order:
                raise TimingPartitionError("paired block arm order differs from executed order")
            if set(block["arms"]) != {"strongest_direct", "packet", "native_direct"}:
                raise TimingPartitionError("paired block omitted a required route")

        final_context = dict(context_reader(source, expected_uuid))
        _validate_context(final_context, expected_uuid, "final")
        if not _same_context(context_before, final_context):
            raise TimingPartitionError("outer Scheduler context changed before partition completion")
        final_build = dict(verify_build(source, runner, build_manifest))
        final_identity = _identity_from_build(final_build)
        if final_identity != identity:
            raise TimingPartitionError("native source or binary identity changed during the partition")
        _verify_qualification_snapshot(qualification)
        harness_after = _harness_snapshot(harness_paths)
        report["harness_provenance"]["after"] = harness_after
        report["harness_provenance"]["unchanged"] = harness_after == harness_before
        if harness_after != harness_before:
            raise TimingPartitionError("timing harness or qualification sidecar changed during execution")
        report["status"] = "observations_complete_pending_independent_review"
        report["completed_blocks"] = PAIRED_BLOCKS
        report["completed_observations"] = PAIRED_BLOCKS * 3
        report["final_outer_execution_context"] = dict(final_context)
        report["build_rechecked"] = True
        for _ in range(8):
            projected = _tree_bytes(packet_root) + len(
                (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
            )
            bounded_peak = max(report["max_observed_peak_scratch_bytes"], projected)
            if projected == report["projected_complete_attempt_bytes"] and \
                    bounded_peak == report["attempt_peak_bound_bytes"]:
                break
            report["projected_complete_attempt_bytes"] = projected
            report["attempt_peak_bound_bytes"] = bounded_peak
        bounded_peak = report["attempt_peak_bound_bytes"]
        final_projection = _tree_bytes(packet_root) + len(
            (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
        )
        if final_projection != report["projected_complete_attempt_bytes"] or \
                max(report["max_observed_peak_scratch_bytes"], final_projection) != bounded_peak:
            raise TimingPartitionError("complete timing report byte projection did not converge")
        if bounded_peak >= MAX_ATTEMPT_BYTES:
            raise TimingPartitionError("complete timing report would exceed the attempt scratch limit")
        return report
    except Exception as exc:
        try:
            harness_after = _harness_snapshot(harness_paths)
            report["harness_provenance"]["after"] = harness_after
            report["harness_provenance"]["unchanged"] = harness_after == harness_before
            if harness_after != harness_before:
                exc = TimingPartitionError(
                    f"{exc}; timing harness or qualification sidecar changed during execution")
        except Exception as harness_exc:
            report["harness_provenance"]["after_error"] = (
                f"{type(harness_exc).__name__}: {harness_exc}")
        report["status"] = "failed_incomplete"
        report["completed_blocks"] = sum(
            1 for block in report["blocks"] if len(block.get("arms", {})) == 3
        )
        report["completed_observations"] = sum(
            len(block.get("arms", {})) for block in report["blocks"]
        )
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise PartitionRunFailure(str(exc), report) from exc


def _load_helper_modules() -> tuple[Any, Any, Any]:
    helper_dir = Path(__file__).resolve().parent
    for path in (str(helper_dir),):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        import bounded_validation  # type: ignore[import-not-found]
        import recorded_qualification  # type: ignore[import-not-found]
        import recorded_receipts  # type: ignore[import-not-found]
    except ImportError as exc:
        raise TimingPartitionError(f"bounded validation helpers are missing from the harness snapshot: {exc}") from exc
    return bounded_validation, recorded_qualification, recorded_receipts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="immutable native source tree from the shared {build}")
    parser.add_argument("--runner", type=Path, required=True,
                        help="candidate runner from the shared immutable native build")
    parser.add_argument("--build-manifest", type=Path, required=True,
                        help="candidate runner build manifest from the shared immutable build")
    parser.add_argument("--workload", required=True)
    parser.add_argument("--phase", choices=("exploration", "confirmation"), required=True)
    parser.add_argument("--expected-uuid", required=True)
    parser.add_argument("--qualification-bundle", type=Path, required=True,
                        help="completed source/build/device-bound dtype+auxiliary+sanitizer qualification bundle")
    parser.add_argument("--artifact-dir", type=Path, required=True,
                        help="new per-partition monitored artifacts directory")
    parser.add_argument("--output", type=Path, required=True,
                        help="new timing partition JSON under this attempt's output directory")
    args = parser.parse_args(argv)
    try:
        harness_paths = _harness_component_paths(args.qualification_bundle)
        harness_before = _harness_snapshot(harness_paths)
        helpers, qualification, receipts = _load_helper_modules()
        expected_modules = (
            (helpers, harness_paths["bounded_validation_sha256"]),
            (qualification, harness_paths["recorded_qualification_sha256"]),
            (receipts, harness_paths["recorded_receipts_sha256"]),
        )
        if any(Path(module.__file__).resolve(strict=True) != path
               for module, path in expected_modules):
            raise TimingPartitionError("helper modules were not imported from the captured harness snapshot")
        workloads = helpers._source_module(
            args.source, "tests/factorized_interactions", "workloads",
        )
        performance_ids = list(workloads.performance_workload_ids())
        if len(performance_ids) != EXPECTED_PERFORMANCE_WORKLOADS or \
                len(set(performance_ids)) != EXPECTED_PERFORMANCE_WORKLOADS:
            raise TimingPartitionError("immutable performance workload list must contain the exact 13 unique cases")
        facts = helpers.workload_facts(
            args.source, args.workload, DTYPES["activation"], DTYPES["parameter"],
        )
        report = collect_partition(
            source=args.source,
            runner=args.runner,
            build_manifest=args.build_manifest,
            workload=args.workload,
            phase=args.phase,
            expected_uuid=args.expected_uuid,
            qualification_bundle=args.qualification_bundle,
            artifact_dir=args.artifact_dir,
            output=args.output,
            verify_build=helpers.verify_build,
            context_reader=helpers.read_context,
            run_validation=helpers.run_validation,
            bundle_validator=qualification.validate_existing_bundle,
            facts=facts,
            performance_ids=performance_ids,
            expected_mode_identity=helpers.expected_mode_identity,
            harness_paths=harness_paths,
            harness_before=harness_before,
        )
        _write_json_exclusive(args.output, report)
        print(json.dumps({
            "schema": SCHEMA,
            "status": report["status"],
            "workload": report["workload"],
            "phase": report["phase"],
            "observations": report["completed_observations"],
            "peak_scratch_bytes": report["max_observed_peak_scratch_bytes"],
        }, sort_keys=True))
        return 0
    except PartitionRunFailure as exc:
        if not args.output.exists() and _report_projection_fits(args.output.parent, exc.partial_report):
            try:
                _write_json_exclusive(args.output, exc.partial_report)
            except Exception as write_exc:
                print(f"failed to persist incomplete timing report: {write_exc}", file=sys.stderr)
        print(f"timing partition failed; completed observations were retained: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"timing partition rejected: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
