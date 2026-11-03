#!/usr/bin/env python3
"""Assemble source-bound CPU input from finished CUDA timing observations.

This tool reads fixed native reports and finished Autobench/Scheduler receipts.
It never launches CUDA work and never derives new timing samples. Its output
retains the original reports, receipts, summaries, and raw timing vectors by
path and SHA-256 for later CPU-only recorded-observation analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any, Mapping

sys.dont_write_bytecode = True
from recorded_qualification import validate_existing_bundle  # noqa: E402
from recorded_receipts import validate_receipt_and_pointer  # noqa: E402

SCHEMA = "cellerator.factorized_interactions.recorded_dataset.v1"
TIMING_SCHEMA = "cellerator.factorized_interactions.timing_partition.v1"
QUALIFICATION_SCHEMA = "cellerator.factorized_interactions.recorded_qualification.v1"
HARNESS_SCHEMA = "cellerator.factorized_interactions.timing_harness_provenance.v1"
PRIMARY_METRIC = "warm_forward_vjp_batch_host_wall_mean"
PHASES = ("exploration", "confirmation")
ARMS = ("strongest_direct", "packet", "native_direct")
PAIR_ORDER = ("strongest_direct", "packet")
BLOCKS = 7
CALLS = 30
_HEX = re.compile(r"^[0-9a-f]{64}$")
_UUID = re.compile(r"^GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$")
_CONTEXT_FIELDS = ("scheduler_job_id", "scheduler_attempt_id", "device_uuid", "cgroup", "unit_name")
_HARNESS_SIDECARS = {
    "timing_collector_sha256": "run_timing_partition.py",
    "bounded_validation_sha256": "bounded_validation.py",
    "recorded_qualification_sha256": "recorded_qualification.py",
    "recorded_receipts_sha256": "recorded_receipts.py",
}
# These pins are refreshed only when the measurement owner freezes the collector
# and its validation sidecars. Reports cannot authenticate themselves by merely
# repeating an arbitrary hash map.
APPROVED_TIMING_HARNESS_SHA256: dict[str, str] = {
    "timing_collector_sha256": "357d68c432dd23981f9c56c770e8e4c0cda7eea73be9413641ad651a4ccfc214",
    "bounded_validation_sha256": "6da91757d8d18501ad8df4c6722c549ae32c9adfb93e78417a016b6849caaf11",
    "recorded_qualification_sha256": "505a83220828e80e22e84e02588918a9df794347fb3b1bee8c3f8190c8f96ea3",
    "recorded_receipts_sha256": "4ea24aedf780caffb5a138d8bfe01213e2ba10e914c6475b4b54b151a60cd1a6",
    "sanitizer_wrapper_sha256": "e5bafd545ecaaae4a093f533da9e96276f6795e6f91691de25180dcf0ad04de7",
}


class DatasetError(ValueError):
    """An observation or its provenance is missing, inconsistent, or changed."""


def _sha_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise DatasetError(f"expected a regular non-symlink file: {path}")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_sha(value: Any) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DatasetError(f"value is not finite canonical JSON: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise DatasetError(f"expected a regular evidence file: {path}")
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise DatasetError(f"duplicate JSON field {key!r} in {path}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise DatasetError(f"non-finite JSON number {value} in {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"),
                          object_pairs_hook=reject_duplicates, parse_constant=reject_constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"cannot read JSON evidence {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise DatasetError(f"evidence root must be an object: {path}")
    return data


def _safe_child(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative:
        raise DatasetError(f"{label} must be a relative POSIX path")
    rel = PurePosixPath(relative)
    if rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts):
        raise DatasetError(f"{label} escapes its retained packet")
    base = root.resolve(strict=True)
    path = base
    for part in rel.parts:
        path = path / part
        if path.is_symlink():
            raise DatasetError(f"{label} includes a symbolic link")
    path = path.resolve(strict=True)
    if not path.is_relative_to(base) or not path.is_file():
        raise DatasetError(f"{label} is missing or outside its packet")
    return path


def _bundle_evidence_path(bundle: Mapping[str, Any], bundle_path: Path,
                          key: str, label: str) -> Path:
    """Resolve a portable bundle member without allowing it to escape its root."""
    value = bundle.get(key)
    if not isinstance(value, str) or not value:
        raise DatasetError(f"qualification bundle omitted {label} path")
    evidence_root = bundle.get("evidence_root")
    if evidence_root == ".":
        return _safe_child(Path(bundle_path).parent, value, label)
    if evidence_root is not None:
        raise DatasetError("qualification evidence_root must be the bundle directory")
    path = Path(value)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise DatasetError(f"legacy qualification {label} path must be an absolute regular file")
    return path.resolve(strict=True)


def _local_harness_hashes(sidecar_root: Path, sanitizer_wrapper: Path) -> dict[str, str]:
    hashes = {}
    for key, filename in _HARNESS_SIDECARS.items():
        path = sidecar_root / filename
        hashes[key] = _sha_file(path)
    hashes["sanitizer_wrapper_sha256"] = _sha_file(sanitizer_wrapper)
    return hashes


def _check_local_harness(sidecar_root: Path, sanitizer_wrapper: Path,
                         approved_hashes: Mapping[str, str]) -> dict[str, str]:
    actual = _local_harness_hashes(sidecar_root, sanitizer_wrapper)
    if actual != dict(approved_hashes):
        raise DatasetError("staged sidecar files differ from the fixed approved timing harness")
    return actual


def _validate_harness_provenance(report: Mapping[str, Any],
                                 approved_hashes: Mapping[str, str],
                                 identity: Mapping[str, str]) -> dict[str, str]:
    provenance = report.get("harness_provenance")
    if not isinstance(provenance, dict) or provenance.get("schema") != HARNESS_SCHEMA or \
            provenance.get("unchanged") is not True:
        raise DatasetError("timing report omitted unchanged sidecar provenance")
    before, after = provenance.get("before"), provenance.get("after")
    if not isinstance(before, dict) or not isinstance(after, dict) or \
            before != dict(approved_hashes) or after != dict(approved_hashes):
        raise DatasetError("timing report sidecar hashes differ from the fixed accepted harness")
    expected_keys = set(_HARNESS_SIDECARS) | {"sanitizer_wrapper_sha256"}
    if set(before) != expected_keys or any(not _HEX.fullmatch(str(value)) for value in before.values()):
        raise DatasetError("timing report sidecar provenance has an unknown or malformed hash")
    if provenance.get("source_closure_sha256") != identity.get("source_closure_sha256") or \
            provenance.get("validator_sha256") != identity.get("validator_sha256"):
        raise DatasetError("timing harness provenance is not tied to the qualified source and validator")
    return dict(before)


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or \
            not math.isfinite(float(value)) or float(value) < 0:
        raise DatasetError(f"{label} must be finite and nonnegative")
    return float(value)


def _context_matches(actual: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    if actual.get("status") != "observed":
        raise DatasetError(f"{label} does not contain an observed execution context")
    for key in _CONTEXT_FIELDS:
        if not isinstance(actual.get(key), str) or not actual.get(key) or \
                actual.get(key) != expected.get(key):
            raise DatasetError(f"{label} differs in {key}")


def _contract(summary: Mapping[str, Any], workload: str, mode: str,
              context: Mapping[str, Any], identity: Mapping[str, str],
              expected_dispatch: tuple[str, str, str]) -> None:
    if summary.get("schema") != "cellerator.factorized_interactions.validation.v1" or \
            summary.get("validated") is not True or summary.get("timed") is not True:
        raise DatasetError("timed validator did not pass")
    if any(summary.get(k) != v for k, v in (
            ("workload", workload), ("activation_storage", "fp16"),
            ("parameter_storage", "fp16"), ("output_storage", "fp16"),
            ("gradient_mode", "overwrite"),
            ("scheduler_job_id", context["scheduler_job_id"]))):
        raise DatasetError("timed validator summary metadata mismatch")
    evidence = summary.get("contract_evidence")
    required = ("y", "dx", "dz", "dalpha", "dbias", "dgamma")
    if not isinstance(evidence, dict) or any(key not in evidence for key in required):
        raise DatasetError("timed validator omitted output or one of its five VJP checks")
    for key in required:
        record = evidence[key]
        ratio = record.get("max_error_to_allowance") if isinstance(record, dict) else None
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or \
                not math.isfinite(float(ratio)) or not 0 <= float(ratio) <= 1:
            raise DatasetError(f"fixed numerical contract failed for {key}")
    native = summary.get("native_execution")
    if not isinstance(native, dict) or any(native.get(k) != v for k, v in (
            ("candidate", mode), ("device_uuid", context["device_uuid"]),
            ("scheduler_job_id", context["scheduler_job_id"]),
            ("requested_route", "packet" if mode == "packet" else "direct"),
            ("selected_backend", expected_dispatch[0]),
            ("actual_implementation", expected_dispatch[1]),
            ("structural_layout", expected_dispatch[2]))):
        raise DatasetError("native execution metadata differs from requested timed mode")
    provenance = summary.get("build_provenance")
    if not isinstance(provenance, dict) or any(provenance.get(k) != v for k, v in (
            ("source_closure_sha256", identity["source_closure_sha256"]),
            ("runner_binary_sha256", identity["candidate_runner_sha256"]),
            ("build_manifest_sha256", identity["candidate_manifest_sha256"]))):
        raise DatasetError("timed validator summary is not bound to the qualified build")


def _validated_observation(report: Mapping[str, Any], report_path: Path,
                           block: Mapping[str, Any], arm_name: str,
                           arm: Mapping[str, Any], workload: str,
                           identity: Mapping[str, str], expected_uuid: str,
                           input_sha: str) -> dict[str, Any]:
    mode = {"strongest_direct": report["direct_control_facts"]["strongest_direct_mode"],
            "packet": "packet", "native_direct": "native_direct"}[arm_name]
    requested_route = "packet" if mode == "packet" else "direct"
    import bounded_validation
    try:
        expected_dispatch = bounded_validation.expected_mode_identity(mode,
                                                                      report["direct_control_facts"])
    except Exception as exc:
        raise DatasetError("timing report selected a mode outside the fixed native dispatch contract") from exc
    if expected_dispatch is None:
        raise DatasetError("timing arm requests an unsupported native mode")
    if arm.get("candidate_argument") != mode or arm.get("requested_route") != requested_route or \
            arm.get("input_case_sha256") != input_sha or arm.get("success_wires_removed") is not True:
        raise DatasetError("timing arm identity or input digest differs from its paired block")
    if arm.get("validator_sha256") != identity["validator_sha256"]:
        raise DatasetError("timing arm used a different fixed validator")
    summary = arm.get("validated_summary")
    if not isinstance(summary, dict):
        raise DatasetError("timing arm omits its fixed-validator summary")
    context = arm.get("execution_context")
    if not isinstance(context, dict):
        raise DatasetError("timing arm omits its observed execution context")
    _context_matches(context, {"status": "observed", "device_uuid": expected_uuid,
                               **{k: report["outer_execution_context"].get(k) for k in _CONTEXT_FIELDS}},
                     "timing arm execution context")
    _contract(summary, workload, mode, context, identity, expected_dispatch)
    native = summary["native_execution"]
    if tuple(native.get(k) for k in ("selected_backend", "actual_implementation", "structural_layout")) != expected_dispatch:
        raise DatasetError("timed arm selected an unexpected native route")
    captured_binary = arm.get("binary_identity")
    captured_to_qualified = {
        "source_closure_sha256": "source_closure_sha256",
        "runner_binary_sha256": "candidate_runner_sha256",
        "build_manifest_sha256": "candidate_manifest_sha256",
        "library_sha256": "library_sha256",
        "validator_sha256": "validator_sha256",
    }
    if not isinstance(captured_binary, dict):
        raise DatasetError("timed arm omitted its recorded binary identity")
    for captured_key, identity_key in captured_to_qualified.items():
        if captured_binary.get(captured_key) != identity[identity_key]:
            raise DatasetError(f"timed arm binary identity mismatch for {captured_key}")
    if arm.get("metrics") != summary.get("metrics"):
        raise DatasetError("raw metrics and fixed-validator summary disagree")
    metrics = arm.get("metrics")
    if not isinstance(metrics, dict) or metrics.get("schema") != "cellerator.factorized_interactions.metrics.v1" or \
            metrics.get("device_uuid") != expected_uuid or \
            metrics.get("scheduler_job_id") != context["scheduler_job_id"] or \
            metrics.get("selected_backend") != expected_dispatch[0] or metrics.get("iterations") != CALLS:
        raise DatasetError("raw metric identity differs from its original timed run")
    phases = metrics.get("phase_ms")
    samples = metrics.get("samples_ms")
    if not isinstance(phases, dict) or not isinstance(samples, dict):
        raise DatasetError("timed arm omitted phase means or raw sample vectors")
    primary = _finite_nonnegative(phases.get(PRIMARY_METRIC), "primary caller-wall mean")
    total = _finite_nonnegative(phases.get("warm_forward_vjp_batch_host_wall"),
                                "primary caller-wall batch total")
    if primary <= 0 or total <= 0 or arm.get("primary_ms") != primary or \
            not math.isclose(primary * CALLS, total, rel_tol=2e-8, abs_tol=1e-9):
        raise DatasetError("primary caller-wall mean does not match the recorded 30-call batch total")
    for phase_name in ("forward", "backward", "forward_vjp", "parameter_refresh"):
        values = samples.get(phase_name)
        if not isinstance(values, list) or len(values) != CALLS:
            raise DatasetError(f"raw metric vector {phase_name} does not contain 30 values")
        for value in values:
            _finite_nonnegative(value, f"raw {phase_name} sample")
    if not _HEX.fullmatch(str(arm.get("output_result_sha256", ""))):
        raise DatasetError("timed arm lacks its output wire digest")

    packet_root = report_path.parent
    helper_path = _safe_child(packet_root, arm.get("helper_evidence_path"), "timed helper evidence")
    if _sha_file(helper_path) != arm.get("helper_evidence_sha256"):
        raise DatasetError("timed helper evidence changed after collection")
    helper = _read(helper_path)
    if helper.get("schema") != "cellerator.factorized_interactions.bounded_case.v1" or \
            helper.get("status") != "pass" or helper.get("success_wires_removed") is not True or \
            helper.get("cleanup_pending") is not False or helper.get("summary") != summary:
        raise DatasetError("durable timed helper record does not attest the successful validator summary")
    if helper.get("workload") != workload or helper.get("mode") != mode or \
            helper.get("input_case_sha256") != input_sha or \
            helper.get("output_result_sha256") != arm["output_result_sha256"]:
        raise DatasetError("timed helper record differs from arm identity or wire digests")
    if summary.get("input_case_sha256") != input_sha:
        raise DatasetError("timed summary input digest differs from its retained wire record")
    helper_context = helper.get("context")
    if not isinstance(helper_context, dict):
        raise DatasetError("timed helper record omitted original execution context")
    _context_matches(helper_context, context, "timed helper context")
    if helper.get("validator_sha256") != identity["validator_sha256"]:
        raise DatasetError("timed helper record used a different validator")
    helper_build = helper.get("build")
    if not isinstance(helper_build, dict) or any(helper_build.get(k) != v for k, v in (
            ("source_closure_sha256", identity["source_closure_sha256"]),
            ("runner_sha256", identity["candidate_runner_sha256"]),
            ("build_manifest_sha256", identity["candidate_manifest_sha256"]),
            ("library_sha256", identity["library_sha256"]),
            ("validator_sha256", identity["validator_sha256"]))):
        raise DatasetError("timed helper evidence is not bound to the qualified build")
    if helper.get("stdout_relative_path") != arm.get("helper_stdout_path") or \
            helper.get("stdout_sha256") != arm.get("helper_stdout_sha256") or \
            helper.get("stderr_relative_path") != arm.get("helper_stderr_path") or \
            helper.get("stderr_sha256") != arm.get("helper_stderr_sha256"):
        raise DatasetError("timing report stdout/stderr references differ from retained helper evidence")
    stdout = _safe_child(packet_root, helper.get("stdout_relative_path", arm.get("helper_stdout_path")),
                         "timed validator stdout")
    stderr = _safe_child(packet_root, helper.get("stderr_relative_path", arm.get("helper_stderr_path")),
                         "timed validator stderr")
    stdout_sha = _sha_file(stdout)
    stderr_sha = _sha_file(stderr)
    if stdout_sha != helper.get("stdout_sha256") or stdout_sha != arm.get("helper_stdout_sha256") or \
            stderr_sha != helper.get("stderr_sha256") or stderr_sha != arm.get("helper_stderr_sha256"):
        raise DatasetError("timed validator stdout/stderr bytes differ from recorded digests")
    if _read(stdout) != summary:
        raise DatasetError("timed validator stdout differs from the retained summary")
    validated = _safe_child(packet_root, helper.get("validated_relative_path"), "pre-cleanup validation record")
    if _sha_file(validated) != helper.get("validated_record_sha256"):
        raise DatasetError("pre-cleanup validation record changed")
    before = _read(validated)
    if before.get("cleanup_pending") is not True or before.get("success_wires_removed") is not False:
        raise DatasetError("pre-cleanup validation record does not prove wires were present before cleanup")
    for key in ("summary", "wire_inventory", "input_case_sha256", "output_result_sha256",
                "context", "build", "validator_sha256", "case_id"):
        if before.get(key) != helper.get(key):
            raise DatasetError(f"pre-cleanup record differs in {key}")
    inventory = helper.get("wire_inventory")
    if not isinstance(inventory, list) or len(inventory) < 2:
        raise DatasetError("timed helper omitted its pre-cleanup wire inventory")
    paths = set()
    result = []
    inputs = []
    for entry in inventory:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or \
                entry["path"] in paths or isinstance(entry.get("bytes"), bool) or \
                not isinstance(entry.get("bytes"), int) or entry["bytes"] < 0 or \
                not _HEX.fullmatch(str(entry.get("sha256", ""))):
            raise DatasetError("pre-cleanup wire inventory is malformed")
        paths.add(entry["path"])
        if entry["path"].endswith(".result"):
            result.append(entry)
        if entry["path"].endswith(".case") or ".case.part" in entry["path"]:
            inputs.append(entry)
    if len(result) != 1 or result[0]["sha256"] != arm["output_result_sha256"] or not inputs:
        raise DatasetError("pre-cleanup input/output digest inventory is incomplete")
    peak = arm.get("peak_scratch_bytes")
    if isinstance(peak, bool) or not isinstance(peak, int) or not 0 <= peak < 128 * 1024 * 1024 or \
            not isinstance(arm.get("peak_measurement"), str) or \
            not arm["peak_measurement"].startswith("50ms sampled whole packet tree"):
        raise DatasetError("timed arm omitted valid measured scratch evidence")

    return {
        "primary_ms": primary,
        "metrics": metrics,
        "raw_metrics_sha256": _canonical_sha(metrics),
        "validated_summary": summary,
        "validation_summary_sha256": _canonical_sha(summary),
        "input_case_sha256": input_sha,
        "output_result_sha256": arm["output_result_sha256"],
        "candidate_argument": mode,
        "requested_route": requested_route,
        "selected_backend": expected_dispatch[0],
        "actual_implementation": expected_dispatch[1],
        "structural_layout": expected_dispatch[2],
        "binary_identity": {
            "source_closure_sha256": identity["source_closure_sha256"],
            "candidate_runner_sha256": identity["candidate_runner_sha256"],
            "candidate_manifest_sha256": identity["candidate_manifest_sha256"],
            "library_sha256": identity["library_sha256"],
            "validator_sha256": identity["validator_sha256"],
        },
        "scheduler_job_id": context["scheduler_job_id"],
        "scheduler_attempt_id": context["scheduler_attempt_id"],
        "autobench_attempt_id": None,
        "observed_context": helper_context,
        "source_fingerprint": None,
        "build_fingerprint": None,
        "build_output_identity": None,
        "device_uuid": expected_uuid,
        "cgroup": context["cgroup"],
        "unit_name": context["unit_name"],
        "peak_scratch_bytes": peak,
        "peak_measurement": arm.get("peak_measurement"),
        "helper_evidence_path": str(helper_path),
        "helper_evidence_sha256": _sha_file(helper_path),
        "validated_record_path": str(validated),
        "validated_record_sha256": _sha_file(validated),
        "validator_stdout_path": str(stdout),
        "validator_stdout_sha256": stdout_sha,
        "validator_stderr_path": str(stderr),
        "validator_stderr_sha256": stderr_sha,
        "wire_inventory": inventory,
    }


def _load_catalog(source: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    path = source / "bench/factorized_interactions/workloads.json"
    catalog = _read(path)
    cases = catalog.get("cases")
    if not isinstance(cases, list) or len(cases) != 13:
        raise DatasetError("immutable workload catalog must contain exactly 13 performance cases")
    rows = {}
    for row in cases:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or row["id"] in rows:
            raise DatasetError("immutable workload catalog has malformed or duplicate cases")
        rows[row["id"]] = row
    return rows, {"path": str(path.resolve(strict=True)), "sha256": _sha_file(path)}


def _expected_arm_facts(source: Path, workload: str, report: Mapping[str, Any]) -> dict[str, Any]:
    facts = report.get("direct_control_facts")
    if not isinstance(facts, dict):
        raise DatasetError("timing report omitted direct-control workload facts")
    # This module is part of the accepted external harness. It recomputes the
    # declared unique-output/stable routing facts from the immutable workload.
    import bounded_validation
    expected = bounded_validation.workload_facts(source, workload, "fp16", "fp16")
    if facts != expected:
        raise DatasetError("timing report direct-control facts differ from the immutable workload")
    return expected


def assemble_dataset(*, items: Mapping[tuple[str, str], tuple[Path, Path, Path]],
                     source: Path, candidate_manifest: Path, baseline_manifest: Path,
                     qualification_bundle: Path, expected_uuid: str,
                     sanitizer_wrapper: Path) -> dict[str, Any]:
    """Verify all 26 finished timing partitions and build the immutable dataset."""
    source, candidate_manifest, baseline_manifest = map(
        lambda p: Path(p).resolve(strict=True), (source, candidate_manifest, baseline_manifest))
    rows, catalog_info = _load_catalog(source)
    if len(items) != 26:
        raise DatasetError("the dataset requires all 13 workloads in both timing phases")
    # Catalog order is canonical and agrees with the fixed native performance workload list.
    order = list(rows)
    expected_keys = {(workload, phase) for workload in order for phase in PHASES}
    if set(items) != expected_keys:
        raise DatasetError("timing reports do not cover the exact 13 by 2 workload/phase matrix")
    bundle_validator = validate_existing_bundle
    bundle = bundle_validator(qualification_bundle, source, candidate_manifest,
                              baseline_manifest, expected_uuid)
    if bundle.get("schema") != QUALIFICATION_SCHEMA or \
            bundle.get("status") != "validated_pending_independent_review" or \
            bundle.get("device_uuid") != expected_uuid:
        raise DatasetError("the complete native qualification bundle is not bound to this device")
    identity = bundle.get("build")
    if not isinstance(identity, dict):
        raise DatasetError("native qualification bundle omitted its exact build identity")
    qpath = Path(qualification_bundle).resolve(strict=True)
    qsha = _sha_file(qpath)
    qual_ref = {"path": str(qpath), "sha256": qsha, "schema": bundle["schema"],
                "status": bundle["status"], "device_uuid": expected_uuid,
                "build": dict(identity), "coverage": dict(bundle["coverage"])}
    input_refs: dict[tuple[str, str, str], dict[str, str]] = {}

    def add_ref(path: Path, role: str) -> dict[str, str]:
        resolved = Path(path).resolve(strict=True)
        digest = _sha_file(resolved)
        identifier = "recorded-" + hashlib.sha256(
            (role + "\0" + str(resolved)).encode("utf-8")).hexdigest()[:20]
        input_refs[(str(resolved), digest, role)] = {
            "id": identifier, "path": str(resolved), "sha256": digest, "role": role}
        return {"path": str(resolved), "sha256": digest}

    add_ref(qpath, "qualification_bundle")
    add_ref(Path(catalog_info["path"]), "fixed_workload_catalog")
    for qrow in bundle.get("reports", []):
        if not isinstance(qrow, dict):
            raise DatasetError("qualification bundle report reference is malformed")
        for role in ("report", "receipt", "pointer"):
            path = _bundle_evidence_path({**qrow, "evidence_root": bundle.get("evidence_root")}, qpath,
                                         f"{role}_path", f"{qrow.get('kind')} {role}")
            if _sha_file(path) != qrow[f"{role}_sha256"]:
                raise DatasetError(f"qualification {role} changed after bundle validation")
            add_ref(path, f"qualification_{qrow['kind']}_{role}")
    supplied_wrapper = Path(sanitizer_wrapper).resolve(strict=True)
    sanitizer_wrapper = _bundle_evidence_path(bundle, qpath, "sanitizer_wrapper_path",
                                               "sanitizer validator wrapper")
    if _sha_file(supplied_wrapper) != bundle.get("sanitizer_wrapper_sha256") or \
            _sha_file(sanitizer_wrapper) != bundle.get("sanitizer_wrapper_sha256"):
        raise DatasetError("sanitizer validator wrapper differs from qualification bundle")
    add_ref(sanitizer_wrapper, "fixed_sanitizer_partition_validator")
    sidecar_root = Path(__file__).resolve(strict=True).parent
    harness_hashes = _check_local_harness(sidecar_root, sanitizer_wrapper,
                                          APPROVED_TIMING_HARNESS_SHA256)
    for key, filename in _HARNESS_SIDECARS.items():
        add_ref(sidecar_root / filename, f"timing_harness_{key.removesuffix('_sha256')}")

    timing_refs = []
    common_input: dict[str, str] = {}
    case_phase_blocks: dict[str, dict[str, list[dict[str, Any]]]] = {
        workload: {phase: [] for phase in PHASES} for workload in order
    }
    jobs, attempts = set(), set()
    report_hashes = set()
    for workload in order:
        expected_facts = None
        for phase in PHASES:
            report_path, receipt_path, pointer_path = (
                Path(item).resolve(strict=True) for item in items[(workload, phase)])
            report = _read(report_path)
            report_sha = _sha_file(report_path)
            if report_sha in report_hashes:
                raise DatasetError("duplicate timing report bytes cannot count as another partition")
            report_hashes.add(report_sha)
            wrapper = _read(receipt_path)
            if wrapper.get("report_sha256") != report_sha:
                raise DatasetError("finished receipt export is not bound to the supplied report bytes")
            original_report_path = wrapper.get("report_path")
            if original_report_path is not None and (not isinstance(original_report_path, str) or not original_report_path):
                raise DatasetError("finished receipt export has a malformed original report path")
            if report.get("schema") != TIMING_SCHEMA or \
                    report.get("status") != "observations_complete_pending_independent_review" or \
                    report.get("workload") != workload or report.get("phase") != phase or \
                    report.get("phase_offset") != (0 if phase == "exploration" else BLOCKS) or \
                    report.get("paired_blocks") != BLOCKS or report.get("arms_per_block") != 3 or \
                    report.get("iterations_per_arm") != CALLS or report.get("primary_metric") != PRIMARY_METRIC or \
                    report.get("activation_storage") != "fp16" or report.get("parameter_storage") != "fp16" or \
                    report.get("output_storage") != "fp16" or report.get("gradient_mode") != "overwrite" or \
                    report.get("build_rechecked") is not True or report.get("completed_blocks") != BLOCKS or \
                    report.get("completed_observations") != BLOCKS * 3:
                raise DatasetError(f"{workload}/{phase} timing partition is incomplete or out of contract")
            if report.get("expected_device_uuid") != expected_uuid or report.get("attempt_scratch_limit_bytes") != 128 * 1024 * 1024:
                raise DatasetError(f"{workload}/{phase} uses the wrong device or scratch limit")
            report_build = report.get("build")
            if not isinstance(report_build, dict) or any(report_build.get(key) != identity.get(key) for key in (
                    "source_closure_sha256", "candidate_runner_sha256", "candidate_manifest_sha256",
                    "baseline_runner_sha256", "baseline_manifest_sha256", "library_sha256", "validator_sha256")):
                raise DatasetError(f"{workload}/{phase} does not use the qualified immutable build")
            report_harness = _validate_harness_provenance(report, harness_hashes, identity)
            report_qualification = report.get("qualification")
            expected_qualification = {
                "sha256": qual_ref["sha256"], "schema": qual_ref["schema"],
                "status": qual_ref["status"], "device_uuid": qual_ref["device_uuid"],
                "build": qual_ref["build"], "coverage": qual_ref["coverage"],
                "verification": {"validator": "recorded_qualification.validate_existing_bundle",
                                 "prerequisite_validator_passed": True,
                                 "schema": qual_ref["schema"], "status": qual_ref["status"]},
                "reports": list(bundle["reports"])}
            if not isinstance(report_qualification, dict) or any(
                    report_qualification.get(key) != value
                    for key, value in expected_qualification.items()):
                raise DatasetError(f"{workload}/{phase} records a different qualification bundle")
            outer = report.get("outer_execution_context")
            final = report.get("final_outer_execution_context")
            if not isinstance(outer, dict) or not isinstance(final, dict) or \
                    outer.get("status") not in ("observed", "observed_inside_active_attempt"):
                raise DatasetError(f"{workload}/{phase} omitted its before/after observed outer context")
            if final.get("status") != "observed":
                raise DatasetError(f"{workload}/{phase} final process context is not observed")
            for key in _CONTEXT_FIELDS:
                if not isinstance(outer.get(key), str) or not outer[key] or final.get(key) != outer[key]:
                    raise DatasetError(f"{workload}/{phase} outer context changed in {key}")
            if outer.get("device_uuid") != expected_uuid or final.get("device_uuid") != expected_uuid:
                raise DatasetError(f"{workload}/{phase} outer context used a different GPU")
            if outer.get("scheduler_receipt_status") != "pending_external_terminal_receipt_and_cleanup":
                raise DatasetError("partition did not preserve the pending receipt boundary")
            if report.get("direct_control_facts") != (expected_facts or _expected_arm_facts(source, workload, report)):
                raise DatasetError("workload direct-control facts changed between phase reports")
            expected_facts = report["direct_control_facts"]
            if report.get("scratch_measurement", {}).get("authority") != \
                    "Autobench attempt monitor enforces the hard 128 MiB cap":
                raise DatasetError("timing report omitted the monitored scratch boundary")
            if isinstance(report.get("max_observed_peak_scratch_bytes"), bool) or \
                    not isinstance(report.get("max_observed_peak_scratch_bytes"), int) or \
                    not 0 <= report["max_observed_peak_scratch_bytes"] < 128 * 1024 * 1024:
                raise DatasetError("timing report exceeded or omitted the monitored scratch bound")
            projected = report.get("projected_complete_attempt_bytes")
            if isinstance(projected, bool) or not isinstance(projected, int) or \
                    not 0 <= projected < 128 * 1024 * 1024:
                raise DatasetError("timing report's complete attempt projection exceeds the monitored scratch bound")
            blocks = report.get("blocks")
            if not isinstance(blocks, list) or len(blocks) != BLOCKS:
                raise DatasetError(f"{workload}/{phase} omitted one or more paired blocks")
            if report.get("input_case_sha256") != common_input.get(workload, report.get("input_case_sha256")):
                raise DatasetError(f"{workload} input case bytes changed between phases")
            phase_input = report.get("input_case_sha256")
            if not _HEX.fullmatch(str(phase_input or "")):
                raise DatasetError("partition omitted canonical workload input bytes digest")
            common_input[workload] = phase_input
            measured_peak = 0
            first_block = blocks[0]
            first_order = first_block.get("execution_order") if isinstance(first_block, dict) else None
            first_arms = first_block.get("arms") if isinstance(first_block, dict) else None
            if not isinstance(first_order, list) or not first_order or not isinstance(first_arms, dict):
                raise DatasetError("first timing block cannot establish its original execution context")
            first_arm = first_arms.get(first_order[0])
            if not isinstance(first_arm, dict):
                raise DatasetError("first timing arm is malformed")
            first_helper_path = _safe_child(report_path.parent, first_arm.get("helper_evidence_path"),
                                            "first timing helper evidence")
            first_helper = _read(first_helper_path)
            report_context = first_helper.get("context")
            if not isinstance(report_context, dict) or any(
                    report_context.get(key) != outer.get(key) for key in _CONTEXT_FIELDS):
                raise DatasetError("first helper context differs from report outer context")
            try:
                receipt_proof = validate_receipt_and_pointer(receipt_path, pointer_path,
                                                             report_context, expected_uuid)
            except Exception as exc:
                raise DatasetError(f"finished Scheduler receipt rejected for {workload}/{phase}: {exc}") from exc
            if receipt_proof["scheduler_job_id"] != outer["scheduler_job_id"] or \
                    receipt_proof["scheduler_attempt_id"] != outer["scheduler_attempt_id"] or \
                    receipt_proof["device_uuid"] != expected_uuid:
                raise DatasetError("outer receipt differs from the timing partition")
            if receipt_proof["scheduler_job_id"] in jobs or receipt_proof["scheduler_attempt_id"] in attempts:
                raise DatasetError("timing partitions must have 26 distinct finished Scheduler attempts")
            jobs.add(receipt_proof["scheduler_job_id"])
            attempts.add(receipt_proof["scheduler_attempt_id"])
            for expected_round, block in enumerate(blocks):
                global_round = expected_round + (0 if phase == "exploration" else BLOCKS)
                pair = list(PAIR_ORDER if global_round % 2 == 0 else reversed(PAIR_ORDER))
                descriptive_pos = global_round % 3
                execution = pair.copy(); execution.insert(descriptive_pos, "native_direct")
                if not isinstance(block, dict) or block.get("round_index") != expected_round or \
                        block.get("global_round_index") != global_round or \
                        block.get("paired_order") != pair or block.get("descriptive_position") != descriptive_pos or \
                        block.get("execution_order") != execution:
                    raise DatasetError(f"{workload}/{phase} paired order or block index is invalid")
                arms = block.get("arms")
                if not isinstance(arms, dict) or set(arms) != set(ARMS):
                    raise DatasetError(f"{workload}/{phase} block {expected_round} is missing one or more arms")
                checked_arms = {}
                for arm_name in ARMS:
                    arm = arms[arm_name]
                    if not isinstance(arm, dict):
                        raise DatasetError("timing arm entry is malformed")
                    checked = _validated_observation(report, report_path, block, arm_name, arm,
                                                     workload, identity, expected_uuid, phase_input)
                    observed = checked.pop("observed_context")
                    if any(observed.get(k) != outer.get(k) for k in _CONTEXT_FIELDS):
                        raise DatasetError("retained helper context differs from the outer report context")
                    if checked["scheduler_job_id"] != outer["scheduler_job_id"] or \
                            checked["scheduler_attempt_id"] != outer["scheduler_attempt_id"]:
                        raise DatasetError("timed arm ran under another outer Scheduler allocation")
                    checked["autobench_attempt_id"] = receipt_proof["autobench_attempt_id"]
                    checked["original_report_sha256"] = report_sha
                    checked["source_fingerprint"] = receipt_proof["source_fingerprint"]
                    checked["build_fingerprint"] = receipt_proof["build_fingerprint"]
                    checked["build_output_identity"] = receipt_proof["build_output_identity"]
                    if checked["peak_scratch_bytes"] != arm.get("peak_scratch_bytes") or \
                            checked["peak_measurement"] != arm.get("peak_measurement"):
                        raise DatasetError("timed arm scratch evidence differs")
                    measured_peak = max(measured_peak, checked["peak_scratch_bytes"])
                    checked_arms[arm_name] = checked
                    for key, role in (("helper_evidence_path", "timed_helper_evidence"),
                                      ("validated_record_path", "pre_cleanup_validation_record"),
                                      ("validator_stdout_path", "validator_stdout"),
                                      ("validator_stderr_path", "validator_stderr")):
                        add_ref(Path(checked[key]), role)
                case_phase_blocks[workload][phase].append({
                    "block_index": expected_round,
                    "global_block_index": global_round,
                    "paired_order": pair,
                    "execution_order": execution,
                    "descriptive_position": descriptive_pos,
                    "arms": checked_arms,
                })
            if report["max_observed_peak_scratch_bytes"] != measured_peak:
                raise DatasetError("partition peak scratch summary differs from its 21 original invocations")
            timing_refs.append({"workload": workload, "phase": phase,
                                "report_path": str(report_path), "report_sha256": report_sha,
                                "receipt_export_report_path": original_report_path,
                                "original_qualification_bundle_path": report_qualification.get("path"),
                                "receipt_path": str(receipt_path), "receipt_sha256": _sha_file(receipt_path),
                                "pointer_path": str(pointer_path), "pointer_sha256": _sha_file(pointer_path),
                                "scheduler_job_id": receipt_proof["scheduler_job_id"],
                                "scheduler_attempt_id": receipt_proof["scheduler_attempt_id"],
                                "autobench_attempt_id": receipt_proof["autobench_attempt_id"],
                                "device_uuid": expected_uuid,
                                "cgroup": receipt_proof["cgroup"], "unit_name": receipt_proof["unit_name"],
                                "source_fingerprint": receipt_proof["source_fingerprint"],
                                "build_fingerprint": receipt_proof["build_fingerprint"],
                                "build_output_identity": receipt_proof["build_output_identity"],
                                "harness_provenance": {"schema": HARNESS_SCHEMA,
                                    "before": report_harness, "after": report_harness,
                                    "unchanged": True,
                                    "source_closure_sha256": identity["source_closure_sha256"],
                                    "validator_sha256": identity["validator_sha256"]}})
            add_ref(report_path, "timing_partition_report")
            add_ref(receipt_path, "finished_autobench_scheduler_receipt")
            add_ref(pointer_path, "finished_autobench_pointer")

    cases = []
    for workload in order:
        descriptor = rows[workload]
        cases.append({"id": workload, "workload_sha256": _canonical_sha(descriptor),
                      "workload_descriptor": descriptor,
                      "input_case_sha256": common_input[workload],
                      "phases": case_phase_blocks[workload]})
    return {
        "schema": SCHEMA,
        "status": "recorded_cuda_observations_verified_pending_independent_review",
        "qualification_status": bundle["status"],
        "qualification_bundle_sha256": qsha,
        "qualification_bundle": qual_ref,
        "sanitizer_wrapper": {"path": str(sanitizer_wrapper), "sha256": _sha_file(sanitizer_wrapper)},
        "identity": {**{key: identity[key] for key in (
            "source_closure_sha256", "candidate_runner_sha256", "candidate_manifest_sha256",
            "baseline_runner_sha256", "baseline_manifest_sha256", "library_sha256", "validator_sha256")},
            "device_uuid": expected_uuid, "workload_catalog": catalog_info,
            "timing_harness": harness_hashes},
        "source_root": str(source),
        "cases": cases,
        "timing_reports": timing_refs,
        "trusted_inputs": sorted(input_refs.values(), key=lambda item: (item["path"], item["role"])),
        "coverage": {"workloads": len(cases), "phases_per_workload": 2,
                     "paired_blocks_per_phase": BLOCKS, "arms_per_block": 3,
                     "unique_cuda_observations": len(cases) * 2 * BLOCKS * 3},
    }


def validate_dataset(dataset: Mapping[str, Any]) -> dict[str, Any]:
    """Validate structural coverage of the normalized (already bound) dataset."""
    if dataset.get("schema") != SCHEMA or \
            dataset.get("status") != "recorded_cuda_observations_verified_pending_independent_review":
        raise DatasetError("recorded observation dataset schema or status is invalid")
    cases = dataset.get("cases")
    if not isinstance(cases, list) or len(cases) != 13:
        raise DatasetError("recorded dataset must contain all 13 fixed workloads")
    ids = [row.get("id") if isinstance(row, dict) else None for row in cases]
    if len(set(ids)) != 13 or any(not isinstance(i, str) for i in ids):
        raise DatasetError("recorded dataset workload IDs are missing or duplicated")
    identity = dataset.get("identity")
    qualification = dataset.get("qualification_bundle")
    if not isinstance(identity, dict) or not isinstance(qualification, dict) or \
            qualification.get("status") != dataset.get("qualification_status") or \
            qualification.get("sha256") != dataset.get("qualification_bundle_sha256") or \
            identity.get("device_uuid") != qualification.get("device_uuid") or \
            not _UUID.fullmatch(str(identity.get("device_uuid", ""))):
        raise DatasetError("dataset build/device/qualification identities disagree")
    source_build_keys = ("source_closure_sha256", "candidate_runner_sha256", "candidate_manifest_sha256",
                         "baseline_runner_sha256", "baseline_manifest_sha256", "library_sha256", "validator_sha256")
    for key in source_build_keys:
        if not _HEX.fullmatch(str(identity.get(key, ""))):
            raise DatasetError(f"dataset identity has an invalid {key}")
    if identity.get("timing_harness") != APPROVED_TIMING_HARNESS_SHA256:
        raise DatasetError("dataset timing sidecar hashes differ from the fixed accepted harness")
    descriptor = identity.get("workload_catalog")
    if not isinstance(descriptor, dict) or not _HEX.fullmatch(str(descriptor.get("sha256", ""))):
        raise DatasetError("dataset does not bind the fixed workload catalog")
    report_rows = dataset.get("timing_reports")
    if not isinstance(report_rows, list) or len(report_rows) != 26:
        raise DatasetError("dataset must retain all 26 original timing reports")
    report_by_key = {}
    job_ids, attempt_ids = set(), set()
    report_hashes, receipt_hashes, pointer_hashes, autobench_ids = set(), set(), set(), set()
    for report in report_rows:
        if not isinstance(report, dict):
            raise DatasetError("timing report provenance is malformed")
        key = (report.get("workload"), report.get("phase"))
        if key in report_by_key or key[1] not in PHASES:
            raise DatasetError("timing report provenance has a duplicate or invalid phase")
        report_by_key[key] = report
        _validate_harness_provenance({"harness_provenance": report.get("harness_provenance")},
                                     APPROVED_TIMING_HARNESS_SHA256, identity)
        for field in ("report_sha256", "receipt_sha256", "pointer_sha256"):
            if not _HEX.fullmatch(str(report.get(field, ""))):
                raise DatasetError(f"timing report provenance has invalid {field}")
        if report["report_sha256"] in report_hashes or report["receipt_sha256"] in receipt_hashes or \
                report["pointer_sha256"] in pointer_hashes:
            raise DatasetError("timing report provenance reuses a report, receipt, or pointer")
        report_hashes.add(report["report_sha256"])
        receipt_hashes.add(report["receipt_sha256"])
        pointer_hashes.add(report["pointer_sha256"])
        job, attempt = report.get("scheduler_job_id"), report.get("scheduler_attempt_id")
        autobench = report.get("autobench_attempt_id")
        if not isinstance(job, str) or not job or not isinstance(attempt, str) or not attempt or \
                not isinstance(autobench, str) or not autobench or job in job_ids or \
                attempt in attempt_ids or autobench in autobench_ids or \
                report.get("device_uuid") != identity["device_uuid"]:
            raise DatasetError("timing report provenance repeats an attempt or changes GPU")
        job_ids.add(job); attempt_ids.add(attempt); autobench_ids.add(autobench)
    if set(report_by_key) != {(case_id, phase) for case_id in ids for phase in PHASES}:
        raise DatasetError("timing report provenance does not cover every workload and phase")
    expected_coverage = {"workloads": 13, "phases_per_workload": 2,
                         "paired_blocks_per_phase": BLOCKS, "arms_per_block": 3,
                         "unique_cuda_observations": 13 * 2 * BLOCKS * 3}
    if dataset.get("coverage") != expected_coverage:
        raise DatasetError("dataset coverage summary does not account for all original observations")
    trusted_rows = dataset.get("trusted_inputs")
    if not isinstance(trusted_rows, list):
        raise DatasetError("dataset omitted its original report and receipt trusted inputs")
    trusted = set()
    trusted_ids = set()
    for row in trusted_rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"] or \
                not isinstance(row.get("role"), str) or not row["role"] or \
                not isinstance(row.get("path"), str) or not row["path"] or \
                not _HEX.fullmatch(str(row.get("sha256", ""))):
            raise DatasetError("dataset trusted input reference is malformed")
        if row["id"] in trusted_ids:
            raise DatasetError("dataset trusted input identifiers are duplicated")
        trusted_ids.add(row["id"])
        trusted.add((row["path"], row["sha256"]))
    for path_key, hash_key in (("path", "sha256"),):
        bundle = dataset.get("qualification_bundle", {})
        if (bundle.get(path_key), bundle.get(hash_key)) not in trusted:
            raise DatasetError("qualification bundle is not retained as a trusted input")
    if (descriptor.get("path"), descriptor.get("sha256")) not in trusted:
        raise DatasetError("fixed workload catalog is not retained as a trusted input")
    if (dataset.get("sanitizer_wrapper", {}).get("path"),
            dataset.get("sanitizer_wrapper", {}).get("sha256")) not in trusted:
        raise DatasetError("fixed sanitizer validator wrapper is not a trusted input")
    for key, filename in _HARNESS_SIDECARS.items():
        role = f"timing_harness_{key.removesuffix('_sha256')}"
        if not any(row.get("role") == role and row.get("sha256") == APPROVED_TIMING_HARNESS_SHA256[key]
                   for row in trusted_rows):
            raise DatasetError(f"approved timing sidecar {filename} is not retained as a trusted input")
    for row in report_rows:
        for prefix in ("report", "receipt", "pointer"):
            if (row.get(f"{prefix}_path"), row.get(f"{prefix}_sha256")) not in trusted:
                raise DatasetError(f"original timing {prefix} is not retained as a trusted input")
    unique_observations = set()
    for case in cases:
        if not isinstance(case.get("workload_descriptor"), dict) or \
                case["workload_descriptor"].get("id") != case.get("id") or \
                _canonical_sha(case.get("workload_descriptor")) != case.get("workload_sha256"):
            raise DatasetError(f"fixed descriptor digest mismatch for {case['id']}")
        if set(case.get("phases", {})) != set(PHASES):
            raise DatasetError(f"{case['id']} is missing an observation phase")
        case_inputs = set()
        for phase in PHASES:
            blocks = case["phases"][phase]
            if not isinstance(blocks, list) or len(blocks) != BLOCKS:
                raise DatasetError(f"{case['id']}/{phase} does not contain seven blocks")
            for index, block in enumerate(blocks):
                global_index = index + (0 if phase == "exploration" else BLOCKS)
                pair = list(PAIR_ORDER if global_index % 2 == 0 else reversed(PAIR_ORDER))
                position = global_index % 3
                execution = pair.copy(); execution.insert(position, "native_direct")
                if block.get("block_index") != index or block.get("global_block_index") != global_index or \
                        block.get("paired_order") != pair or block.get("descriptive_position") != position or \
                        block.get("execution_order") != execution or set(block.get("arms", {})) != set(ARMS):
                    raise DatasetError("recorded blocks have missing or duplicate arms")
                input_values = set()
                for arm_name, arm in block["arms"].items():
                    _finite_nonnegative(arm.get("primary_ms"), "recorded primary time")
                    for key in ("raw_metrics_sha256", "validation_summary_sha256", "original_report_sha256",
                                "input_case_sha256", "output_result_sha256"):
                        if not _HEX.fullmatch(str(arm.get(key, ""))):
                            raise DatasetError(f"recorded arm has invalid {key}")
                    input_values.add(arm["input_case_sha256"])
                    report_ref = report_by_key[(case["id"], phase)]
                    if arm.get("original_report_sha256") != report_ref["report_sha256"] or \
                            arm.get("scheduler_job_id") != report_ref["scheduler_job_id"] or \
                            arm.get("scheduler_attempt_id") != report_ref["scheduler_attempt_id"] or \
                            arm.get("autobench_attempt_id") != report_ref.get("autobench_attempt_id") or \
                            arm.get("device_uuid") != identity["device_uuid"] or \
                            arm.get("source_fingerprint") != report_ref.get("source_fingerprint") or \
                            arm.get("build_fingerprint") != report_ref.get("build_fingerprint") or \
                            arm.get("build_output_identity") != report_ref.get("build_output_identity") or \
                            arm.get("cgroup") != report_ref.get("cgroup") or \
                            arm.get("unit_name") != report_ref.get("unit_name"):
                        raise DatasetError("arm provenance does not bind its exact timing report and receipt")
                    case_inputs.add(arm["input_case_sha256"])
                    metrics = arm.get("metrics")
                    summary = arm.get("validated_summary")
                    if not isinstance(metrics, dict) or not isinstance(summary, dict) or \
                            _canonical_sha(metrics) != arm["raw_metrics_sha256"] or \
                            _canonical_sha(summary) != arm["validation_summary_sha256"] or \
                            summary.get("metrics") != metrics:
                        raise DatasetError("normalized raw metric or validation-summary digest changed")
                    samples = metrics.get("samples_ms")
                    phases = metrics.get("phase_ms")
                    if not isinstance(samples, dict):
                        raise DatasetError("normalized metrics omit raw timing vectors")
                    for phase_name in ("forward", "backward", "forward_vjp", "parameter_refresh"):
                        vector = samples.get(phase_name)
                        if not isinstance(vector, list) or len(vector) != CALLS:
                            raise DatasetError("normalized raw timing vector is incomplete")
                        for value in vector:
                            _finite_nonnegative(value, "normalized raw metric sample")
                    total = _finite_nonnegative(
                        phases.get("warm_forward_vjp_batch_host_wall") if isinstance(phases, dict) else None,
                        "normalized primary batch total")
                    if arm["primary_ms"] != phases.get(PRIMARY_METRIC) or \
                            not math.isclose(float(arm["primary_ms"]) * CALLS, total,
                                             rel_tol=2e-8, abs_tol=1e-9):
                        raise DatasetError("normalized primary mean differs from its 30-call batch total")
                    if summary.get("validated") is not True or summary.get("timed") is not True:
                        raise DatasetError("normalized arm no longer carries a passing timed validation")
                    contract = summary.get("contract_evidence")
                    if not isinstance(contract, dict) or any(k not in contract for k in
                            ("y", "dx", "dz", "dalpha", "dbias", "dgamma")):
                        raise DatasetError("normalized arm lacks output or a VJP contract")
                    for field in ("y", "dx", "dz", "dalpha", "dbias", "dgamma"):
                        ratio = contract[field].get("max_error_to_allowance") if isinstance(contract[field], dict) else None
                        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or \
                                not math.isfinite(float(ratio)) or not 0 <= float(ratio) <= 1:
                            raise DatasetError(f"normalized arm failed the frozen numerical bound for {field}")
                    if arm.get("binary_identity") != {key: identity[key] for key in (
                            "source_closure_sha256", "candidate_runner_sha256", "candidate_manifest_sha256",
                            "library_sha256", "validator_sha256")}:
                        raise DatasetError("normalized arm native binary identity changed")
                    if arm.get("actual_implementation") not in {"packet", "direct", "direct_unique_output", "stable"} or \
                            not _HEX.fullmatch(str(arm.get("helper_evidence_sha256", ""))):
                        raise DatasetError("normalized arm implementation identity is invalid")
                    for path_key, hash_key in (("helper_evidence_path", "helper_evidence_sha256"),
                                               ("validated_record_path", "validated_record_sha256"),
                                               ("validator_stdout_path", "validator_stdout_sha256"),
                                               ("validator_stderr_path", "validator_stderr_sha256")):
                        if (arm.get(path_key), arm.get(hash_key)) not in trusted:
                            raise DatasetError(f"retained arm {path_key} is not a trusted input")
                    observation = (case["id"], phase, arm_name,
                                   arm.get("scheduler_job_id"), arm.get("helper_evidence_sha256"))
                    if observation in unique_observations:
                        raise DatasetError("a recorded timing observation was duplicated")
                    unique_observations.add(observation)
                if len(input_values) != 1:
                    raise DatasetError("paired arms in a workload do not share the canonical input bytes")
        if len(case_inputs) != 1 or case.get("input_case_sha256") not in case_inputs:
            raise DatasetError("workload input digest changed across its phases")
    if len(unique_observations) != 13 * 2 * BLOCKS * 3:
        raise DatasetError("dataset must retain 546 distinct original observations")
    return {"case_ids": ids, "unique_cuda_observations": len(unique_observations),
            "device_uuid": dataset.get("identity", {}).get("device_uuid"),
            "qualification_bundle_sha256": dataset.get("qualification_bundle_sha256")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--qualification-bundle", type=Path, required=True)
    parser.add_argument("--sanitizer-wrapper", type=Path, required=True)
    parser.add_argument("--expected-uuid", required=True)
    parser.add_argument("--item", action="append", nargs=5,
                        metavar=("WORKLOAD", "PHASE", "REPORT", "RECEIPT", "POINTER"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    items = {}
    for workload, phase, report, receipt, pointer in args.item:
        key = (workload, phase)
        if key in items:
            raise DatasetError(f"duplicate timing partition: {workload}/{phase}")
        items[key] = (Path(report), Path(receipt), Path(pointer))
    try:
        result = assemble_dataset(items=items, source=args.source,
                                  candidate_manifest=args.candidate_manifest,
                                  baseline_manifest=args.baseline_manifest,
                                  qualification_bundle=args.qualification_bundle,
                                  expected_uuid=args.expected_uuid,
                                  sanitizer_wrapper=args.sanitizer_wrapper)
        validate_dataset(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        descriptor = os.open(args.output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except (DatasetError, OSError) as exc:
        parser.exit(2, f"recorded dataset assembly failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
