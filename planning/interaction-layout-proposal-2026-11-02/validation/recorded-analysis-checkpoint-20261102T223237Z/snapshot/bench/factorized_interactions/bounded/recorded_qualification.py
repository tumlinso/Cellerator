#!/usr/bin/env python3
"""Bind completed native qualification reports to their real outer receipts.

This CPU-only reader does not create Scheduler claims or replace the independent
oracle. It creates a compact, conditional manifest pointing to retained source
evidence. No timing observation is generated here.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any, Mapping

sys.dont_write_bytecode = True
from bounded_validation import VALIDATOR_SHA256, verify_build  # noqa: E402

SCHEMA = "cellerator.factorized_interactions.recorded_qualification.v1"
KINDS = ("dtype", "aux_packet_regular", "aux_packet_high_fanin_1m",
         "aux_strongest_direct_regular", "aux_strongest_direct_high_fanin_1m",
         "sanitizer_memcheck", "sanitizer_racecheck", "sanitizer_initcheck", "sanitizer_synccheck")
TOOLS = ("memcheck", "racecheck", "initcheck", "synccheck")
SANITIZER_WRAPPER_SHA256 = "e5bafd545ecaaae4a093f533da9e96276f6795e6f91691de25180dcf0ad04de7"
QUALIFICATION_DRIVER_SHA256 = "eed5ed49683c4ac5b79b24225f1d6e37315707a0dbf71cee1464310015cf2210"
BOUNDED_VALIDATION_SHA256 = "6da91757d8d18501ad8df4c6722c549ae32c9adfb93e78417a016b6849caaf11"
MODES = ("packet", "native_direct", "minimal_direct", "direct_unique_output")
AUXILIARY_IDS = (
    "correctness_ordered_tied_zero", "correctness_saturated_gates",
    "correctness_ordered_role_context_variants", "correctness_fma_product_residual_cancellation",
    "correctness_context_cancellation_middle", "correctness_context_cancellation_tail",
    "correctness_high_fanin_100k", "correctness_packet_high_fanin_fp32_100k",
    "correctness_packet_high_fanin_1m", "correctness_empty_mechanisms",
    "correctness_empty_cells", "correctness_empty_channels",
)
_HEX = re.compile(r"^[0-9a-f]{64}$")
_LIMIT = 128 * 1024 * 1024


class QualificationError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise QualificationError(f"expected regular evidence file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise QualificationError(f"expected object in {path}")
    return value


def _relative_file(parent: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise QualificationError("evidence path must be relative")
    path = PurePosixPath(relative)
    if path.is_absolute() or any(part in (".", "..") for part in path.parts):
        raise QualificationError("evidence path escapes its retained artifact")
    base = parent.resolve(strict=True)
    target = base.joinpath(*path.parts)
    if any(component.is_symlink() for component in (target, *target.parents) if component != base.parent):
        raise QualificationError("evidence path contains a symbolic link")
    target = target.resolve(strict=True)
    if not target.is_relative_to(base) or not target.is_file():
        raise QualificationError("evidence file is outside its retained artifact")
    return target


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _HEX.fullmatch(value):
        raise QualificationError(f"{label} is not a SHA256 digest")
    return value


def _finite_contract(evidence: Any, *, repeated: bool) -> None:
    names = ["y", "dx", "dz", "dalpha", "dbias", "dgamma"]
    if repeated:
        names += [name + "_repeat" for name in names[1:]]
    if not isinstance(evidence, dict) or any(name not in evidence for name in names):
        raise QualificationError("oracle evidence omits output or a required VJP")
    for name in names:
        row = evidence[name]
        if not isinstance(row, dict):
            raise QualificationError(f"malformed oracle evidence for {name}")
        ratio = row.get("max_error_to_allowance")
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or \
                not math.isfinite(float(ratio)) or float(ratio) < 0 or float(ratio) > 1:
            raise QualificationError(f"oracle tolerance failed for {name}")


def _binary_identity(source: Path, candidate_manifest: Path,
                     baseline_manifest: Path) -> dict[str, str]:
    candidate = _read(candidate_manifest)
    baseline = _read(baseline_manifest)
    runner_info = candidate.get("binaries", {}).get("runner", {})
    runner = _relative_file(candidate_manifest.parent, runner_info.get("path"))
    build = verify_build(source, runner, candidate_manifest)
    if baseline.get("schema") != "cellerator.factorized_interactions.build.v1" or \
            baseline.get("architecture") != "sm_70" or baseline.get("fast_math") is not False or \
            baseline.get("source_closure_sha256") != build["source_closure_sha256"] or \
            baseline.get("source_files") != candidate.get("source_files"):
        raise QualificationError("baseline and candidate source/build contracts differ")
    parent = baseline_manifest.parent
    binary = baseline.get("binaries", {}).get("runner", {})
    baseline_runner = _relative_file(parent, binary.get("path"))
    if baseline_runner.stat().st_size != binary.get("bytes") or sha256_file(baseline_runner) != binary.get("sha256"):
        raise QualificationError("baseline runner differs from its immutable manifest")
    library = baseline.get("binaries", {}).get("factorized_interaction_cuda", {})
    if library != candidate.get("binaries", {}).get("factorized_interaction_cuda"):
        raise QualificationError("baseline and candidate do not bind the same native library")
    link = baseline.get("candidate_runner", {})
    if not isinstance(link, dict) or \
            _relative_file(parent, link.get("manifest_path")) != candidate_manifest.resolve() or \
            _relative_file(parent, link.get("binary_path")) != runner or \
            link.get("manifest_sha256") != sha256_file(candidate_manifest) or \
            link.get("binary_sha256") != build["runner_sha256"]:
        raise QualificationError("baseline manifest does not bind the exact candidate")
    return {
        "source_closure_sha256": build["source_closure_sha256"],
        "candidate_runner_sha256": build["runner_sha256"],
        "candidate_manifest_sha256": build["build_manifest_sha256"],
        "baseline_runner_sha256": binary["sha256"],
        "baseline_manifest_sha256": sha256_file(baseline_manifest),
        "library_sha256": build["library_sha256"],
        "validator_sha256": build["validator_sha256"],
    }


def _match_report_build(report: Mapping[str, Any], identity: Mapping[str, str]) -> None:
    build = report.get("build")
    if not isinstance(build, dict):
        raise QualificationError("qualification report has no build identity")
    for name, expected in (("source_closure_sha256", identity["source_closure_sha256"]),
                           ("runner_sha256", identity["candidate_runner_sha256"]),
                           ("build_manifest_sha256", identity["candidate_manifest_sha256"])):
        if build.get(name) != expected:
            raise QualificationError(f"qualification report has a different {name}")
    if "library_sha256" in build and build["library_sha256"] != identity["library_sha256"]:
        raise QualificationError("qualification report has a different native library")
    if report.get("validator_sha256", build.get("validator_sha256")) != VALIDATOR_SHA256:
        raise QualificationError("qualification report has a different fixed validator")


def _legacy_dtype(report: Mapping[str, Any], report_path: Path,
                  identity: Mapping[str, str], context: Mapping[str, Any]) -> list[dict[str, Any]]:
    if report.get("schema") != "cellerator.factorized_interactions.dtype_qualification.v1" or \
            report.get("status") != "pass" or report.get("timing_performed") is not False or \
            report.get("expected_case_count") != 32 or report.get("passed_case_count") != 32 or \
            report.get("failures") != []:
        raise QualificationError("dtype matrix did not pass exactly 32 untimed cases")
    _match_report_build(report, identity)
    rows = report.get("records")
    if not isinstance(rows, list) or len(rows) != 32:
        raise QualificationError("dtype report has incomplete records")
    wanted = set(itertools.product(("fp16", "fp32"), repeat=3))
    wanted = {(a, p, o, m) for a, p, o in wanted for m in MODES}
    seen = set()
    inputs_by_dtype = {}
    out = []
    for row in rows:
        if not isinstance(row, dict):
            raise QualificationError("dtype row is malformed")
        key = (row.get("activation"), row.get("parameter"), row.get("output"), row.get("mode"))
        if key not in wanted or key in seen:
            raise QualificationError("dtype matrix has a duplicate or unexpected combination")
        seen.add(key)
        a, p, o, mode = key
        if row.get("workload") != "packet_a4_m32_u16" or row.get("status") != "pass":
            raise QualificationError("dtype matrix used a different fixed workload")
        expected_mode = {
            "packet": ("packet", "packet", "full_packet"),
            "native_direct": ("direct", "direct", "full_packet"),
            "minimal_direct": ("direct", "direct", "canonical_direct"),
            "direct_unique_output": ("direct", "direct_unique_output", "canonical_direct"),
        }[mode]
        if tuple(row.get(field) for field in
                 ("selected_backend", "actual_implementation", "structural_layout")) != expected_mode:
            raise QualificationError("dtype case did not execute its declared implementation")
        for field, value in (("source_closure_sha256", identity["source_closure_sha256"]),
                             ("runner_sha256", identity["candidate_runner_sha256"]),
                             ("build_manifest_sha256", identity["candidate_manifest_sha256"]),
                             ("validator_sha256", identity["validator_sha256"])):
            if row.get(field) != value:
                raise QualificationError(f"dtype case identity mismatch: {field}")
        _finite_contract(row.get("contract_evidence"), repeated=True)
        case = report_path.parent / "cases" / f"{mode}.a{a}.p{p}.o{o}"
        if case.is_symlink() or not case.is_dir():
            raise QualificationError("retained dtype case directory is missing")
        summary_path = case / "validator.stdout.json"
        summary = _read(summary_path)
        if summary.get("validated") is not True or summary.get("timed") is not False or \
                summary.get("workload") != row["workload"] or \
                summary.get("contract_evidence") != row["contract_evidence"] or \
                summary.get("input_case_sha256") != row.get("input_case_sha256") or \
                summary.get("scheduler_job_id") != context["scheduler_job_id"]:
            raise QualificationError("retained dtype validator summary differs from its report")
        native = summary.get("native_execution")
        if not isinstance(native, dict) or native.get("candidate") != mode or \
                native.get("device_uuid") != context["device_uuid"] or \
                native.get("scheduler_job_id") != context["scheduler_job_id"] or any(
                    native.get(field) != row.get(field) for field in
                    ("selected_backend", "actual_implementation", "structural_layout")):
            raise QualificationError("retained dtype native metadata differs from its report")
        inputs = sorted(path for path in case.iterdir() if path.is_file() and
                        (path.name.endswith(".case") or ".case.part" in path.name))
        outputs = [path for path in case.iterdir() if path.is_file() and path.name.endswith(".result")]
        if not inputs or len(outputs) != 1 or len(inputs) != summary.get("input_case_shards"):
            raise QualificationError("retained dtype wire inputs or output are missing")
        stream = hashlib.sha256()
        for path in inputs:
            if path.is_symlink():
                raise QualificationError("dtype input is a symbolic link")
            with path.open("rb") as data:
                for block in iter(lambda: data.read(1024 * 1024), b""):
                    stream.update(block)
        if stream.hexdigest() != row.get("input_case_sha256") or \
                sum(path.stat().st_size for path in inputs) != summary.get("input_case_bytes") or \
                outputs[0].stat().st_size != summary.get("output_bytes"):
            raise QualificationError("retained dtype wire differs from the checked bytes")
        dtype = (a, p, o)
        if dtype in inputs_by_dtype and inputs_by_dtype[dtype] != row["input_case_sha256"]:
            raise QualificationError("dtype modes did not receive identical input bytes")
        inputs_by_dtype[dtype] = row["input_case_sha256"]
        stderr_path = case / "validator.stderr.log"
        if stderr_path.is_symlink() or not stderr_path.is_file():
            raise QualificationError("retained dtype validator stderr is missing")
        out.append({"activation": a, "parameter": p, "output": o, "mode": mode,
                    "input_case_sha256": row["input_case_sha256"],
                    "output_result_sha256": sha256_file(outputs[0]),
                    "validator_summary_sha256": sha256_file(summary_path),
                    "validator_stderr_sha256": sha256_file(stderr_path)})
    if seen != wanted:
        raise QualificationError("dtype matrix is incomplete")
    return out


def _expected_aux_plan(role: str) -> set[tuple[str, str, str, str]]:
    plan = set()
    for workload in AUXILIARY_IDS:
        activation = "fp32" if workload == "correctness_fma_product_residual_cancellation" else "fp16"
        parameter = "fp32" if workload in {
            "correctness_fma_product_residual_cancellation",
            "correctness_packet_high_fanin_fp32_100k",
        } else "fp16"
        plan.add((workload, activation, parameter, "fp16"))
    if role not in {"packet", "strongest-direct"}:
        raise QualificationError("unknown auxiliary role")
    return plan


def _strongest_mode(native: Mapping[str, Any]) -> tuple[str, str]:
    try:
        cells = int(native["cells"])
        channels = int(native["channels"])
        mechanisms = int(native["mechanisms"])
        outputs = int(native["outputs"])
        maxima = native["structural_maxima"]
        if not isinstance(maxima, dict):
            raise TypeError("maxima")
        keys = ("output_mechanisms", "input_roles", "context_terms", "alpha_mechanisms",
                "bias_mechanisms", "gamma_terms", "mechanism_terms")
        if any(isinstance(maxima.get(key), bool) or not isinstance(maxima.get(key), int) or
               maxima[key] < 0 for key in keys) or min(cells, channels, mechanisms, outputs) < 0:
            raise ValueError("negative or malformed extent")
        active = bool(cells and channels and mechanisms)
        stable = active and (
            any(maxima[key] > 64 for key in
                ("output_mechanisms", "input_roles", "context_terms", "mechanism_terms")) or
            maxima["alpha_mechanisms"] * ((cells + 3) // 4) > 64 or
            maxima["bias_mechanisms"] * ((cells + 31) // 32) > 64 or
            maxima["gamma_terms"] * ((cells + 31) // 32) > 64)
        unique = active and not stable and outputs == mechanisms and maxima["output_mechanisms"] == 1
    except (KeyError, TypeError, ValueError) as exc:
        raise QualificationError("native metadata cannot establish the strongest direct mode") from exc
    return ("direct_unique_output" if unique else "minimal_direct", "stable" if stable else "direct")


def _bounded_qualification(report: Mapping[str, Any], report_path: Path,
                           kind: str, identity: Mapping[str, str],
                           context: Mapping[str, Any]) -> list[dict[str, Any]]:
    if kind == "dtype":
        role, partition, count = "dtype-all", None, 32
    elif kind.startswith("aux_packet_"):
        role = "packet"
        partition = "regular" if kind.endswith("_regular") else "high-fanin-1m"
        count = 11 if partition == "regular" else 1
    elif kind.startswith("aux_strongest_direct_"):
        role = "strongest-direct"
        partition = "regular" if kind.endswith("_regular") else "high-fanin-1m"
        count = 11 if partition == "regular" else 1
    else:
        raise QualificationError("unknown bounded qualification kind")
    if report.get("schema") != "cellerator.factorized_interactions.bounded_qualification.v1" or \
            report.get("status") != "pass" or report.get("role") != role or \
            report.get("kind") != ("dtype" if kind == "dtype" else "aux") or \
            report.get("expected_case_count") != count or report.get("passed_case_count") != count or \
            report.get("failures") != []:
        raise QualificationError(f"{kind} report has incomplete or failed coverage")
    if partition is not None and (report.get("partition") != partition or
                                  report.get("full_auxiliary_corpus_count") != 12 or
                                  report.get("full_auxiliary_corpus_qualified_by_this_report") is not False):
        raise QualificationError("auxiliary report does not declare its partial-corpus boundary")
    if report.get("scratch_limit_bytes") != _LIMIT or not isinstance(report.get("peak_scratch_bytes"), int) or \
            not 0 <= report["peak_scratch_bytes"] < _LIMIT:
        raise QualificationError(f"{kind} did not preserve the bounded scratch contract")
    if report.get("harness") != {"qualification_sha256": QUALIFICATION_DRIVER_SHA256,
                                 "bounded_validation_sha256": BOUNDED_VALIDATION_SHA256}:
        raise QualificationError("qualification used unapproved collector or validation helper bytes")
    _match_report_build(report, identity)
    rows = report.get("records")
    if not isinstance(rows, list) or len(rows) != count:
        raise QualificationError(f"{kind} report records are incomplete")
    wanted = ({("packet_a4_m32_u16", a, p, o, m)
               for a, p, o in itertools.product(("fp16", "fp32"), repeat=3) for m in MODES}
              if kind == "dtype" else {
                  case for case in _expected_aux_plan(role)
                  if (case[0] == "correctness_packet_high_fanin_1m") == (partition == "high-fanin-1m")
              })
    seen = set()
    out = []
    for row in rows:
        if not isinstance(row, dict) or row.get("status") != "pass":
            raise QualificationError(f"{kind} contains a failed case")
        basic = (row.get("workload"), row.get("activation"), row.get("parameter"), row.get("output"))
        key = (*basic, row.get("mode")) if kind == "dtype" else basic
        if key not in wanted or key in seen:
            raise QualificationError(f"{kind} has a duplicate or out-of-scope case")
        seen.add(key)
        path = _relative_file(report_path.parent, row.get("evidence_relative_path"))
        if sha256_file(path) != _hash(row.get("evidence_sha256"), "case evidence digest"):
            raise QualificationError("qualification case evidence digest changed")
        evidence = _read(path)
        if evidence.get("schema") != "cellerator.factorized_interactions.bounded_case.v1" or \
                evidence.get("status") != "pass" or evidence.get("success_wires_removed") is not True or \
                evidence.get("cleanup_pending") is not False or \
                evidence.get("case_id") != row.get("case_id") or \
                evidence.get("workload") != row.get("workload") or evidence.get("mode") != row.get("mode"):
            raise QualificationError("case evidence does not attest a checked successful invocation")
        for field in ("scheduler_job_id", "scheduler_attempt_id", "device_uuid", "cgroup", "unit_name"):
            if evidence.get("context", {}).get(field) != context.get(field):
                raise QualificationError(f"case evidence has a different outer {field}")
        _match_report_build({"build": evidence.get("build")}, identity)
        if evidence.get("validator_sha256") != identity["validator_sha256"]:
            raise QualificationError("case evidence used a different validator")
        if evidence.get("bounded_validation_sha256") != BOUNDED_VALIDATION_SHA256:
            raise QualificationError("case evidence used a different bounded validation helper")
        for field in ("input_case_sha256", "output_result_sha256"):
            if evidence.get(field) != _hash(row.get(field), field):
                raise QualificationError(f"case {field} differs between report and evidence")
        summary = evidence.get("summary")
        if not isinstance(summary, dict) or summary.get("validated") is not True or \
                summary.get("timed") is not False or summary.get("workload") != basic[0] or \
                summary.get("activation_storage") != basic[1] or summary.get("parameter_storage") != basic[2] or \
                summary.get("output_storage") != basic[3] or \
                summary.get("input_case_sha256") != evidence["input_case_sha256"] or \
                summary.get("scheduler_job_id") != context["scheduler_job_id"]:
            raise QualificationError("fixed-validator summary differs from the case record")
        _finite_contract(summary.get("contract_evidence"), repeated=True)
        native = summary.get("native_execution")
        if not isinstance(native, dict) or native.get("candidate") != row.get("mode") or \
                native.get("device_uuid") != context["device_uuid"] or \
                native.get("scheduler_job_id") != context["scheduler_job_id"]:
            raise QualificationError("native execution identity differs from the case record")
        direct_mode, direct_backend = _strongest_mode(native)
        if role == "packet" and row.get("mode") != "packet":
            raise QualificationError("packet auxiliary case used another route")
        if role == "strongest-direct" and row.get("mode") != direct_mode:
            raise QualificationError("auxiliary case did not use the applicable strongest direct control")
        expected_backend = ("stable" if direct_backend == "stable" else
                            "packet" if row.get("mode") == "packet" else "direct")
        if native.get("selected_backend") != expected_backend or any(
                native.get(field) != row.get(field) for field in
                ("selected_backend", "actual_implementation", "structural_layout")):
            raise QualificationError("case backend identity differs from the structural contract")
        stdout_path = _relative_file(report_path.parent, evidence.get("stdout_relative_path"))
        stderr_path = _relative_file(report_path.parent, evidence.get("stderr_relative_path"))
        if sha256_file(stdout_path) != evidence.get("stdout_sha256") or \
                sha256_file(stderr_path) != evidence.get("stderr_sha256") or _read(stdout_path) != summary:
            raise QualificationError("retained validator streams differ from their case evidence")
        validated = _relative_file(report_path.parent, evidence.get("validated_relative_path"))
        if sha256_file(validated) != evidence.get("validated_record_sha256"):
            raise QualificationError("pre-cleanup durable validation record is missing or changed")
        prior = _read(validated)
        if prior.get("cleanup_pending") is not True or prior.get("success_wires_removed") is not False or \
                any(prior.get(field) != evidence.get(field) for field in
                    ("summary", "wire_inventory", "input_case_sha256", "output_result_sha256", "build", "context")):
            raise QualificationError("pre-cleanup record does not bind the final case evidence")
        inventory = evidence.get("wire_inventory")
        if not isinstance(inventory, list) or not inventory:
            raise QualificationError("case evidence has no transient wire inventory")
        result_rows = [entry for entry in inventory if isinstance(entry, dict) and
                       isinstance(entry.get("path"), str) and entry["path"].endswith(".result")]
        if len(result_rows) != 1 or result_rows[0].get("sha256") != evidence["output_result_sha256"]:
            raise QualificationError("case output digest is not bound to its wire inventory")
        out.append({"case_id": row["case_id"], "workload": basic[0], "mode": row["mode"],
                    "input_case_sha256": evidence["input_case_sha256"],
                    "output_result_sha256": evidence["output_result_sha256"],
                    "evidence_path": str(path), "evidence_sha256": sha256_file(path),
                    "validator_summary_sha256": sha256_file(stdout_path)})
    if seen != wanted:
        raise QualificationError(f"{kind} omitted required cases")
    return out


def _sanitizer_report(report: Mapping[str, Any], report_path: Path, tool: str,
                      identity: Mapping[str, str], context: Mapping[str, Any],
                      source: Path, wrapper_path: Path) -> dict[str, str]:
    """Independently bind a fixed-suite report and all its retained artifacts."""
    import importlib.util
    if report.get('schema') != 'cellerator.factorized_interactions.sanitizer_tool_partition.v1' or \
            report.get('status') != 'pass' or report.get('tool') != tool or \
            report.get('expected_tools') != list(TOOLS) or \
            report.get('expected_case_count') != 14 or report.get('case_count') != 14 or \
            report.get('passed') != 14 or report.get('timed') is not False or \
            report.get('outer_receipt_verification') != 'pending' or report.get('errors') != [] or \
            report.get('repeat_vjp_checked') is not False:
        raise QualificationError('sanitizer partition did not pass the fixed 14-case suite')
    expected = {
        'source_closure_sha256': identity['source_closure_sha256'],
        'runner_binary_sha256': identity['baseline_runner_sha256'],
        'build_manifest_sha256': identity['baseline_manifest_sha256'],
        'candidate_runner_binary_sha256': identity['candidate_runner_sha256'],
        'candidate_build_manifest_sha256': identity['candidate_manifest_sha256'],
        'component_archive_sha256': identity['library_sha256'],
        'shared_library_sha256': identity['library_sha256'],
        'validator_sha256': identity['validator_sha256'],
        'wrapper_sha256': sha256_file(wrapper_path),
        'suite_sha256': sha256_file(source / 'bench/factorized_interactions/run_sanitizer_suite.py'),
        'context_helper_sha256': sha256_file(source / 'bench/factorized_interactions/experiments/in_job_kda.py'),
    }
    if report.get('library_kind') != 'static_archive' or any(report.get(k) != v for k, v in expected.items()):
        raise QualificationError('sanitizer report identities differ from the qualified native build')
    after = report.get('scheduler_context_after')
    for key in ('scheduler_job_id', 'scheduler_attempt_id', 'device_uuid', 'cgroup', 'unit_name', 'status'):
        if not isinstance(after, dict) or after.get(key) != context.get(key):
            raise QualificationError('sanitizer outer context changed while the suite ran')
    suite_path = _relative_file(report_path.parent, report.get('suite_report_relative_path'))
    if sha256_file(suite_path) != report.get('suite_report_sha256') or _read(suite_path) != report.get('suite_report'):
        raise QualificationError('sanitizer suite report bytes differ from the partition')
    for stream in ('stdout', 'stderr'):
        path = _relative_file(report_path.parent, f'fixed-sanitizer-suite.{stream}.log')
        if sha256_file(path) != report.get(f'suite_{stream}_sha256'):
            raise QualificationError('sanitizer suite stream changed')
    artifact_rel = report.get('artifact_relative_path')
    if not isinstance(artifact_rel, str):
        raise QualificationError('missing sanitizer artifact path')
    # Recompute full inventory; the trusted wrapper rejects links and special files.
    artifact_root = report_path.parent.joinpath(*PurePosixPath(artifact_rel).parts)
    if PurePosixPath(artifact_rel).is_absolute() or '..' in PurePosixPath(artifact_rel).parts or \
            artifact_root.is_symlink() or not artifact_root.is_dir() or \
            not artifact_root.resolve().is_relative_to(report_path.parent.resolve()):
        raise QualificationError('sanitizer artifact directory is outside the report')
    spec = importlib.util.spec_from_file_location('fixed_sanitizer_partition', wrapper_path)
    if spec is None or spec.loader is None:
        raise QualificationError('cannot load fixed sanitizer validator')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.artifact_inventory(artifact_root) != report.get('artifact_inventory'):
        raise QualificationError('sanitizer artifact inventory differs from retained files')
    build = dict(expected)
    module.validate_suite_report(report['suite_report'], tool=tool, context=context, build=build,
                                 tool_binary=Path(report['compute_sanitizer_path']),
                                 tool_version=report['compute_sanitizer_version'])
    for key in ('compute_sanitizer_binary_sha256', 'compute_sanitizer_support_tree_sha256'):
        _hash(report.get(key), key)
    if not isinstance(report.get('compute_sanitizer_version'), str) or not report['compute_sanitizer_version']:
        raise QualificationError('sanitizer version missing')
    return {key: report[key] for key in ('compute_sanitizer_home', 'compute_sanitizer_path',
            'compute_sanitizer_binary_sha256', 'compute_sanitizer_support_tree_sha256',
            'compute_sanitizer_version', 'suite_sha256', 'wrapper_sha256')}


def assemble_qualification(items: Mapping[str, tuple[Path, Path, Path]], source: Path,
                           candidate_manifest: Path, baseline_manifest: Path,
                           expected_uuid: str, sanitizer_wrapper: Path, *,
                           evidence_root: Path | None = None) -> dict[str, Any]:
    from recorded_receipts import validate_receipt_and_pointer
    if set(items) != set(KINDS):
        raise QualificationError('qualification requires exactly nine distinct report kinds')
    original_wrapper = Path(sanitizer_wrapper).absolute()
    source, candidate_manifest, baseline_manifest, sanitizer_wrapper = map(
        lambda p: Path(p).resolve(strict=True), (source, candidate_manifest, baseline_manifest, sanitizer_wrapper))
    raw_root = None if evidence_root is None else Path(evidence_root).absolute()
    if raw_root is not None and raw_root.is_symlink():
        raise QualificationError('portable evidence root cannot be a symbolic link')
    portable_root = None if raw_root is None else raw_root.resolve(strict=True)
    if portable_root is not None and not portable_root.is_dir():
        raise QualificationError('portable evidence root must be a directory')

    def recorded_path(path: Path) -> str:
        if portable_root is None:
            return str(path)
        try:
            relative = path.relative_to(portable_root).as_posix()
        except ValueError as exc:
            raise QualificationError('qualification evidence is outside the portable root') from exc
        if _relative_file(portable_root, relative) != path:
            raise QualificationError('qualification evidence path is not a regular staged file')
        return relative

    def check_staged_path(path: Path) -> None:
        if portable_root is None:
            return
        try:
            relative = path.absolute().relative_to(portable_root).as_posix()
        except ValueError as exc:
            raise QualificationError('qualification evidence is outside the portable root') from exc
        _relative_file(portable_root, relative)

    check_staged_path(original_wrapper)

    identity = _binary_identity(source, candidate_manifest, baseline_manifest)
    wrapper_hash = sha256_file(sanitizer_wrapper)
    if wrapper_hash != SANITIZER_WRAPPER_SHA256:
        raise QualificationError('sanitizer wrapper is not the independently accepted validator')
    rows = []
    job_ids, attempt_ids = set(), set()
    tool_identity = None
    for kind in KINDS:
        if len(items[kind]) != 3:
            raise QualificationError('each qualification item needs report, receipt and pointer')
        for path in items[kind]:
            check_staged_path(Path(path))
        report_path, receipt_path, pointer_path = (Path(p).resolve(strict=True) for p in items[kind])
        report, receipt = _read(report_path), _read(receipt_path)
        digest = sha256_file(report_path)
        if receipt.get('report_sha256') != digest:
            raise QualificationError('official exported report digest differs from supplied bytes')
        context = report.get('scheduler_context')
        if not isinstance(context, dict):
            raise QualificationError('qualification report lacks observed scheduler context')
        proof = validate_receipt_and_pointer(receipt_path, pointer_path, context, expected_uuid)
        if proof['scheduler_job_id'] in job_ids or proof['scheduler_attempt_id'] in attempt_ids:
            raise QualificationError('qualification partitions do not have distinct finished attempts')
        job_ids.add(proof['scheduler_job_id']); attempt_ids.add(proof['scheduler_attempt_id'])
        if kind == 'dtype' and report.get('schema') == 'cellerator.factorized_interactions.dtype_qualification.v1':
            cases = _legacy_dtype(report, report_path, identity, context)
        elif kind.startswith('sanitizer_'):
            this_tool = _sanitizer_report(report, report_path, kind.removeprefix('sanitizer_'),
                                          identity, context, source, sanitizer_wrapper)
            if tool_identity is None:
                tool_identity = this_tool
            elif this_tool != tool_identity:
                raise QualificationError('sanitizer tools did not use the same fixed installation and suite')
            cases = list(range(14))
        else:
            cases = _bounded_qualification(report, report_path, kind, identity, context)
        rows.append({'kind': kind, 'report_path': recorded_path(report_path), 'report_sha256': digest,
                     'receipt_path': recorded_path(receipt_path), 'receipt_sha256': sha256_file(receipt_path),
                     'pointer_path': recorded_path(pointer_path), 'pointer_sha256': sha256_file(pointer_path),
                     'case_count': len(cases), **proof})
    result = {'schema': SCHEMA, 'status': 'validated_pending_independent_review',
            'device_uuid': expected_uuid, 'build': identity,
            'sanitizer_wrapper_path': recorded_path(sanitizer_wrapper), 'sanitizer_wrapper_sha256': wrapper_hash,
            'sanitizer_tool_identity': tool_identity, 'reports': rows,
            'coverage': {'dtype': 32, 'aux_packet': 12, 'aux_strongest_direct': 12,
                         'sanitizer': {tool: 14 for tool in TOOLS}}}
    if portable_root is not None:
        result['evidence_root'] = '.'
    return result


def validate_existing_bundle(bundle_path: Path, source: Path, candidate_manifest: Path,
                             baseline_manifest: Path, expected_uuid: str) -> dict[str, Any]:
    bundle = _read(Path(bundle_path))
    if bundle.get('schema') != SCHEMA or bundle.get('status') != 'validated_pending_independent_review':
        raise QualificationError('qualification bundle schema or conditional status is invalid')
    portable = bundle.get('evidence_root')
    if portable != '.':
        raise QualificationError('portable evidence root must be the bundle directory')
    root = Path(bundle_path).parent.resolve(strict=True)

    def resolve_recorded(value: Any) -> Path:
        return _relative_file(root, value)

    rows = bundle.get('reports')
    if not isinstance(rows, list) or len(rows) != len(KINDS):
        raise QualificationError('qualification bundle does not contain nine reports')
    items = {}
    for row in rows:
        if not isinstance(row, dict) or row.get('kind') not in KINDS or row['kind'] in items:
            raise QualificationError('qualification bundle has duplicate or unknown reports')
        paths = tuple(resolve_recorded(row[f'{name}_path']) for name in ('report', 'receipt', 'pointer'))
        for name, path in zip(('report', 'receipt', 'pointer'), paths):
            if sha256_file(path) != row.get(f'{name}_sha256'):
                raise QualificationError(f'qualification {name} changed after bundling')
        items[row['kind']] = paths
    wrapper = resolve_recorded(bundle['sanitizer_wrapper_path'])
    if sha256_file(wrapper) != bundle.get('sanitizer_wrapper_sha256'):
        raise QualificationError('sanitizer wrapper changed after bundling')
    current = assemble_qualification(items, source, candidate_manifest, baseline_manifest,
                                     expected_uuid, wrapper, evidence_root=root)
    if current != bundle:
        raise QualificationError('qualification bundle differs from reconstructed evidence')
    return current


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--candidate-manifest', type=Path, required=True)
    parser.add_argument('--baseline-manifest', type=Path, required=True)
    parser.add_argument('--expected-uuid', required=True)
    parser.add_argument('--sanitizer-wrapper', type=Path, required=True)
    parser.add_argument('--evidence-root', type=Path, required=True,
                        help='immutable staged directory containing every report and the bundle')
    parser.add_argument('--item', action='append', nargs=4, metavar=('KIND','REPORT','RECEIPT','POINTER'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    items = {}
    for kind, report, receipt, pointer in args.item:
        if kind in items:
            raise QualificationError(f'duplicate qualification item: {kind}')
        items[kind] = (Path(report), Path(receipt), Path(pointer))
    if args.output.parent.resolve(strict=True) != args.evidence_root.resolve(strict=True):
        raise QualificationError('the bundle must be written directly in its portable evidence root')
    result = assemble_qualification(items, args.source, args.candidate_manifest,
                                    args.baseline_manifest, args.expected_uuid, args.sanitizer_wrapper,
                                    evidence_root=args.evidence_root)
    # Exclusive create avoids overwriting any existing authority/evidence.
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(result, output, sort_keys=True, indent=2)
        output.write('\n')
        output.flush()
        os.fsync(output.fileno())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
