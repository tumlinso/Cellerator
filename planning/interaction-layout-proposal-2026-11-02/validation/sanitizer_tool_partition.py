#!/usr/bin/env python3
"""Run one fixed Compute Sanitizer tool partition from an admitted GPU job.

The native source archive and its strict sanitizer suite remain unchanged. This
wrapper selects exactly one of the suite's tools, verifies the captured source
and binaries before and after the run, and writes a receipt-ready evidence
partition under the packet output directory. The Scheduler receipt is external
evidence and is deliberately marked pending here.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence


SCHEMA = "cellerator.factorized_interactions.sanitizer_tool_partition.v1"
EXPECTED_TOOLS = ("memcheck", "racecheck", "initcheck", "synccheck")
EXPECTED_CASES = 14
TOOL_HOME_DEFAULT = Path(
    "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/compute-sanitizer")
TOOL_VERSION_PREFIX = "Version 2025.2.1.0 (build 35969825)"
EVIDENCE_KEYS = ("y", "dx", "dz", "dalpha", "dbias", "dgamma")


class PartitionError(RuntimeError):
    """The requested sanitizer partition failed its trust or result checks."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_inventory(root: Path) -> tuple[list[dict[str, Any]], str]:
    """Return a canonical file inventory, rejecting links and special files."""
    if root.is_symlink() or not root.is_dir():
        raise PartitionError(f"expected a regular directory: {root}")
    rows: list[dict[str, Any]] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise PartitionError(f"symbolic link in immutable tree: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise PartitionError(f"special file in immutable tree: {path}")
        rel = path.relative_to(root).as_posix()
        rows.append({"path": rel, "bytes": path.stat().st_size,
                     "sha256": sha256_file(path)})
    rows.sort(key=lambda row: row["path"])
    digest = hashlib.sha256()
    prior = None
    for row in rows:
        name = row["path"]
        if prior is not None and name <= prior:
            raise PartitionError("tree inventory paths are not unique and sorted")
        prior = name
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(row["bytes"].to_bytes(8, "little"))
        digest.update(bytes.fromhex(row["sha256"]))
    return rows, digest.hexdigest()


def source_entries_digest(entries: Any) -> tuple[str, list[dict[str, Any]]]:
    if not isinstance(entries, list) or not entries:
        raise PartitionError("build manifest omitted its source file closure")
    digest = hashlib.sha256()
    normalized: list[dict[str, Any]] = []
    prior = None
    for row in entries:
        if not isinstance(row, dict):
            raise PartitionError("source closure rows must be objects")
        name, size, file_hash = row.get("path"), row.get("bytes"), row.get("sha256")
        if not isinstance(name, str) or not name or "\\" in name:
            raise PartitionError("source closure path is invalid")
        rel = PurePosixPath(name)
        if rel.is_absolute() or ".." in rel.parts or "." in rel.parts or \
                rel.as_posix() != name:
            raise PartitionError("source closure path escapes the source tree")
        if prior is not None and name <= prior:
            raise PartitionError("source closure paths must be unique and sorted")
        prior = name
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise PartitionError("source closure byte count is invalid")
        if not isinstance(file_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", file_hash):
            raise PartitionError("source closure SHA256 is invalid")
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(size.to_bytes(8, "little"))
        digest.update(bytes.fromhex(file_hash))
        normalized.append({"path": name, "bytes": size, "sha256": file_hash})
    return digest.hexdigest(), normalized


def verify_source_tree(source: Path, entries: Any, expected_closure: str) -> tuple[
        list[dict[str, Any]], str, int]:
    if source.is_symlink() or not source.is_dir():
        raise PartitionError("source must be the extracted immutable source directory")
    computed_closure, normalized = source_entries_digest(entries)
    if computed_closure != expected_closure:
        raise PartitionError("source entry list differs from its closure identity")
    actual_rows, _ = tree_inventory(source)
    actual = {row["path"]: row for row in actual_rows}
    expected = {row["path"]: row for row in normalized}
    if actual.keys() != expected.keys():
        missing = sorted(expected.keys() - actual.keys())[:3]
        extra = sorted(actual.keys() - expected.keys())[:3]
        raise PartitionError(f"source tree file set differs (missing={missing}, extra={extra})")
    for name, expected_row in expected.items():
        actual_row = actual[name]
        if actual_row["bytes"] != expected_row["bytes"] or \
                actual_row["sha256"] != expected_row["sha256"]:
            raise PartitionError(f"source file differs from frozen build closure: {name}")
    return actual_rows, computed_closure, len(normalized)


def _regular_inside(root: Path, relative: str, description: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise PartitionError(f"{description} path is missing")
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts or "." in rel.parts or \
            rel.as_posix() != relative:
        raise PartitionError(f"{description} path escapes its build directory")
    candidate = root.joinpath(*rel.parts)
    if candidate.is_symlink() or not candidate.is_file():
        raise PartitionError(f"{description} is not a regular file: {candidate}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve(strict=True)):
        raise PartitionError(f"{description} resolves outside its build directory")
    return resolved


def verify_build(runner: Path, manifest_path: Path, source: Path) -> dict[str, Any]:
    if runner.is_symlink() or manifest_path.is_symlink():
        raise PartitionError("runner and manifest may not be symbolic links")
    manifest_path = manifest_path.resolve(strict=True)
    runner = runner.resolve(strict=True)
    native_root = manifest_path.parent.resolve(strict=True)
    if not runner.is_relative_to(native_root):
        raise PartitionError("baseline runner must be within its immutable native build")
    baseline_bytes = manifest_path.read_bytes()
    baseline = json.loads(baseline_bytes)
    if baseline.get("schema") != "cellerator.factorized_interactions.build.v1":
        raise PartitionError("baseline manifest schema is not recognized")
    if baseline.get("architecture") != "sm_70" or baseline.get("fast_math") is not False:
        raise PartitionError("native binaries must be the fixed non-fast-math SM70 build")
    source_hash, source_rows = source_entries_digest(baseline.get("source_files"))
    if source_hash != baseline.get("source_closure_sha256"):
        raise PartitionError("baseline source manifest closure hash is inconsistent")
    source_rows_actual, _, source_count = verify_source_tree(
        source, source_rows, source_hash)
    candidate_pointer = baseline.get("candidate_runner")
    if not isinstance(candidate_pointer, dict):
        raise PartitionError("baseline manifest lacks candidate runner identity")
    candidate_manifest_path = _regular_inside(
        native_root, candidate_pointer.get("manifest_path"), "candidate manifest")
    candidate_manifest_bytes = candidate_manifest_path.read_bytes()
    if sha256_bytes(candidate_manifest_bytes) != candidate_pointer.get("manifest_sha256"):
        raise PartitionError("candidate manifest hash differs from baseline parent")
    candidate_manifest = json.loads(candidate_manifest_bytes)
    if candidate_manifest.get("schema") != "cellerator.factorized_interactions.build.v1":
        raise PartitionError("candidate manifest schema is not recognized")
    if candidate_manifest.get("architecture") != "sm_70" or \
            candidate_manifest.get("fast_math") is not False:
        raise PartitionError("candidate build must be fixed non-fast-math SM70")
    candidate_source_hash, candidate_rows = source_entries_digest(
        candidate_manifest.get("source_files"))
    if candidate_source_hash != source_hash or \
            candidate_manifest.get("source_closure_sha256") != source_hash or \
            candidate_rows != source_rows:
        raise PartitionError("candidate and baseline source closures differ")
    candidate_runner = _regular_inside(
        native_root, candidate_pointer.get("binary_path"), "candidate runner")
    candidate_manifest_root = candidate_manifest_path.parent.resolve(strict=True)
    candidate_entry = candidate_manifest.get("binaries", {}).get("runner")
    if not isinstance(candidate_entry, dict):
        raise PartitionError("candidate manifest lacks a runner binary")
    expected_candidate = _regular_inside(
        candidate_manifest_root, candidate_entry.get("path"), "manifest candidate runner")
    candidate_hash = sha256_file(candidate_runner)
    if candidate_runner != expected_candidate or \
            candidate_hash != candidate_pointer.get("binary_sha256") or \
            candidate_hash != candidate_entry.get("sha256"):
        raise PartitionError("candidate runner differs from both pinned manifests")
    baseline_entry = baseline.get("binaries", {}).get("runner")
    if not isinstance(baseline_entry, dict):
        raise PartitionError("baseline manifest lacks its runner binary")
    expected_baseline = _regular_inside(native_root, baseline_entry.get("path"),
                                        "manifest baseline runner")
    baseline_hash = sha256_file(runner)
    if runner != expected_baseline or baseline_hash != baseline_entry.get("sha256"):
        raise PartitionError("baseline runner differs from its build manifest")
    if baseline.get("source_closure_sha256") != candidate_manifest.get("source_closure_sha256"):
        raise PartitionError("baseline and candidate closure identifiers differ")
    if manifest_path == candidate_manifest_path:
        raise PartitionError("baseline and candidate manifests must be distinct")

    archive_entries = []
    for name, manifest in (("baseline", baseline), ("candidate", candidate_manifest)):
        entry = manifest.get("binaries", {}).get("factorized_interaction_cuda")
        if not isinstance(entry, dict):
            raise PartitionError(f"{name} manifest lacks the native component archive")
        archive = _regular_inside(manifest_path.parent if name == "baseline" else
                                  candidate_manifest_path.parent,
                                  entry.get("path"), f"{name} component archive")
        actual_hash = sha256_file(archive)
        if archive.stat().st_size != entry.get("bytes") or actual_hash != entry.get("sha256"):
            raise PartitionError(f"{name} component archive differs from its manifest")
        archive_entries.append((archive, actual_hash))
    if archive_entries[0][1] != archive_entries[1][1]:
        raise PartitionError("baseline and candidate use different native component archives")

    native_rows, native_digest = tree_inventory(native_root)
    return {
        "source_closure_sha256": source_hash,
        "source_file_count": source_count,
        "source_tree_sha256": sha256_bytes(json.dumps(
            source_rows_actual, sort_keys=True, separators=(",", ":")).encode("utf-8")),
        "baseline_runner_path": str(runner),
        "runner_binary_sha256": baseline_hash,
        "build_manifest_path": str(manifest_path),
        "build_manifest_sha256": sha256_bytes(baseline_bytes),
        "candidate_runner_path": str(candidate_runner),
        "candidate_runner_binary_sha256": candidate_hash,
        "candidate_build_manifest_path": str(candidate_manifest_path),
        "candidate_build_manifest_sha256": sha256_bytes(candidate_manifest_bytes),
        "component_archive_path": str(archive_entries[0][0]),
        "component_archive_sha256": archive_entries[0][1],
        "shared_library_sha256": archive_entries[0][1],
        "library_kind": "static_archive",
        "native_build_tree_sha256": native_digest,
        "native_build_file_count": len(native_rows),
        "native_build_rows": native_rows,
        "source_rows": source_rows_actual,
    }


def load_trusted_module(name: str, path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise PartitionError(f"trusted module is not a regular file: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PartitionError(f"cannot load trusted module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def artifact_inventory(root: Path) -> list[dict[str, Any]]:
    rows, _ = tree_inventory(root)
    return rows


def validate_suite_report(report: Mapping[str, Any], *, tool: str,
                          context: Mapping[str, Any], build: Mapping[str, Any],
                          tool_binary: Path, tool_version: str) -> None:
    if report.get("schema") != "cellerator.factorized_interactions.sanitizer_suite.v1":
        raise PartitionError("fixed sanitizer suite report schema mismatch")
    if report.get("scheduler_job_id") != context["scheduler_job_id"] or \
            report.get("device_uuid") != context["device_uuid"]:
        raise PartitionError("fixed sanitizer report belongs to a different GPU job")
    if report.get("compute_sanitizer") != str(tool_binary):
        raise PartitionError("fixed suite used a different Compute Sanitizer executable")
    if report.get("compute_sanitizer_version") != tool_version:
        raise PartitionError("fixed suite Compute Sanitizer version differs")
    identity_fields = {
        "source_closure_sha256": build["source_closure_sha256"],
        "build_manifest_sha256": build["build_manifest_sha256"],
        "runner_binary_sha256": build["runner_binary_sha256"],
        "candidate_build_manifest_sha256": build["candidate_build_manifest_sha256"],
        "candidate_runner_binary_sha256": build["candidate_runner_binary_sha256"],
    }
    for key, expected in identity_fields.items():
        if report.get(key) != expected:
            raise PartitionError(f"fixed suite identity mismatch for {key}")
    if report.get("timed") is not False or report.get("case_count") != EXPECTED_CASES or \
            report.get("passed") != EXPECTED_CASES:
        raise PartitionError("fixed suite did not pass exactly 14 untimed cases")
    rows = report.get("results")
    if not isinstance(rows, list) or len(rows) != EXPECTED_CASES:
        raise PartitionError("fixed suite omitted one or more sanitizer cases")
    tools = {row.get("tool") for row in rows if isinstance(row, dict)}
    if tools != {tool}:
        raise PartitionError("fixed suite did not isolate exactly the requested sanitizer tool")
    baseline_pairs = {(route, workload) for route in ("packet", "direct")
                      for workload in (
                          "correctness_ordered_tied_zero",
                          "correctness_context_cancellation_middle",
                          "correctness_context_cancellation_tail",
                          "correctness_fma_product_residual_cancellation",
                          "a4_m256_u32_fanin64_tied",
                          "correctness_empty_mechanisms")}
    candidate_pairs = {
        ("direct_unique_output", "packet_a4_m32_u16"),
        ("minimal_direct", "correctness_ordered_tied_zero"),
    }
    observed_baseline: set[tuple[str, str]] = set()
    observed_candidates: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict) or row.get("exit_code") != 0:
            raise PartitionError("a fixed sanitizer invocation did not exit successfully")
        validation = row.get("validation")
        if not isinstance(validation, dict) or validation.get("validated") is not True or \
                validation.get("timed") is not False:
            raise PartitionError("a sanitizer invocation lacks a successful fixed validation")
        if validation.get("workload") != row.get("workload") or \
                validation.get("route") != row.get("route") or \
                validation.get("scheduler_job_id") != context["scheduler_job_id"] or \
                validation.get("gradient_mode") != "overwrite":
            raise PartitionError("a validation result does not match its fixed case identity")
        evidence = validation.get("contract_evidence")
        if not isinstance(evidence, dict) or any(name not in evidence for name in EVIDENCE_KEYS):
            raise PartitionError("a fixed case did not check output and all repeated VJPs")
        for name in EVIDENCE_KEYS:
            item = evidence[name]
            if not isinstance(item, dict):
                raise PartitionError(f"missing numerical evidence object for {name}")
            ratio = item.get("max_error_to_allowance")
            if not isinstance(ratio, (int, float)) or not 0 <= ratio <= 1:
                raise PartitionError(f"frozen numerical bound failed for {name}")
        native = validation.get("native_execution")
        if not isinstance(native, dict) or \
                native.get("device_uuid") != context["device_uuid"] or \
                native.get("scheduler_job_id") != context["scheduler_job_id"] or \
                native.get("gradient_mode") != "overwrite":
            raise PartitionError("a native case did not retain its admitted GPU identity")
        if "candidate" in row:
            key = (row.get("candidate"), row.get("workload"))
            if key not in candidate_pairs or key in observed_candidates:
                raise PartitionError(
                    f"unexpected or duplicate candidate sanitizer case: candidate={key[0]!r}, "
                    f"workload={key[1]!r}, seen={key in observed_candidates}")
            observed_candidates.add(key)
            if native.get("candidate") != key[0] or native.get("selected_backend") != "direct":
                raise PartitionError(
                    f"candidate sanitizer case selected candidate={native.get('candidate')!r}, "
                    f"selected_backend={native.get('selected_backend')!r}; expected "
                    f"candidate={key[0]!r}, selected_backend='direct'")
            # The candidate CLI mode names the requested layout, but
            # actual_implementation records the selected execution backend.
            # Minimal direct therefore reports "direct"; the unique-output
            # specialization is the only direct candidate with its own name.
            # Keep candidate identity, selected backend, and canonical layout
            # checks separate so the requested specialization remains explicit.
            expected_impl = "direct" if key[0] == "minimal_direct" else key[0]
            if native.get("actual_implementation") != expected_impl or \
                    native.get("structural_layout") != "canonical_direct":
                raise PartitionError(
                    f"candidate sanitizer case {key[0]!r} did not select its intended layout: "
                    f"actual_implementation={native.get('actual_implementation')!r}, "
                    f"structural_layout={native.get('structural_layout')!r}; expected "
                    f"actual_implementation={expected_impl!r}, "
                    "structural_layout='canonical_direct'")
            expected_dtype = {
                "direct_unique_output": ("packet_a4_m32_u16", "fp16", "fp16", "fp32"),
                "minimal_direct": ("correctness_ordered_tied_zero", "fp16", "fp16", "fp16"),
            }[key[0]]
            expected_validation = {
                "workload": expected_dtype[0],
                "activation_storage": expected_dtype[1],
                "parameter_storage": expected_dtype[2],
                "output_storage": expected_dtype[3],
            }
            if any(validation.get(field) != value
                   for field, value in expected_validation.items()):
                observed = {field: validation.get(field) for field in expected_validation}
                raise PartitionError(
                    f"candidate sanitizer dtype/workload metadata mismatch: "
                    f"observed={observed!r}, expected={expected_validation!r}")
            if validation.get("build_provenance", {}).get("source_closure_sha256") != \
                    build["source_closure_sha256"] or \
                    validation.get("build_provenance", {}).get("runner_binary_sha256") != \
                    build["candidate_runner_binary_sha256"] or \
                    validation.get("build_provenance", {}).get("build_manifest_sha256") != \
                    build["candidate_build_manifest_sha256"]:
                raise PartitionError("candidate validation is not bound to candidate build bytes")
        else:
            key = (row.get("route"), row.get("workload"))
            if key not in baseline_pairs or key in observed_baseline:
                raise PartitionError("unexpected or duplicate baseline sanitizer case")
            observed_baseline.add(key)
            if native.get("requested_route") != key[0]:
                raise PartitionError("baseline sanitizer route metadata mismatch")
            expected_backend = {"packet": {"packet", "stable"},
                                "direct": {"direct", "stable"}}[key[0]]
            if native.get("selected_backend") not in expected_backend:
                raise PartitionError("baseline sanitizer selected a backend for another route")
            expected_dtype = {
                "correctness_ordered_tied_zero": ("fp16", "fp16", "fp16"),
                "correctness_context_cancellation_middle": ("fp32", "fp32", "fp32"),
                "correctness_context_cancellation_tail": ("fp32", "fp32", "fp32"),
                "correctness_fma_product_residual_cancellation": ("fp32", "fp32", "fp32"),
                "a4_m256_u32_fanin64_tied": ("fp16", "fp16", "fp16"),
                "correctness_empty_mechanisms": ("fp16", "fp16", "fp16"),
            }[key[1]]
            expected_validation = {
                "activation_storage": expected_dtype[0],
                "parameter_storage": expected_dtype[1],
                "output_storage": expected_dtype[2],
            }
            if any(validation.get(field) != value
                   for field, value in expected_validation.items()):
                raise PartitionError("baseline sanitizer dtype metadata mismatch")
            if validation.get("build_provenance", {}).get("source_closure_sha256") != \
                    build["source_closure_sha256"] or \
                    validation.get("build_provenance", {}).get("runner_binary_sha256") != \
                    build["runner_binary_sha256"] or \
                    validation.get("build_provenance", {}).get("build_manifest_sha256") != \
                    build["build_manifest_sha256"]:
                raise PartitionError("baseline validation is not bound to baseline build bytes")
    if observed_baseline != baseline_pairs or observed_candidates != candidate_pairs:
        raise PartitionError("fixed suite case matrix is incomplete")


def _validate_output_paths(output_root: Path, artifact_dir: Path,
                           output_path: Path) -> tuple[Path, Path, Path]:
    if output_root.is_symlink() or not output_root.is_dir():
        raise PartitionError("packet output root must already be a regular directory")
    root = output_root.resolve(strict=True)
    artifact = artifact_dir.absolute()
    report = output_path.absolute()
    suite_report = root / "fixed-sanitizer-suite.json"
    if report.parent.resolve(strict=False) != root:
        raise PartitionError("partition report must be directly under packet output root")
    for path, label in ((artifact, "artifact directory"), (report, "partition report"),
                        (suite_report, "suite report")):
        if not path.resolve(strict=False).is_relative_to(root):
            raise PartitionError(f"{label} must remain within packet output root")
        if path.resolve(strict=False) == root:
            raise PartitionError(f"{label} must be below packet output root")
        cursor = path
        while cursor != root and cursor != cursor.parent:
            if cursor.is_symlink():
                raise PartitionError(f"symbolic link in {label} path")
            cursor = cursor.parent
    if artifact.exists():
        if artifact.is_symlink() or not artifact.is_dir() or any(artifact.iterdir()):
            raise PartitionError("artifact directory must be fresh and empty")
    else:
        artifact.mkdir(parents=True)
    log_paths = (root / "fixed-sanitizer-suite.stdout.log",
                 root / "fixed-sanitizer-suite.stderr.log")
    if any(path.exists() or path.is_symlink()
           for path in (report, suite_report, *log_paths)):
        raise PartitionError("output report paths must not already exist")
    return root, artifact, report


def execute_partition(*, source: Path, tool: str, runner: Path,
                       build_manifest: Path, expected_uuid: str,
                       output_root: Path, artifact_dir: Path,
                       output_path: Path,
                       environ: Mapping[str, str] | None = None,
                       cgroup_text: str | None = None,
                       run: Callable[..., Any] = subprocess.run,
                       module_loader: Callable[[str, Path], Any] = load_trusted_module
                       ) -> dict[str, Any]:
    if tool not in EXPECTED_TOOLS:
        raise PartitionError(f"unsupported sanitizer tool: {tool}")
    env = dict(os.environ if environ is None else environ)
    if not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
                        expected_uuid):
        raise PartitionError("expected UUID must be an explicit NVIDIA GPU UUID")
    home = TOOL_HOME_DEFAULT.resolve(strict=True)
    configured_home = env.get("NVCOMPILER_COMPUTE_SANITIZER_HOME", "").strip()
    if configured_home != str(home):
        raise PartitionError("NVCOMPILER_COMPUTE_SANITIZER_HOME must select the pinned CUDA 12.9 tool")
    tool_binary = (home / "compute-sanitizer").resolve(strict=True)
    if not tool_binary.is_file() or not os.access(tool_binary, os.X_OK):
        raise PartitionError("pinned Compute Sanitizer executable is unavailable")
    context_helper = source / "bench" / "factorized_interactions" / "experiments" / "in_job_kda.py"
    context_module = module_loader("factorized_interaction_outer_context", context_helper)
    try:
        cg = (Path("/proc/self/cgroup").read_text(encoding="ascii")
              if cgroup_text is None else cgroup_text)
        context_before = context_module.validate_context(env, cg, expected_uuid)
    except Exception as exc:
        raise PartitionError(f"outer Scheduler context was rejected: {exc}") from exc
    out_root, artifact_dir, output_path = _validate_output_paths(
        output_root, artifact_dir, output_path)
    suite_report_path = out_root / "fixed-sanitizer-suite.json"
    build = verify_build(runner, build_manifest, source)
    source_root = source.resolve(strict=True)
    native_root = Path(build["build_manifest_path"]).parent
    source_before = tree_inventory(source_root)
    native_before = tree_inventory(native_root)
    tool_rows_before, tool_tree_before = tree_inventory(home)
    tool_hash_before = sha256_file(tool_binary)
    version_result = run([str(tool_binary), "--version"], cwd=out_root, env=env,
                         text=True, capture_output=True, check=False)
    if version_result.returncode != 0:
        raise PartitionError(f"Compute Sanitizer --version failed: {version_result.stderr}")
    tool_version = ((version_result.stdout or "") + (version_result.stderr or "")).strip()
    if TOOL_VERSION_PREFIX not in tool_version:
        raise PartitionError(f"unexpected Compute Sanitizer version: {tool_version!r}")

    suite_path = source_root / "bench" / "factorized_interactions" / "run_sanitizer_suite.py"
    validator_path = source_root / "bench" / "factorized_interactions" / "validate_runner.py"
    suite = module_loader("factorized_interaction_sanitizer_suite", suite_path)
    if not validator_path.is_file() or validator_path.is_symlink():
        raise PartitionError("trusted validator is not a regular source file")
    if tuple(suite.TOOLS) != EXPECTED_TOOLS:
        raise PartitionError("captured strict sanitizer suite tool list differs from the frozen four")
    suite.TOOLS = (tool,)
    suite.COMPUTE_SANITIZER = str(tool_binary)
    args = [
        "run_sanitizer_suite.py",
        "--runner", str(Path(build["baseline_runner_path"])),
        "--build-manifest", str(Path(build["build_manifest_path"])),
        "--expected-uuid", expected_uuid,
        "--artifact-dir", str(artifact_dir),
        "--output", str(suite_report_path),
    ]
    previous_argv = sys.argv
    suite_stdout, suite_stderr = io.StringIO(), io.StringIO()
    suite_error: str | None = None
    try:
        sys.argv = args
        with contextlib.redirect_stdout(suite_stdout), contextlib.redirect_stderr(suite_stderr):
            return_code = suite.main()
        if return_code != 0:
            raise PartitionError(f"fixed sanitizer suite returned {return_code}")
    except (Exception, SystemExit) as exc:
        suite_error = f"{type(exc).__name__}: {exc}"
    finally:
        sys.argv = previous_argv
    (out_root / "fixed-sanitizer-suite.stdout.log").write_text(
        suite_stdout.getvalue(), encoding="utf-8")
    (out_root / "fixed-sanitizer-suite.stderr.log").write_text(
        suite_stderr.getvalue(), encoding="utf-8")

    errors: list[str] = []
    suite_report: dict[str, Any] | None = None
    if suite_error:
        errors.append(suite_error)
    if not suite_report_path.is_file():
        errors.append("fixed sanitizer suite did not emit its JSON report")
    else:
        try:
            parsed = json.loads(suite_report_path.read_text(encoding="utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError("suite report is not an object")
            suite_report = parsed
            validate_suite_report(parsed, tool=tool, context=context_before,
                                  build=build, tool_binary=tool_binary,
                                  tool_version=tool_version)
        except Exception as exc:
            errors.append(f"suite report rejected: {type(exc).__name__}: {exc}")

    # Recheck all execution inputs after the candidate subprocesses completed.
    try:
        context_after = context_module.validate_context(env, cg, expected_uuid)
        if context_after != context_before:
            errors.append("Scheduler context changed during the sanitizer partition")
    except Exception as exc:
        context_after = None
        errors.append(f"post-run Scheduler context rejected: {exc}")
    source_after = tree_inventory(source_root)
    native_after = tree_inventory(native_root)
    tool_rows_after, tool_tree_after = tree_inventory(home)
    tool_hash_after = sha256_file(tool_binary)
    if source_after != source_before:
        errors.append("captured source tree changed while sanitizer suite ran")
    if native_after != native_before:
        errors.append("compiled native build changed while sanitizer suite ran")
    if tool_rows_after != tool_rows_before or tool_tree_after != tool_tree_before or \
            tool_hash_after != tool_hash_before:
        errors.append("Compute Sanitizer installation changed while suite ran")

    inventory = artifact_inventory(artifact_dir)
    validator_hash = sha256_file(validator_path)
    suite_hash = sha256_file(suite_path)
    wrapper_hash = sha256_file(Path(__file__).resolve(strict=True))
    context_helper_hash = sha256_file(context_helper)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "pass" if not errors else "fail",
        "tool": tool,
        "expected_tools": list(EXPECTED_TOOLS),
        "expected_case_count": EXPECTED_CASES,
        "case_count": len(suite_report.get("results", [])) if suite_report else 0,
        "passed": suite_report.get("passed", 0) if suite_report else 0,
        "timed": False,
        "repeat_vjp_checked": False,
        "source_closure_sha256": build["source_closure_sha256"],
        "source_file_count": build["source_file_count"],
        "source_tree_sha256": build["source_tree_sha256"],
        "runner_binary_sha256": build["runner_binary_sha256"],
        "build_manifest_sha256": build["build_manifest_sha256"],
        "candidate_runner_binary_sha256": build["candidate_runner_binary_sha256"],
        "candidate_build_manifest_sha256": build["candidate_build_manifest_sha256"],
        "component_archive_sha256": build["component_archive_sha256"],
        "shared_library_sha256": build["shared_library_sha256"],
        "library_kind": build["library_kind"],
        "native_build_tree_sha256": build["native_build_tree_sha256"],
        "native_build_file_count": build["native_build_file_count"],
        "validator_sha256": validator_hash,
        "suite_sha256": suite_hash,
        "context_helper_sha256": context_helper_hash,
        "wrapper_sha256": wrapper_hash,
        "compute_sanitizer_home": str(home),
        "compute_sanitizer_path": str(tool_binary),
        "compute_sanitizer_binary_sha256": tool_hash_before,
        "compute_sanitizer_support_tree_sha256": tool_tree_before,
        "compute_sanitizer_support_files": len(tool_rows_before),
        "compute_sanitizer_version": tool_version,
        "scheduler_context": context_before,
        "scheduler_context_after": context_after,
        "outer_receipt_verification": "pending",
        "suite_report_path": str(suite_report_path),
        "suite_report_relative_path": suite_report_path.relative_to(out_root).as_posix(),
        "suite_report_sha256": sha256_file(suite_report_path)
        if suite_report_path.is_file() else None,
        "artifact_relative_path": artifact_dir.relative_to(out_root).as_posix(),
        "artifact_inventory": inventory,
        "suite_stdout_sha256": sha256_file(out_root / "fixed-sanitizer-suite.stdout.log"),
        "suite_stderr_sha256": sha256_file(out_root / "fixed-sanitizer-suite.stderr.log"),
        "suite_report": suite_report,
        "errors": errors,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    if errors:
        raise PartitionError("; ".join(errors))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tool", choices=EXPECTED_TOOLS, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--expected-uuid", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = execute_partition(source=args.source, tool=args.tool,
                                   runner=args.runner,
                                   build_manifest=args.build_manifest,
                                   expected_uuid=args.expected_uuid,
                                   output_root=args.output_root,
                                   artifact_dir=args.artifact_dir,
                                   output_path=args.output)
    except Exception as exc:
        print(f"sanitizer partition failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"schema": SCHEMA, "status": report["status"],
                      "tool": report["tool"], "case_count": report["case_count"],
                      "passed": report["passed"], "timed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
