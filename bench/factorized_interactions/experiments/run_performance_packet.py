#!/usr/bin/env python3
"""Prepare and evaluate one source-bound native performance packet in-job.

This orchestration is intended to run inside an already admitted Autobench /
Device Scheduler allocation. It never submits or requests GPU resources. The
KDA engine workspace, candidate builds, validator artifacts, and conditional
selection record all live below a fresh packet directory in ``--output-root``.
"""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid
from typing import Any, Callable, Mapping

# Candidate snapshots and Autobench captures are immutable inputs. Never let
# imports add __pycache__ files to the source closure being measured.
sys.dont_write_bytecode = True

from common import (
    AdapterError,
    candidate_tree_identity,
    checked_build,
    digest_file,
    digest_json,
    load_variant_map,
    read_json,
    write_json,
)
from in_job_kda import validate_context


INCUMBENT_ID = "strongest-direct"
CHALLENGER_ID = "packet-candidate"
NATIVE_MANIFEST = "native_candidate/native_build/factorized_interactions_build_manifest.json"
BASELINE_MANIFEST = "native_candidate/native_build/factorized_interactions_baseline_build_manifest.json"
SANITIZER_SUITE_RELATIVE = "bench/factorized_interactions/run_sanitizer_suite.py"
BUILDER_RELATIVE = "bench/factorized_interactions/candidate_integration/build_candidate_runner.py"
IN_JOB_ENTRYPOINT = Path(__file__).resolve().with_name("in_job_kda.py")
KDA_CANDIDATES_FILE = "candidates.jsonl"
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_CONTEXT_KEYS = (
    "scheduler_job_id", "scheduler_attempt_id", "device_uuid", "cgroup", "unit_name",
)
_REGISTERED_SANITIZER_HOME = Path(
    "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/compute-sanitizer"
)
_EXPECTED_SANITIZER_VERSION = "2025.2.1.0"


class PacketError(RuntimeError):
    """The packet cannot safely produce source-bound measurement evidence."""


@dataclass(frozen=True)
class CandidateBuild:
    candidate_id: str
    runner: Path
    manifest: Path
    runner_sha256: str
    library_sha256: str
    source_snapshot_sha256: str
    build_reference: dict[str, Any]
    baseline_runner: Path
    baseline_manifest: Path
    baseline_runner_sha256: str
    baseline_manifest_sha256: str


@dataclass(frozen=True)
class SanitizerConfiguration:
    home: Path
    binary: Path
    binary_sha256: str
    support_tree_sha256: str
    expected_version: str


def _read_sanitizer_registration(suite_path: Path) -> SanitizerConfiguration:
    registered_home = os.environ.get("NVCOMPILER_COMPUTE_SANITIZER_HOME")
    if registered_home != str(_REGISTERED_SANITIZER_HOME):
        raise PacketError(
            "inherited NVCOMPILER_COMPUTE_SANITIZER_HOME differs from the oracle-registered path"
        )
    if suite_path.is_symlink() or not suite_path.is_file():
        raise PacketError("captured strict sanitizer suite is missing or non-regular")
    spec = importlib.util.spec_from_file_location("_cellerator_strict_sanitizer_suite", suite_path)
    if spec is None or spec.loader is None:
        raise PacketError("cannot load the captured strict sanitizer suite configuration")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    binary_value = getattr(module, "COMPUTE_SANITIZER", None)
    if not isinstance(binary_value, str) or not Path(binary_value).is_absolute():
        raise PacketError("strict sanitizer suite does not declare an absolute compute-sanitizer path")
    binary = Path(binary_value)
    if binary.is_symlink() or not binary.is_file():
        raise PacketError("registered compute-sanitizer executable is missing or non-regular")
    home = Path(registered_home)
    if home.is_symlink() or not home.is_dir():
        raise PacketError("registered compute-sanitizer support home is missing or non-regular")
    try:
        support_sha256, _ = candidate_tree_identity(home)
    except Exception as exc:
        raise PacketError(f"cannot identify registered sanitizer support files: {exc}") from exc
    return SanitizerConfiguration(
        home=home.resolve(strict=True), binary=binary.resolve(strict=True),
        binary_sha256=digest_file(binary), support_tree_sha256=support_sha256,
        expected_version=_EXPECTED_SANITIZER_VERSION,
    )


def _read_outer_context(expected_uuid: str) -> dict[str, Any]:
    try:
        cgroup = Path("/proc/self/cgroup").read_text(encoding="ascii")
        return validate_context(os.environ, cgroup, expected_uuid)
    except Exception as exc:
        raise PacketError(f"inherited Autobench/Scheduler allocation is not valid: {exc}") from exc


def _same_context(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    return all(before.get(key) == after.get(key) for key in _CONTEXT_KEYS)


def _require_context_stable(
    expected_uuid: str,
    initial: Mapping[str, Any],
    context_reader: Callable[[str], dict[str, Any]],
    stage: str,
) -> dict[str, Any]:
    current = context_reader(expected_uuid)
    if not _same_context(initial, current):
        raise PacketError(f"outer Scheduler job/attempt/UUID/cgroup changed during {stage}")
    return current


def _valid_id(value: str, label: str) -> None:
    if not _ID_PATTERN.fullmatch(value) or value in {".", ".."}:
        raise PacketError(f"{label} must be a simple KDA identifier")


def _build_variant_map(source: Path, nvcc: Path, task_id: str) -> dict[str, Any]:
    builder = source / BUILDER_RELATIVE
    if builder.is_symlink() or not builder.is_file():
        raise PacketError(f"immutable candidate source lacks the native builder: {builder}")
    return {
        "schema_version": 1,
        "comparison": {
            "incumbent": INCUMBENT_ID,
            "challenger": CHALLENGER_ID,
            "strongest_direct_control": INCUMBENT_ID,
            "same_source_runner_required": True,
            "experiment_group": task_id.replace("-", "_"),
        },
        "builder": {
            "trusted_script": str(builder.resolve(strict=True)),
            "snapshot_script_relative_path": BUILDER_RELATIVE,
            "argv": [
                "--source", "{source}", "--build", "{build}",
                "--parallel", "2", "--nvcc", str(nvcc.resolve(strict=True)),
            ],
            "manifest_relative_path": NATIVE_MANIFEST,
        },
        "runner_iterations": 30,
        "variants": {
            INCUMBENT_ID: {
                "enabled": True,
                "route": "direct",
                "candidate_argument": "minimal_direct",
                "allowed_backends": ["direct", "stable"],
                "allowed_implementations": ["direct", "stable", "direct_unique_output"],
                "allowed_layouts": ["canonical_direct", "full_packet"],
            },
            CHALLENGER_ID: {
                "enabled": True,
                "route": "packet",
                "candidate_argument": "packet",
                "allowed_backends": ["packet", "stable"],
                "allowed_implementations": ["packet", "stable"],
                "allowed_layouts": ["full_packet"],
                "win_backend": "packet",
                "win_layout": "full_packet",
            },
        },
    }


def _write_candidate_manifest(
    source: Path,
    source_manifest: Path,
    destination: Path,
    manifest_factory: Callable[..., dict[str, Any]] | None,
) -> dict[str, Any]:
    if manifest_factory is None:
        from candidate_manifest import make_manifest

        manifest_factory = make_manifest
    manifest = manifest_factory(source, source_manifest, [])
    write_json(destination, manifest, exclusive=True)
    return manifest


def _registered_candidates(
    workspace: Path,
    candidate_ids: list[str],
    expected_manifest_sha256: str,
) -> dict[str, dict[str, Any]]:
    path = workspace / KDA_CANDIDATES_FILE
    if path.is_symlink() or not path.is_file():
        raise PacketError("KDA registration did not create its candidate registry")
    wanted = set(candidate_ids)
    rows: dict[str, dict[str, Any]] = {}
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("record_type") != "candidate":
                continue
            candidate_id = row.get("candidate_id")
            if candidate_id in wanted:
                if candidate_id in rows:
                    raise PacketError(f"candidate {candidate_id!r} appears more than once in KDA registry")
                rows[candidate_id] = row
    except (OSError, json.JSONDecodeError) as exc:
        raise PacketError(f"cannot inspect KDA candidate registry: {exc}") from exc
    if set(rows) != wanted:
        raise PacketError(f"KDA candidate registry differs from requested pair: {sorted(rows)}")
    for candidate_id, row in rows.items():
        if not isinstance(row.get("snapshot_sha256"), str) or \
                not isinstance(row.get("snapshot_dir"), str):
            raise PacketError(f"KDA candidate {candidate_id!r} omitted immutable snapshot identity")
        if row.get("manifest_sha256") != expected_manifest_sha256:
            raise PacketError(f"KDA candidate {candidate_id!r} registered a different source manifest")
    return rows


def _native_library_sha256(manifest_path: Path) -> str:
    manifest = read_json(manifest_path)
    binaries = manifest.get("binaries")
    library = binaries.get("factorized_interaction_cuda") if isinstance(binaries, dict) else None
    relative = library.get("path") if isinstance(library, dict) else None
    if not isinstance(relative, str):
        raise PacketError("native build manifest omitted the linked CUDA library")
    library_path = manifest_path.parent / relative
    if library_path.is_symlink() or not library_path.is_file():
        raise PacketError("native CUDA library is missing or non-regular")
    return digest_file(library_path)


def _resolve_baseline_build(
    build_record: Mapping[str, Any],
    candidate_manifest_path: Path,
    candidate_runner_path: Path,
    candidate_runner_sha256: str,
    expected_source_closure_sha256: str,
) -> tuple[Path, Path, str, str]:
    build_root = Path(build_record.get("directory", "")).resolve(strict=True)
    if not build_root.is_dir():
        raise PacketError("KDA build directory is missing")
    baseline_manifest_path = build_root / BASELINE_MANIFEST
    if baseline_manifest_path.is_symlink() or not baseline_manifest_path.is_file():
        raise PacketError("KDA artifact set omitted the strict baseline build manifest")
    baseline_manifest_path = baseline_manifest_path.resolve(strict=True)
    baseline_manifest = read_json(baseline_manifest_path)
    candidate_manifest = read_json(candidate_manifest_path)
    if baseline_manifest.get("schema") != "cellerator.factorized_interactions.build.v1" or \
            baseline_manifest.get("architecture") != "sm_70" or \
            baseline_manifest.get("fast_math") is not False:
        raise PacketError("baseline build manifest is not the fixed non-fast-math SM70 artifact")
    if baseline_manifest.get("source_closure_sha256") != expected_source_closure_sha256:
        raise PacketError("baseline build manifest differs from the captured source closure")
    baseline_files = baseline_manifest.get("source_files")
    candidate_files = candidate_manifest.get("source_files")
    if not isinstance(baseline_files, list) or baseline_files != candidate_files:
        raise PacketError("baseline and candidate build manifests enumerate different source closures")

    binaries = baseline_manifest.get("binaries")
    baseline_record = binaries.get("runner") if isinstance(binaries, dict) else None
    baseline_library = binaries.get("factorized_interaction_cuda") if isinstance(binaries, dict) else None
    candidate_link = baseline_manifest.get("candidate_runner")
    if not isinstance(baseline_record, dict) or not isinstance(baseline_record.get("path"), str) or \
            not isinstance(baseline_library, dict) or not isinstance(baseline_library.get("path"), str) or \
            not isinstance(candidate_link, dict):
        raise PacketError("baseline manifest omitted runner/library/candidate cross-reference")
    baseline_runner = (baseline_manifest_path.parent / baseline_record["path"]).resolve(strict=True)
    library_path = (baseline_manifest_path.parent / baseline_library["path"]).resolve(strict=True)
    if not baseline_runner.is_relative_to(build_root) or not baseline_runner.is_file() or \
            not library_path.is_relative_to(build_root) or not library_path.is_file():
        raise PacketError("baseline runner or CUDA library escapes the KDA build directory")
    baseline_runner_sha256 = digest_file(baseline_runner)
    library_sha256 = digest_file(library_path)
    if baseline_record.get("sha256") != baseline_runner_sha256 or \
            baseline_library.get("sha256") != library_sha256:
        raise PacketError("baseline runner or CUDA library differs from its manifest SHA256")

    linked_manifest_path = (baseline_manifest_path.parent /
                            str(candidate_link.get("manifest_path", ""))).resolve(strict=True)
    linked_runner_path = (baseline_manifest_path.parent /
                          str(candidate_link.get("binary_path", ""))).resolve(strict=True)
    if linked_manifest_path != candidate_manifest_path.resolve(strict=True) or \
            linked_runner_path != candidate_runner_path.resolve(strict=True) or \
            candidate_link.get("manifest_sha256") != digest_file(candidate_manifest_path) or \
            candidate_link.get("binary_sha256") != candidate_runner_sha256:
        raise PacketError("baseline manifest candidate_runner link differs from the prepared candidate artifacts")
    candidate_library = candidate_manifest.get("binaries", {}).get("factorized_interaction_cuda")
    if not isinstance(candidate_library, dict) or \
            candidate_library.get("sha256") != library_sha256 or \
            candidate_library.get("path") != baseline_library.get("path"):
        raise PacketError("baseline and candidate manifests do not share the exact native CUDA library")

    artifact_rows = build_record.get("artifacts")
    if not isinstance(artifact_rows, list):
        raise PacketError("KDA candidate build omitted its source-bound artifact list")
    artifact_hashes = {row.get("path"): row.get("sha256") for row in artifact_rows
                       if isinstance(row, dict)}
    required_paths = {
        BASELINE_MANIFEST: digest_file(baseline_manifest_path),
        "native_candidate/" + candidate_manifest_path.relative_to(
            build_root / "native_candidate").as_posix(): digest_file(candidate_manifest_path),
        "native_candidate/" + baseline_runner.relative_to(
            build_root / "native_candidate").as_posix(): baseline_runner_sha256,
        "native_candidate/" + candidate_runner_path.relative_to(
            build_root / "native_candidate").as_posix(): candidate_runner_sha256,
        "native_candidate/" + library_path.relative_to(
            build_root / "native_candidate").as_posix(): library_sha256,
    }
    for relative, expected_sha256 in required_paths.items():
        if artifact_hashes.get(relative) != expected_sha256:
            raise PacketError(f"KDA artifact list does not bind required baseline/candidate artifact {relative}")
    return (baseline_runner, baseline_manifest_path,
            baseline_runner_sha256, digest_file(baseline_manifest_path))


def _prepared_builds(
    workspace: Path,
    prepare_response: Mapping[str, Any],
    variant_map: Mapping[str, Any],
    candidate_ids: list[str],
    expected_source_closure_sha256: str,
    expected_candidate_manifest_sha256: str,
    build_checker: Callable[..., tuple[Path, Path, str]],
) -> dict[str, CandidateBuild]:
    results = prepare_response.get("results")
    if not isinstance(results, list):
        raise PacketError("KDA prepare response omitted candidate build results")
    result_by_id = {
        item.get("candidate_id"): item
        for item in results if isinstance(item, dict) and isinstance(item.get("candidate_id"), str)
    }
    if set(result_by_id) != set(candidate_ids):
        raise PacketError("KDA prepare response did not cover the exact candidate pair")
    if any(result_by_id[candidate_id].get("status") != "ok" for candidate_id in candidate_ids):
        raise PacketError("KDA did not successfully prepare both candidate snapshots")
    registered = _registered_candidates(
        workspace, candidate_ids, expected_candidate_manifest_sha256,
    )
    prepared: dict[str, CandidateBuild] = {}
    for candidate_id in candidate_ids:
        candidate_record = registered[candidate_id]
        build_record = result_by_id[candidate_id]
        variant = variant_map["variants"][candidate_id]
        try:
            runner, manifest_path, runner_sha256 = build_checker(
                candidate_record, build_record, variant, candidate_record,
            )
        except Exception as exc:
            raise PacketError(f"KDA prepared build failed identity validation for {candidate_id}: {exc}") from exc
        native_manifest = read_json(manifest_path)
        source_snapshot_sha256 = candidate_record["snapshot_sha256"]
        if native_manifest.get("source_closure_sha256") != expected_source_closure_sha256:
            raise PacketError(
                f"KDA prepared native build for {candidate_id} does not match the captured source closure"
            )
        (baseline_runner, baseline_manifest_path, baseline_runner_sha256,
         baseline_manifest_sha256) = _resolve_baseline_build(
            build_record, manifest_path, runner, runner_sha256,
            expected_source_closure_sha256,
        )
        prepared[candidate_id] = CandidateBuild(
            candidate_id=candidate_id,
            runner=runner,
            manifest=manifest_path,
            runner_sha256=runner_sha256,
            library_sha256=_native_library_sha256(manifest_path),
            source_snapshot_sha256=source_snapshot_sha256,
            build_reference=dict(build_record),
            baseline_runner=baseline_runner,
            baseline_manifest=baseline_manifest_path,
            baseline_runner_sha256=baseline_runner_sha256,
            baseline_manifest_sha256=baseline_manifest_sha256,
        )
        if native_manifest.get("schema") != "cellerator.factorized_interactions.build.v1":
            raise PacketError(f"KDA build for {candidate_id} has an unsupported native manifest")
    if len({item.source_snapshot_sha256 for item in prepared.values()}) != 1:
        raise PacketError("compared KDA candidates do not share one exact source snapshot")
    return prepared


def _qualification_groups(
    prepared: Mapping[str, CandidateBuild],
) -> list[tuple[tuple[str, str], list[CandidateBuild]]]:
    """Deduplicate only byte-identical runner AND linked-library runtimes."""
    grouped: dict[tuple[str, str], list[CandidateBuild]] = {}
    for build in prepared.values():
        grouped.setdefault((build.runner_sha256, build.library_sha256), []).append(build)
    return sorted(grouped.items(), key=lambda row: row[0])


def _sanitizer_groups(
    prepared: Mapping[str, CandidateBuild],
) -> list[tuple[tuple[str, str, str, str, str], list[CandidateBuild]]]:
    """Deduplicate strict qualification only when all five artifacts match."""
    grouped: dict[tuple[str, str, str, str, str], list[CandidateBuild]] = {}
    for build in prepared.values():
        key = (
            build.baseline_runner_sha256,
            build.baseline_manifest_sha256,
            build.runner_sha256,
            digest_file(build.manifest),
            build.library_sha256,
        )
        grouped.setdefault(key, []).append(build)
    return sorted(grouped.items(), key=lambda row: row[0])


def _validate_sanitizer_report(
    report: Mapping[str, Any],
    *,
    build: CandidateBuild,
    expected_source_closure_sha256: str,
    expected_uuid: str,
    scheduler_job_id: str,
    sanitizer: SanitizerConfiguration,
) -> None:
    expected = {
        "schema": "cellerator.factorized_interactions.sanitizer_suite.v1",
        "scheduler_job_id": scheduler_job_id,
        "device_uuid": expected_uuid,
        "build_manifest_sha256": build.baseline_manifest_sha256,
        "source_closure_sha256": expected_source_closure_sha256,
        "runner_binary_sha256": build.baseline_runner_sha256,
        "candidate_build_manifest_sha256": digest_file(build.manifest),
        "candidate_runner_binary_sha256": build.runner_sha256,
        "timed": False,
        "case_count": 56,
        "passed": 56,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise PacketError(f"strict sanitizer report identity/count mismatch for {key}")
    if report.get("compute_sanitizer") != str(sanitizer.binary):
        raise PacketError("strict sanitizer report used a different compute-sanitizer executable")
    version = report.get("compute_sanitizer_version")
    if not isinstance(version, str) or sanitizer.expected_version not in version:
        raise PacketError("strict sanitizer report omitted the registered compute-sanitizer version")


def _add_strict_sanitizer_trusted_inputs(
    task: dict[str, Any],
    task_path: Path,
    sanitizer_suite: Path,
    sanitizer: SanitizerConfiguration,
    kda_root: Path,
    task_loader: Callable[[Path, Path], Any] | None,
) -> None:
    trusted = task.get("trusted_inputs")
    if not isinstance(trusted, list):
        raise PacketError("prepared KDA task omitted trusted inputs")
    additions = [
        {"id": "strict_sanitizer_suite", "path": str(sanitizer_suite), "role": "validator"},
        {"id": "compute_sanitizer_binary", "path": str(sanitizer.binary), "role": "toolchain"},
        {"id": "compute_sanitizer_support_files", "path": str(sanitizer.home), "role": "toolchain"},
    ]
    by_id = {item.get("id"): item for item in trusted if isinstance(item, dict)}
    for item in additions:
        existing = by_id.get(item["id"])
        if existing is not None:
            if existing.get("path") != item["path"] or existing.get("role") != item["role"]:
                raise PacketError(f"KDA task has a conflicting trusted input {item['id']}")
            continue
        trusted.append(item)
        by_id[item["id"]] = item
    write_json(task_path, task)
    if task_loader is None:
        sys.path.insert(0, str(kda_root))
        from kda_experiments.contracts import load_task

        load_task(task_path)
    else:
        task_loader(task_path, kda_root)


def _run_strict_sanitizer_groups(
    *,
    prepared: Mapping[str, CandidateBuild],
    source_closure_sha256: str,
    expected_uuid: str,
    scheduler_job_id: str,
    sanitizer_suite: Path,
    sanitizer_suite_sha256: str,
    sanitizer: SanitizerConfiguration,
    packet_dir: Path,
    kda_root: Path,
    command_env: Mapping[str, str],
    log_root: Path,
    max_seconds: float,
    context_reader: Callable[[str], dict[str, Any]],
    initial_context: Mapping[str, Any],
    invoke: Callable[..., subprocess.CompletedProcess],
) -> dict[str, Any]:
    evidence_root = packet_dir / "strict-sanitizer"
    evidence_root.mkdir(exist_ok=False)
    groups: list[dict[str, Any]] = []
    for identities, builds in _sanitizer_groups(prepared):
        baseline_runner_sha, baseline_manifest_sha, candidate_runner_sha, candidate_manifest_sha, library_sha = identities
        representative = builds[0]
        group_id = hashlib.sha256(":".join(identities).encode("ascii")).hexdigest()[:16]
        artifact_dir = evidence_root / "artifacts" / group_id
        report_path = evidence_root / f"{group_id}.json"
        label = f"strict-sanitizer-{group_id}"
        argv = [
            sys.executable, str(sanitizer_suite),
            "--runner", str(representative.baseline_runner),
            "--build-manifest", str(representative.baseline_manifest),
            "--expected-uuid", expected_uuid,
            "--artifact-dir", str(artifact_dir),
            "--output", str(report_path),
        ]
        _require_context_stable(
            expected_uuid, initial_context, context_reader,
            f"before strict sanitizer group {group_id}",
        )
        result = _invoke_logged(
            label=label, argv=argv, cwd=kda_root, env=command_env,
            log_root=log_root, invoke=invoke, timeout=max_seconds,
        )
        try:
            summary = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise PacketError(f"strict sanitizer group {group_id} did not return JSON") from exc
        if not isinstance(summary, dict) or summary.get("suite") != \
                "cellerator.factorized_interactions.sanitizer_suite.v1" or \
                summary.get("case_count") != 56 or summary.get("passed") != 56 or \
                summary.get("timed") is not False:
            raise PacketError(f"strict sanitizer group {group_id} did not pass all 56 untimed cases")
        if report_path.is_symlink() or not report_path.is_file():
            raise PacketError(f"strict sanitizer group {group_id} did not write its report")
        report = read_json(report_path)
        _validate_sanitizer_report(
            report, build=representative,
            expected_source_closure_sha256=source_closure_sha256,
            expected_uuid=expected_uuid, scheduler_job_id=scheduler_job_id,
            sanitizer=sanitizer,
        )
        _require_context_stable(
            expected_uuid, initial_context, context_reader,
            f"after strict sanitizer group {group_id}",
        )
        groups.append({
            "group_id": group_id,
            "candidate_ids": [build.candidate_id for build in builds],
            "baseline_runner_sha256": baseline_runner_sha,
            "baseline_build_manifest_sha256": baseline_manifest_sha,
            "candidate_runner_sha256": candidate_runner_sha,
            "candidate_build_manifest_sha256": candidate_manifest_sha,
            "native_library_sha256": library_sha,
            "report_path": str(report_path),
            "report_sha256": digest_file(report_path),
            "artifact_dir": str(artifact_dir),
            "command_log_dir": str(log_root / label),
            "case_count": 56,
            "passed": 56,
            "timed": False,
            "status": "pass",
        })

    if not groups:
        raise PacketError("strict sanitizer qualification found no prepared runtime groups")
    index = {
        "schema": "cellerator.factorized_interactions.strict_sanitizer_qualification.v1",
        "status": "pass",
        "suite_path": str(sanitizer_suite),
        "suite_sha256": sanitizer_suite_sha256,
        "compute_sanitizer_path": str(sanitizer.binary),
        "compute_sanitizer_sha256": sanitizer.binary_sha256,
        "inherited_compute_sanitizer_home_selector": command_env.get(
            "NVCOMPILER_COMPUTE_SANITIZER_HOME"),
        "compute_sanitizer_support_home": str(sanitizer.home),
        "compute_sanitizer_support_tree_sha256": sanitizer.support_tree_sha256,
        "expected_compute_sanitizer_version": sanitizer.expected_version,
        "scheduler_job_id": scheduler_job_id,
        "device_uuid": expected_uuid,
        "deduplication": "only when baseline runner, baseline manifest, candidate runner, candidate manifest, and shared CUDA library SHA256 identities all match",
        "groups": groups,
    }
    index_path = evidence_root / "index.json"
    write_json(index_path, index, exclusive=True)
    return {"path": index_path, "sha256": digest_file(index_path), "index": index}


def _read_kda_response(stdout: str, label: str) -> dict[str, Any]:
    try:
        response = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise PacketError(f"{label} did not return one JSON response: {exc}") from exc
    if not isinstance(response, dict) or response.get("status") != "ok":
        raise PacketError(f"{label} did not complete successfully")
    return response


def _invoke_logged(
    *,
    label: str,
    argv: list[str],
    cwd: Path,
    env: Mapping[str, str],
    log_root: Path,
    invoke: Callable[..., subprocess.CompletedProcess],
    timeout: float,
) -> subprocess.CompletedProcess:
    log_dir = log_root / label
    log_dir.mkdir(parents=True, exist_ok=False)
    write_json(log_dir / "command.json", {"argv": argv, "cwd": str(cwd)}, exclusive=True)
    result = invoke(
        argv, cwd=cwd, env=dict(env), capture_output=True, text=True,
        check=False, timeout=timeout,
    )
    (log_dir / "stdout.log").write_text(result.stdout or "", encoding="utf-8")
    (log_dir / "stderr.log").write_text(result.stderr or "", encoding="utf-8")
    write_json(log_dir / "exit.json", {"returncode": result.returncode}, exclusive=True)
    if result.returncode != 0:
        raise PacketError(f"{label} exited with status {result.returncode}; see {log_dir}")
    return result


def _build_env(kda_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(kda_root) + (os.pathsep + existing if existing else "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _verify_unchanged_inputs(
    source: Path,
    source_manifest: Path,
    source_tree_sha256: str,
    source_manifest_sha256: str,
    site_policy: Path,
    site_policy_sha256: str,
    nvcc: Path,
    nvcc_sha256: str,
    sanitizer_suite: Path,
    sanitizer_suite_sha256: str,
    sanitizer: SanitizerConfiguration,
) -> None:
    source_tree_hash, _ = candidate_tree_identity(source)
    if source_tree_hash != source_tree_sha256:
        raise PacketError("immutable candidate source changed during packet execution")
    if digest_file(source_manifest) != source_manifest_sha256:
        raise PacketError("source closure manifest changed during packet execution")
    if digest_file(site_policy) != site_policy_sha256 or digest_file(nvcc) != nvcc_sha256:
        raise PacketError("toolchain or site policy changed during packet execution")
    if digest_file(sanitizer_suite) != sanitizer_suite_sha256 or \
            digest_file(sanitizer.binary) != sanitizer.binary_sha256 or \
            candidate_tree_identity(sanitizer.home)[0] != sanitizer.support_tree_sha256:
        raise PacketError("strict sanitizer suite, executable, or support files changed during packet execution")


def run_packet(
    *,
    source: Path,
    source_manifest: Path,
    expected_uuid: str,
    output_root: Path,
    site_policy: Path,
    nvcc: Path,
    kda_root: Path,
    task_id: str,
    candidate_win_cases: list[str],
    max_seconds: float,
    context_reader: Callable[[str], dict[str, Any]] = _read_outer_context,
    invoke: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    manifest_factory: Callable[..., dict[str, Any]] | None = None,
    task_factory: Callable[..., dict[str, Any]] | None = None,
    registration: Callable[..., dict[str, Any]] | None = None,
    build_checker: Callable[..., tuple[Path, Path, str]] = checked_build,
    dtype_runner: Callable[..., dict[str, Any]] | None = None,
    selection_runner: Callable[..., dict[str, Any]] | None = None,
    sanitizer_config_reader: Callable[[Path], SanitizerConfiguration] = _read_sanitizer_registration,
    task_loader: Callable[[Path, Path], Any] | None = None,
    sanitizer_runner: Callable[..., subprocess.CompletedProcess] | None = None,
) -> dict[str, Any]:
    _valid_id(task_id, "task ID")
    if not candidate_win_cases or len(candidate_win_cases) != len(set(candidate_win_cases)):
        raise PacketError("at least one unique --candidate-win-case is required")
    for case_id in candidate_win_cases:
        _valid_id(case_id, "candidate-win case ID")
    if not math.isfinite(max_seconds) or max_seconds <= 0 or max_seconds > 86400:
        raise PacketError("--max-seconds must be finite, positive, and no greater than one day")

    context_before = context_reader(expected_uuid)
    source = Path(source).expanduser().absolute()
    source_manifest = Path(source_manifest).expanduser().absolute()
    site_policy = Path(site_policy).expanduser().absolute()
    nvcc = Path(nvcc).expanduser().absolute()
    kda_root = Path(kda_root).expanduser().absolute()
    output_root = Path(output_root).expanduser().absolute()
    if any(path.is_symlink() for path in
           (source, source_manifest, site_policy, nvcc, kda_root, output_root)):
        raise PacketError("source, inputs, toolchain, KDA root, and managed output must not be symlinks")
    source = source.resolve(strict=True)
    source_manifest = source_manifest.resolve(strict=True)
    site_policy = site_policy.resolve(strict=True)
    nvcc = nvcc.resolve(strict=True)
    kda_root = kda_root.resolve(strict=True)
    output_root = output_root.resolve(strict=True)
    for path, label in ((source_manifest, "source manifest"), (site_policy, "site policy"),
                        (nvcc, "nvcc")):
        if path.is_symlink() or not path.is_file():
            raise PacketError(f"{label} must be a regular non-symlink file")
    if not source.is_dir() or not output_root.is_dir() or not kda_root.is_dir():
        raise PacketError("source and managed output root must be existing non-symlink directories")
    source_tree_sha256, source_entries = candidate_tree_identity(source)
    source_manifest_sha256 = digest_file(source_manifest)
    site_policy_sha256 = digest_file(site_policy)
    nvcc_sha256 = digest_file(nvcc)
    sanitizer_suite = source / SANITIZER_SUITE_RELATIVE
    sanitizer = sanitizer_config_reader(sanitizer_suite)
    sanitizer_suite_sha256 = digest_file(sanitizer_suite)
    source_manifest_payload = read_json(source_manifest)
    expected_closure_hash = source_manifest_payload.get("source_closure_sha256")
    if not isinstance(expected_closure_hash, str) or len(expected_closure_hash) != 64:
        raise PacketError("source manifest omitted its exact source_closure_sha256")
    packet_dir = output_root / f"factorized-{task_id}"
    if packet_dir.is_relative_to(source) or source.is_relative_to(packet_dir):
        raise PacketError("source and packet output paths must be disjoint")
    packet_dir.mkdir(exist_ok=False)
    workspace = packet_dir / "kda-workspace"
    workspace.mkdir()
    log_root = packet_dir / "commands"
    log_root.mkdir()

    context_sha256 = digest_json(dict(context_before))
    driver_path = Path(__file__).resolve(strict=True)
    packet_record: dict[str, Any] = {
        "schema": "cellerator.factorized_interactions.performance_packet.v1",
        "status": "preparing",
        "task_id": task_id,
        "expected_device_uuid": expected_uuid,
        "outer_execution_context": dict(context_before),
        "outer_execution_context_sha256": context_sha256,
        "source": {
            "root": str(source), "source_tree_sha256": source_tree_sha256,
            "closure_sha256": expected_closure_hash,
            "closure_manifest_path": str(source_manifest),
            "closure_manifest_sha256": source_manifest_sha256,
            "file_count": len(source_entries),
            "packet_driver_sha256": digest_file(driver_path),
        },
        "toolchain": {
            "nvcc_path": str(nvcc), "nvcc_sha256": nvcc_sha256,
            "inherited_compute_sanitizer_home_selector": os.environ.get(
                "NVCOMPILER_COMPUTE_SANITIZER_HOME"),
            "compute_sanitizer_path": str(sanitizer.binary),
            "compute_sanitizer_sha256": sanitizer.binary_sha256,
            "compute_sanitizer_support_home": str(sanitizer.home),
            "compute_sanitizer_support_tree_sha256": sanitizer.support_tree_sha256,
            "strict_sanitizer_suite_path": str(sanitizer_suite),
            "strict_sanitizer_suite_sha256": sanitizer_suite_sha256,
            "strict_sanitizer_expected_version": sanitizer.expected_version,
            "site_policy_path": str(site_policy),
            "site_policy_sha256": site_policy_sha256,
            "kda_root": str(kda_root), "python": sys.executable,
            "max_seconds_semantics": "KDA task budget and per-process timeout; outer Scheduler duration must also cover registration, CPU preparation, dtype qualification, and the full KDA run",
        },
        "storage_estimate": {
            "minimum_paired_raw_samples_bytes": 1_439_344_284,
            "basis": "13 performance cases x 14 paired exploration/confirmation blocks x 3 direct-control/direct-packet/descriptive-direct arms; excludes build trees, correctness oracle, dtype matrix, caches, and logs",
            "evidence_retention": "all KDA attempt records, per-validator stdout/stderr, dtype reports and caches remain under this packet output; no pruning",
        },
    }
    write_json(packet_dir / "packet.json", packet_record, exclusive=True)

    try:
        variant_map = _build_variant_map(source, nvcc, task_id)
        variant_map_path = packet_dir / "variant-map.json"
        write_json(variant_map_path, variant_map, exclusive=True)
        variant_map = load_variant_map(variant_map_path)
        candidate_manifest_path = packet_dir / "candidate-manifest.json"
        candidate_manifest = _write_candidate_manifest(
            source, source_manifest, candidate_manifest_path, manifest_factory,
        )
        if candidate_manifest["provenance"]["kda_source_tree_sha256"] != source_tree_sha256:
            raise PacketError("candidate manifest and source tree identities disagree")
        if candidate_manifest["provenance"]["source_closure_sha256"] != expected_closure_hash:
            raise PacketError("candidate manifest does not retain the captured closure identity")
        _require_context_stable(expected_uuid, context_before, context_reader, "manifest preparation")

        if task_factory is None:
            from prepare_task import make_task

            task_factory = make_task
        task_path = packet_dir / "task.json"
        task = task_factory(
            output=task_path, variant_map_path=variant_map_path,
            site_policy_path=site_policy, nvcc=nvcc,
            expected_device_uuid=expected_uuid,
            candidate_win_cases=candidate_win_cases,
            max_seconds=max_seconds, task_id=task_id, kda_root=kda_root,
        )
        _add_strict_sanitizer_trusted_inputs(
            task, task_path, sanitizer_suite, sanitizer, kda_root, task_loader,
        )
        _require_context_stable(expected_uuid, context_before, context_reader, "task preparation")

        if registration is None:
            from register_candidate import _checked_registration

            registration = _checked_registration
        registration_rows = []
        for candidate_id, baseline, hypothesis, change in (
            (INCUMBENT_ID, True,
             "Use the strongest exact-record direct mode as the paired native control.",
             "Source-bound direct candidate and unique-output specialization from one native runner."),
            (CHALLENGER_ID, False,
             "Measure the prepared packet physical layout against its strongest applicable direct control.",
             "Source-bound factorized packet plan and packet candidate runner; numerical law unchanged."),
        ):
            registration_rows.append(registration(
                task=task_path, workspace=workspace, candidate_id=candidate_id,
                source_dir=source, manifest_path=candidate_manifest_path,
                variant_map_path=variant_map_path, hypothesis=hypothesis,
                change=change, baseline=baseline, kda_root=kda_root,
            ))
            _require_context_stable(
                expected_uuid, context_before, context_reader,
                f"registration of {candidate_id}",
            )
        write_json(packet_dir / "registration.json", {
            "candidate_ids": [INCUMBENT_ID, CHALLENGER_ID],
            "registrations": registration_rows,
            "source_tree_sha256": source_tree_sha256,
            "candidate_manifest_sha256": digest_file(candidate_manifest_path),
        }, exclusive=True)

        command_env = _build_env(kda_root)
        prepare_argv = [
            sys.executable, "-m", "kda_experiments",
            "--task", str(task_path), "--workspace", str(workspace),
            "prepare", INCUMBENT_ID, CHALLENGER_ID, "--jobs", "1",
        ]
        prepare_result = _invoke_logged(
            label="kda-prepare", argv=prepare_argv, cwd=kda_root,
            env=command_env, log_root=log_root, invoke=invoke,
            timeout=max_seconds,
        )
        prepare_response = _read_kda_response(prepare_result.stdout, "KDA prepare")
        candidate_manifest_sha256 = digest_file(candidate_manifest_path)
        prepared = _prepared_builds(
            workspace, prepare_response, variant_map,
            [INCUMBENT_ID, CHALLENGER_ID], expected_closure_hash,
            candidate_manifest_sha256, build_checker,
        )
        _require_context_stable(expected_uuid, context_before, context_reader, "CPU-only prepare")

        if sanitizer_runner is None:
            sanitizer_runner = invoke
        strict_sanitizer = _run_strict_sanitizer_groups(
            prepared=prepared,
            source_closure_sha256=expected_closure_hash,
            expected_uuid=expected_uuid,
            scheduler_job_id=str(context_before["scheduler_job_id"]),
            sanitizer_suite=sanitizer_suite,
            sanitizer_suite_sha256=sanitizer_suite_sha256,
            sanitizer=sanitizer,
            packet_dir=packet_dir,
            kda_root=kda_root,
            command_env=command_env,
            log_root=log_root,
            max_seconds=max_seconds,
            context_reader=context_reader,
            initial_context=context_before,
            invoke=sanitizer_runner,
        )
        packet_record.update({
            "phase": "strict_sanitizer_qualified",
            "strict_sanitizer_qualification_path": str(strict_sanitizer["path"]),
            "strict_sanitizer_qualification_sha256": strict_sanitizer["sha256"],
        })
        write_json(packet_dir / "packet.json", packet_record)
        _verify_unchanged_inputs(
            source, source_manifest, source_tree_sha256, source_manifest_sha256,
            site_policy, site_policy_sha256, nvcc, nvcc_sha256,
            sanitizer_suite, sanitizer_suite_sha256, sanitizer,
        )

        if dtype_runner is None:
            from run_dtype_qualification import run_matrix

            dtype_runner = run_matrix
        dtype_records = []
        for (runner_sha256, library_sha256), builds in _qualification_groups(prepared):
            group_id = hashlib.sha256(
                f"{runner_sha256}:{library_sha256}".encode("ascii")
            ).hexdigest()[:16]
            representative = builds[0]
            artifact_dir = packet_dir / "dtype-qualification" / group_id
            report_path = packet_dir / "dtype-qualification" / f"{group_id}.json"
            cache_path = artifact_dir / "reference-cache"

            def bounded_invoke(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
                child_env = dict(kwargs.get("env", {}))
                child_env["PYTHONDONTWRITEBYTECODE"] = "1"
                kwargs["env"] = child_env
                return invoke(command, timeout=max_seconds, **kwargs)

            _require_context_stable(expected_uuid, context_before, context_reader,
                                    f"before dtype group {group_id}")
            dtype_result = dtype_runner(
                runner=representative.runner,
                build_manifest=representative.manifest,
                expected_uuid=expected_uuid,
                artifact_dir=artifact_dir,
                output=report_path,
                reference_cache_dir=cache_path,
                invoke=bounded_invoke,
                context_reader=context_reader,
            )
            if dtype_result.get("status") != "pass" or dtype_result.get("passed_case_count") != 32:
                raise PacketError(f"prepared native runner failed the full 32-case dtype matrix: {group_id}")
            if dtype_result.get("build", {}).get("runner_sha256") != runner_sha256:
                raise PacketError("dtype matrix report is bound to a different prepared runner binary")
            dtype_records.append({
                "group_id": group_id,
                "candidate_ids": [build.candidate_id for build in builds],
                "runner_sha256": runner_sha256,
                "native_library_sha256": library_sha256,
                "build_manifest_sha256": dtype_result["build"]["build_manifest_sha256"],
                "report_path": str(report_path),
                "report_sha256": digest_file(report_path),
                "status": dtype_result["status"],
                "passed_case_count": dtype_result["passed_case_count"],
            })
            _require_context_stable(expected_uuid, context_before, context_reader,
                                    f"dtype qualification group {group_id}")
        dtype_index_path = packet_dir / "dtype-qualification.json"
        write_json(dtype_index_path, {
            "schema": "cellerator.factorized_interactions.prepared_dtype_matrix.v1",
            "workload": "packet_a4_m32_u16",
            "modes": ["packet", "native_direct", "minimal_direct", "direct_unique_output"],
            "dtype_combination_count": 8,
            "mode_count": 4,
            "expected_cases_per_distinct_runtime": 32,
            "runner_and_library_grouping": "deduplicated only when both runner SHA256 and linked CUDA library SHA256 match",
            "prepared_candidate_builds": {
                candidate_id: {
                    "runner_sha256": build.runner_sha256,
                    "native_library_sha256": build.library_sha256,
                    "build_manifest_sha256": digest_file(build.manifest),
                    "source_snapshot_sha256": build.source_snapshot_sha256,
                }
                for candidate_id, build in prepared.items()
            },
            "groups": dtype_records,
            "status": "pass" if dtype_records and all(row["status"] == "pass" for row in dtype_records)
            else "fail",
        }, exclusive=True)

        _require_context_stable(expected_uuid, context_before, context_reader,
                                "before guarded KDA evaluation")
        evaluate_argv = [
            sys.executable, str(IN_JOB_ENTRYPOINT.resolve(strict=True)),
            "--task", str(task_path), "--workspace", str(workspace),
            "--expected-uuid", expected_uuid,
            "--candidates", CHALLENGER_ID,
            "--incumbent", INCUMBENT_ID,
        ]
        evaluate_result = _invoke_logged(
            label="guarded-kda-evaluate", argv=evaluate_argv, cwd=kda_root,
            env=command_env, log_root=log_root, invoke=invoke,
            timeout=max_seconds,
        )
        # The guarded entrypoint receives the original inherited environment and
        # execs KDA in place. Using it as a subprocess keeps this driver alive to
        # audit the exact result and write the conditional selection artifact.
        evaluate_response = _read_kda_response(evaluate_result.stdout, "guarded KDA evaluate")
        run_id = evaluate_response.get("run_id")
        try:
            uuid.UUID(str(run_id))
        except ValueError as exc:
            raise PacketError("guarded KDA response omitted its run UUID") from exc
        summary_path_value = evaluate_response.get("summary_path")
        if not isinstance(summary_path_value, str):
            raise PacketError("guarded KDA response omitted its finalized summary path")
        summary_path = Path(summary_path_value)
        if summary_path.is_symlink() or not summary_path.is_file():
            raise PacketError("guarded KDA finalized summary is missing or non-regular")
        expected_summary = workspace / "runs" / "kda-experiments" / "attempts" / str(run_id) / "summary.json"
        if summary_path.resolve(strict=True) != expected_summary.resolve(strict=True):
            raise PacketError("guarded KDA summary is outside the exact packet workspace/run")
        summary = read_json(summary_path)
        if summary.get("run_id") != run_id or summary.get("status") != "complete":
            raise PacketError("KDA finalized summary identity or status is invalid")
        evaluated_builds = _prepared_builds(
            workspace,
            {"results": [
                {"candidate_id": candidate_id, "status": "ok", **build_record}
                for candidate_id, build_record in summary.get("builds", {}).items()
            ]},
            variant_map, [INCUMBENT_ID, CHALLENGER_ID],
            expected_closure_hash, candidate_manifest_sha256, build_checker,
        )
        for candidate_id in [INCUMBENT_ID, CHALLENGER_ID]:
            actual, qualified = evaluated_builds[candidate_id], prepared[candidate_id]
            if (actual.runner_sha256, actual.library_sha256) != \
                    (qualified.runner_sha256, qualified.library_sha256):
                raise PacketError(
                    f"KDA evaluation reused different native bytes than dtype-qualified for {candidate_id}"
                )
            if (actual.baseline_runner_sha256, actual.baseline_manifest_sha256) != \
                    (qualified.baseline_runner_sha256, qualified.baseline_manifest_sha256):
                raise PacketError(
                    f"KDA evaluation reused different baseline bytes than strict-sanitizer-qualified for {candidate_id}"
                )

        if selection_runner is None:
            from selection_guard import guard_run

            selection_runner = guard_run
        selection_path = packet_dir / "conditional-selection.json"
        selection = selection_runner(
            task_path, workspace, str(run_id), variant_map_path, selection_path,
        )
        if selection.get("status") not in {"conditional", "rejected"}:
            raise PacketError("selection guard did not produce a conditional or rejected record")
        if selection.get("outer_receipt_verification") != "pending":
            raise PacketError("selection guard record must remain conditional on outer receipt/cleanup")
        _require_context_stable(expected_uuid, context_before, context_reader,
                                "conditional selection")

        _verify_unchanged_inputs(
            source, source_manifest, source_tree_sha256, source_manifest_sha256,
            site_policy, site_policy_sha256, nvcc, nvcc_sha256,
            sanitizer_suite, sanitizer_suite_sha256, sanitizer,
        )
        packet_record.update({
            "status": selection["status"],
            "phase": "complete",
            "completed_at_unix": time.time(),
            "kda_workspace": str(workspace),
            "task_path": str(task_path),
            "task_sha256": digest_file(task_path),
            "variant_map_path": str(variant_map_path),
            "variant_map_sha256": digest_file(variant_map_path),
            "candidate_manifest_sha256": digest_file(candidate_manifest_path),
            "prepared_builds": {
                candidate_id: {
                    "runner_sha256": build.runner_sha256,
                    "native_library_sha256": build.library_sha256,
                    "build_manifest_path": str(build.manifest),
                    "build_manifest_sha256": digest_file(build.manifest),
                    "baseline_runner_sha256": build.baseline_runner_sha256,
                    "baseline_build_manifest_path": str(build.baseline_manifest),
                    "baseline_build_manifest_sha256": build.baseline_manifest_sha256,
                    "source_snapshot_sha256": build.source_snapshot_sha256,
                }
                for candidate_id, build in prepared.items()
            },
            "strict_sanitizer_qualification_path": str(strict_sanitizer["path"]),
            "strict_sanitizer_qualification_sha256": strict_sanitizer["sha256"],
            "dtype_qualification_path": str(dtype_index_path),
            "dtype_qualification_sha256": digest_file(dtype_index_path),
            "kda_run_id": str(run_id),
            "kda_summary_path": str(summary_path),
            "kda_summary_sha256": digest_file(summary_path),
            "conditional_selection_path": str(selection_path),
            "conditional_selection_sha256": digest_file(selection_path),
            "outer_receipt_verification": "pending; oracle must match the finished Scheduler receipt and cleanup to the recorded outer job/attempt",
            "production_route_changed": False,
        })
        write_json(packet_dir / "packet.json", packet_record)
        return packet_record
    except Exception as exc:
        packet_record.update({
            "status": "failed",
            "phase": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "failed_at_unix": time.time(),
        })
        try:
            write_json(packet_dir / "packet.json", packet_record)
        except Exception:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="immutable source closure directory")
    parser.add_argument("--source-manifest", type=Path, required=True,
                        help="exact Autobench-captured source closure manifest")
    parser.add_argument("--expected-uuid", required=True,
                        help="one exact inherited Scheduler GPU UUID")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="existing Autobench-managed packet output directory")
    parser.add_argument("--site-policy", type=Path, required=True)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--kda-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--candidate-win-case", action="append", required=True)
    parser.add_argument("--max-seconds", type=float, required=True,
                        help="finite KDA budget and per-process timeout; outer allocation must cover the full packet workflow")
    args = parser.parse_args(argv)
    try:
        result = run_packet(
            source=args.source, source_manifest=args.source_manifest,
            expected_uuid=args.expected_uuid, output_root=args.output_root,
            site_policy=args.site_policy, nvcc=args.nvcc, kda_root=args.kda_root,
            task_id=args.task_id, candidate_win_cases=args.candidate_win_case,
            max_seconds=args.max_seconds,
        )
    except Exception as exc:
        print(json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"},
                         sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({
        "status": result["status"],
        "task_id": result["task_id"],
        "outer_scheduler_job_id": result["outer_execution_context"]["scheduler_job_id"],
        "outer_scheduler_attempt_id": result["outer_execution_context"]["scheduler_attempt_id"],
        "kda_run_id": result.get("kda_run_id"),
        "packet_record": str((args.output_root / f"factorized-{args.task_id}" / "packet.json").resolve()),
        "outer_receipt_verification": result.get("outer_receipt_verification"),
        "production_route_changed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
