#!/usr/bin/env python3
"""CPU-only KDA adapter for already-recorded CUDA observations.

This module never launches a native runner or queries CUDA/Scheduler state.  It
copies hash-bound prebuilt files into KDA's build directory, reuses the accepted
validation summaries, and serves one stored timing observation for each exact
KDA (case, phase, round, candidate) request.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any, Mapping, Sequence


SCHEMA = "cellerator.factorized_interactions.recorded_dataset.v1"
TASK_SCHEMA = "cellerator.factorized_interactions.recorded_kda_cpu_task.v1"
RECEIPT_SCHEMA = "cellerator.factorized_interactions.recorded_kda_cpu_receipt.v1"
NATIVE_BUILD_SCHEMA = "cellerator.factorized_interactions.build.v1"
NATIVE_ARCHIVE_SCHEMA = "cellerator.factorized_interactions.archived_source_closure.v1"
KDA_SOURCE_DESCRIPTOR_SCHEMA = "cellerator.factorized_interactions.kda_native_source_descriptor.v1"
KDA_SOURCE_ARCHIVE_NAME = "native_source.tar.gz"
KDA_SOURCE_DESCRIPTOR_NAME = "analysis_descriptor.json"
QUALIFICATION_STATUS = "validated_pending_independent_review"
PHASES = ("exploration", "confirmation")
BLOCKS_PER_PHASE = 7
CANDIDATES = {"strongest-direct": "strongest_direct", "packet-candidate": "packet"}
ARMS = ("strongest_direct", "packet", "native_direct")
PROMOTION = {
    "min_improvement": 0.0,
    "max_regression": 1.0 - 1.0 / 1.05,
    "confidence": 0.95,
    "bootstrap_samples": 10_000,
}
TASK_BUDGET_SECONDS = 600.0

_HEX = set("0123456789abcdef")
_MODULE_PATH = Path(__file__).resolve()


class ReplayError(RuntimeError):
    """The recorded dataset or replay request violates the fixed contract."""


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ReplayError(f"expected a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ReplayError(f"value is not canonical JSON: {exc}") from exc


def digest_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReplayError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ReplayError(f"non-finite JSON value is not allowed: {value}")


def read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReplayError(f"expected a regular non-symlink JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"),
                           object_pairs_hook=_reject_duplicate_keys,
                           parse_constant=_reject_constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReplayError(f"expected a JSON object in {path}")
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise ReplayError(f"refusing to replace existing analysis input or evidence: {path}") from exc


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64 and
            all(char in _HEX for char in value))


def _finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayError(f"{label} must be a positive finite number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ReplayError(f"{label} must be a positive finite number")
    return number


def _all_values(mapping: Any, keys: set[str]) -> list[Any]:
    found: list[Any] = []
    if isinstance(mapping, dict):
        for key, value in mapping.items():
            if key in keys:
                found.append(value)
            found.extend(_all_values(value, keys))
    elif isinstance(mapping, list):
        for value in mapping:
            found.extend(_all_values(value, keys))
    return found


def _one_recursive(mapping: Mapping[str, Any], key: str, label: str) -> Any:
    values = _all_values(mapping, {key})
    distinct = {_canonical_json(value) for value in values}
    if len(distinct) != 1:
        raise ReplayError(f"{label} must contain exactly one consistent {key}")
    return values[0]


def _recorded_device_uuid(dataset: Mapping[str, Any]) -> str:
    identity = dataset["identity"]
    direct = _all_values(identity, {"device_uuid", "gpu_uuid"})
    if not direct:
        for case in dataset["cases"]:
            for phase in PHASES:
                block = case["phases"][phase][0]
                for arm in ARMS:
                    direct.extend(_all_values(block["arms"][arm], {"device_uuid", "gpu_uuid"}))
                if direct:
                    break
            if direct:
                break
    unique = {value for value in direct if isinstance(value, str) and value}
    if len(unique) != 1:
        raise ReplayError("recorded dataset must bind exactly one original CUDA device UUID")
    return next(iter(unique))


def _fixed_workload_catalog(source_root: Path) -> list[dict[str, Any]]:
    """Read the fixed workload descriptors from the captured native source."""
    catalog_path = source_root / "bench" / "factorized_interactions" / "workloads.json"
    catalog = read_json(catalog_path)
    cases = catalog.get("cases")
    if not isinstance(cases, list):
        raise ReplayError("fixed workload manifest has no cases array")
    if len(cases) != 13 or any(not isinstance(row, dict) for row in cases):
        raise ReplayError("fixed performance workload manifest must contain exactly 13 descriptors")
    ids = [row.get("id") for row in cases]
    if any(not isinstance(item, str) or not item for item in ids) or len(set(ids)) != 13:
        raise ReplayError("fixed performance workload manifest must contain exactly 13 unique cases")
    return [dict(row) for row in cases]


def _validate_native_dataset_identity(dataset: Mapping[str, Any], source_root: Path,
                                      bundle: Mapping[str, Any]) -> dict[str, str]:
    identity = dataset.get("identity")
    if not isinstance(identity, dict):
        raise ReplayError("dataset identity must be an object")
    assets = {row["role"]: row["sha256"] for row in bundle["assets"]}
    workload_path = source_root / "bench" / "factorized_interactions" / "workloads.json"
    validator_path = source_root / "bench" / "factorized_interactions" / "validate_runner.py"
    expected = {
        "source_closure_sha256": bundle["source_closure_sha256"],
        "candidate_runner_sha256": assets["candidate_runner"],
        "candidate_manifest_sha256": assets["candidate_manifest"],
        "baseline_runner_sha256": assets["baseline_runner"],
        "baseline_manifest_sha256": assets["baseline_manifest"],
        "library_sha256": assets["shared_cuda_library"],
        "validator_sha256": sha256_file(validator_path),
    }
    for key, value in expected.items():
        if identity.get(key) != value:
            raise ReplayError(f"dataset native identity differs for {key}")
    catalog_identity = identity.get("workload_catalog")
    if not isinstance(catalog_identity, dict) or \
            catalog_identity.get("path") != "bench/factorized_interactions/workloads.json" or \
            catalog_identity.get("sha256") != sha256_file(workload_path):
        raise ReplayError("dataset workload catalog identity differs from captured source")
    return expected | {
        "workload_catalog_path": "bench/factorized_interactions/workloads.json",
        "workload_catalog_sha256": sha256_file(workload_path),
    }


def _recorded_trusted_inputs(dataset: Mapping[str, Any], *, verify_files: bool) -> list[dict[str, str]]:
    """Normalize dataset-retained provenance files into stable KDA inputs."""
    rows = dataset.get("trusted_inputs")
    if not isinstance(rows, list) or not rows:
        raise ReplayError("recorded dataset must retain its source reports and validator inputs")
    normalized: list[dict[str, str]] = []
    ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ReplayError("recorded dataset trusted input row is malformed")
        original_id, raw_path = row.get("id"), row.get("path")
        expected_sha, recorded_role = row.get("sha256"), row.get("role")
        if not isinstance(original_id, str) or not original_id or original_id in ids:
            raise ReplayError("recorded dataset trusted input IDs must be unique nonempty strings")
        if not isinstance(raw_path, str) or not raw_path:
            raise ReplayError(f"recorded trusted input {original_id} has no path")
        if not _is_sha256(expected_sha) or not isinstance(recorded_role, str) or not recorded_role:
            raise ReplayError(f"recorded trusted input {original_id} has an invalid hash or role")
        path = Path(raw_path)
        if not path.is_absolute():
            raise ReplayError(f"recorded trusted input paths must be absolute: {raw_path}")
        if path.is_symlink() or not path.is_file():
            raise ReplayError(f"recorded trusted input is not a regular file: {path}")
        resolved = path.resolve(strict=True)
        if verify_files and sha256_file(resolved) != expected_sha:
            raise ReplayError(f"recorded trusted input changed: {original_id}")
        # KDA has a small fixed role vocabulary. Keep each source role in the
        # config while assigning a broad engine role for integrity snapshots.
        if recorded_role == "fixed_sanitizer_partition_validator":
            kda_role = "validator"
        elif recorded_role in {"timed_helper_evidence", "pre_cleanup_validation_record",
                               "validator_stdout", "validator_stderr"}:
            kda_role = "benchmark"
        else:
            kda_role = "reference"
        stable_id = "recorded_evidence_" + hashlib.sha256(original_id.encode("utf-8")).hexdigest()
        normalized.append({
            "id": stable_id,
            "dataset_id": original_id,
            "path": str(resolved),
            "sha256": expected_sha,
            "role": kda_role,
            "recorded_role": recorded_role,
        })
        ids.add(original_id)
    normalized.sort(key=lambda item: item["id"])
    return normalized


def _check_config_native_identity(dataset: Mapping[str, Any],
                                  expected: Mapping[str, Any]) -> None:
    identity = dataset.get("identity")
    if not isinstance(identity, dict):
        raise ReplayError("dataset identity must be an object")
    for key, value in expected.items():
        if key == "workload_catalog_path":
            catalog = identity.get("workload_catalog")
            if not isinstance(catalog, dict) or catalog.get("path") != value:
                raise ReplayError("dataset workload catalog path differs from recorded task config")
        elif key == "workload_catalog_sha256":
            catalog = identity.get("workload_catalog")
            if not isinstance(catalog, dict) or catalog.get("sha256") != value:
                raise ReplayError("dataset workload catalog hash differs from recorded task config")
        elif identity.get(key) != value:
            raise ReplayError(f"dataset native identity differs from task config for {key}")


def validate_dataset(dataset: Mapping[str, Any], *,
                     expected_workloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate the compact recorded-data envelope without redoing CUDA work."""
    if dataset.get("schema") != SCHEMA:
        raise ReplayError(f"dataset schema must be {SCHEMA}")
    identity = dataset.get("identity")
    if not isinstance(identity, dict):
        raise ReplayError("dataset identity must be an object")
    qualification_status = dataset.get("qualification_status", identity.get("qualification_status"))
    bundle_sha = dataset.get("qualification_bundle_sha256", identity.get("qualification_bundle_sha256"))
    if qualification_status != QUALIFICATION_STATUS:
        raise ReplayError(f"dataset qualification_status must be {QUALIFICATION_STATUS}")
    if not _is_sha256(bundle_sha):
        raise ReplayError("dataset qualification_bundle_sha256 must be a SHA-256 digest")

    cases = dataset.get("cases")
    catalog = [dict(row) for row in expected_workloads]
    expected_ids = [row.get("id") for row in catalog]
    if len(expected_ids) != 13 or len(set(expected_ids)) != 13 or \
            any(not isinstance(item, str) or not item for item in expected_ids):
        raise ReplayError("expected workload descriptors must be the 13 unique rows from captured source")
    if not isinstance(cases, list) or [row.get("id") if isinstance(row, dict) else None for row in cases] != expected_ids:
        raise ReplayError("dataset must contain the exact 13 fixed performance cases in catalog order")

    uuid_values: set[str] = set()
    for case in cases:
        workload_sha = case.get("workload_sha256")
        if not _is_sha256(workload_sha):
            raise ReplayError(f"case {case['id']} has an invalid workload_sha256")
        expected_workload = catalog[expected_ids.index(case["id"])]
        if case.get("workload_descriptor") != expected_workload or \
                workload_sha != digest_json(expected_workload):
            raise ReplayError(f"case {case['id']} workload descriptor differs from captured source catalog")
        phases = case.get("phases")
        if not isinstance(phases, dict) or set(phases) != set(PHASES):
            raise ReplayError(f"case {case['id']} must have exploration and confirmation phases")
        for phase in PHASES:
            blocks = phases[phase]
            if not isinstance(blocks, list) or len(blocks) != BLOCKS_PER_PHASE:
                raise ReplayError(f"case {case['id']} phase {phase} must contain exactly seven blocks")
            for round_index, block in enumerate(blocks):
                if not isinstance(block, dict) or block.get("block_index") != round_index:
                    raise ReplayError(f"case {case['id']} phase {phase} block order must be zero-based and complete")
                arms = block.get("arms")
                if not isinstance(arms, dict) or set(arms) != set(ARMS):
                    raise ReplayError(f"case {case['id']} phase {phase} block {round_index} must contain all three arms")
                for arm_name in ARMS:
                    arm = arms[arm_name]
                    if not isinstance(arm, dict):
                        raise ReplayError(f"case {case['id']} phase {phase} block {round_index} arm {arm_name} is malformed")
                    _finite_positive(arm.get("primary_ms"), f"{case['id']}/{phase}/{round_index}/{arm_name}.primary_ms")
                    for key in ("raw_metrics_sha256", "validation_summary_sha256", "original_report_sha256"):
                        if not _is_sha256(arm.get(key)):
                            raise ReplayError(f"{case['id']}/{phase}/{round_index}/{arm_name} has invalid {key}")
                    for key in ("scheduler_job_id", "scheduler_attempt_id", "autobench_attempt_id"):
                        value = _one_recursive(arm, key, f"{case['id']}/{phase}/{round_index}/{arm_name}")
                        if not isinstance(value, str) or not value:
                            raise ReplayError(f"{case['id']}/{phase}/{round_index}/{arm_name} has invalid {key}")
                    device_uuid = _one_recursive(arm, "device_uuid", f"{case['id']}/{phase}/{round_index}/{arm_name}")
                    if not isinstance(device_uuid, str) or not device_uuid:
                        raise ReplayError(f"{case['id']}/{phase}/{round_index}/{arm_name} has invalid device_uuid")
                    uuid_values.add(device_uuid)
    uuid_values.update(value for value in _all_values(identity, {"device_uuid", "gpu_uuid"})
                       if isinstance(value, str) and value)
    if len(uuid_values) != 1:
        raise ReplayError("all recorded arms and dataset identity must bind the same CUDA device UUID")
    return {
        "qualification_status": qualification_status,
        "qualification_bundle_sha256": bundle_sha,
        "device_uuid": next(iter(uuid_values)),
        "case_ids": expected_ids,
        "workload_sha256": {case["id"]: case["workload_sha256"] for case in cases},
    }


def _compact_observations(dataset: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only immutable timing/provenance fields needed by KDA requests."""
    result: dict[str, Any] = {}
    for case in dataset["cases"]:
        case_row: dict[str, Any] = {
            "recorded_workload_sha256": case["workload_sha256"],
            "phases": {},
        }
        for phase in PHASES:
            blocks = []
            for block in case["phases"][phase]:
                arms: dict[str, Any] = {}
                for arm_name in ARMS:
                    arm = block["arms"][arm_name]
                    arms[arm_name] = {
                        "primary_ms": _finite_positive(arm["primary_ms"], "recorded primary_ms"),
                        "raw_metrics_sha256": arm["raw_metrics_sha256"],
                        "validation_summary_sha256": arm["validation_summary_sha256"],
                        "original_report_sha256": arm["original_report_sha256"],
                        "recorded_observation_sha256": digest_json(arm),
                        "device_uuid": _one_recursive(arm, "device_uuid", "recorded observation"),
                        "original_receipt_identity": {
                            key: _one_recursive(arm, key, "recorded observation")
                            for key in ("scheduler_job_id", "scheduler_attempt_id", "autobench_attempt_id")
                        },
                    }
                blocks.append({"block_index": block["block_index"], "arms": arms})
            case_row["phases"][phase] = blocks
        result[case["id"]] = case_row
    return result


def _validate_compact_observations(config: Mapping[str, Any]) -> None:
    observations = config.get("recorded_observations")
    ids = config.get("case_ids")
    if not isinstance(ids, list) or len(ids) != 13 or \
            not isinstance(observations, dict) or set(observations) != set(ids):
        raise ReplayError("task config must bind the fixed 13-case observation table")
    for case_id in ids:
        case = observations.get(case_id)
        if not isinstance(case, dict) or not _is_sha256(case.get("recorded_workload_sha256")):
            raise ReplayError(f"task config has an invalid observation descriptor for {case_id}")
        phases = case.get("phases")
        if not isinstance(phases, dict) or set(phases) != set(PHASES):
            raise ReplayError(f"task config has incomplete observations for {case_id}")
        for phase in PHASES:
            blocks = phases[phase]
            if not isinstance(blocks, list) or len(blocks) != BLOCKS_PER_PHASE:
                raise ReplayError(f"task config {case_id}/{phase} needs exactly seven blocks")
            for index, block in enumerate(blocks):
                if not isinstance(block, dict) or block.get("block_index") != index or \
                        not isinstance(block.get("arms"), dict) or set(block["arms"]) != set(ARMS):
                    raise ReplayError(f"task config {case_id}/{phase}/{index} has incomplete arms")
                for arm_name, arm in block["arms"].items():
                    if not isinstance(arm, dict):
                        raise ReplayError(f"task config {case_id}/{phase}/{index}/{arm_name} is malformed")
                    _finite_positive(arm.get("primary_ms"), f"{case_id}/{phase}/{index}/{arm_name}.primary_ms")
                    for key in ("raw_metrics_sha256", "validation_summary_sha256",
                                "original_report_sha256", "recorded_observation_sha256"):
                        if not _is_sha256(arm.get(key)):
                            raise ReplayError(f"task config {case_id}/{phase}/{index}/{arm_name} has invalid {key}")
                    if arm.get("device_uuid") != config.get("recorded_device_uuid"):
                        raise ReplayError("task config observation changes the recorded GPU identity")
                    receipt = arm.get("original_receipt_identity")
                    if not isinstance(receipt, dict) or any(
                            not isinstance(receipt.get(key), str) or not receipt[key]
                            for key in ("scheduler_job_id", "scheduler_attempt_id", "autobench_attempt_id")):
                        raise ReplayError(f"task config {case_id}/{phase}/{index}/{arm_name} lacks original receipt identity")


def _source_tree(source_root: Path) -> tuple[str, list[dict[str, str]]]:
    if source_root.is_symlink():
        raise ReplayError("source snapshot may not be a symlink")
    root = source_root.resolve(strict=True)
    if not root.is_dir():
        raise ReplayError("source snapshot must be a directory")
    entries: list[dict[str, str]] = []
    for current, dirs, files in os.walk(root, topdown=True, followlinks=False):
        parent = Path(current)
        for name in dirs:
            if (parent / name).is_symlink():
                raise ReplayError("source snapshot contains a symlink directory")
        for name in files:
            path = parent / name
            if path.is_symlink() or not path.is_file():
                raise ReplayError("source snapshot contains a nonregular file")
            entries.append({"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)})
    entries.sort(key=lambda item: item["path"])
    if not entries:
        raise ReplayError("source snapshot is empty")
    return digest_json(entries), entries


def _safe_relative(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ReplayError(f"{label} must be a nonempty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ReplayError(f"{label} escapes its declared root")
    return path


def _regular_within(root: Path, relative: Any, label: str) -> Path:
    rel = _safe_relative(relative, label)
    candidate = root.joinpath(*rel.parts)
    current = root
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise ReplayError(f"{label} contains a symlink: {relative}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve(strict=True)) or not resolved.is_file():
        raise ReplayError(f"{label} is missing or outside its root: {relative}")
    return resolved


def _source_closure(source_root: Path, source_files: Any) -> tuple[str, list[dict[str, str]]]:
    if not isinstance(source_files, list) or not source_files:
        raise ReplayError("native manifest must include its complete source_files inventory")
    digest = hashlib.sha256()
    normalized: list[dict[str, str]] = []
    previous = ""
    for row in source_files:
        if not isinstance(row, dict):
            raise ReplayError("native source inventory has a malformed row")
        name, size, expected = row.get("path"), row.get("bytes"), row.get("sha256")
        if not isinstance(name, str) or name <= previous or not _is_sha256(expected):
            raise ReplayError("native source inventory paths must be sorted and hashes valid")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ReplayError(f"native source inventory has invalid byte size for {name}")
        source = _regular_within(source_root, name, "native source file")
        if source.stat().st_size != size or sha256_file(source) != expected:
            raise ReplayError(f"native source input changed: {name}")
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(size.to_bytes(8, "little"))
        digest.update(bytes.fromhex(expected))
        normalized.append({"path": name, "sha256": expected})
        previous = name
    tree_hash, tree_entries = _source_tree(source_root)
    if normalized != tree_entries:
        raise ReplayError("native source_files differ from the complete candidate source snapshot")
    return digest.hexdigest(), tree_entries


def _verify_native_source_archive(
    archive_path: Path, index_path: Path, *, expected_archive_sha256: str,
    expected_index_sha256: str, source_files: Any, source_closure_sha256: str,
) -> dict[str, Any]:
    """Verify the operator-pinned TAR byte-for-byte against the native manifest.

    This is intentionally a read-only verifier: it never extracts archive members
    to the filesystem.  The accepted TAR is data for the CPU KDA candidate
    snapshot, not something this adapter executes or rebuilds.
    """
    for path, label in ((archive_path, "accepted native source TAR"),
                        (index_path, "accepted native source member index")):
        if path.is_symlink() or not path.is_file():
            raise ReplayError(f"{label} must be a regular non-symlink file: {path}")
    archive_path = archive_path.resolve(strict=True)
    index_path = index_path.resolve(strict=True)
    if not _is_sha256(expected_archive_sha256) or sha256_file(archive_path) != expected_archive_sha256:
        raise ReplayError("accepted native source TAR differs from its operator-pinned SHA-256")
    if not _is_sha256(expected_index_sha256) or sha256_file(index_path) != expected_index_sha256:
        raise ReplayError("accepted native source member index differs from its operator-pinned SHA-256")

    index = read_json(index_path)
    if index.get("schema") != NATIVE_ARCHIVE_SCHEMA:
        raise ReplayError("accepted native source member index has an unknown schema")
    rows = index.get("files")
    if not isinstance(rows, list) or not rows or rows != source_files:
        raise ReplayError("accepted native archive member index differs from qualified native source_files")
    if index.get("archive_path") != archive_path.name or \
            index.get("archive_sha256") != expected_archive_sha256 or \
            index.get("archive_bytes") != archive_path.stat().st_size:
        raise ReplayError("accepted native source archive identity differs from its member index")
    if index.get("source_closure_sha256") != source_closure_sha256 or \
            index.get("file_count") != len(rows) or \
            index.get("source_bytes") != sum(row.get("bytes", -1) for row in rows):
        raise ReplayError("accepted native source archive index disagrees with the qualified closure")
    members_sha256 = digest_json(rows)

    names = [row.get("path") if isinstance(row, dict) else None for row in rows]
    if any(not isinstance(name, str) for name in names) or names != sorted(names) or len(set(names)) != len(names):
        raise ReplayError("accepted native source member paths must be unique and sorted")
    for row in rows:
        name = row["path"]
        if _safe_relative(name, "native archive member").as_posix() != name or \
                isinstance(row.get("bytes"), bool) or not isinstance(row.get("bytes"), int) or \
                row["bytes"] < 0 or not _is_sha256(row.get("sha256")):
            raise ReplayError(f"accepted native source member row is invalid: {name!r}")

    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            members = archive.getmembers()
            if len(members) != len(rows):
                raise ReplayError("accepted native source TAR member count differs from its index")
            if [member.name for member in members] != names:
                raise ReplayError("accepted native source TAR order/names differ from its sorted index")
            for member, row in zip(members, rows):
                normalized_pax = not member.pax_headers or member.pax_headers == {"path": member.name}
                if not member.isfile() or member.type != tarfile.REGTYPE or member.issparse() or \
                        member.name != row["path"] or member.size != row["bytes"] or \
                        member.mode != 0o444 or member.uid != 0 or member.gid != 0 or \
                        member.mtime != 0 or member.uname != "" or member.gname != "" or \
                        not normalized_pax:
                    raise ReplayError(f"accepted native TAR member metadata is not the qualified normalized form: {member.name}")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ReplayError(f"accepted native TAR member has no regular payload: {member.name}")
                digest = hashlib.sha256()
                byte_count = 0
                with stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        byte_count += len(chunk)
                        digest.update(chunk)
                if byte_count != row["bytes"] or digest.hexdigest() != row["sha256"]:
                    raise ReplayError(f"accepted native TAR payload differs from qualified source: {member.name}")
    except (OSError, tarfile.TarError) as exc:
        raise ReplayError(f"cannot read accepted native source TAR: {exc}") from exc

    return {
        "native_source_archive_tar_path": str(archive_path),
        "native_source_archive_tar_sha256": expected_archive_sha256,
        "native_source_archive_tar_bytes": archive_path.stat().st_size,
        "native_source_archive_manifest_path": str(index_path),
        "native_source_archive_manifest_sha256": expected_index_sha256,
        "native_source_archive_member_count": len(rows),
        "native_source_archive_members_sha256": members_sha256,
        "native_source_archive_source_bytes": index["source_bytes"],
    }


def _make_kda_source_descriptor(*, source_tree_sha256: str,
                                source_closure_sha256: str,
                                archive_identity: Mapping[str, Any],
                                native_identity: Mapping[str, Any],
                                bundle: Mapping[str, Any],
                                qualification_bundle_sha256: str) -> dict[str, Any]:
    assets = {row["role"]: row["sha256"] for row in bundle["assets"]}
    return {
        "schema": KDA_SOURCE_DESCRIPTOR_SCHEMA,
        "native_source_tree_sha256": source_tree_sha256,
        "source_closure_sha256": source_closure_sha256,
        "native_source_archive_tar_path": Path(archive_identity["native_source_archive_tar_path"]).name,
        "native_source_archive_tar_sha256": archive_identity["native_source_archive_tar_sha256"],
        "native_source_archive_tar_bytes": archive_identity["native_source_archive_tar_bytes"],
        "native_source_archive_manifest_path": Path(archive_identity["native_source_archive_manifest_path"]).name,
        "native_source_archive_manifest_sha256": archive_identity["native_source_archive_manifest_sha256"],
        "native_source_archive_member_count": archive_identity["native_source_archive_member_count"],
        "native_source_archive_members_sha256": archive_identity["native_source_archive_members_sha256"],
        "native_source_archive_source_bytes": archive_identity["native_source_archive_source_bytes"],
        "candidate_manifest_sha256": bundle["candidate_manifest_sha256"],
        "baseline_manifest_sha256": bundle["baseline_manifest_sha256"],
        "candidate_runner_sha256": assets["candidate_runner"],
        "baseline_runner_sha256": assets["baseline_runner"],
        "library_sha256": assets["shared_cuda_library"],
        "validator_sha256": native_identity["validator_sha256"],
        "workload_catalog_sha256": native_identity["workload_catalog_sha256"],
        "qualification_bundle_sha256": qualification_bundle_sha256,
    }


def _publish_kda_source(candidate_source_dir: Path, archive_path: Path,
                        descriptor: Mapping[str, Any]) -> dict[str, Any]:
    """Create a fresh two-file candidate source snapshot without touching inputs."""
    if candidate_source_dir.is_symlink() or candidate_source_dir.exists():
        raise ReplayError(f"candidate source output must be a fresh absent directory: {candidate_source_dir}")
    candidate_source_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{candidate_source_dir.name}.", dir=candidate_source_dir.parent))
    try:
        archive_copy = staging / KDA_SOURCE_ARCHIVE_NAME
        with archive_path.open("rb") as input_stream, archive_copy.open("xb") as raw_output:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_output,
                               compresslevel=6, mtime=0) as compressed:
                shutil.copyfileobj(input_stream, compressed, length=1024 * 1024)
        compressed_sha = sha256_file(archive_copy)
        with archive_copy.open("rb") as compressed_header_stream:
            gzip_header = compressed_header_stream.read(10)
        if gzip_header != bytes.fromhex("1f8b08000000000000ff"):
            raise ReplayError("candidate source gzip header is not the deterministic mtime-zero/no-name form")
        descriptor_value = dict(descriptor)
        descriptor_value.update({
            "native_source_transport_path": KDA_SOURCE_ARCHIVE_NAME,
            "native_source_transport_compression": "gzip",
            "native_source_transport_gzip_mtime": 0,
            "native_source_transport_sha256": compressed_sha,
            "native_source_transport_bytes": archive_copy.stat().st_size,
        })
        decompressed = hashlib.sha256()
        decompressed_bytes = 0
        with gzip.open(archive_copy, "rb") as compressed:
            for chunk in iter(lambda: compressed.read(1024 * 1024), b""):
                decompressed.update(chunk)
                decompressed_bytes += len(chunk)
        if decompressed.hexdigest() != descriptor_value["native_source_archive_tar_sha256"] or \
                decompressed_bytes != descriptor_value["native_source_archive_tar_bytes"]:
            raise ReplayError("gzip candidate source does not decompress to the pinned accepted TAR")
        descriptor_path = staging / KDA_SOURCE_DESCRIPTOR_NAME
        _write_json_exclusive(descriptor_path, descriptor_value)
        os.replace(staging, candidate_source_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    tree_sha, entries = _source_tree(candidate_source_dir)
    expected_names = [KDA_SOURCE_DESCRIPTOR_NAME, KDA_SOURCE_ARCHIVE_NAME]
    if [row["path"] for row in entries] != expected_names:
        raise ReplayError("candidate KDA source directory must contain exactly its TAR and descriptor")
    published_archive = candidate_source_dir / KDA_SOURCE_ARCHIVE_NAME
    published_descriptor = candidate_source_dir / KDA_SOURCE_DESCRIPTOR_NAME
    file_records = [{"path": row["path"], "sha256": row["sha256"],
                     "bytes": (candidate_source_dir / row["path"]).stat().st_size}
                    for row in entries]
    return {"candidate_source_root": str(candidate_source_dir.resolve(strict=True)),
            "candidate_source_tree_sha256": tree_sha,
            "candidate_source_files": file_records,
            "candidate_source_descriptor": descriptor_value,
            "candidate_source_descriptor_path": str(published_descriptor.resolve(strict=True)),
            "candidate_source_descriptor_sha256": sha256_file(published_descriptor),
            "native_source_archive_gzip_path": str(published_archive.resolve(strict=True)),
            "native_source_archive_gzip_sha256": compressed_sha,
            "native_source_archive_gzip_bytes": published_archive.stat().st_size,
            "native_source_archive_gzip_compression": "gzip",
            "native_source_archive_gzip_mtime": 0}


def _binary(manifest_root: Path, record: Any, label: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(record, dict):
        raise ReplayError(f"{label} identity is missing")
    relative = record.get("path")
    path = _regular_within(manifest_root, relative, f"{label} path")
    expected = record.get("sha256")
    actual = sha256_file(path)
    if not _is_sha256(expected) or expected != actual:
        raise ReplayError(f"{label} SHA-256 differs from its native manifest")
    byte_count = record.get("bytes")
    if byte_count is not None and (isinstance(byte_count, bool) or byte_count != path.stat().st_size):
        raise ReplayError(f"{label} byte count differs from its native manifest")
    return path, {"path": relative, "sha256": actual, "bytes": path.stat().st_size}


def verify_native_bundle(
    source_root: Path, candidate_manifest_path: Path, baseline_manifest_path: Path,
    *, expected_source_sha256: str | None = None,
) -> dict[str, Any]:
    """Check native manifest links and return the exact five-file payload."""
    if candidate_manifest_path.is_symlink() or baseline_manifest_path.is_symlink():
        raise ReplayError("native manifests may not be symlinks")
    candidate_manifest_path = candidate_manifest_path.resolve(strict=True)
    baseline_manifest_path = baseline_manifest_path.resolve(strict=True)
    candidate = read_json(candidate_manifest_path)
    baseline = read_json(baseline_manifest_path)
    for label, manifest in (("candidate", candidate), ("baseline", baseline)):
        if manifest.get("schema") != NATIVE_BUILD_SCHEMA:
            raise ReplayError(f"{label} native manifest has an unknown schema")
        if manifest.get("architecture") != "sm_70" or manifest.get("fast_math") is not False:
            raise ReplayError(f"{label} native manifest is not the approved SM70 non-fast-math build")

    source_sha_candidate, source_entries = _source_closure(source_root, candidate.get("source_files"))
    source_sha_baseline, baseline_entries = _source_closure(source_root, baseline.get("source_files"))
    if source_sha_candidate != source_sha_baseline or source_entries != baseline_entries:
        raise ReplayError("candidate and baseline manifests do not share the full source closure")
    declared_closure = candidate.get("source_closure_sha256")
    if declared_closure != source_sha_candidate or baseline.get("source_closure_sha256") != source_sha_candidate:
        raise ReplayError("native source closure digest does not match the complete source snapshot")
    source_tree_sha, source_entries_simple = _source_tree(source_root)
    if expected_source_sha256 is not None and expected_source_sha256 != source_tree_sha:
        raise ReplayError("source snapshot differs from its expected KDA tree identity")

    candidate_binaries = candidate.get("binaries")
    baseline_binaries = baseline.get("binaries")
    if not isinstance(candidate_binaries, dict) or not isinstance(baseline_binaries, dict):
        raise ReplayError("native manifests must contain binary identities")
    candidate_runner, candidate_runner_record = _binary(
        candidate_manifest_path.parent, candidate_binaries.get("runner"), "candidate runner")
    candidate_library, candidate_library_record = _binary(
        candidate_manifest_path.parent, candidate_binaries.get("factorized_interaction_cuda"), "shared CUDA library")
    baseline_runner, baseline_runner_record = _binary(
        baseline_manifest_path.parent, baseline_binaries.get("runner"), "baseline runner")
    baseline_library, _ = _binary(
        baseline_manifest_path.parent, baseline_binaries.get("factorized_interaction_cuda"), "baseline CUDA library")
    if candidate_runner == baseline_runner or candidate_library != baseline_library:
        raise ReplayError("direct and candidate manifests must bind distinct runners and one shared CUDA library")
    if candidate_binaries["factorized_interaction_cuda"].get("path") != baseline_binaries["factorized_interaction_cuda"].get("path") or \
            candidate_library_record["sha256"] != baseline_binaries["factorized_interaction_cuda"].get("sha256"):
        raise ReplayError("baseline and candidate manifests do not share identical CUDA library bytes")

    candidate_link = baseline.get("candidate_runner")
    if not isinstance(candidate_link, dict):
        raise ReplayError("baseline manifest is missing its candidate runner links")
    linked_manifest = _regular_within(baseline_manifest_path.parent,
                                      candidate_link.get("manifest_path"),
                                      "baseline candidate-manifest link")
    linked_runner = _regular_within(baseline_manifest_path.parent,
                                    candidate_link.get("binary_path"),
                                    "baseline candidate-runner link")
    if linked_manifest != candidate_manifest_path or linked_runner != candidate_runner or \
            candidate_link.get("manifest_sha256") != sha256_file(candidate_manifest_path) or \
            candidate_link.get("binary_sha256") != candidate_runner_record["sha256"]:
        raise ReplayError("baseline links do not match the candidate manifest and runner")

    artifact_root = Path(os.path.commonpath([candidate_manifest_path.parent,
                                             baseline_manifest_path.parent])).resolve(strict=True)
    roles_and_paths = [
        ("candidate_manifest", candidate_manifest_path),
        ("baseline_manifest", baseline_manifest_path),
        ("candidate_runner", candidate_runner),
        ("baseline_runner", baseline_runner),
        ("shared_cuda_library", candidate_library),
    ]
    assets: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for role, path in roles_and_paths:
        if not path.resolve(strict=True).is_relative_to(artifact_root):
            raise ReplayError(f"{role} is outside the shared native artifact root")
        relative = path.relative_to(artifact_root).as_posix()
        if relative in seen_paths:
            raise ReplayError("native manifest/binary roles unexpectedly alias one file")
        seen_paths.add(relative)
        assets.append({"role": role, "path": relative,
                       "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return {
        "source_tree_sha256": source_tree_sha,
        "source_closure_sha256": source_sha_candidate,
        "source_file_count": len(source_entries_simple),
        "source_files": candidate["source_files"],
        "artifact_root": str(artifact_root),
        "candidate_manifest_path": candidate_manifest_path.relative_to(artifact_root).as_posix(),
        "baseline_manifest_path": baseline_manifest_path.relative_to(artifact_root).as_posix(),
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "baseline_manifest_sha256": sha256_file(baseline_manifest_path),
        "assets": assets,
    }


def _task_case(case_id: str, recorded_workload_sha256: str, dataset_sha: str) -> dict[str, Any]:
    case = {
        "id": case_id,
        "weight": 1.0,
        "input": {
            "recorded_dataset_sha256": dataset_sha,
            "recorded_workload_sha256": recorded_workload_sha256,
        },
        "parameters": {"observation_scope": "one unique recorded CUDA block mean"},
    }
    case["workload_sha256"] = digest_json(case)
    return case


def _trusted(identifier: str, path: Path, role: str) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise ReplayError(f"trusted task input must be a regular file: {path}")
    return {"id": identifier, "path": str(resolved), "role": role}


def make_task(
    *, dataset_path: Path, source_root: Path, candidate_manifest_path: Path,
    baseline_manifest_path: Path, native_source_archive_path: Path,
    native_source_archive_manifest_path: Path,
    expected_native_source_archive_sha256: str,
    expected_native_source_archive_manifest_sha256: str,
    candidate_source_dir: Path, output_dir: Path,
    kda_root: Path = Path("/home/tumlinson/kda"),
    python: Path | None = None,
) -> dict[str, Any]:
    """Create an isolated CPU KDA task and its exact KDA source manifest."""
    if dataset_path.is_symlink() or source_root.is_symlink() or output_dir.is_symlink() or \
            candidate_source_dir.is_symlink():
        raise ReplayError("dataset, immutable source, candidate source and analysis output may not be symlinks")
    dataset_path = dataset_path.resolve(strict=True)
    source_root = source_root.resolve(strict=True)
    native_source_archive_path = native_source_archive_path.resolve(strict=True)
    native_source_archive_manifest_path = native_source_archive_manifest_path.resolve(strict=True)
    candidate_source_dir = candidate_source_dir.resolve()
    output_dir = output_dir.resolve()
    kda_root = kda_root.resolve(strict=True)
    python = (python or Path(sys.executable)).resolve(strict=True)
    if output_dir.is_relative_to(source_root):
        raise ReplayError("analysis output must stay outside the immutable source snapshot")
    if candidate_source_dir.is_relative_to(output_dir) or output_dir.is_relative_to(candidate_source_dir):
        raise ReplayError("candidate source package and analysis attempt must use separate directories")
    if candidate_source_dir.is_relative_to(source_root) or source_root.is_relative_to(candidate_source_dir):
        raise ReplayError("candidate source package must be separate from the extracted native source tree")
    workload_catalog = _fixed_workload_catalog(source_root)
    dataset = read_json(dataset_path)
    dataset_facts = validate_dataset(dataset, expected_workloads=workload_catalog)
    dataset_sha = sha256_file(dataset_path)
    recorded_inputs = _recorded_trusted_inputs(dataset, verify_files=True)
    bundle = verify_native_bundle(source_root, candidate_manifest_path,
                                  baseline_manifest_path)
    archive_identity = _verify_native_source_archive(
        native_source_archive_path, native_source_archive_manifest_path,
        expected_archive_sha256=expected_native_source_archive_sha256,
        expected_index_sha256=expected_native_source_archive_manifest_sha256,
        source_files=bundle["source_files"],
        source_closure_sha256=bundle["source_closure_sha256"],
    )
    native_identity = _validate_native_dataset_identity(dataset, source_root, bundle)
    native_source_tree_sha256 = bundle["source_tree_sha256"]
    source_descriptor = _make_kda_source_descriptor(
        source_tree_sha256=native_source_tree_sha256,
        source_closure_sha256=bundle["source_closure_sha256"],
        archive_identity=archive_identity,
        native_identity=native_identity,
        bundle=bundle,
        qualification_bundle_sha256=dataset_facts["qualification_bundle_sha256"],
    )
    published_source = _publish_kda_source(candidate_source_dir,
                                            native_source_archive_path,
                                            source_descriptor)

    output_dir.mkdir(parents=True, exist_ok=True)
    task_path = output_dir / "recorded_task.json"
    config_path = output_dir / "recorded_config.json"
    source_manifest_path = output_dir / "kda_source_manifest.json"
    recorded_inputs_path = output_dir / "recorded_trusted_inputs.json"
    candidate_source_sha = published_source["candidate_source_tree_sha256"]
    source_entries = published_source["candidate_source_files"]
    source_manifest = {
        "schema_version": 1,
        "files": [row["path"] for row in source_entries],
        "dependencies": [{"name": "cellerator-qualified-native-source-closure",
                           "revision": f"sha256:{bundle['source_closure_sha256']}"},
                          {"name": "cellerator-recorded-cuda-observations",
                           "revision": f"sha256:{dataset_sha}"}],
        "provenance": {
            "candidate_source_tree_sha256": candidate_source_sha,
            "native_source_tree_sha256": native_source_tree_sha256,
            "source_closure_sha256": bundle["source_closure_sha256"],
            "native_source_archive_tar_sha256": archive_identity["native_source_archive_tar_sha256"],
            "native_source_archive_manifest_sha256": archive_identity["native_source_archive_manifest_sha256"],
            "native_source_archive_gzip_sha256": published_source["native_source_archive_gzip_sha256"],
            "candidate_source_descriptor_sha256": published_source["candidate_source_descriptor_sha256"],
            "qualification_bundle_sha256": dataset_facts["qualification_bundle_sha256"],
        },
    }
    _write_json_exclusive(source_manifest_path, source_manifest)
    _write_json_exclusive(recorded_inputs_path, {
        "schema": "cellerator.factorized_interactions.recorded_trusted_inputs.v1",
        "inputs": recorded_inputs,
    })

    cases = [
        _task_case(case["id"], case["workload_sha256"], dataset_sha)
        for case in dataset["cases"]
    ]
    python_string = str(python)
    adapter_prefix = [python_string, str(_MODULE_PATH), "--config", str(config_path)]
    command_timeout = 120
    task_id = f"factorized-recorded-cuda-{dataset_sha[:12]}"
    task = {
        "schema_version": 1,
        "task_id": task_id,
        "recorded_analysis_schema": TASK_SCHEMA,
        "cases": cases,
        "commands": {
            "build": {"argv": adapter_prefix + ["build", "{request}", "{result}"],
                      "cpu_only": True, "timeout_seconds": command_timeout},
            "validate": {"argv": adapter_prefix + ["validate", "{request}", "{result}"],
                         "timeout_seconds": command_timeout},
            "benchmark": {"argv": adapter_prefix + ["benchmark", "{request}", "{result}"],
                          "timeout_seconds": command_timeout},
        },
        "trusted_inputs": [],
        "numerical_requirements": {
            "mode": "recorded-cuda-observation-analysis",
            "meaning": "No native runner or CUDA binary is executed; the fixed validator result is reused from the independently verified original CUDA packet.",
            "qualification_status": QUALIFICATION_STATUS,
            "dataset_sha256": dataset_sha,
            "qualification_bundle_sha256": dataset_facts["qualification_bundle_sha256"],
            "scored_arms": {"strongest-direct": "strongest_direct", "packet-candidate": "packet"},
            "descriptive_only_arm": "native_direct",
            "root_selection_remains_external": True,
        },
        "probe": {"argv": adapter_prefix + ["probe", "{request}", "{result}"],
                  "timeout_seconds": 60},
        "timing": {
            "metric": "recorded_primary_caller_wall_time",
            "unit": "ms_per_fwd_vjp_step",
            "direction": "minimize",
            "warmup": 0,
            "repetitions": 1,
            "rounds": BLOCKS_PER_PHASE,
            "confirmation_rounds": BLOCKS_PER_PHASE,
            "seed": 20261102,
            "scope": "one originally measured resident caller-wall block mean for one forward/VJP step; replayed as data on CPU",
        },
        "promotion": dict(PROMOTION),
        "target": {"kind": "cpu", "architecture": platform.machine()},
        "toolchain": {},
        "execution": {"mode": "direct"},
        "budgets": {
            "max_candidates": 2,
            "max_build_jobs": 1,
            "max_seconds": TASK_BUDGET_SECONDS,
        },
    }

    assets = list(bundle["assets"])
    config = {
        "schema": TASK_SCHEMA,
        "task_id": task_id,
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha,
        "qualification_status": dataset_facts["qualification_status"],
        "qualification_bundle_sha256": dataset_facts["qualification_bundle_sha256"],
        "recorded_device_uuid": dataset_facts["device_uuid"],
        "recorded_observations": _compact_observations(dataset),
        "source_root": str(source_root),
        "native_source_tree_sha256": native_source_tree_sha256,
        "source_tree_sha256": candidate_source_sha,
        "candidate_source_root": published_source["candidate_source_root"],
        "candidate_source_tree_sha256": candidate_source_sha,
        "candidate_source_files": published_source["candidate_source_files"],
        "native_source_descriptor_path": published_source["candidate_source_descriptor_path"],
        "native_source_descriptor_sha256": published_source["candidate_source_descriptor_sha256"],
        "native_source_descriptor": published_source["candidate_source_descriptor"],
        "source_closure_sha256": bundle["source_closure_sha256"],
        **archive_identity,
        "native_source_archive_gzip_path": published_source["native_source_archive_gzip_path"],
        "native_source_archive_gzip_sha256": published_source["native_source_archive_gzip_sha256"],
        "native_source_archive_gzip_bytes": published_source["native_source_archive_gzip_bytes"],
        "native_source_archive_gzip_compression": published_source["native_source_archive_gzip_compression"],
        "native_source_archive_gzip_mtime": published_source["native_source_archive_gzip_mtime"],
        "workload_catalog": workload_catalog,
        "native_identity": native_identity,
        "source_manifest_path": str(source_manifest_path.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "native_artifact_root": bundle["artifact_root"],
        "candidate_manifest_path": bundle["candidate_manifest_path"],
        "baseline_manifest_path": bundle["baseline_manifest_path"],
        "candidate_manifest_sha256": bundle["candidate_manifest_sha256"],
        "baseline_manifest_sha256": bundle["baseline_manifest_sha256"],
        "assets": assets,
        "recorded_trusted_inputs_manifest_path": str(recorded_inputs_path.resolve()),
        "recorded_trusted_inputs_manifest_sha256": sha256_file(recorded_inputs_path),
        "recorded_trusted_inputs_digest": digest_json(recorded_inputs),
        "recorded_trusted_input_count": len(recorded_inputs),
        "candidate_ids": dict(CANDIDATES),
        "kda_root": str(kda_root),
        "python": python_string,
        "case_ids": [case["id"] for case in cases],
        "task_cases": {
            case["id"]: {"task_workload_sha256": case["workload_sha256"],
                         "recorded_workload_sha256": dataset_facts["workload_sha256"][case["id"]]}
            for case in cases
        },
    }
    _write_json_exclusive(config_path, config)

    trusted = [
        _trusted("recorded_engine", _MODULE_PATH, "harness"),
        _trusted("recorded_dataset", dataset_path, "reference"),
        _trusted("recorded_configuration", config_path, "harness"),
        _trusted("kda_source_manifest", source_manifest_path, "harness"),
    ]
    for index, item in enumerate(assets):
        trusted.append(_trusted(f"native_artifact_{index}_{item['role']}",
                                Path(bundle["artifact_root"]) / item["path"],
                                "toolchain"))
    trusted.extend([
        _trusted("native_source_archive_gzip",
                 Path(published_source["native_source_archive_gzip_path"]), "toolchain"),
        _trusted("native_source_archive_member_index",
                 native_source_archive_manifest_path, "reference"),
        _trusted("native_source_descriptor",
                 Path(published_source["candidate_source_descriptor_path"]), "harness"),
    ])
    for item in recorded_inputs:
        trusted.append(_trusted(item["id"], Path(item["path"]), item["role"]))
    trusted.append(_trusted("recorded_trusted_input_manifest", recorded_inputs_path, "harness"))
    task["trusted_inputs"] = trusted
    _write_json_exclusive(task_path, task)
    return {
        "task_path": str(task_path),
        "config_path": str(config_path),
        "source_manifest_path": str(source_manifest_path),
        "task_sha256": sha256_file(task_path),
        "config_sha256": sha256_file(config_path),
        "source_tree_sha256": candidate_source_sha,
        "native_source_tree_sha256": native_source_tree_sha256,
        "candidate_source_root": published_source["candidate_source_root"],
        "candidate_source_tree_sha256": candidate_source_sha,
        "candidate_source_files": published_source["candidate_source_files"],
        "native_source_archive_gzip_sha256": published_source["native_source_archive_gzip_sha256"],
        "native_source_archive_tar_sha256": archive_identity["native_source_archive_tar_sha256"],
        "source_closure_sha256": bundle["source_closure_sha256"],
        "dataset_sha256": dataset_sha,
        "qualification_status": dataset_facts["qualification_status"],
        "candidate_ids": list(CANDIDATES),
    }


def _load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema") != TASK_SCHEMA:
        raise ReplayError(f"config schema must be {TASK_SCHEMA}")
    if config.get("candidate_ids") != CANDIDATES:
        raise ReplayError("config candidate IDs differ from the fixed strongest-direct/packet comparison")
    if not _is_sha256(config.get("dataset_sha256")):
        raise ReplayError("task config omitted its canonical recorded dataset digest")
    for key in ("native_source_tree_sha256", "source_tree_sha256",
                "candidate_source_tree_sha256", "source_closure_sha256",
                "native_source_archive_tar_sha256", "native_source_archive_manifest_sha256",
                "native_source_archive_members_sha256", "native_source_archive_gzip_sha256",
                "native_source_descriptor_sha256"):
        if not _is_sha256(config.get(key)):
            raise ReplayError(f"task config omitted a valid source identity: {key}")
    if config.get("source_tree_sha256") != config.get("candidate_source_tree_sha256"):
        raise ReplayError("KDA source tree and candidate source tree identities disagree")
    if not isinstance(config.get("candidate_source_root"), str) or \
            not isinstance(config.get("candidate_source_files"), list) or \
            len(config["candidate_source_files"]) != 2:
        raise ReplayError("task config omitted the two-file immutable candidate source package")
    workload_catalog = config.get("workload_catalog")
    if not isinstance(workload_catalog, list):
        raise ReplayError("config must bind the captured fixed workload descriptors")
    if len(workload_catalog) != 13 or \
            config.get("case_ids") != [row.get("id") for row in workload_catalog]:
        raise ReplayError("config workload catalog differs from the fixed 13-case source")
    _validate_compact_observations(config)
    if isinstance(config.get("recorded_trusted_input_count"), bool) or \
            not isinstance(config.get("recorded_trusted_input_count"), int) or \
            config["recorded_trusted_input_count"] < 1 or \
            not _is_sha256(config.get("recorded_trusted_inputs_digest")) or \
            not _is_sha256(config.get("recorded_trusted_inputs_manifest_sha256")) or \
            not isinstance(config.get("recorded_trusted_inputs_manifest_path"), str) or \
            not Path(config["recorded_trusted_inputs_manifest_path"]).is_absolute():
        raise ReplayError("config omitted retained recorded evidence identities")
    if sha256_file(Path(config["source_manifest_path"])) != config.get("source_manifest_sha256"):
        raise ReplayError("KDA source manifest changed after task creation")
    if config.get("native_source_archive_gzip_compression") != "gzip" or \
            config.get("native_source_archive_gzip_mtime") != 0:
        raise ReplayError("task config does not bind the deterministic gzip source transport")
    return config


def _verify_recorded_dataset(config: Mapping[str, Any], *, verify_evidence_files: bool) -> dict[str, Any]:
    """Revalidate original dataset/files at a candidate or KDA task boundary."""
    dataset_path = Path(config["dataset_path"])
    if sha256_file(dataset_path) != config["dataset_sha256"]:
        raise ReplayError("canonical recorded dataset changed after task creation")
    dataset = read_json(dataset_path)
    facts = validate_dataset(dataset, expected_workloads=config["workload_catalog"])
    recorded_inputs = _recorded_trusted_inputs(dataset, verify_files=verify_evidence_files)
    manifest_path = Path(config["recorded_trusted_inputs_manifest_path"])
    manifest = read_json(manifest_path)
    if sha256_file(manifest_path) != config["recorded_trusted_inputs_manifest_sha256"] or \
            manifest.get("schema") != "cellerator.factorized_interactions.recorded_trusted_inputs.v1" or \
            manifest.get("inputs") != recorded_inputs or \
            digest_json(recorded_inputs) != config["recorded_trusted_inputs_digest"] or \
            len(recorded_inputs) != config["recorded_trusted_input_count"]:
        raise ReplayError("recorded dataset trusted-input identities changed after task creation")
    _check_config_native_identity(dataset, config.get("native_identity", {}))
    for key, config_key in (("qualification_status", "qualification_status"),
                            ("qualification_bundle_sha256", "qualification_bundle_sha256"),
                            ("device_uuid", "recorded_device_uuid")):
        if config.get(config_key) != facts[key]:
            raise ReplayError(f"recorded dataset {key} changed after task creation")
    if config.get("case_ids") != facts["case_ids"] or \
            _compact_observations(dataset) != config.get("recorded_observations"):
        raise ReplayError("recorded observation table changed after task creation")
    return facts


def _verify_native_inputs(config: Mapping[str, Any], source_root: Path) -> dict[str, Any]:
    if source_root.resolve(strict=True) != Path(config["source_root"]).resolve(strict=True):
        raise ReplayError("native source verifier was given the KDA candidate snapshot instead of the original source tree")
    artifact_root = Path(config["native_artifact_root"])
    candidate_manifest = artifact_root / _safe_relative(config["candidate_manifest_path"], "candidate manifest")
    baseline_manifest = artifact_root / _safe_relative(config["baseline_manifest_path"], "baseline manifest")
    bundle = verify_native_bundle(source_root, candidate_manifest, baseline_manifest,
                                  expected_source_sha256=config["native_source_tree_sha256"])
    expected = {(item["role"], item["path"], item["sha256"]) for item in config["assets"]}
    actual = {(item["role"], item["path"], item["sha256"]) for item in bundle["assets"]}
    if expected != actual or bundle["source_closure_sha256"] != config["source_closure_sha256"]:
        raise ReplayError("native manifests or prebuilt assets changed after task creation")
    archive_identity = _verify_native_source_archive(
        Path(config["native_source_archive_tar_path"]),
        Path(config["native_source_archive_manifest_path"]),
        expected_archive_sha256=config["native_source_archive_tar_sha256"],
        expected_index_sha256=config["native_source_archive_manifest_sha256"],
        source_files=bundle["source_files"],
        source_closure_sha256=bundle["source_closure_sha256"],
    )
    for key, value in archive_identity.items():
        if config.get(key) != value:
            raise ReplayError(f"accepted native source archive identity changed: {key}")
    _verify_candidate_source_package(config)
    return bundle


def _verify_request_source(config: Mapping[str, Any], candidate: Mapping[str, Any]) -> Path:
    if candidate.get("id") not in CANDIDATES:
        raise ReplayError(f"unexpected KDA candidate id {candidate.get('id')!r}")
    source_value = candidate.get("source_dir")
    if not isinstance(source_value, str):
        raise ReplayError("KDA candidate request omitted source_dir")
    source_root = Path(source_value)
    source_sha, entries = _source_tree(source_root)
    if source_sha != config["candidate_source_tree_sha256"] or candidate.get("snapshot_sha256") != source_sha:
        raise ReplayError("KDA candidate snapshot differs from the gzip TAR+descriptor source identity")
    if [{"path": row["path"], "sha256": row["sha256"],
         "bytes": (source_root / row["path"]).stat().st_size} for row in entries] != \
            config["candidate_source_files"]:
        raise ReplayError("KDA candidate snapshot source members differ from the fixed two-file package")
    descriptor_path = source_root / KDA_SOURCE_DESCRIPTOR_NAME
    if sha256_file(descriptor_path) != config["native_source_descriptor_sha256"] or \
            read_json(descriptor_path) != config["native_source_descriptor"]:
        raise ReplayError("KDA candidate snapshot descriptor differs from the task-bound archive identity")
    return source_root.resolve(strict=True)


def _verify_candidate_source_package(config: Mapping[str, Any]) -> None:
    source_root = Path(config["candidate_source_root"])
    source_sha, entries = _source_tree(source_root)
    if source_sha != config["candidate_source_tree_sha256"]:
        raise ReplayError("immutable KDA candidate source package changed after task creation")
    normalized = [{"path": row["path"], "sha256": row["sha256"],
                   "bytes": (source_root / row["path"]).stat().st_size} for row in entries]
    if normalized != config["candidate_source_files"]:
        raise ReplayError("KDA candidate source package has unexpected files or byte sizes")
    descriptor_path = source_root / KDA_SOURCE_DESCRIPTOR_NAME
    if sha256_file(descriptor_path) != config["native_source_descriptor_sha256"] or \
            read_json(descriptor_path) != config["native_source_descriptor"]:
        raise ReplayError("native source descriptor changed after task creation")
    descriptor = config["native_source_descriptor"]
    if descriptor.get("native_source_transport_path") != KDA_SOURCE_ARCHIVE_NAME or \
            descriptor.get("native_source_transport_compression") != config["native_source_archive_gzip_compression"] or \
            descriptor.get("native_source_transport_gzip_mtime") != config["native_source_archive_gzip_mtime"] or \
            descriptor.get("native_source_transport_sha256") != config["native_source_archive_gzip_sha256"] or \
            descriptor.get("native_source_transport_bytes") != config["native_source_archive_gzip_bytes"]:
        raise ReplayError("candidate source descriptor and gzip transport identities disagree")
    archive_path = source_root / KDA_SOURCE_ARCHIVE_NAME
    if archive_path.resolve(strict=True) != Path(config["native_source_archive_gzip_path"]).resolve(strict=True) or \
            descriptor_path.resolve(strict=True) != Path(config["native_source_descriptor_path"]).resolve(strict=True) or \
            archive_path.stat().st_size != config["native_source_archive_gzip_bytes"] or \
            sha256_file(archive_path) != config["native_source_archive_gzip_sha256"]:
        raise ReplayError("deterministic gzip source transport changed after task creation")


def _native_payload_paths(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return list(config["assets"])


def _copy_prebuilt_payload(config: Mapping[str, Any],
                           build_dir: Path) -> list[dict[str, str]]:
    source_root = Path(config["source_root"])
    _verify_native_inputs(config, source_root)
    if build_dir.is_symlink() or not build_dir.is_dir():
        raise ReplayError("KDA build directory must be an existing regular directory")
    payload_root = build_dir / "prebuilt-native"
    payload_root.mkdir(exist_ok=False)
    source_artifact_root = Path(config["native_artifact_root"])
    copied: list[dict[str, str]] = []
    for item in _native_payload_paths(config):
        rel = _safe_relative(item["path"], "native artifact path")
        source = _regular_within(source_artifact_root, rel.as_posix(), item["role"])
        if source.stat().st_size != item["bytes"] or sha256_file(source) != item["sha256"]:
            raise ReplayError(f"prebuilt native artifact changed: {item['role']}")
        destination = payload_root.joinpath(*rel.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if destination.stat().st_size != item["bytes"] or sha256_file(destination) != item["sha256"]:
            raise ReplayError(f"prebuilt native artifact copy failed verification: {item['role']}")
        copied.append({"path": (Path("prebuilt-native") / Path(*rel.parts)).as_posix(),
                       "sha256": item["sha256"]})
    return copied


def _verify_candidate_build(config: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
    build = candidate.get("build")
    if not isinstance(build, dict) or not isinstance(build.get("directory"), str):
        raise ReplayError("KDA candidate request omitted its prepared build directory")
    build_dir = Path(build["directory"])
    artifact_records = build.get("artifacts")
    if not isinstance(artifact_records, list):
        raise ReplayError("KDA candidate request omitted prepared artifact identities")
    artifact_map = {row.get("path"): row.get("sha256") for row in artifact_records if isinstance(row, dict)}
    for item in config["assets"]:
        relative = (Path("prebuilt-native") / _safe_relative(item["path"], "native artifact path")).as_posix()
        path = _regular_within(build_dir, relative, "copied native artifact")
        if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise ReplayError(f"prepared native artifact differs from the verified source: {item['role']}")
        if artifact_map.get(relative) != item["sha256"]:
            raise ReplayError(f"KDA build artifact list omitted or changed {item['role']}")


def _task_case_contract(config: Mapping[str, Any], case: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(case, dict) or not isinstance(case.get("id"), str):
        raise ReplayError("KDA case request is malformed")
    case_id = case["id"]
    expected = config["task_cases"].get(case_id)
    if not isinstance(expected, dict):
        raise ReplayError(f"unexpected KDA workload {case_id!r}")
    descriptor = {key: value for key, value in case.items() if key != "workload_sha256"}
    if digest_json(descriptor) != expected["task_workload_sha256"] or \
            case.get("workload_sha256") != expected["task_workload_sha256"]:
        raise ReplayError(f"KDA workload descriptor changed for case {case_id}")
    if case.get("input", {}).get("recorded_dataset_sha256") != config["dataset_sha256"] or \
            case.get("input", {}).get("recorded_workload_sha256") != expected["recorded_workload_sha256"]:
        raise ReplayError(f"KDA case {case_id} is not bound to the canonical recorded dataset row")
    return case_id, expected


def _dispatch(config: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
    request_id = request.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise ReplayError("KDA request_id is missing")
    operation = request.get("operation")
    if operation in {"probe", "build", "validate"}:
        dataset_facts = _verify_recorded_dataset(config, verify_evidence_files=(operation == "probe"))
    else:
        dataset_facts = {
            "device_uuid": config["recorded_device_uuid"],
            "qualification_status": config["qualification_status"],
            "qualification_bundle_sha256": config["qualification_bundle_sha256"],
        }
    if operation == "probe":
        return {
            "schema_version": 1,
            "request_id": request_id,
            "status": "ok",
            "analysis_kind": "recorded_cuda_observation_replay",
            "environment": {
                "kind": "cpu",
                "architecture": platform.machine(),
                "tools": {
                    "python": platform.python_version(),
                    "recorded_engine_sha256": sha256_file(_MODULE_PATH),
                },
                "analysis_kind": "recorded_cuda_observation_replay",
                "process": {"pid": os.getpid(), "parent_pid": os.getppid()},
                "recorded_gpu_observation": {
                    "device_uuid": dataset_facts["device_uuid"],
                    "qualification_status": dataset_facts["qualification_status"],
                    "qualification_bundle_sha256": dataset_facts["qualification_bundle_sha256"],
                },
                "cuda_executed": False,
                "scheduler_admission_requested": False,
            },
        }

    if operation == "build":
        candidate = request.get("candidate")
        if not isinstance(candidate, dict) or candidate.get("id") not in CANDIDATES:
            raise ReplayError("KDA request candidate is not part of the fixed two-arm comparison")
        candidate_id = candidate["id"]
        source_root = _verify_request_source(config, candidate)
        build = request.get("build")
        if not isinstance(build, dict) or not isinstance(build.get("directory"), str):
            raise ReplayError("KDA build request omitted its output directory")
        build_dir = Path(build["directory"])
        copied = _copy_prebuilt_payload(config, build_dir)
        return {
            "schema_version": 1,
            "request_id": request_id,
            "status": "ok",
            "candidate_id": candidate_id,
            "artifacts": copied,
            "provenance": {
                "analysis_kind": "cpu_copy_of_prebuilt_native_artifacts",
                "executed_any_binary": False,
                "source_tree_sha256": config["source_tree_sha256"],
                "source_closure_sha256": config["source_closure_sha256"],
                "dataset_sha256": config["dataset_sha256"],
                "qualification_bundle_sha256": config["qualification_bundle_sha256"],
            },
        }

    candidate = request.get("candidate")
    if not isinstance(candidate, dict) or candidate.get("id") not in CANDIDATES:
        raise ReplayError("KDA request candidate is not part of the fixed two-arm comparison")
    candidate_id = candidate["id"]
    if candidate.get("snapshot_sha256") != config["candidate_source_tree_sha256"]:
        raise ReplayError("KDA candidate snapshot differs from the accepted source archive package")
    if operation == "validate":
        _verify_request_source(config, candidate)
        _verify_candidate_build(config, candidate)
        requested_cases = request.get("cases")
        if not isinstance(requested_cases, list) or len(requested_cases) != len(config["task_cases"]):
            raise ReplayError("KDA validation must request all 13 canonical workloads exactly once")
        checked: dict[str, dict[str, Any]] = {}
        for case in requested_cases:
            case_id, expected = _task_case_contract(config, case)
            if case_id in checked:
                raise ReplayError(f"duplicate validation workload {case_id}")
            checked[case_id] = {"case_id": case_id,
                                "workload_sha256": expected["task_workload_sha256"],
                                "status": "pass"}
        if set(checked) != set(config["task_cases"]):
            raise ReplayError("KDA validation request omitted fixed workloads")
        return {
            "schema_version": 1,
            "request_id": request_id,
            "status": "ok",
            "candidate_id": candidate_id,
            "cases": list(checked.values()),
            "validation_scope": "reused independently verified original CUDA qualification; no runner execution",
            "qualification_status": dataset_facts["qualification_status"],
            "qualification_bundle_sha256": dataset_facts["qualification_bundle_sha256"],
        }

    if request.get("operation") == "benchmark":
        case_id, expected = _task_case_contract(config, request.get("case"))
        phase = request.get("phase")
        round_index = request.get("round")
        if phase not in PHASES:
            raise ReplayError(f"unsupported recorded observation phase {phase!r}")
        if isinstance(round_index, bool) or not isinstance(round_index, int) or not 0 <= round_index < BLOCKS_PER_PHASE:
            raise ReplayError("recorded observation round must be an integer in [0, 6]")
        metric = request.get("timing")
        if not isinstance(metric, dict) or metric.get("metric") != "recorded_primary_caller_wall_time" or \
                metric.get("unit") != "ms_per_fwd_vjp_step" or metric.get("repetitions") != 1:
            raise ReplayError("KDA timing request differs from the one-sample recorded-observation contract")
        case_observations = config["recorded_observations"].get(case_id)
        if not isinstance(case_observations, dict) or \
                case_observations.get("recorded_workload_sha256") != expected["recorded_workload_sha256"]:
            raise ReplayError(f"compact observation table changed for {case_id}")
        block = case_observations["phases"][phase][round_index]
        arm_name = CANDIDATES[candidate_id]
        arm = block["arms"][arm_name]
        sample = _finite_positive(arm["primary_ms"], "recorded primary_ms")
        provenance = {
            "dataset_sha256": config["dataset_sha256"],
            "qualification_status": dataset_facts["qualification_status"],
            "qualification_bundle_sha256": dataset_facts["qualification_bundle_sha256"],
            "recorded_workload_sha256": expected["recorded_workload_sha256"],
            "recorded_observation_sha256": arm["recorded_observation_sha256"],
            "raw_metrics_sha256": arm["raw_metrics_sha256"],
            "validation_summary_sha256": arm["validation_summary_sha256"],
            "original_report_sha256": arm["original_report_sha256"],
            "recorded_cuda_device_uuid": arm["device_uuid"],
            "original_receipt_identity": arm["original_receipt_identity"],
            "scored_arm": arm_name,
            "native_direct_is_descriptive_only": True,
        }
        return {
            "schema_version": 1,
            "request_id": request_id,
            "status": "ok",
            "case_id": case_id,
            "workload_sha256": expected["task_workload_sha256"],
            "metric": "recorded_primary_caller_wall_time",
            "unit": "ms_per_fwd_vjp_step",
            "timing_scope": "one originally measured resident caller-wall block mean for one forward/VJP step; replayed as data on CPU",
            "warmup": 0,
            "repetitions": 1,
            "samples": [sample],
            "recorded_observation": provenance,
        }

    raise ReplayError(f"unsupported KDA operation {request.get('operation')!r}")


def _handle(config_path: Path, request_path: Path, result_path: Path, operation: str) -> None:
    request = read_json(request_path)
    if request.get("operation") != operation:
        raise ReplayError(f"request operation must be {operation!r}")
    config = _load_config(config_path)
    result = _dispatch(config, request)
    _write_json_exclusive(result_path, result)


def _run_kda_command(argv: list[str], *, kda_root: Path, timeout: float) -> dict[str, Any]:
    env = os.environ.copy()
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(kda_root) + (os.pathsep + old if old else "")
    started_ns = time.time_ns()
    process = subprocess.Popen(argv, cwd=kda_root, env=env, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               shell=False, start_new_session=True)
    try:
        stdout_bytes, stderr_bytes = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout_bytes, stderr_bytes = process.communicate()
        raise ReplayError(f"KDA CPU command exceeded {timeout:.0f}s: {argv[0]}")
    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    try:
        response = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ReplayError(f"KDA command did not return one JSON response: {stderr[-1000:]}") from exc
    if process.returncode != 0 or not isinstance(response, dict) or response.get("status") != "ok":
        raise ReplayError(f"KDA command failed ({process.returncode}): {stderr[-1600:]} {response!r}")
    return {
        "argv": argv,
        "pid": process.pid,
        "started_unix_ns": started_ns,
        "finished_unix_ns": time.time_ns(),
        "returncode": process.returncode,
        "stdout_sha256": hashlib.sha256(stdout_bytes).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr_bytes).hexdigest(),
        "response": response,
    }


def orchestrate(*, task_path: Path, config_path: Path, workspace: Path,
                receipt_path: Path | None = None,
                kda_root: Path | None = None) -> dict[str, Any]:
    """Register, prepare, and evaluate a fresh public KDA CPU workspace."""
    task_path = task_path.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    workspace = workspace.resolve()
    task = read_json(task_path)
    config = _load_config(config_path)
    if task.get("recorded_analysis_schema") != TASK_SCHEMA:
        raise ReplayError("task is not a recorded-observation CPU analysis")
    kda_root = (kda_root or Path(config["kda_root"])).resolve(strict=True)
    source_root = Path(config["source_root"])
    source_sha, _ = _source_tree(source_root)
    if source_sha != config["native_source_tree_sha256"]:
        raise ReplayError("original native source tree changed after task preparation")
    _verify_candidate_source_package(config)
    _verify_native_inputs(config, source_root)
    if workspace.exists() and any(workspace.iterdir()):
        raise ReplayError("KDA workspace must be fresh and empty; existing evidence is preserved")
    workspace.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_path or workspace.parent / "recorded_kda_cpu_receipt.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        raise ReplayError(f"refusing to overwrite existing CPU analysis receipt: {receipt_path}")

    python = config["python"]
    prefix = [python, "-m", "kda_experiments", "--task", str(task_path),
              "--workspace", str(workspace)]
    group = f"recorded-cuda-{config['dataset_sha256'][:12]}"
    source = config["candidate_source_root"]
    source_manifest = config["source_manifest_path"]
    candidates = list(CANDIDATES)
    commands: list[dict[str, Any]] = []
    for candidate_id in candidates:
        baseline = candidate_id == "strongest-direct"
        argv = prefix + ["register", candidate_id, source, "--manifest", source_manifest,
                         "--hypothesis", ("Strongest qualified direct control from recorded CUDA samples"
                                          if baseline else "Factorized packet candidate from recorded CUDA samples"),
                         "--change", "CPU-only replay of immutable recorded GPU observations"]
        if baseline:
            argv.append("--baseline")
        argv.extend(["--experiment-group", group])
        commands.append(_run_kda_command(argv, kda_root=kda_root, timeout=TASK_BUDGET_SECONDS))

    commands.append(_run_kda_command(prefix + ["prepare", *candidates, "--jobs", "1"],
                                     kda_root=kda_root, timeout=TASK_BUDGET_SECONDS))
    evaluation = _run_kda_command(prefix + ["evaluate", "--candidates", *candidates,
                                             "--incumbent", "strongest-direct"],
                                   kda_root=kda_root, timeout=TASK_BUDGET_SECONDS)
    commands.append(evaluation)
    response = evaluation["response"]
    run_id = response.get("run_id")
    summary_path_value = response.get("summary_path")
    if not isinstance(run_id, str) or not run_id or not isinstance(summary_path_value, str):
        raise ReplayError("KDA evaluate response omitted its completed run and summary path")
    summary_path = Path(summary_path_value)
    if summary_path.is_symlink() or not summary_path.is_file():
        raise ReplayError("KDA evaluate summary path does not match its completed workspace attempt")
    summary_path = summary_path.resolve(strict=True)
    if not summary_path.is_relative_to(workspace.resolve(strict=True)) or \
            summary_path.name != "summary.json" or summary_path.parent.name != run_id:
        raise ReplayError("KDA evaluate summary path does not match its completed workspace attempt")
    summary_sha = sha256_file(summary_path)
    summary = read_json(summary_path)
    if summary.get("status") != "complete" or summary.get("run_id") != run_id or \
            summary.get("task_sha256") != sha256_file(task_path):
        raise ReplayError("KDA evaluate summary identity does not match this CPU analysis task")
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "provisional_cpu_analysis_completed",
        "external_cpu_scheduler_receipt_and_cleanup": "pending",
        "outer_receipt_verification": "pending",
        "cleanup_verification": "pending",
        "selection_authority": "provisional KDA analysis only; external root policy and independent review remain required",
        "analysis_kind": "recorded_cuda_observation_replay",
        "cuda_executed": False,
        "scheduler_admission_requested": False,
        "host_process": {
            "pid": os.getpid(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "architecture": platform.machine(),
        },
        "task_path": str(task_path),
        "task_sha256": sha256_file(task_path),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "workspace": str(workspace),
        "recorded_cuda_observation": {
            "device_uuid": config["recorded_device_uuid"],
            "qualification_status": config["qualification_status"],
            "qualification_bundle_sha256": config["qualification_bundle_sha256"],
            "dataset_sha256": config["dataset_sha256"],
        },
        "commands": commands,
        "provisional_result": {
            "run_id": run_id,
            "decision": response.get("decision"),
            "rankings": response.get("rankings"),
            "summary_path": str(summary_path.resolve(strict=True)),
            "summary_sha256": summary_sha,
        },
    }
    _write_json_exclusive(receipt_path, receipt)
    return {"receipt_path": str(receipt_path), "receipt_sha256": sha256_file(receipt_path),
            "result": receipt}


def _command_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)

    for name in ("probe", "build", "validate", "benchmark"):
        command = commands.add_parser(name)
        command.add_argument("request", type=Path)
        command.add_argument("result", type=Path)

    make = commands.add_parser("make-task")
    make.add_argument("--dataset", type=Path, required=True)
    make.add_argument("--source", type=Path, required=True)
    make.add_argument("--candidate-manifest", type=Path, required=True)
    make.add_argument("--baseline-manifest", type=Path, required=True)
    make.add_argument("--native-source-archive", type=Path, required=True)
    make.add_argument("--native-source-archive-sha256", required=True)
    make.add_argument("--native-source-archive-manifest", type=Path, required=True)
    make.add_argument("--native-source-archive-manifest-sha256", required=True)
    make.add_argument("--candidate-source-dir", type=Path, required=True)
    make.add_argument("--output-dir", type=Path, required=True)
    make.add_argument("--kda-root", type=Path, default=Path("/home/tumlinson/kda"))
    make.add_argument("--python", type=Path)

    run = commands.add_parser("orchestrate")
    run.add_argument("--task", type=Path, required=True)
    run.add_argument("--config-path", type=Path, required=True)
    run.add_argument("--workspace", type=Path, required=True)
    run.add_argument("--receipt", type=Path)
    run.add_argument("--kda-root", type=Path)

    args = parser.parse_args(argv)
    try:
        if args.command == "make-task":
            result = make_task(
                dataset_path=args.dataset, source_root=args.source,
                candidate_manifest_path=args.candidate_manifest,
                baseline_manifest_path=args.baseline_manifest,
                native_source_archive_path=args.native_source_archive,
                native_source_archive_manifest_path=args.native_source_archive_manifest,
                expected_native_source_archive_sha256=args.native_source_archive_sha256,
                expected_native_source_archive_manifest_sha256=args.native_source_archive_manifest_sha256,
                candidate_source_dir=args.candidate_source_dir,
                output_dir=args.output_dir, kda_root=args.kda_root,
                python=args.python,
            )
        elif args.command == "orchestrate":
            result = orchestrate(task_path=args.task, config_path=args.config_path,
                                 workspace=args.workspace, receipt_path=args.receipt,
                                 kda_root=args.kda_root)
        else:
            if args.config is None:
                raise ReplayError("KDA command execution requires --config")
            _handle(args.config, args.request, args.result, args.command)
            return 0
    except Exception as exc:
        parser.error(f"recorded KDA operation failed: {type(exc).__name__}: {exc}")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_command_main())
