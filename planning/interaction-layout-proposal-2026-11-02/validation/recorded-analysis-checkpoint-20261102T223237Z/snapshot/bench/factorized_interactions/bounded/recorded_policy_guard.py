#!/usr/bin/env python3
"""Apply the predeclared exact-case policy to a completed CPU replay.

The guard reads KDA and the recorded CUDA evidence. It never runs a runner,
queries a GPU, changes KDA's preferred candidate, or creates a Scheduler claim.
Its output remains conditional until an operator binds the finished CPU attempt
receipt and independent reviewers accept the evidence.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
from typing import Any, Mapping

sys.dont_write_bytecode = True

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import recorded_dataset  # noqa: E402
import recorded_engine  # noqa: E402
import recorded_qualification  # noqa: E402
import recorded_receipts  # noqa: E402

SCHEMA = "cellerator.factorized_interactions.recorded_policy_guard.v1"
DATASET_SCHEMA = "cellerator.factorized_interactions.recorded_dataset.v1"
TASK_SCHEMA = "cellerator.factorized_interactions.recorded_kda_cpu_task.v1"
ANALYSIS_SCHEMA = "cellerator.factorized_interactions.recorded_kda_cpu_receipt.v1"
PROMOTION = {
    "min_improvement": 0.0,
    "max_regression": 1.0 - 1.0 / 1.05,
    "confidence": 0.95,
    "bootstrap_samples": 10_000,
}
MIN_PACKET_SPEEDUP = 1.10
MAX_PACKET_TIME_RATIO = 1.05
BLOCKS = 7
DIRECT = "strongest_direct"
PACKET = "packet"
CANDIDATES = {"strongest-direct": DIRECT, "packet-candidate": PACKET}
_HEX = re.compile(r"^[0-9a-f]{64}$")
_NATIVE_ARCHIVE_SCHEMA = "cellerator.factorized_interactions.archived_source_closure.v1"
_NATIVE_DESCRIPTOR_SCHEMA = "cellerator.factorized_interactions.kda_native_source_descriptor.v1"
_NATIVE_SOURCE_TAR_SHA256 = "203fe5e743a1a2928afcdcb1b8085e6b2c86defebbbdc0a6e497d178a43e0078"
_NATIVE_SOURCE_TAR_BYTES = 23_162_880
_NATIVE_SOURCE_INDEX_SHA256 = "84071adc04a37f3c50f0f5c73b96af1ae78e2656b8dd29896f45d4aaf82f6b6a"
_NATIVE_SOURCE_GZIP_SHA256 = "f7c9440ea4acd2a75082a9518d6c8c0329b51b55026fa9120b0f110877265c42"
_NATIVE_SOURCE_GZIP_BYTES = 3_915_473
_NATIVE_SOURCE_CLOSURE_SHA256 = "1f8e34ce89594fc3d8fb3d63c46600091c24f3016d6a9ddfd9dc3630eb08543c"
_NATIVE_SOURCE_TREE_SHA256 = "85467fee280b00194168160f52a4e7ab4f2753032e5599b86b40073636fdb900"
_NATIVE_SOURCE_MEMBERS_SHA256 = "03461f085ce4878a292582bb58778ee5b4422b73ec4b293d29df5d7594ca1b66"
_NATIVE_SOURCE_MEMBER_COUNT = 3_979
_NATIVE_SOURCE_BYTES = 20_035_237


class PolicyError(ValueError):
    """Evidence is incomplete, inconsistent, or outside the fixed policy."""


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise PolicyError(f"expected regular evidence file: {path}")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _harness_identities() -> dict[str, str]:
    modules = {
        "policy_guard": Path(__file__),
        "recorded_dataset": Path(recorded_dataset.__file__),
        "recorded_engine": Path(recorded_engine.__file__),
        "recorded_qualification": Path(recorded_qualification.__file__),
        "recorded_receipts": Path(recorded_receipts.__file__),
    }
    return {name: _sha(path.resolve(strict=True)) for name, path in modules.items()}


def _digest_json(value: Any) -> str:
    try:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PolicyError(f"value is not canonical JSON: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PolicyError(f"expected regular JSON evidence file: {path}")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"),
                         object_pairs_hook=_reject_duplicates,
                         parse_constant=lambda item: (_ for _ in ()).throw(
                             PolicyError(f"non-finite JSON number: {item}")))
    except (OSError, json.JSONDecodeError) as exc:
        raise PolicyError(f"cannot read JSON evidence {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise PolicyError(f"expected JSON object in {path}")
    return obj


def _reject_duplicates(items: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in items:
        if key in out:
            raise PolicyError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyError(f"{label} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise PolicyError(f"{label} must be a positive finite number")
    return result


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n != BLOCKS:
        raise PolicyError("each confirmation arm must contain exactly seven paired blocks")
    middle = n // 2
    return ordered[middle]


def _safe_artifact(root: Path, relative: str, label: str) -> Path:
    rel = PurePosixPath(relative)
    if not isinstance(relative, str) or not relative or "\\" in relative or \
            rel.is_absolute() or any(part in ("", ".", "..") for part in rel.parts):
        raise PolicyError(f"{label} path must stay relative to its KDA attempt")
    base = root.resolve(strict=True)
    candidate = base
    for part in rel.parts:
        candidate /= part
        if candidate.is_symlink():
            raise PolicyError(f"{label} contains a symbolic link")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(base) or not resolved.is_file():
        raise PolicyError(f"{label} escapes its KDA attempt")
    return resolved


def _public_kda_path(workspace: Path, path: Path, label: str) -> Path:
    """Resolve a KDA-owned output without following workspace symlinks."""
    root = workspace.resolve(strict=True)
    raw = Path(path)
    if not raw.is_absolute():
        raise PolicyError(f"{label} must be an absolute path")
    try:
        relative = raw.relative_to(root)
    except ValueError as exc:
        raise PolicyError(f"{label} is outside its KDA workspace") from exc
    if not relative.parts or any(part in ("", ".", "..") for part in relative.parts):
        raise PolicyError(f"{label} has an unsafe KDA workspace path")
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise PolicyError(f"{label} contains a symbolic link")
    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise PolicyError(f"{label} escapes its KDA workspace")
    return resolved


def _task_trusted_inputs(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Reproduce KDA's trusted-input identities for its completed summary."""
    result = []
    trusted = task.get("trusted_inputs")
    if not isinstance(trusted, list) or not trusted:
        raise PolicyError("KDA task has no trusted inputs")
    for row in trusted:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or \
                not isinstance(row.get("role"), str) or not isinstance(row.get("path"), str):
            raise PolicyError("malformed KDA trusted input")
        path = Path(row["path"])
        if path.is_symlink():
            raise PolicyError("KDA trusted input may not be a symbolic link")
        if path.is_dir():
            files = []
            for current, dirs, names in os.walk(path, followlinks=False):
                base = Path(current)
                if any((base / name).is_symlink() for name in dirs):
                    raise PolicyError("symbolic link in a KDA trusted-input directory")
                for name in sorted(names):
                    child = base / name
                    if child.is_symlink() or not child.is_file():
                        raise PolicyError("non-regular KDA trusted input")
                    files.append({"path": child.relative_to(path).as_posix(), "sha256": _sha(child)})
            digest = _digest_json(files)
            result.append({"id": row["id"], "role": row["role"], "path": str(path),
                           "files": files, "sha256": digest})
        else:
            result.append({"id": row["id"], "role": row["role"], "path": str(path),
                           "sha256": _sha(path)})
    return result


def _verify_rebuilt_dataset(dataset: Mapping[str, Any], config: Mapping[str, Any],
                            dataset_path: Path) -> None:
    """Reopen the source-bound reports, receipts, and bundle before scoring."""
    from recorded_dataset import assemble_dataset

    source = Path(config["source_root"]).resolve(strict=True)
    candidate = Path(config["native_artifact_root"]) / config["candidate_manifest_path"]
    baseline = Path(config["native_artifact_root"]) / config["baseline_manifest_path"]
    qualification_path = Path(dataset["qualification_bundle"]["path"])
    sanitizer_wrapper = Path(dataset["sanitizer_wrapper"]["path"])
    items: dict[tuple[str, str], tuple[Path, Path, Path]] = {}
    for row in dataset["timing_reports"]:
        key = (row["workload"], row["phase"])
        if key in items:
            raise PolicyError("dataset repeats an original timing partition")
        items[key] = (Path(row["report_path"]), Path(row["receipt_path"]), Path(row["pointer_path"]))
    rebuilt = assemble_dataset(items=items, source=source,
                               candidate_manifest=candidate, baseline_manifest=baseline,
                               qualification_bundle=qualification_path,
                               expected_uuid=dataset["identity"]["device_uuid"],
                               sanitizer_wrapper=sanitizer_wrapper)
    recorded_dataset.validate_dataset(rebuilt)
    if rebuilt != dataset:
        raise PolicyError("normalized dataset differs from the original reports and receipts")
    if _sha(dataset_path) != config.get("dataset_sha256"):
        raise PolicyError("dataset file differs from the KDA task configuration")
    for ref in dataset.get("trusted_inputs", []):
        if not isinstance(ref, dict) or not isinstance(ref.get("path"), str) or \
                not _HEX.fullmatch(str(ref.get("sha256", ""))):
            raise PolicyError("dataset has a malformed trusted-input reference")
        path = Path(ref["path"])
        if path.is_symlink():
            raise PolicyError("dataset source evidence contains a symbolic link")
        if path.is_file():
            actual = _sha(path)
        elif path.is_dir():
            entries = []
            for current, dirs, names in os.walk(path, followlinks=False):
                base = Path(current)
                if any((base / name).is_symlink() for name in dirs):
                    raise PolicyError("symbolic link in retained dataset evidence")
                for name in sorted(names):
                    child = base / name
                    if child.is_symlink() or not child.is_file():
                        raise PolicyError("non-regular retained dataset evidence")
                    entries.append({"path": child.relative_to(path).as_posix(), "sha256": _sha(child)})
            actual = _digest_json(entries)
        else:
            raise PolicyError(f"retained dataset evidence is missing: {path}")
        if actual != ref["sha256"]:
            raise PolicyError("retained dataset evidence changed after collection")


def _expected_compact_observations(dataset: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild the exact KDA config lookup table from all 546 source rows."""
    result: dict[str, Any] = {}
    for case in dataset.get("cases", []):
        if not isinstance(case, dict) or not isinstance(case.get("id"), str):
            raise PolicyError("recorded dataset contains a malformed observation case")
        phases = {}
        for phase in ("exploration", "confirmation"):
            blocks = case.get("phases", {}).get(phase)
            if not isinstance(blocks, list) or len(blocks) != BLOCKS:
                raise PolicyError(f"recorded observation table lacks {case['id']}/{phase} blocks")
            phase_rows = []
            for block_index, block in enumerate(blocks):
                if not isinstance(block, dict) or block.get("block_index") != block_index:
                    raise PolicyError("recorded observation table block indices are malformed")
                arms = block.get("arms")
                if not isinstance(arms, dict) or set(arms) != {DIRECT, PACKET, "native_direct"}:
                    raise PolicyError("recorded observation table has incomplete arms")
                compact_arms = {}
                for arm_name in (DIRECT, PACKET, "native_direct"):
                    arm = arms[arm_name]
                    identity = {key: arm.get(key) for key in
                                ("scheduler_job_id", "scheduler_attempt_id", "autobench_attempt_id")}
                    if any(not isinstance(value, str) or not value for value in identity.values()):
                        raise PolicyError("recorded observation table arm lacks original receipt identity")
                    compact_arms[arm_name] = {
                        "primary_ms": _finite_positive(arm.get("primary_ms"),
                                                        f"{case['id']} {phase} primary mean"),
                        "raw_metrics_sha256": arm.get("raw_metrics_sha256"),
                        "validation_summary_sha256": arm.get("validation_summary_sha256"),
                        "original_report_sha256": arm.get("original_report_sha256"),
                        "recorded_observation_sha256": _digest_json(arm),
                        "device_uuid": arm.get("device_uuid"),
                        "original_receipt_identity": identity,
                    }
                    for key in ("raw_metrics_sha256", "validation_summary_sha256",
                                "original_report_sha256"):
                        if not _HEX.fullmatch(str(compact_arms[arm_name][key])):
                            raise PolicyError(f"recorded table arm has invalid {key}")
                    if compact_arms[arm_name]["device_uuid"] != dataset.get("identity", {}).get("device_uuid"):
                        raise PolicyError("recorded table changes the original CUDA device")
                phase_rows.append({"block_index": block_index, "arms": compact_arms})
            phases[phase] = phase_rows
        result[case["id"]] = {"recorded_workload_sha256": case.get("workload_sha256"),
                               "phases": phases}
    if len(result) != 13:
        raise PolicyError("compact recorded observation table must contain all 13 workloads")
    return result


def _verify_compact_observations(dataset: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    expected = _expected_compact_observations(dataset)
    if config.get("recorded_observations") != expected:
        raise PolicyError("KDA compact observation table differs from the 546 original observations")


def _expected_recorded_trusted_inputs(dataset: Mapping[str, Any]) -> list[dict[str, str]]:
    rows = dataset.get("trusted_inputs")
    if not isinstance(rows, list) or not rows:
        raise PolicyError("recorded dataset has no preserved upstream report and receipt inputs")
    normalized = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or \
                not isinstance(row.get("path"), str) or not _HEX.fullmatch(str(row.get("sha256", ""))) or \
                not isinstance(row.get("role"), str):
            raise PolicyError("recorded dataset trusted-input row is malformed")
        original_id = row["id"]
        if not original_id or original_id in seen:
            raise PolicyError("recorded dataset trusted-input IDs must be unique")
        seen.add(original_id)
        source_role = row["role"]
        kda_role = ("validator" if source_role == "fixed_sanitizer_partition_validator" else
                    "benchmark" if source_role in {"timed_helper_evidence", "pre_cleanup_validation_record",
                                                    "validator_stdout", "validator_stderr"} else
                    "reference")
        normalized.append({
            "id": "recorded_evidence_" + hashlib.sha256(original_id.encode("utf-8")).hexdigest(),
            "dataset_id": original_id,
            "path": str(Path(row["path"]).resolve(strict=True)),
            "sha256": row["sha256"],
            "role": kda_role,
            "recorded_role": source_role,
        })
    normalized.sort(key=lambda item: item["id"])
    return normalized


def _verify_recorded_trusted_inputs(dataset: Mapping[str, Any], task: Mapping[str, Any],
                                    config: Mapping[str, Any]) -> None:
    expected = _expected_recorded_trusted_inputs(dataset)
    manifest_raw = config.get("recorded_trusted_inputs_manifest_path")
    manifest_sha = config.get("recorded_trusted_inputs_manifest_sha256")
    if not isinstance(manifest_raw, str) or not Path(manifest_raw).is_absolute() or \
            not _HEX.fullmatch(str(manifest_sha)):
        raise PolicyError("KDA config omits the retained-evidence sidecar identity")
    manifest_path = Path(manifest_raw)
    manifest = _read(manifest_path)
    if _sha(manifest_path) != manifest_sha or \
            manifest.get("schema") != "cellerator.factorized_interactions.recorded_trusted_inputs.v1" or \
            manifest.get("inputs") != expected or \
            config.get("recorded_trusted_inputs_digest") != _digest_json(expected) or \
            config.get("recorded_trusted_input_count") != len(expected):
        raise PolicyError("KDA retained-evidence sidecar differs from the canonical dataset")
    task_rows = task.get("trusted_inputs")
    if not isinstance(task_rows, list):
        raise PolicyError("KDA task omits normalized upstream evidence inputs")
    expected_task = [{"id": item["id"], "path": item["path"], "role": item["role"]}
                     for item in expected]
    actual_task = [{"id": row.get("id"), "path": row.get("path"), "role": row.get("role")}
                   for row in task_rows if isinstance(row, dict) and
                   isinstance(row.get("id"), str) and row["id"].startswith("recorded_evidence_")]
    if actual_task != expected_task:
        raise PolicyError("KDA task does not trust every preserved upstream report and receipt")
    sidecar_rows = [{"id": row.get("id"), "path": row.get("path"), "role": row.get("role")}
                    for row in task_rows if isinstance(row, dict) and
                    row.get("id") == "recorded_trusted_input_manifest"]
    expected_sidecar = [{"id": "recorded_trusted_input_manifest",
                         "path": str(manifest_path.resolve(strict=True)), "role": "harness"}]
    if sidecar_rows != expected_sidecar:
        raise PolicyError("KDA task does not trust its retained-evidence sidecar")
    # The dataset reader authenticates these five pinned sidecars against the
    # approved timing harness. Repeat the identity check at the KDA boundary
    # so none can be omitted while creating the replay task.
    approved = recorded_dataset.APPROVED_TIMING_HARNESS_SHA256
    required = {f"timing_harness_{key.removesuffix('_sha256')}": value
                for key, value in approved.items() if key != "sanitizer_wrapper_sha256"}
    required["fixed_sanitizer_partition_validator"] = approved["sanitizer_wrapper_sha256"]
    dataset_rows = dataset.get("trusted_inputs", [])
    task_by_id = {row["id"]: row for row in task_rows if isinstance(row, dict) and
                  isinstance(row.get("id"), str)}
    for role, expected_sha in required.items():
        matching = [row for row in dataset_rows if isinstance(row, dict) and
                    row.get("role") == role and row.get("sha256") == expected_sha]
        if len(matching) != 1:
            raise PolicyError(f"dataset does not retain exactly one approved {role} sidecar")
        item = matching[0]
        path = Path(item["path"])
        if _sha(path) != expected_sha:
            raise PolicyError(f"approved {role} sidecar bytes differ from their pinned hash")
        normalized_id = "recorded_evidence_" + hashlib.sha256(item["id"].encode("utf-8")).hexdigest()
        bound = task_by_id.get(normalized_id)
        if not isinstance(bound, dict) or bound.get("path") != str(path.resolve(strict=True)) or \
                bound.get("role") != ("validator" if role == "fixed_sanitizer_partition_validator" else "reference"):
            raise PolicyError(f"KDA task does not bind the approved {role} sidecar")


def _absolute_regular(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise PolicyError(f"{label} must be an absolute file path")
    path = Path(value)
    if path.is_symlink() or not path.is_file():
        raise PolicyError(f"{label} must be a regular non-symlink file")
    return path.resolve(strict=True)


def _root_file(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise PolicyError(f"{label} must be a relative POSIX path")
    rel = PurePosixPath(relative)
    if rel.is_absolute() or any(part in ("", ".", "..") for part in rel.parts):
        raise PolicyError(f"{label} escapes its declared native artifact root")
    if root.is_symlink():
        raise PolicyError(f"{label} root is a symbolic link")
    base = root.resolve(strict=True)
    current = base
    for part in rel.parts:
        current /= part
        if current.is_symlink():
            raise PolicyError(f"{label} contains a symbolic link")
    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(base) or not resolved.is_file():
        raise PolicyError(f"{label} is missing or outside its declared native artifact root")
    return resolved


def _archive_source_closure(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    previous = ""
    for row in rows:
        if not isinstance(row, dict):
            raise PolicyError("native source index contains a malformed member row")
        name, size, expected = row.get("path"), row.get("bytes"), row.get("sha256")
        if not isinstance(name, str) or name <= previous or \
                isinstance(size, bool) or not isinstance(size, int) or size < 0 or \
                not _HEX.fullmatch(str(expected)):
            raise PolicyError("native source index paths, sizes, or hashes are invalid")
        rel = PurePosixPath(name)
        if "\\" in name or rel.is_absolute() or any(part in ("", ".", "..") for part in rel.parts):
            raise PolicyError("native source index contains an unsafe member path")
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(size.to_bytes(8, "little"))
        digest.update(bytes.fromhex(expected))
        previous = name
    return digest.hexdigest()


def _verify_gzipped_native_tar(gzip_path: Path, rows: list[dict[str, Any]]) -> None:
    """Hash and inspect the exact accepted source TAR without extracting it."""
    header = gzip_path.open("rb")
    try:
        gzip_header = header.read(10)
    finally:
        header.close()
    if gzip_header != bytes.fromhex("1f8b08000000000000ff"):
        raise PolicyError("native source gzip header is not the accepted deterministic mtime-zero form")
    decompressed_hash = hashlib.sha256()
    decompressed_bytes = 0
    try:
        with gzip.open(gzip_path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                decompressed_bytes += len(chunk)
                decompressed_hash.update(chunk)
    except (OSError, EOFError) as exc:
        raise PolicyError(f"native source gzip stream is invalid: {exc}") from exc
    if decompressed_hash.hexdigest() != _NATIVE_SOURCE_TAR_SHA256 or \
            decompressed_bytes != _NATIVE_SOURCE_TAR_BYTES:
        raise PolicyError("native source gzip does not decompress to the pinned accepted TAR")

    try:
        with gzip.open(gzip_path, "rb") as compressed:
            with tarfile.open(fileobj=compressed, mode="r|") as archive:
                count = 0
                for member in archive:
                    if count >= len(rows):
                        raise PolicyError("native source TAR contains unindexed extra members")
                    row = rows[count]
                    if not member.isfile() or member.type != tarfile.REGTYPE or \
                            member.issparse() or member.name != row["path"] or \
                            member.size != row["bytes"] or member.mode != 0o444 or \
                            member.uid != 0 or member.gid != 0 or member.mtime != 0 or \
                            member.uname != "" or member.gname != "" or \
                            member.pax_headers not in ({}, {"path": member.name}):
                        raise PolicyError(f"native source TAR member differs from its normalized index: {member.name}")
                    payload = archive.extractfile(member)
                    if payload is None:
                        raise PolicyError(f"native source TAR member has no regular payload: {member.name}")
                    digest = hashlib.sha256()
                    byte_count = 0
                    with payload:
                        for chunk in iter(lambda: payload.read(1 << 20), b""):
                            digest.update(chunk)
                            byte_count += len(chunk)
                    if byte_count != row["bytes"] or digest.hexdigest() != row["sha256"]:
                        raise PolicyError(f"native source TAR payload differs from its qualified source: {member.name}")
                    count += 1
    except PolicyError:
        raise
    except (OSError, EOFError, tarfile.TarError) as exc:
        raise PolicyError(f"cannot inspect native source TAR: {exc}") from exc
    if count != len(rows) or count != _NATIVE_SOURCE_MEMBER_COUNT:
        raise PolicyError("native source TAR member count differs from its pinned source inventory")


def _verify_native_source_archive(dataset: Mapping[str, Any], task: Mapping[str, Any],
                                  config: Mapping[str, Any]) -> None:
    """Bind KDA's two-file candidate snapshot to the accepted native source archive."""
    identity = dataset.get("identity")
    if not isinstance(identity, dict) or \
            identity.get("source_closure_sha256") != _NATIVE_SOURCE_CLOSURE_SHA256 or \
            config.get("source_closure_sha256") != _NATIVE_SOURCE_CLOSURE_SHA256:
        raise PolicyError("recorded dataset/config source closure is not the qualified native source")

    tar_path = _absolute_regular(config.get("native_source_archive_tar_path"),
                                 "accepted native source TAR")
    index_path = _absolute_regular(config.get("native_source_archive_manifest_path"),
                                    "accepted native source member index")
    gzip_path = _absolute_regular(config.get("native_source_archive_gzip_path"),
                                  "KDA native source gzip")
    descriptor_path = _absolute_regular(config.get("native_source_descriptor_path"),
                                         "KDA native source descriptor")
    if _sha(tar_path) != _NATIVE_SOURCE_TAR_SHA256 or \
            tar_path.stat().st_size != _NATIVE_SOURCE_TAR_BYTES or \
            config.get("native_source_archive_tar_sha256") != _NATIVE_SOURCE_TAR_SHA256 or \
            config.get("native_source_archive_tar_bytes") != _NATIVE_SOURCE_TAR_BYTES:
        raise PolicyError("accepted native source TAR bytes differ from the qualified pinned source")
    if _sha(index_path) != _NATIVE_SOURCE_INDEX_SHA256 or \
            config.get("native_source_archive_manifest_sha256") != _NATIVE_SOURCE_INDEX_SHA256:
        raise PolicyError("native source member index differs from its accepted pinned bytes")
    if _sha(gzip_path) != _NATIVE_SOURCE_GZIP_SHA256 or \
            gzip_path.stat().st_size != _NATIVE_SOURCE_GZIP_BYTES or \
            config.get("native_source_archive_gzip_sha256") != _NATIVE_SOURCE_GZIP_SHA256 or \
            config.get("native_source_archive_gzip_bytes") != _NATIVE_SOURCE_GZIP_BYTES or \
            config.get("native_source_archive_gzip_compression") != "gzip" or \
            config.get("native_source_archive_gzip_mtime") != 0:
        raise PolicyError("KDA native source transport differs from the pinned deterministic gzip")

    index = _read(index_path)
    rows = index.get("files")
    if index.get("schema") != _NATIVE_ARCHIVE_SCHEMA or not isinstance(rows, list) or \
            len(rows) != _NATIVE_SOURCE_MEMBER_COUNT or \
            index.get("archive_path") != tar_path.name or \
            index.get("archive_sha256") != _NATIVE_SOURCE_TAR_SHA256 or \
            index.get("archive_bytes") != _NATIVE_SOURCE_TAR_BYTES or \
            index.get("source_closure_sha256") != _NATIVE_SOURCE_CLOSURE_SHA256 or \
            index.get("file_count") != _NATIVE_SOURCE_MEMBER_COUNT or \
            index.get("source_bytes") != _NATIVE_SOURCE_BYTES:
        raise PolicyError("accepted native source member index disagrees with the qualified archive")
    if _archive_source_closure(rows) != _NATIVE_SOURCE_CLOSURE_SHA256 or \
            _digest_json(rows) != _NATIVE_SOURCE_MEMBERS_SHA256 or \
            config.get("native_source_archive_members_sha256") != _NATIVE_SOURCE_MEMBERS_SHA256 or \
            config.get("native_source_archive_member_count") != _NATIVE_SOURCE_MEMBER_COUNT or \
            config.get("native_source_archive_source_bytes") != _NATIVE_SOURCE_BYTES:
        raise PolicyError("native source member inventory does not reproduce the qualified closure")
    native_tree_sha = _digest_json([{"path": row["path"], "sha256": row["sha256"]}
                                    for row in rows])
    if native_tree_sha != _NATIVE_SOURCE_TREE_SHA256 or \
            config.get("native_source_tree_sha256") != _NATIVE_SOURCE_TREE_SHA256:
        raise PolicyError("native source tree digest differs from its complete accepted member inventory")

    artifact_root_raw = config.get("native_artifact_root")
    if not isinstance(artifact_root_raw, str) or not Path(artifact_root_raw).is_absolute():
        raise PolicyError("native artifact root is missing from recorded analysis config")
    artifact_root = Path(artifact_root_raw)
    candidate_manifest_path = _root_file(artifact_root, config.get("candidate_manifest_path"),
                                         "candidate native manifest")
    baseline_manifest_path = _root_file(artifact_root, config.get("baseline_manifest_path"),
                                        "baseline native manifest")
    candidate_manifest = _read(candidate_manifest_path)
    baseline_manifest = _read(baseline_manifest_path)
    if candidate_manifest.get("source_files") != rows or baseline_manifest.get("source_files") != rows or \
            candidate_manifest.get("source_closure_sha256") != _NATIVE_SOURCE_CLOSURE_SHA256 or \
            baseline_manifest.get("source_closure_sha256") != _NATIVE_SOURCE_CLOSURE_SHA256 or \
            _sha(candidate_manifest_path) != config.get("candidate_manifest_sha256") or \
            _sha(baseline_manifest_path) != config.get("baseline_manifest_sha256"):
        raise PolicyError("native manifests differ from the accepted source index or recorded identity")

    candidate_root_raw = config.get("candidate_source_root")
    if not isinstance(candidate_root_raw, str) or not Path(candidate_root_raw).is_absolute():
        raise PolicyError("KDA candidate source root is missing")
    candidate_root = Path(candidate_root_raw)
    if candidate_root.is_symlink() or not candidate_root.is_dir():
        raise PolicyError("KDA candidate source root is missing or a symbolic link")
    expected_names = ["analysis_descriptor.json", "native_source.tar.gz"]
    children = sorted(candidate_root.iterdir(), key=lambda path: path.name)
    if [path.name for path in children] != expected_names or any(
            path.is_symlink() or not path.is_file() for path in children):
        raise PolicyError("KDA candidate source snapshot must contain exactly the descriptor and gzip")
    if gzip_path != (candidate_root / "native_source.tar.gz").resolve(strict=True) or \
            descriptor_path != (candidate_root / "analysis_descriptor.json").resolve(strict=True):
        raise PolicyError("KDA candidate source paths escape their two-file snapshot")
    candidate_files = [{"path": path.name, "sha256": _sha(path), "bytes": path.stat().st_size}
                       for path in children]
    if config.get("candidate_source_files") != candidate_files:
        raise PolicyError("KDA candidate source file inventory differs from its snapshot bytes")
    candidate_tree_sha = _digest_json([{"path": row["path"], "sha256": row["sha256"]}
                                       for row in candidate_files])
    if config.get("candidate_source_tree_sha256") != candidate_tree_sha or \
            config.get("source_tree_sha256") != candidate_tree_sha:
        raise PolicyError("KDA source snapshot digest differs from its exact two-file candidate input")

    source_manifest_path = _absolute_regular(config.get("source_manifest_path"),
                                             "KDA candidate source manifest")
    source_manifest = _read(source_manifest_path)
    if _sha(source_manifest_path) != config.get("source_manifest_sha256") or \
            source_manifest.get("schema_version") != 1 or \
            source_manifest.get("files") != expected_names:
        raise PolicyError("KDA source manifest does not list the exact descriptor and gzip inputs")
    source_provenance = source_manifest.get("provenance")
    if not isinstance(source_provenance, dict) or any(source_provenance.get(key) != value for key, value in (
            ("candidate_source_tree_sha256", candidate_tree_sha),
            ("native_source_tree_sha256", native_tree_sha),
            ("source_closure_sha256", _NATIVE_SOURCE_CLOSURE_SHA256),
            ("native_source_archive_tar_sha256", _NATIVE_SOURCE_TAR_SHA256),
            ("native_source_archive_manifest_sha256", _NATIVE_SOURCE_INDEX_SHA256),
            ("native_source_archive_gzip_sha256", _NATIVE_SOURCE_GZIP_SHA256),
            ("candidate_source_descriptor_sha256", config.get("native_source_descriptor_sha256")),
            ("qualification_bundle_sha256", dataset.get("qualification_bundle_sha256")))):
        raise PolicyError("KDA source manifest provenance does not bind the accepted source snapshot")

    descriptor = _read(descriptor_path)
    if _sha(descriptor_path) != config.get("native_source_descriptor_sha256") or \
            descriptor != config.get("native_source_descriptor"):
        raise PolicyError("KDA native source descriptor differs from its saved identity")
    assets = config.get("assets")
    if not isinstance(assets, list):
        raise PolicyError("native artifact identity list is missing")
    asset_hashes = {row.get("role"): row.get("sha256") for row in assets if isinstance(row, dict)}
    native_identity = config.get("native_identity")
    if not isinstance(native_identity, dict):
        raise PolicyError("qualified native identity is missing")
    expected_descriptor = {
        "schema": _NATIVE_DESCRIPTOR_SCHEMA,
        "native_source_tree_sha256": native_tree_sha,
        "source_closure_sha256": _NATIVE_SOURCE_CLOSURE_SHA256,
        "native_source_archive_tar_path": tar_path.name,
        "native_source_archive_tar_sha256": _NATIVE_SOURCE_TAR_SHA256,
        "native_source_archive_tar_bytes": _NATIVE_SOURCE_TAR_BYTES,
        "native_source_archive_manifest_path": index_path.name,
        "native_source_archive_manifest_sha256": _NATIVE_SOURCE_INDEX_SHA256,
        "native_source_archive_member_count": _NATIVE_SOURCE_MEMBER_COUNT,
        "native_source_archive_members_sha256": _NATIVE_SOURCE_MEMBERS_SHA256,
        "native_source_archive_source_bytes": _NATIVE_SOURCE_BYTES,
        "candidate_manifest_sha256": config.get("candidate_manifest_sha256"),
        "baseline_manifest_sha256": config.get("baseline_manifest_sha256"),
        "candidate_runner_sha256": asset_hashes.get("candidate_runner"),
        "baseline_runner_sha256": asset_hashes.get("baseline_runner"),
        "library_sha256": asset_hashes.get("shared_cuda_library"),
        "validator_sha256": identity.get("validator_sha256"),
        "workload_catalog_sha256": identity.get("workload_catalog", {}).get("sha256"),
        "qualification_bundle_sha256": dataset.get("qualification_bundle_sha256"),
        "native_source_transport_path": "native_source.tar.gz",
        "native_source_transport_compression": "gzip",
        "native_source_transport_gzip_mtime": 0,
        "native_source_transport_sha256": _NATIVE_SOURCE_GZIP_SHA256,
        "native_source_transport_bytes": _NATIVE_SOURCE_GZIP_BYTES,
    }
    if descriptor != expected_descriptor or \
            native_identity.get("source_closure_sha256") != _NATIVE_SOURCE_CLOSURE_SHA256 or \
            native_identity.get("candidate_runner_sha256") != asset_hashes.get("candidate_runner") or \
            native_identity.get("baseline_runner_sha256") != asset_hashes.get("baseline_runner") or \
            native_identity.get("library_sha256") != asset_hashes.get("shared_cuda_library") or \
            native_identity.get("validator_sha256") != identity.get("validator_sha256") or \
            native_identity.get("workload_catalog_sha256") != identity.get("workload_catalog", {}).get("sha256"):
        raise PolicyError("KDA source descriptor does not bind the qualified native build and archive")

    task_rows = task.get("trusted_inputs")
    if not isinstance(task_rows, list):
        raise PolicyError("KDA task omits source archive trusted inputs")
    required_inputs = {
        "native_source_archive_gzip": (gzip_path, _NATIVE_SOURCE_GZIP_SHA256, "toolchain"),
        "native_source_archive_member_index": (index_path, _NATIVE_SOURCE_INDEX_SHA256, "reference"),
        "native_source_descriptor": (descriptor_path, _sha(descriptor_path), "harness"),
    }
    for identifier, (path, digest, role) in required_inputs.items():
        matching = [row for row in task_rows if isinstance(row, dict) and row.get("id") == identifier]
        if len(matching) != 1 or matching[0].get("path") != str(path) or \
                matching[0].get("role") != role or _sha(path) != digest or Path(matching[0]["path"]).is_dir():
            raise PolicyError(f"KDA task does not bind the individual {identifier} file")
    manifest_rows = [row for row in task_rows if isinstance(row, dict) and
                     row.get("id") == "kda_source_manifest"]
    if len(manifest_rows) != 1 or manifest_rows[0].get("path") != str(source_manifest_path) or \
            manifest_rows[0].get("role") != "harness":
        raise PolicyError("KDA task does not bind its candidate source manifest")

    _verify_gzipped_native_tar(gzip_path, rows)


def _verify_kda_engine(task: Mapping[str, Any], task_path: Path, workspace: Path,
                       summary: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    root = Path(config.get("kda_root", "")).resolve(strict=True)
    if not (root / "kda_experiments" / "engine.py").is_file():
        raise PolicyError("configured KDA source tree is unavailable")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from kda_experiments.contracts import digest_json as kda_digest_json
        import kda_experiments.contracts as contracts_module
        import kda_experiments.engine as engine_module
        from kda_experiments.engine import _command_identities, package_fingerprint
        if not Path(engine_module.__file__).resolve(strict=True).is_relative_to(root) or \
                not Path(contracts_module.__file__).resolve(strict=True).is_relative_to(root):
            raise PolicyError("imported KDA modules are outside the task's configured source tree")
        expected_commands = _command_identities(task, task_path.parent, workspace)
        fingerprint = package_fingerprint()
    except Exception as exc:
        raise PolicyError(f"cannot verify the installed KDA engine identity: {exc}") from exc
    if summary.get("engine_fingerprint") != fingerprint:
        raise PolicyError("KDA engine source differs from the completed summary")
    if kda_digest_json(summary.get("command_identities")) != kda_digest_json(expected_commands):
        raise PolicyError("KDA command executables differ from the completed summary")


def _verify_builds(config: Mapping[str, Any], summary: Mapping[str, Any], workspace: Path) -> None:
    candidate_source_sha = config.get("candidate_source_tree_sha256")
    if not _HEX.fullmatch(str(candidate_source_sha)) or \
            candidate_source_sha != config.get("source_tree_sha256"):
        raise PolicyError("KDA config does not bind one exact candidate source snapshot")
    try:
        if str(Path(config.get("kda_root", "")).resolve(strict=True)) not in sys.path:
            sys.path.insert(0, str(Path(config["kda_root"]).resolve(strict=True)))
        from kda_experiments.store import workspace_paths
        store_paths = workspace_paths(workspace)
        builds_root = _public_kda_path(workspace, store_paths["builds"], "KDA builds root")
    except Exception as exc:
        raise PolicyError(f"cannot resolve the public KDA workspace layout: {exc}") from exc
    builds = summary.get("builds")
    if not isinstance(builds, dict) or set(builds) != set(CANDIDATES):
        raise PolicyError("KDA summary does not contain both prepared direct and packet builds")
    expected = []
    for asset in config.get("assets", []):
        if not isinstance(asset, dict) or not isinstance(asset.get("path"), str) or \
                not _HEX.fullmatch(str(asset.get("sha256", ""))):
            raise PolicyError("KDA config has a malformed prebuilt asset")
        rel = PurePosixPath(asset["path"])
        if rel.is_absolute() or any(part in ("", ".", "..") for part in rel.parts):
            raise PolicyError("KDA prebuilt asset path escapes its payload")
        expected.append({"path": (Path("prebuilt-native") / Path(*rel.parts)).as_posix(),
                         "sha256": asset["sha256"]})
    if not expected:
        raise PolicyError("KDA config does not pin any native binary artifacts")
    for candidate_id in CANDIDATES:
        build = builds[candidate_id]
        if not isinstance(build, dict) or build.get("artifacts") != expected:
            raise PolicyError(f"KDA {candidate_id} build record differs from pinned native artifacts")
        build_dir = Path(build.get("directory", ""))
        if build_dir.is_symlink() or not build_dir.is_dir():
            raise PolicyError("KDA build directory is missing or is a symbolic link")
        resolved_build = _public_kda_path(workspace, build_dir, "KDA build directory")
        candidate_root = _public_kda_path(workspace, builds_root / candidate_id,
                                          "KDA candidate build root")
        if not resolved_build.is_relative_to(candidate_root):
            raise PolicyError("KDA build directory is outside its public candidate build root")
        if build.get("fingerprint") != build_dir.parent.name:
            raise PolicyError("KDA build fingerprint does not match its content-addressed directory")
        manifest_path = Path(build.get("manifest_path", ""))
        if manifest_path.is_symlink() or not manifest_path.is_file() or \
                manifest_path.parent.resolve(strict=True) != build_dir.resolve(strict=True) or \
                _sha(manifest_path) != build.get("manifest_sha256"):
            raise PolicyError("KDA build manifest differs from its recorded identity")
        manifest = _read(manifest_path)
        if manifest.get("candidate_id") != candidate_id or \
                manifest.get("build_fingerprint") != build["fingerprint"] or \
                manifest.get("artifacts") != expected or \
                manifest.get("snapshot_sha256") != candidate_source_sha:
            raise PolicyError("KDA build manifest does not bind the expected candidate and bytes")
        for artifact in expected:
            path = build_dir / artifact["path"]
            if path.is_symlink() or not path.is_file() or _sha(path) != artifact["sha256"]:
                raise PolicyError("KDA copied native artifact changed or is missing")


def _validate_observed_replays(dataset: Mapping[str, Any], task: Mapping[str, Any],
                               config: Mapping[str, Any], summary: Mapping[str, Any],
                               run_dir: Path) -> None:
    """Check every KDA request/result against exactly one retained observation."""
    cases = {row["id"]: row for row in dataset["cases"]}
    task_cases = {row["id"]: row for row in task["cases"]}
    if set(cases) != set(task_cases) or len(task_cases) != 13:
        raise PolicyError("KDA task case coverage differs from the recorded workload set")
    records = summary.get("benchmark_records")
    confirmation = summary.get("confirmation")
    phases = {"exploration"}
    if confirmation is not None:
        phases.add("confirmation")
    expected_records = 13 * BLOCKS * 2 * len(phases)
    if not isinstance(records, list) or len(records) != expected_records:
        raise PolicyError("KDA summary has missing or duplicate replay observations")
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            raise PolicyError("KDA benchmark record is malformed")
        candidate_id, case_id, phase, round_index = (record.get(key) for key in
            ("candidate_id", "case_id", "phase", "round"))
        if candidate_id not in CANDIDATES or case_id not in cases or phase not in phases or \
                isinstance(round_index, bool) or not isinstance(round_index, int) or not 0 <= round_index < BLOCKS:
            raise PolicyError("KDA benchmark record names an unsupported request")
        key = (candidate_id, case_id, phase, round_index)
        if key in seen or (record.get("run_id") is not None and
                           record.get("run_id") != summary.get("run_id")):
            raise PolicyError("KDA benchmark records repeat or cross a different run")
        seen.add(key)
        case = cases[case_id]
        task_case = task_cases[case_id]
        if record.get("workload_sha256") is not None and \
                record.get("workload_sha256") != task_case.get("workload_sha256"):
            raise PolicyError("KDA benchmark record workload hash differs from its fixed task")
        request_id = record.get("request_id")
        if not isinstance(request_id, str) or not re.fullmatch(r"[0-9a-f]{8}-[0-9a-f-]{27,}", request_id):
            raise PolicyError("KDA benchmark record omitted its invocation UUID")
        case_component = hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:20]
        stage = run_dir / "stages" / candidate_id / phase / f"round-{round_index:03d}" / case_component
        request_path = _safe_artifact(stage, f"requests/{request_id}.json", "KDA request")
        response_path = _safe_artifact(stage, f"responses/{request_id}.json", "KDA response")
        response_sha = _sha(response_path)
        if response_sha != record.get("response_sha256"):
            raise PolicyError("KDA benchmark response differs from the summary digest")
        request, response = _read(request_path), _read(response_path)
        if request.get("request_id") != request_id or request.get("operation") != "benchmark" or \
                request.get("candidate", {}).get("id") != candidate_id or \
                request.get("case") != task_case or request.get("phase") != phase or \
                request.get("round") != round_index or request.get("timing") != task.get("timing"):
            raise PolicyError("KDA request bytes do not match its summary record and fixed task")
        if response.get("request_id") != request_id or response.get("case_id") != case_id or \
                response.get("workload_sha256") != task_case["workload_sha256"] or \
                response.get("status") != "ok" or response.get("metric") != task["timing"]["metric"] or \
                response.get("unit") != task["timing"]["unit"] or response.get("repetitions") != 1:
            raise PolicyError("KDA response does not match its source-bound task")
        arm_name = CANDIDATES[candidate_id]
        source_block = case["phases"][phase][round_index]
        arm = source_block["arms"][arm_name]
        expected_observation = {
            "dataset_sha256": config["dataset_sha256"],
            "qualification_status": dataset["qualification_status"],
            "qualification_bundle_sha256": dataset["qualification_bundle_sha256"],
            "recorded_workload_sha256": case["workload_sha256"],
            "recorded_observation_sha256": _digest_json(arm),
            "raw_metrics_sha256": arm["raw_metrics_sha256"],
            "validation_summary_sha256": arm["validation_summary_sha256"],
            "original_report_sha256": arm["original_report_sha256"],
            "recorded_cuda_device_uuid": arm["device_uuid"],
            "original_receipt_identity": {name: arm[name] for name in
                ("scheduler_job_id", "scheduler_attempt_id", "autobench_attempt_id")},
            "scored_arm": arm_name,
            "native_direct_is_descriptive_only": True,
        }
        if response.get("samples") != [arm["primary_ms"]] or \
                response.get("timing_scope") != "one originally measured resident caller-wall block mean for one forward/VJP step; replayed as data on CPU" or \
                response.get("recorded_observation") != expected_observation:
            raise PolicyError("KDA replay sample does not reproduce its exact original CUDA observation")
    if len(seen) != expected_records:
        raise PolicyError("KDA benchmark record matrix is incomplete")


def _execution_identity(arm: Mapping[str, Any]) -> tuple[Any, ...] | None:
    fields = ("candidate_argument", "requested_route", "selected_backend",
              "actual_implementation", "structural_layout")
    values = tuple(arm.get(field) for field in fields)
    if any(not isinstance(value, str) or not value for value in values):
        return None
    summary = arm.get("validated_summary")
    metrics = arm.get("metrics")
    if not isinstance(summary, dict) or not isinstance(metrics, dict):
        return None
    native = summary.get("native_execution")
    if not isinstance(native, dict):
        return None
    nested = (native.get("candidate"), native.get("requested_route"),
              native.get("selected_backend"), native.get("actual_implementation"),
              native.get("structural_layout"))
    if nested != values or metrics.get("selected_backend") != values[2]:
        return None
    return values


def _packet_execution_contract(case: Mapping[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    """Require 14 consistent actual routes before a packet case can be selected."""
    direct_ids: list[tuple[Any, ...] | None] = []
    packet_ids: list[tuple[Any, ...] | None] = []
    for phase in recorded_dataset.PHASES:
        blocks = case.get("phases", {}).get(phase)
        if not isinstance(blocks, list) or len(blocks) != BLOCKS:
            return False, "execution metadata lacks the complete 14-block route record", {}
        for block in blocks:
            arms = block.get("arms") if isinstance(block, dict) else None
            if not isinstance(arms, dict):
                return False, "execution metadata omits a scored arm", {}
            direct = arms.get(DIRECT)
            packet = arms.get(PACKET)
            direct_ids.append(_execution_identity(direct) if isinstance(direct, dict) else None)
            packet_ids.append(_execution_identity(packet) if isinstance(packet, dict) else None)

    direct_valid = all(identity is not None for identity in direct_ids)
    packet_valid = all(identity is not None for identity in packet_ids)
    direct_unique = set(direct_ids)
    expected_packet = ("packet", "packet", "packet", "packet", "full_packet")
    if not direct_valid:
        return False, "strongest direct control metadata is missing or inconsistent", {
            "strongest_direct": None, "packet": None}
    if len(direct_unique) != 1:
        return False, "strongest direct control identity drifts across the 14 paired blocks", {
            "strongest_direct": None, "packet": None}
    direct_identity = direct_ids[0]
    assert direct_identity is not None
    direct_modes = {
        ("minimal_direct", "direct", "direct", "direct", "canonical_direct"),
        ("direct_unique_output", "direct", "direct", "direct_unique_output", "canonical_direct"),
    }
    if direct_identity not in direct_modes:
        return False, "strongest direct control used a stable/shared or non-direct route", {
            "strongest_direct": list(direct_identity), "packet": None}
    if not packet_valid:
        return False, "packet challenger metadata is missing or inconsistent", {
            "strongest_direct": list(direct_identity), "packet": None}
    if any(identity != packet_ids[0] for identity in packet_ids):
        return False, "packet challenger identity drifts across the 14 paired blocks", {
            "strongest_direct": list(direct_identity), "packet": None}
    packet_identity = packet_ids[0]
    assert packet_identity is not None
    if packet_identity != expected_packet:
        if packet_identity[2:4] == ("stable", "stable"):
            reason = "packet challenger was routed to the stable backend; retain strongest direct"
        elif packet_identity[2:4] == ("shared", "shared"):
            reason = "packet challenger used a shared route; retain strongest direct"
        else:
            reason = "packet challenger did not execute packet/full_packet in every block"
        return False, reason, {"strongest_direct": list(direct_identity),
                               "packet": list(packet_identity)}
    return True, "packet and strongest direct identities match their qualified execution routes", {
        "strongest_direct": list(direct_identity), "packet": list(packet_identity)}


def select_exact_cases(dataset: Mapping[str, Any], task: Mapping[str, Any],
                       summary: Mapping[str, Any]) -> dict[str, Any]:
    """Compute the fixed route map from seven confirmation blocks per case."""
    try:
        recorded_dataset.validate_dataset(dataset)
    except Exception as exc:
        raise PolicyError(f"recorded dataset failed structural validation: {exc}") from exc
    if task.get("recorded_analysis_schema") != TASK_SCHEMA or \
            task.get("target", {}).get("kind") != "cpu" or \
            task.get("execution", {}).get("mode") != "direct" or \
            task.get("promotion") != PROMOTION or \
            task.get("timing", {}).get("direction") != "minimize" or \
            task.get("timing", {}).get("rounds") != BLOCKS or \
            task.get("timing", {}).get("confirmation_rounds") != BLOCKS:
        raise PolicyError("KDA task does not use the fixed CPU replay and selection contract")
    if summary.get("status") != "complete" or summary.get("trusted_inputs_unchanged") is not True or \
            summary.get("tools_unchanged") is not True:
        raise PolicyError("KDA run did not complete with unchanged trusted inputs and commands")
    ids = [case["id"] for case in dataset["cases"]]
    if set(summary.get("candidate_ids", [])) != set(CANDIDATES) or \
            summary.get("incumbent_id") != "strongest-direct" or \
            set(summary.get("requested_candidate_ids", [])) != set(CANDIDATES):
        raise PolicyError("KDA run did not compare the approved strongest-direct and packet candidates")
    if summary.get("benchmark_failures") != {}:
        raise PolicyError("KDA run contains a benchmark failure")
    validation = summary.get("validation")
    if not isinstance(validation, dict) or set(validation) != set(CANDIDATES):
        raise PolicyError("KDA run omitted one or more candidate validation results")
    for candidate_id in CANDIDATES:
        result = validation[candidate_id]
        if not isinstance(result, dict) or result.get("status") != "pass" or \
                set(result.get("case_status", {})) != set(ids) or \
                any(status != "pass" for status in result["case_status"].values()):
            raise PolicyError(f"KDA candidate {candidate_id} failed or skipped correctness validation")

    confirmation = summary.get("confirmation")
    per_case = None
    if confirmation is not None:
        if not isinstance(confirmation, dict) or not isinstance(confirmation.get("per_case"), dict) or \
                set(confirmation["per_case"]) != set(ids):
            raise PolicyError("KDA confirmation omits one or more fixed workload cases")
        challenger = next((row for row in summary.get("rankings", [])
                           if isinstance(row, dict) and row.get("candidate_id") == "packet-candidate"), None)
        if challenger is None:
            raise PolicyError("KDA confirmation exists without a packet challenger ranking")
        per_case = confirmation["per_case"]

    case_results = []
    for case in dataset["cases"]:
        case_id = case["id"]
        route_eligible, route_reason, route_identities = _packet_execution_contract(case)
        phase = case["phases"].get("confirmation")
        if not isinstance(phase, list) or len(phase) != BLOCKS:
            raise PolicyError(f"{case_id} does not have seven captured confirmation blocks")
        direct = [_finite_positive(block["arms"][DIRECT]["primary_ms"], f"{case_id} direct sample")
                  for block in phase]
        packet = [_finite_positive(block["arms"][PACKET]["primary_ms"], f"{case_id} packet sample")
                  for block in phase]
        direct_median, packet_median = _median(direct), _median(packet)
        speedup = direct_median / packet_median
        time_ratio = packet_median / direct_median
        if not math.isfinite(speedup) or not math.isfinite(time_ratio):
            raise PolicyError(f"{case_id} confirmation speed ratio is not finite")

        summary_case = None
        low = None
        kda_score = None
        kda_case_eligible = False
        eligible = False
        if per_case is not None:
            summary_case = per_case[case_id]
            if not isinstance(summary_case, dict):
                raise PolicyError(f"KDA confirmation for {case_id} is malformed")
            kda_score = _finite_positive(summary_case.get("score"), f"{case_id} KDA point score")
            low = _finite_positive(summary_case.get("ci_low"), f"{case_id} KDA CI lower bound")
            high = _finite_positive(summary_case.get("ci_high"), f"{case_id} KDA CI upper bound")
            if low > high or not math.isclose(kda_score, speedup, rel_tol=2e-12, abs_tol=2e-12):
                raise PolicyError(f"KDA point score/interval differs from the seven recorded medians for {case_id}")
            kda_case_eligible = bool(summary_case.get("eligible"))
            expected_kda_gate = low >= (1.0 - PROMOTION["max_regression"])
            if summary_case.get("eligible") is not expected_kda_gate:
                raise PolicyError(f"KDA per-case regression gate is inconsistent for {case_id}")
            eligible = (route_eligible and speedup >= MIN_PACKET_SPEEDUP and low > 1.0 and
                        time_ratio <= MAX_PACKET_TIME_RATIO)

        case_results.append({
            "case_id": case_id,
            "workload_sha256": case["workload_sha256"],
            "input_case_sha256": case["input_case_sha256"],
            "confirmation_primary_ms": {DIRECT: direct, PACKET: packet,
                                        "native_direct_descriptive": [
                                            _finite_positive(block["arms"]["native_direct"]["primary_ms"],
                                                             f"{case_id} descriptive sample")
                                            for block in phase]},
            "median_ms": {DIRECT: direct_median, PACKET: packet_median},
            "packet_speedup": speedup,
            "packet_time_ratio": time_ratio,
            "kda_point_score": kda_score,
            "kda_ci_low": low,
            "kda_ci_high": None if summary_case is None else high,
            "kda_per_case_regression_gate": kda_case_eligible,
            "packet_execution_contract_eligible": route_eligible,
            "execution_identity": route_identities,
            "packet_policy_eligible": eligible,
            "selected_route": PACKET if eligible else DIRECT,
            "selection_reason": (
                "packet cleared 1.10x point gain, paired 95% CI lower bound, and 5% time cap"
                if eligible else route_reason if not route_eligible else
                "strongest direct retained; packet did not clear every fixed per-case gate"),
        })

    return {
        "schema": SCHEMA,
        "status": "conditional_pending_external_cpu_receipt_and_independent_review",
        "selection_authority": "benchmark-scoped evidence only; no Cellerator production route is changed",
        "cuda_executed_by_analysis": False,
        "external_cpu_receipt_verification": "pending",
        "cleanup_verification": "pending",
        "independent_review_status": "pending",
        "qualification_status": dataset["qualification_status"],
        "qualification_bundle_sha256": dataset["qualification_bundle_sha256"],
        "recorded_device_uuid": dataset["identity"]["device_uuid"],
        "dataset_sha256": _digest_json(dataset),
        "task_sha256": summary.get("task_sha256"),
        "kda_run_id": summary.get("run_id"),
        "kda_engine_fingerprint": summary.get("engine_fingerprint"),
        "kda_summary_sha256": None,
        "fixed_policy": {"min_packet_speedup": MIN_PACKET_SPEEDUP,
                         "max_packet_time_ratio": MAX_PACKET_TIME_RATIO,
                         "per_case_paired_confidence_lower_bound_strictly_greater_than": 1.0,
                         "confirmation_blocks": BLOCKS,
                         "kda_promotion_contract": dict(PROMOTION)},
        "packet_regimes": [row["case_id"] for row in case_results if row["packet_policy_eligible"]],
        "strongest_direct_regimes": [row["case_id"] for row in case_results if not row["packet_policy_eligible"]],
        "unmeasured_workloads": [],
        "cases": case_results,
    }


def _verify_and_create(dataset_path: Path, task_path: Path, config_path: Path,
                       receipt_path: Path, summary_path: Path) -> dict[str, Any]:
    raw_summary_path = Path(summary_path)
    dataset_path, task_path, config_path, receipt_path, summary_path = (
        Path(p).resolve(strict=True) for p in
        (dataset_path, task_path, config_path, receipt_path, summary_path))
    dataset, task, config, receipt, summary = map(_read,
        (dataset_path, task_path, config_path, receipt_path, summary_path))
    harness_before = _harness_identities()
    dataset_sha, task_sha, config_sha = _sha(dataset_path), _sha(task_path), _sha(config_path)
    if dataset.get("schema") != DATASET_SCHEMA or task.get("recorded_analysis_schema") != TASK_SCHEMA or \
            config.get("schema") != TASK_SCHEMA or receipt.get("schema") != ANALYSIS_SCHEMA:
        raise PolicyError("recorded data, task, config, or CPU analysis schema is invalid")
    engine_rows = [row for row in task.get("trusted_inputs", [])
                   if isinstance(row, dict) and row.get("id") == "recorded_engine"]
    try:
        engine_path = Path(engine_rows[0].get("path", "")).resolve(strict=True) if len(engine_rows) == 1 else None
    except OSError:
        engine_path = None
    if engine_path != Path(recorded_engine.__file__).resolve(strict=True) or \
            engine_path is None or _sha(engine_path) != harness_before["recorded_engine"]:
        raise PolicyError("live recorded KDA adapter differs from the task's trusted input bytes")
    if receipt.get("status") != "provisional_cpu_analysis_completed" or \
            receipt.get("outer_receipt_verification") != "pending" or \
            receipt.get("cleanup_verification") != "pending" or \
            receipt.get("cuda_executed") is not False or receipt.get("scheduler_admission_requested") is not False:
        raise PolicyError("internal analysis record claims an external receipt or GPU execution")
    if receipt.get("task_path") != str(task_path) or receipt.get("task_sha256") != task_sha or \
            receipt.get("config_path") != str(config_path) or receipt.get("config_sha256") != config_sha:
        raise PolicyError("analysis record task/config identity differs from the supplied files")
    if config.get("dataset_path") != str(dataset_path) or config.get("dataset_sha256") != dataset_sha or \
            config.get("qualification_bundle_sha256") != dataset.get("qualification_bundle_sha256") or \
            config.get("recorded_device_uuid") != dataset.get("identity", {}).get("device_uuid"):
        raise PolicyError("task config is not bound to the supplied recorded dataset")
    _verify_compact_observations(dataset, config)
    _verify_recorded_trusted_inputs(dataset, task, config)
    _verify_native_source_archive(dataset, task, config)
    observed = receipt.get("recorded_cuda_observation")
    if not isinstance(observed, dict) or observed.get("dataset_sha256") != dataset_sha or \
            observed.get("qualification_bundle_sha256") != dataset["qualification_bundle_sha256"] or \
            observed.get("device_uuid") != dataset["identity"]["device_uuid"]:
        raise PolicyError("CPU analysis record names different original CUDA evidence")
    provisional = receipt.get("provisional_result")
    if not isinstance(provisional, dict) or provisional.get("summary_path") != str(summary_path) or \
            provisional.get("summary_sha256") != _sha(summary_path):
        raise PolicyError("KDA final summary path or digest differs from the analysis record")
    commands = receipt.get("commands")
    if not isinstance(commands, list) or len(commands) != 4:
        raise PolicyError("internal CPU analysis record omits its four KDA commands")
    for command in commands:
        if not isinstance(command, dict) or command.get("returncode") != 0 or \
                not isinstance(command.get("argv"), list) or not command["argv"] or \
                not _HEX.fullmatch(str(command.get("stdout_sha256", ""))) or \
                not _HEX.fullmatch(str(command.get("stderr_sha256", ""))) or \
                not isinstance(command.get("response"), dict) or command["response"].get("status") != "ok":
            raise PolicyError("a recorded KDA command did not complete with a digested response")
    evaluate_result = commands[-1]["response"]
    if evaluate_result.get("run_id") != provisional.get("run_id") or \
            evaluate_result.get("summary_path") != provisional.get("summary_path"):
        raise PolicyError("KDA evaluate command response differs from the analysis record")
    run_id = provisional.get("run_id")
    workspace = Path(receipt.get("workspace", "")).resolve(strict=True)
    try:
        kda_root = Path(config["kda_root"]).resolve(strict=True)
        if str(kda_root) not in sys.path:
            sys.path.insert(0, str(kda_root))
        from kda_experiments.store import workspace_paths
        attempt_root = _public_kda_path(workspace, workspace_paths(workspace)["attempts"],
                                        "KDA attempts root")
    except Exception as exc:
        raise PolicyError(f"cannot resolve the public KDA attempt layout: {exc}") from exc
    if not isinstance(run_id, str) or not run_id or "/" in run_id or "\\" in run_id:
        raise PolicyError("KDA run id is malformed")
    run_dir = _public_kda_path(workspace, attempt_root / run_id, "KDA run directory")
    if not raw_summary_path.is_absolute() or \
            _public_kda_path(workspace, raw_summary_path, "KDA summary") != summary_path or \
            not run_dir.is_relative_to(attempt_root) or summary_path.parent.resolve(strict=True) != run_dir:
        raise PolicyError("KDA summary is outside the identified experiment attempt")
    if summary.get("run_id") != run_id or summary.get("status") != "complete" or \
            summary.get("task_id") != task.get("task_id") or summary.get("task_path") != str(task_path) or \
            summary.get("task_sha256") != task_sha:
        raise PolicyError("KDA summary identity/status differs from the completed task")
    if provisional.get("decision") != summary.get("decision") or \
            provisional.get("rankings") != summary.get("rankings"):
        raise PolicyError("analysis record decision/rankings differ from the hashed KDA summary")
    if summary.get("trusted_inputs_unchanged") is not True or summary.get("tools_unchanged") is not True:
        raise PolicyError("KDA summary does not attest unchanged trusted inputs and commands")
    if task.get("target", {}).get("kind") != "cpu" or summary.get("environment", {}).get("kind") != "cpu" or \
            summary.get("probe", {}).get("environment", {}).get("kind") != "cpu":
        raise PolicyError("KDA analysis did not run with an explicitly CPU target")
    actual_trusted = _task_trusted_inputs(task)
    if summary.get("trusted_inputs") != actual_trusted:
        raise PolicyError("KDA summary trusted-input hashes differ from the current task inputs")
    _verify_kda_engine(task, task_path, workspace, summary, config)
    _verify_builds(config, summary, workspace)
    _verify_rebuilt_dataset(dataset, config, dataset_path)
    _validate_observed_replays(dataset, task, config, summary, run_dir)
    result = select_exact_cases(dataset, task, summary)
    result.update({"dataset_path": str(dataset_path), "task_path": str(task_path),
                   "config_path": str(config_path), "analysis_record_path": str(receipt_path),
                   "kda_summary_path": str(summary_path),
                   "dataset_sha256": dataset_sha,
                   "dataset_file_sha256": dataset_sha, "task_file_sha256": task_sha,
                   "config_file_sha256": config_sha,
                   "analysis_record_sha256": _sha(receipt_path),
                   "kda_summary_sha256": _sha(summary_path),
                   "source_closure_sha256": dataset["identity"]["source_closure_sha256"],
                   "candidate_runner_sha256": dataset["identity"]["candidate_runner_sha256"],
                   "baseline_runner_sha256": dataset["identity"]["baseline_runner_sha256"],
                   "shared_cuda_library_sha256": dataset["identity"]["library_sha256"],
                   "task_selection_policy_sha256": _digest_json(task["promotion"]),
                   "kda_confirmation_present": summary.get("confirmation") is not None})
    harness_after = _harness_identities()
    if harness_after != harness_before:
        raise PolicyError("CPU analysis harness changed while verifying the evidence")
    result["analysis_harness_sha256"] = harness_after
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--analysis-record", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = _verify_and_create(args.dataset, args.task, args.config,
                                args.analysis_record, args.summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
