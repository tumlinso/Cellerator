#!/usr/bin/env python3
"""Build and collect the fixed, three-job grouping probe packet.

This adapter does not compile CUDA code or register Scheduler work. ``build``
verifies and copies the two already-qualified binaries. ``run`` is the direct
Scheduler command; timing phases are wrapped in the archived study's serial GPU
monitor, while ``collect`` is its child and tracks each native campaign process.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import shutil
import stat
import statistics
import subprocess
import sys
import tarfile
import time
from pathlib import Path


SANITIZER = Path(
    "/home/tumlinson/Software/cuda-sanitizer-12.9/usr/local/cuda-12.9/bin/compute-sanitizer"
)
SANITIZERS = ("memcheck", "racecheck", "initcheck", "synccheck")
ARMS = ("masked-direct-two-tile", "cohort-direct-two-tile")
COMMON_CONTENT_SHA256 = "c735b9f73b23be0e91554a1b17932b6150846d0b4bd750246068cda3683e8f03"
CASES = {
    "probe-grouping-coherent-hard-frozen-r1-change0": {
        "reuse_horizon": 1,
        "workload_sha256": "3547f22a38de21cd8c5fe2b762ec3d4d352fd3a3d7172314fd7d305cdeb711c6",
    },
    "probe-grouping-coherent-hard-frozen-r16-change0": {
        "reuse_horizon": 16,
        "workload_sha256": "2d5de22ae75633558f820c9366e0cdc40a0cc25bf48619093b1da63338733c4f",
    },
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ZERO_ERROR_RE = re.compile(r"ERROR SUMMARY:\s*0 errors\b")
ZERO_HAZARD_RE = re.compile(r"0 hazards displayed \(0 errors\b")


class PacketError(RuntimeError):
    """A packet input or output did not satisfy the frozen contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_json(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PacketError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PacketError(f"JSON root is not an object: {path}")
    return value


def require_sha(value: object, field: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise PacketError(f"{field} must be a lowercase SHA-256 digest")
    return value


def extract_archive(archive: Path, target: Path, expected_sha256: str) -> dict:
    archive = Path(archive)
    target = Path(target)
    if archive.is_symlink() or not archive.is_file():
        raise PacketError("source archive must be a regular, non-symlink file")
    if sha256(archive) != require_sha(expected_sha256, "archive SHA-256"):
        raise PacketError("source archive bytes differ from the registered identity")
    if target.exists():
        if target.is_symlink() or not target.is_dir() or any(target.iterdir()):
            raise PacketError("source extraction directory must be new or empty")
    else:
        target.mkdir(parents=True)
    names: list[str] = []
    try:
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            names = [member.name for member in members]
            if len(set(names)) != len(names) or "SOURCE-MANIFEST.json" not in names:
                raise PacketError("source archive has duplicate members or no manifest")
            for member in members:
                relative = Path(member.name)
                if (not member.isfile() or relative.is_absolute() or ".." in relative.parts or
                        not relative.parts or member.issym() or member.islnk()):
                    raise PacketError(f"unsafe source archive member: {member.name}")
                destination = target / relative
                if not destination.resolve().is_relative_to(target.resolve()):
                    raise PacketError(f"source archive member escaped extraction root: {member.name}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    raise PacketError(f"cannot read source archive member: {member.name}")
                with source, destination.open("xb") as output:
                    shutil.copyfileobj(source, output)
                destination.chmod(member.mode & 0o777)
    except (OSError, tarfile.TarError) as exc:
        raise PacketError(f"source archive extraction failed: {exc}") from exc

    manifest = read_json(target / "SOURCE-MANIFEST.json")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(names) != set(files) | {"SOURCE-MANIFEST.json"}:
        raise PacketError("source archive membership differs from SOURCE-MANIFEST.json")
    for name, entry in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not isinstance(entry, dict):
            raise PacketError(f"invalid source manifest entry: {name}")
        path = target / relative
        if (path.is_symlink() or not path.is_file() or path.stat().st_size != entry.get("size") or
                sha256(path) != entry.get("sha256")):
            raise PacketError(f"source archive content differs from its manifest: {name}")
    return manifest


def _regular_file(path: Path, label: str) -> Path:
    path = Path(path).expanduser()
    if path.is_symlink() or not path.is_file():
        raise PacketError(f"{label} must be a regular, non-symlink file: {path}")
    return path.resolve(strict=True)


def _original_receipt(path: Path, archive_sha: str, binary: Path, expected_target: str) -> dict:
    receipt = read_json(path)
    if receipt.get("schema") != "cgs.build.v1":
        raise PacketError(f"{expected_target} receipt is not cgs.build.v1")
    if receipt.get("target") != expected_target:
        raise PacketError(f"build receipt target must be {expected_target}")
    if receipt.get("archive_sha256") != archive_sha:
        raise PacketError(f"{expected_target} receipt names a different source archive")
    binary_sha = require_sha(receipt.get("binary_sha256"), f"{expected_target} receipt binary SHA")
    if sha256(binary) != binary_sha:
        raise PacketError(f"{expected_target} binary bytes differ from its original build receipt")
    return receipt


def build(args: argparse.Namespace) -> dict:
    archive = _regular_file(Path(args.archive), "source archive")
    binary_source = _regular_file(Path(args.binary), "cgs_study binary")
    binary_receipt_source = _regular_file(Path(args.receipt), "cgs_study build receipt")
    qualifier_source = _regular_file(Path(args.qualification_binary), "qualification binary")
    qualifier_receipt_source = _regular_file(Path(args.qualification_receipt), "qualification build receipt")
    archive_sha = require_sha(args.archive_sha256, "--archive-sha256")
    if sha256(archive) != archive_sha:
        raise PacketError("archive bytes differ from --archive-sha256")
    study_receipt = _original_receipt(binary_receipt_source, archive_sha, binary_source, "cgs_study")
    qualifier_receipt = _original_receipt(
        qualifier_receipt_source, archive_sha, qualifier_source, "cgs_grouping_probe_qualification"
    )

    output = Path(args.build).expanduser().resolve()
    if output.exists():
        if output.is_symlink() or not output.is_dir() or any(output.iterdir()):
            raise PacketError("build directory must be new or empty; preserving its contents")
    else:
        output.mkdir(parents=True)
    source_root = output / "source"
    manifest = extract_archive(archive, source_root, archive_sha)
    expected_source_files = (
        "Cellerator/bench/chromatin_geometry/scripts/run_packet.py",
        "Cellerator/bench/chromatin_geometry/scripts/kda_driver.py",
    )
    for relative in expected_source_files:
        if relative not in manifest.get("files", {}):
            raise PacketError(f"source closure is missing required runtime file: {relative}")
    copied_study = output / "cgs_study"
    copied_qualifier = output / "cgs_grouping_probe_qualification"
    shutil.copy2(binary_source, copied_study)
    shutil.copy2(qualifier_source, copied_qualifier)
    copied_study.chmod(copied_study.stat().st_mode | stat.S_IXUSR)
    copied_qualifier.chmod(copied_qualifier.stat().st_mode | stat.S_IXUSR)
    if sha256(copied_study) != study_receipt["binary_sha256"]:
        raise PacketError("copied cgs_study failed its digest check")
    if sha256(copied_qualifier) != qualifier_receipt["binary_sha256"]:
        raise PacketError("copied qualification binary failed its digest check")

    copied_study_receipt = output / "original-cgs-study-build.json"
    copied_qualifier_receipt = output / "original-qualification-build.json"
    shutil.copy2(binary_receipt_source, copied_study_receipt)
    shutil.copy2(qualifier_receipt_source, copied_qualifier_receipt)
    result = {
        "schema": "cgs.grouping-probe-build.v1",
        "archive_sha256": archive_sha,
        "binary_sha256": study_receipt["binary_sha256"],
        "binary_path": str(copied_study.resolve()),
        "archive_source_path": str(archive),
        "source_root": "source",
        "source_manifest_sha256": sha256(source_root / "SOURCE-MANIFEST.json"),
        "cgs_study": {
            "target": "cgs_study", "path": "cgs_study", "binary_sha256": study_receipt["binary_sha256"],
            "original_receipt_path": copied_study_receipt.name,
            "original_receipt_sha256": sha256(copied_study_receipt),
        },
        "qualification_binary": {
            "target": "cgs_grouping_probe_qualification", "path": "cgs_grouping_probe_qualification",
            "binary_sha256": qualifier_receipt["binary_sha256"],
            "original_receipt_path": copied_qualifier_receipt.name,
            "original_receipt_sha256": sha256(copied_qualifier_receipt),
        },
    }
    write_json(output / "build.json", result)
    return result


def validate_build(build_dir: Path) -> tuple[dict, Path, Path, Path]:
    build_dir = Path(build_dir).resolve(strict=True)
    if build_dir.is_symlink() or not build_dir.is_dir():
        raise PacketError("build path must be a directory")
    manifest_path = build_dir / "build.json"
    if manifest_path.is_symlink():
        raise PacketError("build manifest must not be a symlink")
    manifest = read_json(manifest_path)
    if manifest.get("schema") != "cgs.grouping-probe-build.v1":
        raise PacketError("unsupported grouping probe build manifest")
    archive_sha = require_sha(manifest.get("archive_sha256"), "build.archive_sha256")
    source = build_dir / manifest.get("source_root", "")
    if not source.is_dir() or source.is_symlink():
        raise PacketError("verified source closure is missing")
    source_manifest_path = source / "SOURCE-MANIFEST.json"
    if sha256(source_manifest_path) != manifest.get("source_manifest_sha256"):
        raise PacketError("source manifest changed after build")
    source_manifest = read_json(source_manifest_path)
    files = source_manifest.get("files")
    if not isinstance(files, dict):
        raise PacketError("source manifest has no file map")
    for name, entry in files.items():
        path = source / name
        if (path.is_symlink() or not path.is_file() or path.stat().st_size != entry.get("size") or
                sha256(path) != entry.get("sha256")):
            raise PacketError(f"retained source closure changed: {name}")
    study = manifest.get("cgs_study", {})
    qualifier = manifest.get("qualification_binary", {})
    if study.get("target") != "cgs_study" or qualifier.get("target") != "cgs_grouping_probe_qualification":
        raise PacketError("build manifest has unexpected binary targets")
    study_binary = build_dir / study.get("path", "")
    qualifier_binary = build_dir / qualifier.get("path", "")
    if (manifest.get("binary_sha256") != study.get("binary_sha256") or
            manifest.get("binary_path") != str(study_binary.resolve())):
        raise PacketError("top-level build identity differs from the cgs_study copy")
    for path, entry, label in ((study_binary, study, "cgs_study"),
                               (qualifier_binary, qualifier, "qualification binary")):
        if (path.is_symlink() or not path.is_file() or not path.stat().st_mode & stat.S_IXUSR or
                sha256(path) != require_sha(entry.get("binary_sha256"), f"{label} binary SHA")):
            raise PacketError(f"retained {label} is missing or changed")
        receipt_path = build_dir / entry.get("original_receipt_path", "")
        if (receipt_path.is_symlink() or not receipt_path.is_file() or
                sha256(receipt_path) != require_sha(entry.get("original_receipt_sha256"), f"{label} receipt SHA")):
            raise PacketError(f"retained original {label} build receipt is missing or changed")
        receipt = read_json(receipt_path)
        if (receipt.get("schema") != "cgs.build.v1" or receipt.get("archive_sha256") != archive_sha or
                receipt.get("target") != entry["target"] or receipt.get("binary_sha256") != entry["binary_sha256"]):
            raise PacketError(f"retained original {label} receipt no longer binds source and binary")
    for relative in (
        "Cellerator/bench/chromatin_geometry/scripts/run_packet.py",
        "Cellerator/bench/chromatin_geometry/scripts/kda_driver.py",
    ):
        if relative not in files:
            raise PacketError(f"verified source closure is missing {relative}")
    return manifest, study_binary, qualifier_binary, source


def scheduler_identity(uuid: str) -> tuple[str, str]:
    job = os.environ.get("AUTOBENCH_SCHEDULER_JOB_ID", "").strip()
    attempt = os.environ.get("AUTOBENCH_SCHEDULER_ATTEMPT_ID", "").strip()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not job or not attempt:
        raise PacketError("GPU execution requires an admitted Scheduler job and attempt ID")
    if not uuid.startswith("GPU-") or "," in uuid or visible != uuid:
        raise PacketError("--uuid must exactly match the single Scheduler-visible GPU UUID")
    return job, attempt


def packet_directory(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    if path.name != "probe-packet":
        raise PacketError("--output must be the dedicated {output}/probe-packet directory")
    if path.exists():
        if path.is_symlink() or not path.is_dir() or any(path.iterdir()):
            raise PacketError("probe-packet output must be new or empty; preserving existing files")
    else:
        path.mkdir(parents=True)
    return path


def expected_orders() -> list[list[str]]:
    # A is the masked control, B the cohort candidate. This order is frozen
    # before measurement; balanced random orders are a different protocol.
    return [list(ARMS) if index % 2 == 0 else list(reversed(ARMS))
            for index in range(6)]


def paired_timing_summary(rounds: list[dict]) -> dict:
    """Use the same paired estimator for the point, interval and decision."""
    if len(rounds) != 6:
        raise PacketError("paired analysis requires exactly six independent rounds")
    ratios, control, candidate = [], [], []
    for row in rounds:
        a = row["arms"][ARMS[0]]["resident_median_ms"]
        b = row["arms"][ARMS[1]]["resident_median_ms"]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
               not math.isfinite(value) or value <= 0 for value in (a, b)):
            raise PacketError("paired resident medians must be finite and positive")
        control.append(a)
        candidate.append(b)
        ratios.append(b / a)
    point = statistics.median(ratios)
    rng = random.Random(20261102)
    draws = sorted(statistics.median(rng.choices(ratios, k=6)) for _ in range(10000))

    def percentile(fraction: float) -> float:
        position = fraction * (len(draws) - 1)
        low = int(position)
        high = min(low + 1, len(draws) - 1)
        return draws[low] + (draws[high] - draws[low]) * (position - low)

    lower, upper = percentile(0.025), percentile(0.975)
    practical = point < 0.95 - 1e-12
    interval_supports = upper < 1.0
    return {
        "paired_ratios_cohort_over_masked": ratios,
        "median_paired_ratio_cohort_over_masked": point,
        "median_paired_latency_reduction": 1.0 - point,
        "paired_bootstrap_95_ratio_interval": [lower, upper],
        "paired_bootstrap_95_latency_reduction_interval": [1.0 - upper, 1.0 - lower],
        "paired_bootstrap_seed": 20261102,
        "paired_bootstrap_samples": 10000,
        "median_cohort_latency_over_masked_latency":
            statistics.median(candidate) / statistics.median(control),
        "separate_arm_median_ratio_scope": "diagnostic only; not the paired estimator",
        "practical_latency_reduction_over_5_percent": practical,
        "paired_95_interval_supports_improvement": interval_supports,
        "grouping_beneficial_for_probe": practical and interval_supports,
    }


def campaign_argv(binary: Path, case_id: str, backend: str, output: Path, warmups: int, repeats: int) -> list[str]:
    return [str(binary), "--mode", "campaign", "--case", case_id, "--backend", backend,
            "--gate", "supplied", "--body", "product", "--precision", "fp16",
            "--condition-update", "none", "--tile", "32", "--stages", "1",
            "--warmups", str(warmups), "--repeats", str(repeats), "--output", str(output)]


def validate_campaign(data: dict, case_id: str, backend: str, uuid: str, timing: bool) -> None:
    case = CASES.get(case_id)
    if case is None or backend not in ARMS:
        raise PacketError("campaign case or backend is outside the frozen probe")
    exact = {
        "schema": "cgs.campaign.v1", "status": "ok", "accuracy_accepted": True,
        "validation_scope": "first-and-last-complete-steps", "device_uuid": uuid,
        "case": case_id, "backend": backend, "gate": "supplied", "body": "product",
        "precision": "fp16", "condition_update": "none", "input": "synthetic",
        "reuse_horizon": case["reuse_horizon"], "tile": 32, "stages": 1,
        "cell_tiles_per_cta": 2, "workload_sha256": case["workload_sha256"],
        "workload_content_sha256": COMMON_CONTENT_SHA256,
    }
    for field, expected in exact.items():
        if data.get(field) != expected:
            raise PacketError(f"campaign response has wrong {field}: {data.get(field)!r}")
    expected_descriptor = ("exact-gate-planes-no-materialized-workset" if backend == ARMS[0]
                           else "exact-work-membership-and-raw-group-identity")
    if data.get("descriptor_validation") != expected_descriptor:
        raise PacketError("campaign exact descriptor validation does not match the selected arm")
    if data.get("cell_order") != ("contiguous" if backend == ARMS[0] else "exact-program-cohorts"):
        raise PacketError("campaign cell ownership differs from the frozen arm")
    if data.get("per_lane_group_mask") is not (backend == ARMS[0]):
        raise PacketError("campaign per-lane mask identity differs from the frozen arm")
    if data.get("program_sorting") is not (backend == ARMS[1]):
        raise PacketError("campaign program-sorting identity differs from the frozen arm")
    contract = data.get("numerical_contract")
    if not isinstance(contract, dict) or contract.get("input_storage") != "fp16" or contract.get("accumulation") != "fp32":
        raise PacketError("campaign numerical contract differs from FP16 storage and FP32 accumulation")
    if not timing:
        return
    resident = data.get("resident_complete_horizon_ms")
    recurrent = data.get("recurrent_complete_horizon_ms")
    recovery = data.get("output_recovery_ms")
    outputs = data.get("output_and_vjp_sha256")
    if not isinstance(resident, list) or len(resident) != 3:
        raise PacketError("timing response must contain three resident_complete_horizon_ms samples")
    if not isinstance(recurrent, list) or len(recurrent) != 3 or recurrent != resident:
        raise PacketError("resident and recurrent complete-horizon samples must agree")
    if not isinstance(recovery, list) or len(recovery) != 3:
        raise PacketError("timing response must retain three separate output recovery samples")
    if not isinstance(outputs, list) or len(outputs) != 3 or any(
            not isinstance(value, str) or not SHA256_RE.fullmatch(value) for value in outputs):
        raise PacketError("timing response must retain an output/VJP digest for each measured sequence")
    for values, field in ((resident, "resident horizon"), (recovery, "output recovery")):
        if any(not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in values):
            raise PacketError(f"timing response contains invalid {field} measurements")
    for field in ("cold_execution_preparation_ms", "output_recovery_ms"):
        if field == "cold_execution_preparation_ms":
            value = data.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise PacketError("timing response omits a finite cold execution preparation value")


def _save_process_logs(base: Path, proc: subprocess.CompletedProcess, output_root: Path | None = None) -> dict:
    stdout_path = base.with_suffix(base.suffix + ".stdout.txt")
    stderr_path = base.with_suffix(base.suffix + ".stderr.txt")
    stdout_path.write_text(proc.stdout or "")
    stderr_path.write_text(proc.stderr or "")
    if output_root is None:
        stdout_name, stderr_name = stdout_path.name, stderr_path.name
    else:
        stdout_name = stdout_path.relative_to(output_root).as_posix()
        stderr_name = stderr_path.relative_to(output_root).as_posix()
    return {"stdout_path": stdout_name, "stdout_sha256": sha256(stdout_path),
            "stderr_path": stderr_name, "stderr_sha256": sha256(stderr_path),
            "returncode": proc.returncode}


def _run_subprocess(argv: list[str], env: dict, base: Path,
                    output_root: Path | None = None) -> tuple[subprocess.CompletedProcess, dict]:
    base.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(argv, env=env, text=True, capture_output=True, check=False)
    except OSError as exc:
        proc = subprocess.CompletedProcess(argv, 127, "", f"launch failed: {exc}\n")
    return proc, _save_process_logs(base, proc, output_root)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PacketError(f"cannot load verified source module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_qualification(build_info: dict, study_binary: Path, qualifier_binary: Path,
                      output: Path, uuid: str) -> dict:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid)
    raw_root = output / "raw" / "qualification"
    raw_root.mkdir(parents=True, exist_ok=False)
    rows: list[dict] = []

    def flush_progress() -> None:
        write_json(raw_root / "qualification-progress.json", {
            "schema": "cgs.grouping-probe-qualification-progress.v1",
            "status": "in-progress" if not rows or rows[-1].get("status") == "pass" else "fail",
            "checks": rows,
        })

    normal_path = raw_root / "standalone-qualification.stdout.txt"
    normal_err = raw_root / "standalone-qualification.stderr.txt"
    try:
        normal = subprocess.run([str(qualifier_binary)], env=env, text=True, capture_output=True, check=False)
    except OSError as exc:
        normal = subprocess.CompletedProcess([str(qualifier_binary)], 127, "", f"launch failed: {exc}\n")
    normal_path.write_text(normal.stdout or "")
    normal_err.write_text(normal.stderr or "")
    normal_record = {
        "kind": "standalone", "argv": [str(qualifier_binary)], "returncode": normal.returncode,
        "stdout_path": str(normal_path.relative_to(output)), "stdout_sha256": sha256(normal_path),
        "stderr_path": str(normal_err.relative_to(output)), "stderr_sha256": sha256(normal_err),
    }
    if normal.returncode != 0:
        rows.append({**normal_record, "status": "fail", "reason": "standalone qualification binary failed"})
        flush_progress()
        raise PacketError("standalone qualification binary failed")
    try:
        standalone = json.loads((normal.stdout or "").strip())
    except json.JSONDecodeError as exc:
        rows.append({**normal_record, "status": "fail", "reason": "standalone output is not one JSON object"})
        flush_progress()
        raise PacketError("standalone qualification output is not one JSON object") from exc
    if (not isinstance(standalone, dict) or standalone.get("schema") != "cgs.grouping-probe-qualification.v1" or
            standalone.get("status") != "pass" or
            (standalone.get("device_uuid") is not None and standalone.get("device_uuid") != uuid)):
        rows.append({**normal_record, "status": "fail", "reason": "standalone qualification JSON failed its contract"})
        flush_progress()
        raise PacketError("standalone qualification JSON failed its contract")
    normal_record.update(status="pass", parsed_result=standalone)
    rows.append(normal_record)
    flush_progress()

    for sanitizer in SANITIZERS:
        base = raw_root / f"sanitizer-{sanitizer}"
        argv = [str(SANITIZER), "--tool", sanitizer, "--error-exitcode", "99",
                "--target-processes", "all", str(qualifier_binary)]
        proc, logs = _run_subprocess(argv, env, base, output)
        output_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        zero_summary = bool(ZERO_ERROR_RE.search(output_text) or ZERO_HAZARD_RE.search(output_text))
        record = {
            "kind": "compute-sanitizer", "tool": sanitizer, "argv": argv,
            "status": "pass" if proc.returncode == 0 and zero_summary else "fail",
            "zero_error_summary": zero_summary, **logs,
        }
        rows.append(record)
        flush_progress()
        if record["status"] != "pass":
            raise PacketError(f"Compute Sanitizer {sanitizer} failed or omitted a zero-error summary")

    campaign_root = raw_root / "campaign-oracles"
    campaign_root.mkdir()
    campaign_results: dict[str, dict] = {}
    for case_id, case_info in CASES.items():
        per_case = []
        for backend in ARMS:
            stem = f"{case_id}-{backend}"
            raw = campaign_root / f"{stem}.json"
            stdout = campaign_root / f"{stem}.stdout.txt"
            stderr = campaign_root / f"{stem}.stderr.txt"
            argv = campaign_argv(study_binary, case_id, backend, raw, 0, 1)
            proc, logs = _run_subprocess(argv, env, campaign_root / stem, output)
            record = {"case": case_id, "backend": backend, "argv": argv, **logs}
            if proc.returncode != 0:
                record.update(status="fail", reason="native independent campaign oracle returned nonzero")
                rows.append(record)
                flush_progress()
                raise PacketError(f"qualification campaign failed for {case_id}/{backend}")
            if not raw.is_file():
                record.update(status="fail", reason="native campaign omitted its JSON output")
                rows.append(record)
                flush_progress()
                raise PacketError(f"qualification campaign omitted output for {case_id}/{backend}")
            data = read_json(raw)
            try:
                validate_campaign(data, case_id, backend, uuid, timing=False)
            except PacketError as exc:
                record.update(status="fail", reason=str(exc), raw_path=str(raw.relative_to(output)),
                              raw_sha256=sha256(raw))
                rows.append(record)
                flush_progress()
                raise
            record.update(status="pass", raw_path=str(raw.relative_to(output)), raw_sha256=sha256(raw),
                          workload_sha256=data["workload_sha256"],
                          workload_content_sha256=data["workload_content_sha256"],
                          descriptor_validation=data["descriptor_validation"])
            rows.append(record)
            flush_progress()
            per_case.append(data)
        if per_case[0]["workload_sha256"] != per_case[1]["workload_sha256"]:
            flush_progress()
            raise PacketError(f"qualification arms disagree on workload digest for {case_id}")
        if any(x["workload_content_sha256"] != COMMON_CONTENT_SHA256 for x in per_case):
            flush_progress()
            raise PacketError(f"qualification content digest differs from the independent host proof for {case_id}")
        campaign_results[case_id] = {
            "workload_sha256": case_info["workload_sha256"],
            "workload_content_sha256": COMMON_CONTENT_SHA256,
            "arms": {row["backend"]: {"raw_path": row["raw_path"], "raw_sha256": row["raw_sha256"]}
                     for row in rows if row.get("case") == case_id and row.get("status") == "pass"},
        }
    content_digests = {row["workload_content_sha256"] for row in campaign_results.values()}
    if content_digests != {COMMON_CONTENT_SHA256}:
        flush_progress()
        raise PacketError("r1 and r16 qualification content digests differ")
    write_json(raw_root / "qualification-progress.json", {
        "schema": "cgs.grouping-probe-qualification-progress.v1", "status": "pass", "checks": rows,
    })

    proof = {
        "schema": "cgs.grouping-probe-proof.v1", "status": "pass",
        "archive_sha256": build_info["archive_sha256"],
        "binary_sha256": build_info["cgs_study"]["binary_sha256"],
        "qualification_binary_sha256": build_info["qualification_binary"]["binary_sha256"],
        "cgs_study_build_receipt_sha256": build_info["cgs_study"]["original_receipt_sha256"],
        "qualification_build_receipt_sha256": build_info["qualification_binary"]["original_receipt_sha256"],
        "device_uuid": uuid, "scheduler_job_id": os.environ["AUTOBENCH_SCHEDULER_JOB_ID"],
        "scheduler_attempt_id": os.environ["AUTOBENCH_SCHEDULER_ATTEMPT_ID"],
        "campaigns": campaign_results,
        "sanitizers": {row["tool"]: {"status": row["status"], "zero_error_summary": row["zero_error_summary"],
                                      "stdout_sha256": row["stdout_sha256"], "stderr_sha256": row["stderr_sha256"]}
                       for row in rows if row.get("kind") == "compute-sanitizer"},
        "qualification_checks": rows,
    }
    write_json(output / "qualification-proof.json", proof)
    return {"schema": "cgs.grouping-probe-qualification-result.v1", "status": "pass",
            "checks": rows, "qualification_proof_path": "qualification-proof.json",
            "qualification_proof_sha256": sha256(output / "qualification-proof.json")}


def validate_proof(path: Path, build_info: dict, uuid: str) -> tuple[dict, str]:
    path = _regular_file(path, "qualification proof")
    proof = read_json(path)
    if proof.get("schema") != "cgs.grouping-probe-proof.v1" or proof.get("status") != "pass":
        raise PacketError("timing requires a separately retained successful qualification proof")
    expected = {
        "archive_sha256": build_info["archive_sha256"],
        "binary_sha256": build_info["cgs_study"]["binary_sha256"],
        "qualification_binary_sha256": build_info["qualification_binary"]["binary_sha256"],
        "cgs_study_build_receipt_sha256": build_info["cgs_study"]["original_receipt_sha256"],
        "qualification_build_receipt_sha256": build_info["qualification_binary"]["original_receipt_sha256"],
        "device_uuid": uuid,
    }
    for field, value in expected.items():
        if proof.get(field) != value:
            raise PacketError(f"qualification proof differs from current {field}")
    if (proof.get("scheduler_job_id") == os.environ.get("AUTOBENCH_SCHEDULER_JOB_ID") or
            proof.get("scheduler_attempt_id") == os.environ.get("AUTOBENCH_SCHEDULER_ATTEMPT_ID")):
        raise PacketError("qualification proof must come from an independent Scheduler job attempt")
    campaigns = proof.get("campaigns")
    if not isinstance(campaigns, dict) or set(campaigns) != set(CASES):
        raise PacketError("qualification proof does not cover both frozen probe cases")
    for case_id, case_info in CASES.items():
        case = campaigns[case_id]
        if (case.get("workload_sha256") != case_info["workload_sha256"] or
                case.get("workload_content_sha256") != COMMON_CONTENT_SHA256 or
                set(case.get("arms", {})) != set(ARMS)):
            raise PacketError(f"qualification proof does not bind the exact workload and arms for {case_id}")
        for arm, record in case["arms"].items():
            raw_name = record.get("raw_path")
            if (not isinstance(raw_name, str) or Path(raw_name).is_absolute() or ".." in Path(raw_name).parts or
                    not SHA256_RE.fullmatch(str(record.get("raw_sha256", "")))):
                raise PacketError(f"qualification proof has an invalid retained campaign record for {case_id}/{arm}")
    sanitizers = proof.get("sanitizers", {})
    if set(sanitizers) != set(SANITIZERS) or any(
            value.get("status") != "pass" or value.get("zero_error_summary") is not True
            for value in sanitizers.values()):
        raise PacketError("qualification proof does not contain all four zero-error sanitizer results")
    checks = proof.get("qualification_checks")
    if not isinstance(checks, list) or not checks or any(row.get("status") != "pass" for row in checks):
        raise PacketError("qualification proof does not retain all passing raw qualification checks")
    return proof, sha256(path)


def _case_object(case_id: str) -> dict:
    info = CASES[case_id]
    return {
        "id": case_id,
        "parameters": {"gate": "supplied", "body": "product", "precision": "fp16",
                       "condition_update": "none", "tile": 32, "stages": 1},
        "input": {"campaign_workload_sha256": info["workload_sha256"]},
    }


def _validate_timing_record(data: dict, case_id: str, backend: str, uuid: str) -> None:
    validate_campaign(data, case_id, backend, uuid, timing=True)


def run_timing(build_info: dict, study_binary: Path, source: Path, output: Path,
               uuid: str, case_id: str, proof_path: Path) -> dict:
    proof, proof_sha = validate_proof(proof_path, build_info, uuid)
    run_packet_module = _load_module(
        "cgs_grouping_probe_run_packet",
        source / "Cellerator/bench/chromatin_geometry/scripts/run_packet.py",
    )
    driver = _load_module(
        "cgs_grouping_probe_kda_driver",
        source / "Cellerator/bench/chromatin_geometry/scripts/kda_driver.py",
    )
    case = _case_object(case_id)
    expected_workload = proof["campaigns"][case_id]["workload_sha256"]
    if case["input"]["campaign_workload_sha256"] != expected_workload:
        raise PacketError("frozen KDA case digest differs from the qualification proof")

    raw_root = output / "raw" / "timing"
    raw_root.mkdir(parents=True, exist_ok=False)
    orders = expected_orders()
    rounds = []
    original_run_tracked = driver.run_tracked

    def tracked_with_logs(argv, events, lock_path, metadata, raw_path):
        try:
            completed = original_run_tracked(argv, events, lock_path, metadata, raw_path)
        except BaseException as exc:
            log_base = Path(raw_path).with_suffix("")
            Path(str(log_base) + ".stdout.txt").write_text("")
            Path(str(log_base) + ".stderr.txt").write_text(f"tracked native launch failed: {exc}\n")
            raise
        log_base = Path(raw_path).with_suffix("")
        Path(str(log_base) + ".stdout.txt").write_text(completed.stdout or "")
        Path(str(log_base) + ".stderr.txt").write_text(completed.stderr or "")
        return completed

    driver.run_tracked = tracked_with_logs
    try:
        for round_index, order in enumerate(orders):
            round_dir = raw_root / f"round-{round_index:02d}"
            round_dir.mkdir()
            arms: dict[str, dict] = {}
            for position, backend in enumerate(order):
                raw = round_dir / f"{backend}.json"
                record = {"backend": backend, "position": position, "raw_path": str(raw.relative_to(output)),
                          "status": "failed"}
                try:
                    data = driver.run_one(study_binary, backend, case, 1, 3, raw)
                    _validate_timing_record(data, case_id, backend, uuid)
                    if data["workload_content_sha256"] != proof["campaigns"][case_id]["workload_content_sha256"]:
                        raise PacketError("timing campaign content differs from successful qualification")
                    record.update(
                        status="pass", raw_sha256=sha256(raw),
                        stdout_path=str(Path(str(raw.with_suffix("")) + ".stdout.txt").relative_to(output)),
                        stderr_path=str(Path(str(raw.with_suffix("")) + ".stderr.txt").relative_to(output)),
                        stdout_sha256=sha256(Path(str(raw.with_suffix("")) + ".stdout.txt")),
                        stderr_sha256=sha256(Path(str(raw.with_suffix("")) + ".stderr.txt")),
                        workload_sha256=data["workload_sha256"],
                        workload_content_sha256=data["workload_content_sha256"],
                        resident_complete_horizon_ms=data["resident_complete_horizon_ms"],
                        resident_median_ms=sorted(data["resident_complete_horizon_ms"])[1],
                        output_recovery_ms=data["output_recovery_ms"],
                        cold_execution_preparation_ms=data["cold_execution_preparation_ms"],
                        output_and_vjp_sha256=data["output_and_vjp_sha256"],
                        device_uuid=data["device_uuid"],
                    )
                    arms[backend] = record
                except Exception as exc:
                    if raw.is_file():
                        record["raw_sha256"] = sha256(raw)
                    record["reason"] = str(exc)
                    arms[backend] = record
                    round_row = {"round_id": f"round-{round_index:02d}", "order": order,
                                 "arms": arms, "status": "fail"}
                    rounds.append(round_row)
                    write_json(output / "timing-progress.json", {
                        "schema": "cgs.grouping-probe-timing-progress.v1", "status": "fail",
                        "case": case_id, "rounds": rounds,
                    })
                    raise PacketError(f"timing failed in {case_id} round {round_index:02d} arm {backend}: {exc}") from exc
            if set(arms) != set(ARMS) or any(row["status"] != "pass" for row in arms.values()):
                raise PacketError(f"timing round {round_index:02d} did not complete both arms")
            rounds.append({"round_id": f"round-{round_index:02d}", "order": order,
                           "arms": arms, "status": "pass",
                           "paired_ratio_cohort_over_masked":
                               arms[ARMS[1]]["resident_median_ms"] / arms[ARMS[0]]["resident_median_ms"]})
            write_json(output / "timing-progress.json", {
                "schema": "cgs.grouping-probe-timing-progress.v1", "status": "in-progress",
                "case": case_id, "rounds": rounds,
            })
    finally:
        driver.run_tracked = original_run_tracked
    if len(rounds) != 6:
        raise PacketError("timing did not retain all six independent paired rounds")
    return {
        "schema": "cgs.grouping-probe-timing.v1", "status": "pass", "case": case_id,
        "workload_sha256": CASES[case_id]["workload_sha256"],
        "workload_content_sha256": proof["campaigns"][case_id]["workload_content_sha256"],
        "qualification_proof_sha256": proof_sha, "device_uuid": uuid,
        "warmups_per_invocation": 1, "measured_sequences_per_invocation": 3,
        "round_order_rule": "AB,BA,AB,BA,AB,BA", "rounds": rounds,
        **paired_timing_summary(rounds),
        "engine_discovery_or_evaluation_run": False,
        "cost_scope": "resident_complete_horizon_ms only; cold preparation and output recovery retained separately",
        "selection_status": "paired probe estimator; raw samples require receipt and telemetry qualification",
    }


def _packet_metrics(build_info: dict, build_manifest_path: Path, job: str, attempt: str,
                    uuid: str, phase: str, success: bool, **extra) -> dict:
    study = build_info["cgs_study"]
    qualifier = build_info["qualification_binary"]
    return {
        "schema": "cgs.packet.v1", "status": "pass" if success else "fail", "success": success,
        "phase": phase, "scheduler_job_id": job, "scheduler_attempt_id": attempt, "uuid": uuid,
        "archive_sha256": build_info["archive_sha256"], "binary_sha256": study["binary_sha256"],
        "binary_path": str((build_manifest_path.parent / study["path"]).resolve()),
        "qualification_binary_sha256": qualifier["binary_sha256"],
        "cgs_study_build_receipt_sha256": study["original_receipt_sha256"],
        "qualification_build_receipt_sha256": qualifier["original_receipt_sha256"],
        "build_manifest_sha256": sha256(build_manifest_path), **extra,
    }


def run(args: argparse.Namespace) -> dict:
    packet = packet_directory(Path(args.output))
    job, attempt = scheduler_identity(args.uuid)
    build_info, study_binary, qualifier_binary, source = validate_build(Path(args.build))
    build_manifest_path = Path(args.build).resolve() / "build.json"
    proof_sha = None
    extra: dict = {}
    success = False
    if args.phase == "qualification":
        try:
            result = run_qualification(build_info, study_binary, qualifier_binary, packet, args.uuid)
            success = result.get("status") == "pass"
            extra["qualification_result"] = result
            if success:
                extra["qualification_proof_path"] = result["qualification_proof_path"]
                extra["qualification_proof_sha256"] = result["qualification_proof_sha256"]
        except Exception as exc:
            extra.update({"failure": str(exc),
                          "qualification_checks_path": "raw/qualification/qualification-progress.json"})
    else:
        if args.suite != "kda":
            raise PacketError("timing phases must be registered with --suite kda for serialized GPU guarding")
        if not args.qualification_proof:
            raise PacketError("timing phases require a separately retained --qualification-proof")
        proof_path = _regular_file(Path(args.qualification_proof), "qualification proof")
        proof_sha = sha256(proof_path)
        retained_proof = packet / "qualification-proof.json"
        shutil.copyfile(proof_path, retained_proof)
        if sha256(retained_proof) != proof_sha:
            raise PacketError("retained qualification proof differs from the supplied bytes")
        proof_path = retained_proof
        if os.environ.get("CGS_LIFETIME_EVENTS") or os.environ.get("CGS_PROCESS_LOCK"):
            raise PacketError("outer timing command must not reuse a prior native lifetime ledger")
        command = [sys.executable, str(Path(__file__).resolve()), "collect", "--phase", args.phase,
                   "--build", str(Path(args.build).resolve()), "--uuid", args.uuid,
                   "--output", str(packet), "--qualification-proof", str(proof_path)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.uuid)
        run_packet = _load_module("cgs_probe_archived_run_packet",
                                  source / "Cellerator/bench/chromatin_geometry/scripts/run_packet.py")
        process, serial_clean = run_packet.run_serial_monitored(
            command, env, args.uuid, packet, build_info["cgs_study"]["binary_sha256"])
        parent_stdout = packet / "collector-parent.stdout.txt"
        parent_stderr = packet / "collector-parent.stderr.txt"
        parent_stdout.write_text(process.stdout or "")
        parent_stderr.write_text(process.stderr or "")
        collection_path = packet / "collection.json"
        collection = read_json(collection_path) if collection_path.is_file() else {}
        success = process.returncode == 0 and serial_clean is True and collection.get("status") == "pass"
        extra.update({"collection_path": "collection.json" if collection_path.is_file() else None,
                      "collection_sha256": sha256(collection_path) if collection_path.is_file() else None,
                      "collector_parent_stdout_sha256": sha256(parent_stdout),
                      "collector_parent_stderr_sha256": sha256(parent_stderr),
                      "serial_telemetry_clean": serial_clean,
                      "serial_telemetry_path": "serial-telemetry.json",
                      "native_lifetimes_path": "native-child-lifetimes.jsonl",
                      "failure": None if success else "timing collection or serialized telemetry failed"})
    metrics = _packet_metrics(build_info, build_manifest_path, job, attempt, args.uuid,
                              args.phase, success, **extra)
    if proof_sha:
        metrics["qualification_proof_path"] = "qualification-proof.json"
        metrics["qualification_proof_sha256"] = proof_sha
    write_json(packet / "metrics.json", metrics)
    return metrics


def collect(args: argparse.Namespace) -> dict:
    output = Path(args.output).resolve(strict=True)
    if output.name != "probe-packet":
        raise PacketError("collector output must be the dedicated probe-packet directory")
    job, attempt = scheduler_identity(args.uuid)
    build_info, study_binary, _, source = validate_build(Path(args.build))
    try:
        if args.phase == "qualification":
            result = run_qualification(build_info, study_binary,
                                       Path(args.build).resolve() / "cgs_grouping_probe_qualification",
                                       output, args.uuid)
        else:
            if not os.environ.get("CGS_LIFETIME_EVENTS") or not os.environ.get("CGS_PROCESS_LOCK"):
                raise PacketError("timing collector must run inside the monitored native-lifetime guard")
            case_id = next(case for case in CASES if CASES[case]["reuse_horizon"] == (1 if args.phase == "r1" else 16))
            result = run_timing(build_info, study_binary, source, output, args.uuid,
                                case_id, Path(args.qualification_proof))
    except Exception as exc:
        result = {"schema": "cgs.grouping-probe-collection.v1", "status": "fail",
                  "phase": args.phase, "failure": str(exc)}
        write_json(output / "collection.json", result)
        raise
    result.update({"scheduler_job_id": job, "scheduler_attempt_id": attempt, "device_uuid": args.uuid})
    write_json(output / "collection.json", result)
    if result.get("status") != "pass":
        raise PacketError("probe collection did not pass")
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build", help="verify and copy the two qualified binaries; do not compile")
    build_parser.add_argument("--archive", required=True)
    build_parser.add_argument("--archive-sha256", required=True)
    build_parser.add_argument("--binary", required=True)
    build_parser.add_argument("--receipt", required=True)
    build_parser.add_argument("--qualification-binary", required=True)
    build_parser.add_argument("--qualification-receipt", required=True)
    build_parser.add_argument("--build", required=True)
    run_parser = sub.add_parser("run", help="direct registered Scheduler packet command")
    run_parser.add_argument("--phase", choices=("qualification", "r1", "r16"), required=True)
    run_parser.add_argument("--build", required=True)
    run_parser.add_argument("--uuid", required=True)
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--suite", choices=("kda",), help="required for serialized timing phases")
    run_parser.add_argument("--qualification-proof")
    collect_parser = sub.add_parser("collect", help="internal child launched by the reviewed parent guard")
    collect_parser.add_argument("--phase", choices=("qualification", "r1", "r16"), required=True)
    collect_parser.add_argument("--build", required=True)
    collect_parser.add_argument("--uuid", required=True)
    collect_parser.add_argument("--output", required=True)
    collect_parser.add_argument("--qualification-proof")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "build":
            result = build(args)
            print(json.dumps(result, sort_keys=True))
            return 0
        if args.command == "run":
            result = run(args)
        else:
            result = collect(args)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        if args.command == "run" and getattr(args, "output", None):
            try:
                packet = Path(args.output).expanduser().resolve()
                if packet.is_dir() and packet.name == "probe-packet" and (packet / "build.json").exists():
                    build_info, _, _, _ = validate_build(Path(args.build))
                    job = os.environ.get("AUTOBENCH_SCHEDULER_JOB_ID", "unknown")
                    attempt = os.environ.get("AUTOBENCH_SCHEDULER_ATTEMPT_ID", "unknown")
                    uuid = getattr(args, "uuid", "unknown")
                    write_json(packet / "metrics.json", _packet_metrics(
                        build_info, Path(args.build).resolve() / "build.json", job, attempt,
                        uuid, args.phase, False, failure=str(exc)))
            except Exception:
                pass
        print(f"probe_packet: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
