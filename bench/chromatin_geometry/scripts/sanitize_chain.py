#!/usr/bin/env python3
"""Build and run the complete-chain sanitizer packet on an admitted GPU.

This adapter intentionally does not build CUDA code. ``build`` verifies and
copies an already-qualified cgs_study binary; ``run`` invokes that binary only
inside the Autobench Scheduler worker which owns the requested GPU UUID.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

SANITIZER = Path(
    "/home/tumlinson/Software/cuda-sanitizer-12.9/usr/local/cuda-12.9/bin/compute-sanitizer"
)
SANITIZERS = ("memcheck", "racecheck", "initcheck", "synccheck")
CASE_ID = "dag2304-64root-complete-consumer-diverse-dense-low-update-r1-change100"
GATES = ("supplied", "ste", "teacher")
BODIES = ("product", "linear")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ZERO_ERROR_SUMMARY_RE = re.compile(r"ERROR SUMMARY:\s*0 errors\b")
ZERO_HAZARDS_RE = re.compile(r"0 hazards displayed \(0 errors\b")


def quiescence(uuid: str) -> list[dict]:
    samples = []
    consecutive = 0
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        text = subprocess.check_output(['nvidia-smi', '-i', uuid, '--query-gpu=uuid,utilization.gpu',
                                        '--format=csv,noheader,nounits'], text=True).strip()
        fields = [x.strip() for x in text.split(',')]
        if len(fields) != 2 or fields[0] != uuid:
            raise AdapterError('UUID-scoped idle telemetry unavailable')
        value = int(fields[1])
        samples.append({'uuid': uuid, 'utilization_percent': value, 'time': time.time()})
        consecutive = consecutive + 1 if value == 0 else 0
        if consecutive == 3:
            return samples
        time.sleep(1)
    raise AdapterError('GPU did not produce three consecutive idle observations')


class AdapterError(RuntimeError):
    """An input or packet-level contract was not satisfied."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AdapterError(f"JSON root must be an object: {path}")
    return value


def require_sha(value: object, field: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise AdapterError(f"{field} must be a lowercase SHA-256 digest")
    return value


def prepare_empty_directory(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if path.exists():
        if not path.is_dir():
            raise AdapterError(f"{label} path exists and is not a directory: {path}")
        if any(path.iterdir()):
            raise AdapterError(f"{label} directory is not empty; preserving its contents: {path}")
    else:
        path.mkdir(parents=True)
    return path


def build(args: argparse.Namespace) -> dict:
    archive = Path(args.archive).expanduser().resolve(strict=True)
    binary_source = Path(args.binary).expanduser().resolve(strict=True)
    original_receipt_path = Path(args.receipt).expanduser().resolve(strict=True)
    expected_archive_sha = require_sha(args.archive_sha256, "--archive-sha256")

    if not archive.is_file() or not binary_source.is_file():
        raise AdapterError("source archive and qualified binary must be regular files")
    original = read_json(original_receipt_path)
    if original.get("schema") != "cgs.build.v1":
        raise AdapterError("original build receipt is not cgs.build.v1")
    if original.get("target") != "cgs_study":
        raise AdapterError("original build receipt does not identify target cgs_study")
    receipt_archive_sha = require_sha(original.get("archive_sha256"), "receipt.archive_sha256")
    receipt_binary_sha = require_sha(original.get("binary_sha256"), "receipt.binary_sha256")
    if receipt_archive_sha != expected_archive_sha:
        raise AdapterError("requested source archive SHA differs from original build receipt")
    actual_archive_sha = sha256(archive)
    if actual_archive_sha != expected_archive_sha:
        raise AdapterError("source archive bytes do not match --archive-sha256")
    actual_binary_sha = sha256(binary_source)
    if actual_binary_sha != receipt_binary_sha:
        raise AdapterError("qualified binary bytes do not match original build receipt")

    build_dir = prepare_empty_directory(Path(args.build), "build")
    binary_copy = build_dir / "cgs_study"
    shutil.copy2(binary_source, binary_copy)
    binary_copy.chmod(binary_copy.stat().st_mode | stat.S_IXUSR)
    copied_binary_sha = sha256(binary_copy)
    if copied_binary_sha != receipt_binary_sha:
        raise AdapterError("copied binary SHA differs from original build receipt")

    receipt = {
        "schema": "cgs.build.v1",
        "suite": "sanitize-chain",
        "target": "cgs_study",
        "archive_sha256": expected_archive_sha,
        "source_archive_path": str(archive),
        "binary_sha256": copied_binary_sha,
        "binary_relative_path": "cgs_study",
        "qualified_binary_source_path": str(binary_source),
        "original_build_receipt_path": str(original_receipt_path),
        "original_build_receipt_sha256": sha256(original_receipt_path),
    }
    (build_dir / "build.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def validate_build(build_dir: Path) -> tuple[dict, Path]:
    build_dir = build_dir.expanduser().resolve(strict=True)
    receipt = read_json(build_dir / "build.json")
    if receipt.get("schema") != "cgs.build.v1" or receipt.get("target") != "cgs_study":
        raise AdapterError("adapter build receipt is not a cgs_study cgs.build.v1 receipt")
    archive_sha = require_sha(receipt.get("archive_sha256"), "build.archive_sha256")
    binary_sha = require_sha(receipt.get("binary_sha256"), "build.binary_sha256")
    if receipt.get("binary_relative_path") != "cgs_study":
        raise AdapterError("adapter build receipt must name the local cgs_study binary")
    binary = build_dir / "cgs_study"
    if binary.is_symlink() or not binary.is_file():
        raise AdapterError("adapter build binary is missing or is a symlink")
    if sha256(binary) != binary_sha:
        raise AdapterError("adapter build binary changed after build receipt creation")
    if not binary.stat().st_mode & stat.S_IXUSR:
        raise AdapterError("adapter build binary is not owner-executable")
    return receipt, binary


def expected_cases(backend: str = 'cohort-shared') -> list[dict]:
    rows = []
    for gate in GATES:
        for body in BODIES:
            update = "none" if gate == "ste" else "incremental"
            rows.append({
                "case": CASE_ID,
                "gate": gate,
                "body": body,
                "backend": backend,
                "precision": "fp16",
                "condition_update": update,
                "tile": 64,
                "stages": 2 if backend == 'cohort-shared' else 1,
                "warmups": 0,
                "repeats": 1,
            })
    return rows


def command_for(binary: Path, case: dict, output_json: Path, sanitizer: str) -> list[str]:
    cgs = [
        str(binary),
        "--mode", "campaign",
        "--case", case["case"],
        "--backend", case["backend"],
        "--gate", case["gate"],
        "--body", case["body"],
        "--precision", case["precision"],
        "--condition-update", case["condition_update"],
        "--tile", str(case["tile"]),
        "--stages", str(case["stages"]),
        "--warmups", str(case["warmups"]),
        "--repeats", str(case["repeats"]),
        "--output", str(output_json),
    ]
    return [str(SANITIZER), "--tool", sanitizer, "--target-processes", "all",
            "--error-exitcode", "99", *cgs]


def validate_campaign(path: Path, expected: dict, uuid: str) -> tuple[dict | None, list[str]]:
    issues = []
    try:
        record = read_json(path)
    except AdapterError as exc:
        return None, [str(exc)]

    required_equal = {
        "schema": "cgs.campaign.v1",
        "status": "ok",
        "accuracy_accepted": True,
        "validation_scope": "first-and-last-complete-steps",
        "device_uuid": uuid,
        "case": expected["case"],
        "gate": expected["gate"],
        "body": expected["body"],
        "backend": expected["backend"],
        "precision": expected["precision"],
        "condition_update": expected["condition_update"],
        "tile": expected["tile"],
        "stages": expected["stages"],
    }
    for field, wanted in required_equal.items():
        if record.get(field) != wanted:
            issues.append(f"campaign {field} expected {wanted!r}, got {record.get(field)!r}")
    digest = record.get("workload_sha256")
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
        issues.append("campaign workload_sha256 is absent or invalid")
    if expected["backend"] in ("cohort-shared", "cohort-direct-two-tile") and record.get("descriptor_validation") != "exact-work-membership-and-raw-group-identity":
        issues.append("campaign did not report exact cohort work-membership validation")
    return record, issues


def run(args: argparse.Namespace) -> dict:
    scheduler_job_id = os.environ.get("AUTOBENCH_SCHEDULER_JOB_ID", "").strip()
    scheduler_attempt_id = os.environ.get("AUTOBENCH_SCHEDULER_ATTEMPT_ID", "").strip()
    if not scheduler_job_id or not scheduler_attempt_id:
        raise AdapterError("GPU execution requires Autobench Scheduler job and attempt IDs")

    uuid = args.uuid.strip()
    if not uuid or "," in uuid or any(ord(ch) < 32 for ch in uuid):
        raise AdapterError("--uuid must be one nonempty Scheduler GPU UUID")
    current_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if current_visible and current_visible != uuid:
        raise AdapterError("CUDA_VISIBLE_DEVICES conflicts with the Scheduler-assigned UUID")

    if args.sanitizer not in SANITIZERS:
        raise AdapterError(f"unsupported Compute Sanitizer mode: {args.sanitizer}")
    receipt, binary = validate_build(Path(args.build))
    if not SANITIZER.is_file():
        raise AdapterError(f"Compute Sanitizer executable is unavailable: {SANITIZER}")

    output = prepare_empty_directory(Path(args.output), "output")
    idle = quiescence(uuid)
    child_env = dict(os.environ)
    child_env["CUDA_VISIBLE_DEVICES"] = uuid
    rows = []
    for index, expected in enumerate(expected_cases(getattr(args, 'backend', 'cohort-shared'))):
        stem = f"{index:02d}-{expected['gate']}-{expected['body']}"
        case_dir = output / stem
        case_dir.mkdir()
        result_path = case_dir / "campaign.json"
        stdout_path = case_dir / "sanitizer.stdout"
        stderr_path = case_dir / "sanitizer.stderr"
        argv = command_for(binary, expected, result_path, args.sanitizer)
        returncode = None
        launch_error = None
        try:
            proc = subprocess.run(argv, env=child_env, text=True, capture_output=True, check=False)
            returncode = proc.returncode
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
        except OSError as exc:
            stdout = ""
            stderr = f"failed to launch Compute Sanitizer: {exc}\n"
            launch_error = str(exc)
        stdout_path.write_text(stdout)
        stderr_path.write_text(stderr)

        campaign, validation_issues = validate_campaign(result_path, expected, uuid)
        summary_text = stdout + "\n" + stderr
        has_clean_summary = bool(
            ZERO_ERROR_SUMMARY_RE.search(summary_text) or ZERO_HAZARDS_RE.search(summary_text)
        )
        issues = list(validation_issues)
        if launch_error:
            issues.append("Compute Sanitizer launch failed: " + launch_error)
        if returncode != 0:
            issues.append(f"Compute Sanitizer returned {returncode!r}")
        if not has_clean_summary:
            issues.append("Compute Sanitizer output lacks an explicit zero-error summary")
        row = {
            **expected,
            "index": index,
            "status": "pass" if not issues else "fail",
            "success": not issues,
            "returncode": returncode,
            "argv": argv,
            "campaign_path": result_path.relative_to(output).as_posix() if result_path.exists() else None,
            "campaign_sha256": sha256(result_path) if result_path.is_file() else None,
            "sanitizer_stdout_path": stdout_path.relative_to(output).as_posix(),
            "sanitizer_stdout_sha256": sha256(stdout_path),
            "sanitizer_stderr_path": stderr_path.relative_to(output).as_posix(),
            "sanitizer_stderr_sha256": sha256(stderr_path),
            "sanitizer_zero_error_summary": has_clean_summary,
            "campaign": campaign,
            "errors": issues,
        }
        rows.append(row)

    metrics = {
        "schema": "cgs.packet.v1",
        "suite": "sanitize-chain",
        "binary_path": str(binary),
        "binary_sha256": receipt["binary_sha256"],
        "archive_sha256": receipt["archive_sha256"],
        "source_archive_path": receipt.get("source_archive_path"),
        "uuid": uuid,
        "sanitizer": args.sanitizer,
        "scheduler_job_id": scheduler_job_id,
        "scheduler_attempt_id": scheduler_attempt_id,
        "quiescence": idle,
        "success": all(row["success"] for row in rows),
        "case_count": len(rows),
        "cases": rows,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build", help="verify and stage a qualified immutable binary")
    build_parser.add_argument("--binary", required=True, type=Path)
    build_parser.add_argument("--receipt", required=True, type=Path)
    build_parser.add_argument("--archive", required=True, type=Path)
    build_parser.add_argument("--archive-sha256", required=True)
    build_parser.add_argument("--build", required=True, type=Path)
    run_parser = sub.add_parser("run", help="run six complete-chain Compute Sanitizer cases")
    run_parser.add_argument("--build", required=True, type=Path)
    run_parser.add_argument("--uuid", required=True)
    run_parser.add_argument("--sanitizer", required=True, choices=SANITIZERS)
    run_parser.add_argument('--backend', choices=('cohort-shared', 'cohort-direct-two-tile'), default='cohort-shared')
    run_parser.add_argument("--output", required=True, type=Path)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "build":
            receipt = build(args)
            print(json.dumps(receipt, sort_keys=True))
            return 0
        metrics = run(args)
        print(json.dumps(metrics, sort_keys=True))
        return 0 if metrics["success"] else 1
    except (AdapterError, OSError) as exc:
        print(f"sanitize_chain: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
