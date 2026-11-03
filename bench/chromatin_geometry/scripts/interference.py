#!/usr/bin/env python3
"""Run and validate bounded isolated/concurrent chromatin-study probes.

GPU work is permitted only inside a Scheduler-admitted Autobench packet. This
module does not submit jobs; it records per-job samples and joins them later to
the Scheduler's retained cleanup receipts.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


CONFIG_SCHEMA = "cgs.interference-config.v1"
RUN_SCHEMA = "cgs.interference-run.v1"
RESULT_SCHEMA = "cgs.interference.v1"
UUID_RE = re.compile(r"^GPU-[A-Za-z0-9-]+$")
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
RUNNER_FIELDS = ("case", "backend", "gate", "body", "precision", "condition_update",
                 "seed", "tile", "stages", "input")
IDENTITY_FIELDS = ("source_archive_sha256", "binary_sha256", "workload_sha256", *RUNNER_FIELDS,
                   "replay_sha256")
SUCCESS_STATES = {"succeeded", "completed"}
LAYOUT_HOST_FIELD = "host_layout_preparation_ms"
LAYOUT_DEVICE_FIELD = "cold_device_preparation_ms"
REPLAY_PREPARATION_FIELD = "replay_input_preparation_ms"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _no_symlink_components(path: Path, *, allow_missing: bool = False) -> Path:
    """Return an absolute path after rejecting symlinks in every component."""
    path = Path(os.path.abspath(path))
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            if allow_missing:
                continue
            raise RuntimeError(f"path component does not exist: {current}")
        if current.is_symlink():
            raise RuntimeError(f"symlink path is forbidden: {current}")
        if current != path and not current.is_dir():
            raise RuntimeError(f"non-directory path component: {current}")
        if current == path and not (current.is_dir() or current.is_file()):
            raise RuntimeError(f"unsupported path type: {current}")
        _ = mode
    return path


def _mkdir_safe(path: Path) -> Path:
    path = Path(os.path.abspath(path))
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            current.mkdir(mode=0o700)
        except FileExistsError:
            pass
        if current.is_symlink() or not current.is_dir():
            raise RuntimeError(f"unsafe directory path: {current}")
    return path


def _read_regular(path: Path) -> bytes:
    _no_symlink_components(path)
    if not path.is_file():
        raise RuntimeError(f"expected regular file: {path}")
    return path.read_bytes()


def _atomic_json(path: Path, obj: Any) -> None:
    _no_symlink_components(path.parent)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to replace existing evidence: {path}")
    data = (json.dumps(obj, sort_keys=True, indent=2) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    data = _read_regular(path)
    try:
        cfg = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON configuration: {exc}") from exc
    if not isinstance(cfg, dict) or cfg.get("schema") != CONFIG_SCHEMA:
        raise RuntimeError(f"configuration must use {CONFIG_SCHEMA}")
    required = ("source_archive_sha256", "binary_sha256", "workload_sha256", "case",
                "backend", "gate", "body", "precision", "condition_update", "seed",
                "tile", "stages", "input", "primary_uuid", "concurrent_uuids",
                "independent_repetitions", "warmups")
    missing = [key for key in required if key not in cfg]
    if missing:
        raise RuntimeError(f"configuration lacks required fields: {', '.join(missing)}")
    for key in ("source_archive_sha256", "binary_sha256", "workload_sha256"):
        if not isinstance(cfg[key], str) or not HASH_RE.fullmatch(cfg[key]):
            raise RuntimeError(f"{key} must be a lowercase SHA-256")
    if not isinstance(cfg["case"], str) or not cfg["case"]:
        raise RuntimeError("case must be a nonempty frozen workload ID")
    allowed = {
        "backend": {"scalar", "bitmap", "direct", "cohort-direct", "cohort-direct-two-tile", "cohort-shared", "library", "library-batched"},
        "gate": {"supplied", "ste", "teacher"}, "body": {"product", "linear"},
        "precision": {"fp16", "fp32"}, "condition_update": {"none", "rebuild", "incremental"},
        "input": {"synthetic", "10x-pbmc-cgatac01"},
    }
    for key, values in allowed.items():
        if cfg[key] not in values:
            raise RuntimeError(f"unsupported {key}: {cfg[key]!r}")
    if cfg["backend"].startswith("library") and cfg["body"] != "linear":
        raise RuntimeError("library backend supports the grouped-linear body only")
    if cfg["gate"] == "ste" and cfg["condition_update"] != "none":
        raise RuntimeError("STE inputs must use condition_update=none")
    if type(cfg["seed"]) is not int or cfg["seed"] < 0:
        raise RuntimeError("seed must be a nonnegative integer")
    if cfg["tile"] not in (32, 64, 128) or cfg["stages"] not in (1, 2):
        raise RuntimeError("tile or stages is outside the frozen study choices")
    if type(cfg["independent_repetitions"]) is not int or not 2 <= cfg["independent_repetitions"] <= 32:
        raise RuntimeError("independent_repetitions must be in [2, 32]")
    if type(cfg["warmups"]) is not int or not 0 <= cfg["warmups"] <= 100:
        raise RuntimeError("warmups must be in [0, 100]")
    if not isinstance(cfg["primary_uuid"], str) or not UUID_RE.fullmatch(cfg["primary_uuid"]):
        raise RuntimeError("primary_uuid must be an exact NVIDIA GPU UUID")
    peers = cfg["concurrent_uuids"]
    if (not isinstance(peers, list) or len(peers) != 2 or len(set(peers)) != 2 or
            any(not isinstance(x, str) or not UUID_RE.fullmatch(x) for x in peers) or
            cfg["primary_uuid"] not in peers):
        raise RuntimeError("concurrent_uuids must contain exactly two distinct UUIDs including primary_uuid")
    tolerance = cfg.get("practical_tolerance_percent", 5.0)
    if (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or
            not math.isfinite(tolerance) or not 0 < tolerance < 100):
        raise RuntimeError("practical_tolerance_percent must be finite and between 0 and 100")
    if cfg["input"] == "synthetic":
        if cfg.get("replay_path") or cfg.get("replay_sha256"):
            raise RuntimeError("synthetic config cannot declare replay input")
    else:
        rel = cfg.get("replay_path")
        digest = cfg.get("replay_sha256")
        if not isinstance(rel, str) or not rel or Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise RuntimeError("replay_path must be a source-root-relative file")
        if not isinstance(digest, str) or not HASH_RE.fullmatch(digest):
            raise RuntimeError("replay input requires a SHA-256")
    return cfg, sha256_bytes(data)


def _required_env() -> tuple[str, str, str]:
    job = os.environ.get("AUTOBENCH_SCHEDULER_JOB_ID", "")
    attempt = os.environ.get("AUTOBENCH_SCHEDULER_ATTEMPT_ID", "")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not job or not attempt:
        raise RuntimeError("GPU execution requires AUTOBENCH_SCHEDULER_JOB_ID and AUTOBENCH_SCHEDULER_ATTEMPT_ID")
    if not SAFE_ID_RE.fullmatch(job) or not SAFE_ID_RE.fullmatch(attempt):
        raise RuntimeError("Scheduler job and attempt identifiers contain unsupported characters")
    if not UUID_RE.fullmatch(visible):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly one GPU UUID")
    return job, attempt, visible


def _nvidia_snapshot() -> dict[str, Any]:
    gpu_argv = ["nvidia-smi", "--query-gpu=uuid,utilization.gpu", "--format=csv,noheader,nounits"]
    proc_argv = ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"]
    sampled = time.time_ns()
    try:
        gpu = subprocess.run(gpu_argv, text=True, capture_output=True)
        procs = subprocess.run(proc_argv, text=True, capture_output=True)
    except OSError as exc:
        return {"sampled_at_epoch_ns": sampled, "complete": False, "gpus": [],
                "compute_processes": [], "commands": [{"error": str(exc), "gpu_argv": gpu_argv,
                                                          "process_argv": proc_argv}]}
    rows: list[dict[str, Any]] = []
    for row in csv.reader(gpu.stdout.splitlines(), skipinitialspace=True):
        if not row:
            continue
        if len(row) != 2 or not UUID_RE.fullmatch(row[0].strip()):
            continue
        util = row[1].strip()
        try:
            percent: int | None = int(util)
        except ValueError:
            percent = None
        rows.append({"uuid": row[0].strip(), "utilization_percent": percent})
    apps: list[dict[str, Any]] = []
    no_apps = {"[no running processes found]", "no running processes found"}
    if procs.stdout.strip().lower() not in no_apps:
        for row in csv.reader(procs.stdout.splitlines(), skipinitialspace=True):
            if len(row) >= 2 and UUID_RE.fullmatch(row[0].strip()):
                try:
                    pid: int | None = int(row[1].strip())
                except ValueError:
                    pid = None
                apps.append({"gpu_uuid": row[0].strip(), "pid": pid,
                             "process_name": ",".join(row[2:]).strip() if len(row) > 2 else ""})
    unique = len({r["uuid"] for r in rows}) == len(rows)
    return {
        "sampled_at_epoch_ns": sampled,
        "complete": gpu.returncode == 0 and procs.returncode == 0 and bool(rows) and unique,
        "gpus": sorted(rows, key=lambda r: r["uuid"]), "compute_processes": apps,
        "commands": [
            {"argv": gpu_argv, "returncode": gpu.returncode, "stdout": gpu.stdout, "stderr": gpu.stderr},
            {"argv": proc_argv, "returncode": procs.returncode, "stdout": procs.stdout, "stderr": procs.stderr},
        ],
    }


def _write_arrival(barrier_dir: Path, name: str, obj: dict[str, Any]) -> None:
    _atomic_json(barrier_dir / name, obj)


def _wait_for_pair(study_root: Path, barrier_id: str, repetition: int, config: dict[str, Any],
                   config_sha: str, job: str, attempt: str, uuid: str, timeout: int) -> dict[str, Any]:
    if not SAFE_ID_RE.fullmatch(barrier_id):
        raise RuntimeError("barrier ID must be a bounded safe identifier")
    _no_symlink_components(study_root)
    barrier_root = study_root / "runs" / "barriers"
    _mkdir_safe(barrier_root)
    _no_symlink_components(barrier_root)
    barrier_dir = barrier_root / barrier_id / f"rep-{repetition:03d}"
    _mkdir_safe(barrier_dir)
    job_key = hashlib.sha256(job.encode()).hexdigest()[:24]
    arrival = {"job_id": job, "attempt_id": attempt, "uuid": uuid,
               "hostname": socket.gethostname(), "repetition": repetition,
               "config_sha256": config_sha, "source_archive_sha256": config["source_archive_sha256"],
               "binary_sha256": config["binary_sha256"], "workload_sha256": config["workload_sha256"],
               "arrived_epoch_ns": time.time_ns()}
    _write_arrival(barrier_dir, f"arrival-{job_key}.json", arrival)
    deadline = time.monotonic() + timeout
    while True:
        _no_symlink_components(barrier_dir)
        files = sorted(barrier_dir.glob("arrival-*.json"))
        arrivals = [json.loads(_read_regular(p)) for p in files]
        if len(arrivals) > 2:
            raise RuntimeError("concurrent barrier has more than two Scheduler jobs")
        if len(arrivals) == 2:
            jobs = {x.get("job_id") for x in arrivals}
            attempts = {x.get("attempt_id") for x in arrivals}
            uuids = {x.get("uuid") for x in arrivals}
            if len(jobs) != 2 or len(attempts) != 2 or len(uuids) != 2 or uuids != set(config["concurrent_uuids"]):
                raise RuntimeError("concurrent barrier participants are not two distinct declared jobs/UUIDs")
            if any(x.get("hostname") != socket.gethostname() for x in arrivals):
                raise RuntimeError("concurrent GPU timing requires both jobs on the same host")
            for row in arrivals:
                for field in ("repetition", "config_sha256", "source_archive_sha256", "binary_sha256", "workload_sha256"):
                    expected = repetition if field == "repetition" else config_sha if field == "config_sha256" else config[field]
                    if row.get(field) != expected:
                        raise RuntimeError(f"concurrent barrier participant differs on {field}")
            release = {"barrier_id": barrier_id, "repetition": repetition,
                       "participants": sorted(arrivals, key=lambda x: x["job_id"]),
                       "released_epoch_ns": time.time_ns()}
            release_path = barrier_dir / "release.json"
            try:
                _atomic_json(release_path, release)
            except RuntimeError:
                if not release_path.is_file():
                    raise
            joined = json.loads(_read_regular(release_path))
            members = {x["job_id"] for x in joined.get("participants", [])}
            if members != jobs or job not in members:
                raise RuntimeError("barrier was released to a different participant pair")
            return {"barrier_id": barrier_id, "directory": str(barrier_dir),
                    "release": joined, "participant_job_ids": sorted(jobs),
                    "participant_uuids": sorted(uuids)}
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(f"concurrent peer missing after {timeout}s; fail closed")
        time.sleep(min(0.1, remaining))


def _campaign_command(cfg: dict[str, Any], binary: Path, output: Path,
                      warmups: int, replay: Path | None) -> list[str]:
    argv = [str(binary), "--mode", "campaign", "--case", cfg["case"],
            "--backend", cfg["backend"], "--gate", cfg["gate"], "--body", cfg["body"],
            "--precision", cfg["precision"], "--condition-update", cfg["condition_update"],
            "--tile", str(cfg["tile"]), "--stages", str(cfg["stages"]),
            "--warmups", str(warmups), "--repeats", "1", "--output", str(output)]
    if replay is not None:
        argv.extend(["--replay", str(replay)])
    return argv


def _run_one(binary: Path, output: Path, cfg: dict[str, Any], warmups: int,
             replay: Path | None) -> tuple[dict[str, Any], str, str, str, int, list[dict[str, Any]], int, int, list[str], int, int]:
    argv = _campaign_command(cfg, binary, output, warmups, replay)
    start_ns = time.time_ns()
    started = time.monotonic_ns()
    proc = subprocess.Popen(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    telemetry: list[dict[str, Any]] = []
    telemetry.append({"phase": "during", **_nvidia_snapshot()})
    while proc.poll() is None:
        time.sleep(0.5)
        if proc.poll() is None:
            telemetry.append({"phase": "during", **_nvidia_snapshot()})
    stdout, stderr = proc.communicate()
    ended = time.monotonic_ns()
    end_ns = time.time_ns()
    try:
        raw = _read_regular(output)
    except RuntimeError as exc:
        raise RuntimeError(f"campaign output is missing: {exc}; returncode={proc.returncode}; stdout={stdout!r}; stderr={stderr!r}; argv={argv!r}") from exc
    try:
        campaign = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cgs_study emitted invalid JSON: {exc}; stdout={stdout!r}; stderr={stderr!r}") from exc
    return campaign, raw.decode(), stdout, stderr, proc.returncode, telemetry, start_ns, end_ns, argv, proc.pid, ended - started


def _validate_campaign(campaign: dict[str, Any], cfg: dict[str, Any], uuid: str) -> float:
    if campaign.get("schema") != "cgs.campaign.v1" or campaign.get("status") != "ok":
        raise RuntimeError("runner did not emit a successful cgs.campaign.v1 record")
    if campaign.get("accuracy_accepted") is not True or campaign.get("validation_scope") != "first-and-last-complete-steps":
        raise RuntimeError("runner correctness gate is absent or incomplete")
    expected = {"workload_sha256": "workload_sha256", "device_uuid": None}
    for key, value in expected.items():
        wanted = uuid if value is None else cfg[value]
        if campaign.get(key) != wanted:
            raise RuntimeError(f"runner {key} differs from the frozen interference config")
    for field in RUNNER_FIELDS:
        if campaign.get(field) != cfg.get(field):
            raise RuntimeError(f"runner {field} differs from the frozen interference config")
    for field in (REPLAY_PREPARATION_FIELD, LAYOUT_HOST_FIELD, LAYOUT_DEVICE_FIELD,
                  "cold_execution_preparation_ms"):
        val = campaign.get(field)
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0:
            raise RuntimeError(f"runner lacks a valid {field}")
    component_sum = (float(campaign[REPLAY_PREPARATION_FIELD]) +
                     float(campaign[LAYOUT_HOST_FIELD]) + float(campaign[LAYOUT_DEVICE_FIELD]))
    if not math.isclose(float(campaign["cold_execution_preparation_ms"]), component_sum,
                        rel_tol=1e-8, abs_tol=1e-6):
        raise RuntimeError("cold execution preparation does not equal replay input plus host layout plus cold device preparation")
    for field in ("recurrent_complete_horizon_ms", "output_recovery_ms"):
        values = campaign.get(field)
        if not isinstance(values, list) or len(values) != 1:
            raise RuntimeError(f"runner must report exactly one measured {field} sample")
        val = values[0]
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0:
            raise RuntimeError(f"runner has invalid {field}")
    return (float(campaign["cold_execution_preparation_ms"]) +
            float(campaign["recurrent_complete_horizon_ms"][0]) +
            float(campaign["output_recovery_ms"][0]))


def run_packet(args: argparse.Namespace) -> dict[str, Any]:
    cfg, config_sha = _load_config(Path(args.config))
    job, attempt, uuid = _required_env()
    if args.mode not in ("isolated", "concurrent"):
        raise RuntimeError("mode must be isolated or concurrent")
    if not 0 <= args.repetition < cfg["independent_repetitions"]:
        raise RuntimeError("repetition index is outside the frozen config")
    if args.mode == "isolated" and uuid != cfg["primary_uuid"]:
        raise RuntimeError("isolated measurements must use the declared primary UUID")
    if args.mode == "concurrent" and uuid not in cfg["concurrent_uuids"]:
        raise RuntimeError("concurrent measurement UUID is not one of the two declared devices")
    binary = Path(os.path.abspath(args.binary))
    _no_symlink_components(binary)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError("cgs_study binary must be an executable regular file")
    binary_sha = sha256_file(binary)
    if binary_sha != cfg["binary_sha256"]:
        raise RuntimeError("cgs_study binary hash differs from frozen config")
    replay = None
    if cfg["input"] != "synthetic":
        if not args.source_root:
            raise RuntimeError("replay case needs --source-root")
        source_root = _no_symlink_components(Path(args.source_root))
        replay = _no_symlink_components(source_root / cfg["replay_path"])
        if not replay.is_file() or sha256_file(replay) != cfg["replay_sha256"]:
            raise RuntimeError("replay input is absent or differs from its frozen digest")
    elif args.source_root:
        _no_symlink_components(Path(args.source_root))
    out_dir = _mkdir_safe(Path(args.output_dir))
    if list(out_dir.iterdir()):
        raise RuntimeError("output directory must be fresh and empty")
    if args.mode == "concurrent":
        if not args.shared_study_root or not args.barrier_id:
            raise RuntimeError("concurrent mode requires --shared-study-root and --barrier-id")
        if not 1 <= args.barrier_timeout <= 60:
            raise RuntimeError("barrier timeout must be within 1..60 seconds")
    elif args.barrier_id or args.shared_study_root:
        raise RuntimeError("isolated mode does not accept a barrier")
    barrier = None
    telemetry: dict[str, Any] = {"before": None, "during": [], "after": None,
                                 "limitations": ["nvidia-smi utilization/process visibility is observational and does not prove absence of short unobserved activity"]}
    raw_path = out_dir / "cgs.campaign.raw.json"
    result: dict[str, Any] = {"schema": RUN_SCHEMA, "status": "failed", "mode": args.mode,
                              "config_sha256": config_sha, "config": cfg,
                              **{key: cfg.get(key) for key in IDENTITY_FIELDS},
                              "source_archive_sha256": cfg["source_archive_sha256"],
                              "binary_sha256": binary_sha, "binary_path": str(binary),
                              "workload_sha256": cfg["workload_sha256"],
                              "scheduler_job_id": job, "scheduler_attempt_id": attempt,
                              "device_uuid": uuid, "hostname": socket.gethostname(),
                              "repetition": args.repetition, "barrier": barrier,
                              "packet_started_at_epoch_ns": time.time_ns(),
                              "packet_started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                              "telemetry": telemetry}
    try:
        if args.mode == "concurrent":
            barrier = _wait_for_pair(Path(args.shared_study_root), args.barrier_id, args.repetition,
                                     cfg, config_sha, job, attempt, uuid, args.barrier_timeout)
            result["barrier"] = barrier
        telemetry["before"] = _nvidia_snapshot()
        observed_uuids = {x["uuid"] for x in telemetry["before"].get("gpus", [])}
        if not telemetry["before"].get("complete") or not set(cfg["concurrent_uuids"]) <= observed_uuids:
            raise RuntimeError("all-device nvidia-smi pre-snapshot unavailable or missing target UUID")
        campaign, raw, stdout, stderr, returncode, during, start_ns, end_ns, command, runner_pid, elapsed_ns = _run_one(
            binary, raw_path, cfg, cfg["warmups"], replay)
        result["telemetry"]["during"] = during
        result["campaign"] = campaign
        result["campaign_raw_json"] = raw
        result["stdout"] = stdout
        result["stderr"] = stderr
        result["runner_returncode"] = returncode
        result["command"] = command
        result["runner_pid"] = runner_pid
        result["runner_elapsed_monotonic_ns"] = elapsed_ns
        result["runner_started_epoch_ns"] = start_ns
        result["runner_ended_epoch_ns"] = end_ns
        result["operational_total_ms"] = _validate_campaign(campaign, cfg, uuid)
        if returncode != 0:
            raise RuntimeError(f"cgs_study returned {returncode}")
        result["status"] = "ok"
    except Exception as exc:
        result["error"] = str(exc)
        if raw_path.exists() and not raw_path.is_symlink():
            result["campaign_raw_json"] = raw_path.read_text(errors="replace")
    finally:
        result["telemetry"]["after"] = _nvidia_snapshot()
        result["packet_ended_at_epoch_ns"] = time.time_ns()
        result["packet_ended_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        _atomic_json(out_dir / "interference-run.json", result)
    return result


def _parse_timestamp(value: Any) -> float:
    if not isinstance(value, str):
        raise RuntimeError("Scheduler receipt timestamp is missing")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError(f"invalid Scheduler timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise RuntimeError("Scheduler timestamp must include a timezone")
    return parsed.timestamp()


def _load_receipt(receipt_root: Path, packet: dict[str, Any]) -> dict[str, Any]:
    job_id = packet.get("scheduler_job_id")
    if not isinstance(job_id, str) or not SAFE_ID_RE.fullmatch(job_id):
        raise RuntimeError("packet lacks a safe Scheduler job ID")
    path = _no_symlink_components(receipt_root / f"{job_id}.json")
    raw = _read_regular(path)
    data = json.loads(raw)
    job = data.get("job", data)
    if job.get("id") != job_id or job.get("cleanup_verified") is not True:
        raise RuntimeError(f"Scheduler receipt {job_id} lacks exact identity or verified cleanup")
    if job.get("state") not in SUCCESS_STATES:
        raise RuntimeError(f"Scheduler job {job_id} did not succeed")
    attempt = job.get("attempt", {})
    if attempt.get("id") != packet.get("scheduler_attempt_id"):
        raise RuntimeError(f"Scheduler attempt identity differs for {job_id}")
    started = _parse_timestamp(attempt.get("created_at"))
    ended = _parse_timestamp(attempt.get("finished_at"))
    if ended <= started:
        raise RuntimeError(f"Scheduler receipt {job_id} has an invalid execution interval")
    allocations = attempt.get("execution_context", {}).get("allocations", [])
    gpu_ids = [x.get("resource_id") for x in allocations
               if isinstance(x, dict) and str(x.get("resource_id", "")).startswith("gpu:")]
    expected_uuid = packet.get("device_uuid")
    if len(gpu_ids) != 1 or not str(gpu_ids[0]).endswith(":" + expected_uuid):
        raise RuntimeError(f"Scheduler receipt {job_id} does not grant exactly the packet UUID")
    start_ns = packet.get("runner_started_epoch_ns")
    end_ns = packet.get("runner_ended_epoch_ns")
    if (type(start_ns) is not int or type(end_ns) is not int or end_ns <= start_ns or
            start_ns / 1e9 < started - 2.0 or end_ns / 1e9 > ended + 2.0):
        raise RuntimeError(f"packet timestamps are not enclosed by Scheduler attempt {job_id}")
    return {"job_id": job_id, "attempt_id": attempt["id"], "state": job["state"],
            "cleanup_verified": True, "gpu_resource_id": gpu_ids[0], "device_uuid": expected_uuid,
            "scheduler_started_epoch_s": started, "scheduler_finished_epoch_s": ended,
            "receipt_sha256": sha256_bytes(raw), "receipt_path": str(path)}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return max(a0, b0) < min(a1, b1)


def _telemetry_isolated(packet: dict[str, Any], target_uuids: set[str], active_uuid: str) -> tuple[bool, list[str]]:
    telemetry = packet.get("telemetry", {})
    samples = [telemetry.get("before"), *telemetry.get("during", []), telemetry.get("after")]
    issues: list[str] = []
    valid_samples = 0
    for sample in samples:
        if not isinstance(sample, dict) or not sample.get("complete"):
            issues.append("incomplete all-device nvidia-smi observation")
            continue
        valid_samples += 1
        gpus = {x.get("uuid"): x for x in sample.get("gpus", [])}
        if not target_uuids <= set(gpus):
            issues.append("telemetry does not include both declared target UUIDs")
            continue
        for uuid, gpu in gpus.items():
            if uuid == active_uuid:
                continue
            if gpu.get("utilization_percent") != 0:
                issues.append(f"non-target GPU activity observed on {uuid}")
                break
        for app in sample.get("compute_processes", []):
            if app.get("gpu_uuid") != active_uuid:
                issues.append(f"non-target GPU compute process observed on {app.get('gpu_uuid')}")
                break
    if valid_samples == 0:
        issues.append("no complete telemetry samples")
    return not issues, sorted(set(issues))


def _t_critical_95(df: int) -> float:
    # Two-sided 95% Student t critical values. The long tail for tiny samples
    # deliberately makes a two-repeat check conservative.
    table = [0.0, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306,
             2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110,
             2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
             2.052, 2.048, 2.045, 2.042]
    return table[df] if df < len(table) else 1.96


def _ratio_report(pairs: list[dict[str, Any]], tolerance_percent: float) -> dict[str, Any]:
    ratios = [float(row["concurrent_ms"]) / float(row["isolated_ms"]) for row in pairs]
    logs = [math.log(x) for x in ratios]
    n = len(logs)
    median = math.exp(__import__("statistics").median(logs))
    uncertainty_note = None
    if n >= 2:
        mean = sum(logs) / n
        sd = __import__("statistics").stdev(logs)
        if sd == 0.0 and n < 5:
            ci = None
            uncertainty_note = "zero observed variance with fewer than five independent pairs cannot support an equivalence claim"
        else:
            half = _t_critical_95(n - 1) * sd / math.sqrt(n)
            ci = [math.exp(mean - half), math.exp(mean + half)]
    else:
        ci = None
        uncertainty_note = "fewer than two independent pairs"
    tolerance = tolerance_percent / 100.0
    lower, upper = 1.0 - tolerance, 1.0 + tolerance
    equivalent = bool(ci and ci[0] >= lower and ci[1] <= upper)
    return {"metric": "cold_execution_preparation_ms (= replay_input_preparation_ms + host_layout_preparation_ms + cold_device_preparation_ms) + recurrent_complete_horizon_ms + output_recovery_ms",
            "independent_pairs": n, "raw_pairs": pairs, "concurrent_over_isolated_ratios": ratios,
            "median_concurrent_over_isolated_ratio": median,
            "median_latency_change_percent": (median - 1.0) * 100.0,
            "paired_log_ratio_t_95_percent_interval": ci,
            "uncertainty_note": uncertainty_note,
            "practical_tolerance_percent": tolerance_percent,
            "tolerance_boundary_ratio": [lower, upper],
            "interval_within_tolerance": equivalent,
            "uncertainty_method": "two-sided paired Student t interval on log latency ratios; small-sample intervals are intentionally wide"}


def _serialized_policy(cfg: dict[str, Any], policy_path: Path | None,
                       receipt_root: Path, packets: list[dict[str, Any]]) -> dict[str, Any] | None:
    if policy_path is None:
        return None
    raw = _read_regular(policy_path)
    policy = json.loads(raw)
    if policy.get("schema") != "cgs.serialized-policy.v1":
        raise RuntimeError("serialized policy must use cgs.serialized-policy.v1")
    if (not isinstance(policy.get("policy_id"), str) or not policy["policy_id"] or
            not isinstance(policy.get("policy_description"), str) or not policy["policy_description"].strip() or
            policy.get("phase_scope") != "whole_scheduler_job" or
            not isinstance(policy.get("phases"), list) or not policy["phases"] or
            not isinstance(policy.get("job_ids"), list) or len(policy["job_ids"]) < 2):
        raise RuntimeError("serialized policy must name its whole-job policy, phases, and at least two receipts")
    receipt_map = {p.get("scheduler_job_id"): _load_receipt(receipt_root, p) for p in packets}
    ids = policy["job_ids"]
    if len(set(ids)) != len(ids) or not set(ids) <= set(receipt_map):
        raise RuntimeError("serialized policy job IDs must identify supplied retained run packets")
    rows = [receipt_map[j] for j in ids]
    uuids = {x["device_uuid"] for x in rows}
    if uuids != {cfg["primary_uuid"]}:
        raise RuntimeError("serialized policy receipts must use the same primary GPU UUID")
    rows.sort(key=lambda x: x["scheduler_started_epoch_s"])
    for left, right in zip(rows, rows[1:]):
        if _overlap(left["scheduler_started_epoch_s"], left["scheduler_finished_epoch_s"],
                    right["scheduler_started_epoch_s"], right["scheduler_finished_epoch_s"]):
            raise RuntimeError("serialized policy receipts overlap; serialization is unproven")
    if (policy.get("source_archive_sha256") != cfg["source_archive_sha256"] or
            policy.get("binary_sha256") != cfg["binary_sha256"] or
            policy.get("workload_sha256") != cfg["workload_sha256"]):
        raise RuntimeError("serialized policy does not bind the same source, binary and workload")
    return {"policy": policy, "policy_sha256": sha256_bytes(raw),
            "receipt_identities": rows,
            "proof_scope": "Scheduler whole-job windows only; declared internal phases are not separately instrumented"}


def _analyze_validated(args: argparse.Namespace) -> dict[str, Any]:
    cfg, config_sha = _load_config(Path(args.config))
    receipt_root = _no_symlink_components(Path(args.receipts_root))
    closures_root = _no_symlink_components(Path(args.closures_root))
    builds_root = _no_symlink_components(Path(args.builds_root))
    closure = _no_symlink_components(closures_root / f"{cfg['source_archive_sha256']}.tar.gz")
    binary = _no_symlink_components(builds_root / cfg["binary_sha256"] / "binary")
    if sha256_file(closure) != cfg["source_archive_sha256"]:
        raise RuntimeError("retained immutable source closure does not match the frozen source identity")
    if sha256_file(binary) != cfg["binary_sha256"]:
        raise RuntimeError("retained campaign binary does not match the frozen binary identity")
    build_receipts = []
    for path in sorted((builds_root / cfg["binary_sha256"]).glob("*.json")):
        raw_receipt = _read_regular(path)
        try:
            build = json.loads(raw_receipt)
        except json.JSONDecodeError:
            continue
        if build.get("archive_sha256") == cfg["source_archive_sha256"] and build.get("binary_sha256") == cfg["binary_sha256"]:
            build_receipts.append({"path": str(path), "sha256": sha256_bytes(raw_receipt)})
    if not build_receipts:
        raise RuntimeError("no preserved build receipt binds the source closure and campaign binary")
    isolated_paths = [Path(x) for x in args.isolated]
    concurrent_paths = [Path(x) for x in args.concurrent]
    if not isolated_paths or not concurrent_paths:
        raise RuntimeError("analysis needs isolated and concurrent run packets")
    packet_paths = isolated_paths + concurrent_paths
    packets: list[dict[str, Any]] = []
    receipts: dict[str, dict[str, Any]] = {}
    for path in packet_paths:
        data = json.loads(_read_regular(path))
        if data.get("schema") != RUN_SCHEMA or data.get("status") != "ok":
            raise RuntimeError(f"run packet is not a successful {RUN_SCHEMA}: {path}")
        if data.get("config_sha256") != config_sha or data.get("config") != cfg:
            raise RuntimeError("run packet configuration differs from the frozen analyzer config")
        for field in IDENTITY_FIELDS:
            if data.get(field) != cfg.get(field):
                raise RuntimeError(f"run packet identity differs on {field}")
        campaign = data.get("campaign")
        raw_campaign = data.get("campaign_raw_json")
        if not isinstance(campaign, dict) or not isinstance(raw_campaign, str):
            raise RuntimeError("run packet lacks the full raw cgs campaign JSON")
        try:
            if json.loads(raw_campaign) != campaign:
                raise RuntimeError("embedded cgs campaign object differs from its retained raw JSON")
        except json.JSONDecodeError as exc:
            raise RuntimeError("embedded cgs campaign JSON is invalid") from exc
        actual_total = _validate_campaign(campaign, cfg, data.get("device_uuid"))
        if not _finite(data.get("operational_total_ms")) or not math.isclose(
                float(data["operational_total_ms"]), actual_total, rel_tol=1e-9, abs_tol=1e-9):
            raise RuntimeError("packet operational cost does not match raw cgs cost-boundary fields")
        command = data.get("command")
        if (not isinstance(command, list) or not command or
                os.path.abspath(command[0]) != os.path.abspath(data.get("binary_path", ""))):
            raise RuntimeError("packet command does not identify the executed cgs binary")
        for flag, expected in (("--mode", "campaign"), ("--case", cfg["case"]),
                               ("--backend", cfg["backend"]), ("--gate", cfg["gate"]),
                               ("--body", cfg["body"]), ("--precision", cfg["precision"]),
                               ("--condition-update", cfg["condition_update"]),
                               ("--tile", str(cfg["tile"])), ("--stages", str(cfg["stages"])),
                               ("--warmups", str(cfg["warmups"])), ("--repeats", "1")):
            try:
                actual = command[command.index(flag) + 1]
            except (ValueError, IndexError) as exc:
                raise RuntimeError(f"packet command omits {flag}") from exc
            if actual != expected:
                raise RuntimeError(f"packet command differs from frozen config for {flag}")
        if data.get("repetition") not in range(cfg["independent_repetitions"]):
            raise RuntimeError("run packet repetition is outside the frozen config")
        receipt = _load_receipt(receipt_root, data)
        job_id = data["scheduler_job_id"]
        if job_id in receipts:
            raise RuntimeError("duplicate Scheduler job packet")
        receipts[job_id] = receipt
        data["_packet_path"] = str(path)
        data["_receipt"] = receipt
        packets.append(data)
    isolated = []
    concurrent = []
    for source_path, packet in zip(packet_paths, packets):
        expected_mode = "isolated" if source_path in isolated_paths else "concurrent"
        if packet.get("mode") != expected_mode:
            raise RuntimeError(f"packet mode does not match its input list: {source_path}")
        (isolated if expected_mode == "isolated" else concurrent).append(packet)
    # Every independent repetition is a distinct admitted job. This also
    # prevents repeated samples from one process from masquerading as trials.
    primary_isolated = [x for x in isolated if x.get("device_uuid") == cfg["primary_uuid"]]
    iso_rep = [x.get("repetition") for x in primary_isolated]
    wanted_reps = list(range(cfg["independent_repetitions"]))
    if sorted(iso_rep) != wanted_reps or len({x.get("scheduler_job_id") for x in primary_isolated}) != len(wanted_reps):
        raise RuntimeError("need one distinct isolated primary-UUID Scheduler job per independent repetition")
    for x in isolated:
        if x.get("device_uuid") != cfg["primary_uuid"]:
            raise RuntimeError("isolated packets must be on the declared primary UUID; cross-device timing is forbidden")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in concurrent:
        barrier = row.get("barrier") or {}
        group = barrier.get("barrier_id")
        if not isinstance(group, str) or not group:
            raise RuntimeError("concurrent run lacks its file-barrier group identity")
        groups.setdefault(group, []).append(row)
    if len(groups) != cfg["independent_repetitions"]:
        raise RuntimeError("need one concurrent two-job barrier group per independent repetition")
    primary_concurrent: list[dict[str, Any]] = []
    concurrent_group_evidence = []
    seen_reps: set[int] = set()
    for group_id, members in sorted(groups.items()):
        if len(members) != 2:
            raise RuntimeError(f"concurrent barrier {group_id} must have exactly two job packets")
        jobs = {x["scheduler_job_id"] for x in members}
        uuids = {x["device_uuid"] for x in members}
        reps = {x["repetition"] for x in members}
        if len(jobs) != 2 or uuids != set(cfg["concurrent_uuids"]) or len(reps) != 1:
            raise RuntimeError(f"concurrent barrier {group_id} has mismatched job/device/repetition membership")
        repetition = next(iter(reps))
        if repetition in seen_reps:
            raise RuntimeError("concurrent repetition duplicated")
        seen_reps.add(repetition)
        if any(x.get("hostname") != members[0].get("hostname") for x in members):
            raise RuntimeError("concurrent pair did not run on the same host")
        a, b = members
        a0, a1 = a["runner_started_epoch_ns"] / 1e9, a["runner_ended_epoch_ns"] / 1e9
        b0, b1 = b["runner_started_epoch_ns"] / 1e9, b["runner_ended_epoch_ns"] / 1e9
        if not _overlap(a0, a1, b0, b1):
            raise RuntimeError(f"concurrent barrier {group_id} did not produce overlapping runner windows")
        ar, br = a["_receipt"], b["_receipt"]
        if not _overlap(ar["scheduler_started_epoch_s"], ar["scheduler_finished_epoch_s"],
                        br["scheduler_started_epoch_s"], br["scheduler_finished_epoch_s"]):
            raise RuntimeError(f"concurrent barrier {group_id} Scheduler attempt windows do not overlap")
        primary = next(x for x in members if x["device_uuid"] == cfg["primary_uuid"])
        primary_concurrent.append(primary)
        concurrent_group_evidence.append({"barrier_id": group_id, "repetition": repetition,
                                          "job_ids": sorted(jobs), "uuids": sorted(uuids),
                                          "runner_overlap_seconds": min(a1, b1) - max(a0, b0),
                                          "scheduler_overlap_seconds": min(ar["scheduler_finished_epoch_s"], br["scheduler_finished_epoch_s"]) -
                                                                      max(ar["scheduler_started_epoch_s"], br["scheduler_started_epoch_s"])})
    if seen_reps != set(wanted_reps):
        raise RuntimeError("concurrent repetitions do not match the frozen config")
    if sorted(x["repetition"] for x in primary_concurrent) != wanted_reps:
        raise RuntimeError("concurrent primary-UUID packets are incomplete")
    # Isolated whole-job windows must not overlap each other. nvidia-smi
    # observations additionally look for unrelated GPU activity.
    iso_sorted = sorted(primary_isolated, key=lambda x: x["packet_started_at_epoch_ns"])
    for left, right in zip(iso_sorted, iso_sorted[1:]):
        if _overlap(left["packet_started_at_epoch_ns"] / 1e9, left["packet_ended_at_epoch_ns"] / 1e9,
                    right["packet_started_at_epoch_ns"] / 1e9, right["packet_ended_at_epoch_ns"] / 1e9):
            raise RuntimeError("isolated primary jobs overlap; isolated timing is not established")
    for solo in primary_isolated:
        for busy in concurrent:
            if _overlap(solo["packet_started_at_epoch_ns"] / 1e9, solo["packet_ended_at_epoch_ns"] / 1e9,
                        busy["packet_started_at_epoch_ns"] / 1e9, busy["packet_ended_at_epoch_ns"] / 1e9):
                raise RuntimeError("isolated and concurrent measurement packet windows overlap")
            sr, cr = solo["_receipt"], busy["_receipt"]
            if _overlap(sr["scheduler_started_epoch_s"], sr["scheduler_finished_epoch_s"],
                        cr["scheduler_started_epoch_s"], cr["scheduler_finished_epoch_s"]):
                raise RuntimeError("isolated and concurrent Scheduler attempt windows overlap")
    target_uuids = set(cfg["concurrent_uuids"])
    telemetry_checks = {x["scheduler_job_id"]: _telemetry_isolated(x, target_uuids, cfg["primary_uuid"])
                        for x in primary_isolated}
    telemetry_clean = all(ok for ok, _ in telemetry_checks.values())
    pairs = []
    by_rep_iso = {x["repetition"]: x for x in primary_isolated}
    by_rep_conc = {x["repetition"]: x for x in primary_concurrent}
    for rep in wanted_reps:
        left, right = by_rep_iso[rep], by_rep_conc[rep]
        pairs.append({"repetition": rep, "device_uuid": cfg["primary_uuid"],
                      "isolated_job_id": left["scheduler_job_id"],
                      "concurrent_job_id": right["scheduler_job_id"],
                      "isolated_ms": left["operational_total_ms"],
                      "concurrent_ms": right["operational_total_ms"]})
    tolerance_source = "config" if "practical_tolerance_percent" in cfg else "default-5-percent"
    tolerance = float(cfg.get("practical_tolerance_percent", 5.0))
    report = _ratio_report(pairs, tolerance)
    serialized = _serialized_policy(cfg, Path(args.serialized_policy) if args.serialized_policy else None,
                                    receipt_root, packets)
    # Whole-process windows include oracle validation and other setup. Their
    # overlap does not establish overlap of the measured phases. Likewise a
    # policy document and past sequential receipts cannot enforce future jobs.
    # Retain the numerical comparison as exploratory, but fail closed until
    # measured phase windows or an independently enforced runtime policy exist.
    status = "inconclusive"
    reason = ("runner and Scheduler windows alone do not prove overlap of measured phases; "
              "a document describing serialization does not enforce future jobs")
    all_receipts = {job: receipt for job, receipt in sorted(receipts.items())}
    primary_iso_jobs = sorted(x["scheduler_job_id"] for x in primary_isolated)
    primary_concurrent_jobs = sorted(x["scheduler_job_id"] for x in primary_concurrent)
    partner_jobs = sorted(x["scheduler_job_id"] for x in concurrent if x["device_uuid"] != cfg["primary_uuid"])
    receipt_lists = [set(primary_iso_jobs), set(primary_concurrent_jobs), set(partner_jobs)]
    if any(receipt_lists[i] & receipt_lists[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("receipt IDs must be unique across isolated, concurrent-primary, and partner lists")
    result = {"schema": RESULT_SCHEMA, "status": status, "analysis_complete": True, "reason": reason,
              "device_uuid": cfg["primary_uuid"], "primary_uuid": cfg["primary_uuid"],
              "concurrent_uuids": list(cfg["concurrent_uuids"]),
              "source_archive_sha256": cfg["source_archive_sha256"], "binary_sha256": cfg["binary_sha256"],
              "workload_sha256": cfg["workload_sha256"], "config_sha256": config_sha,
              "case_identity": {k: cfg.get(k) for k in IDENTITY_FIELDS},
              "tolerance_source": tolerance_source, "measurement": report,
              "independent_repetitions_required": cfg["independent_repetitions"],
              "concurrent_groups": concurrent_group_evidence,
              "isolated_telemetry_clean": telemetry_clean,
              "isolated_telemetry_by_job": {job: {"clean": ok, "issues": issues}
                                            for job, (ok, issues) in telemetry_checks.items()},
              "receipt_identities": all_receipts,
              "isolated_receipts": primary_iso_jobs,
              "concurrent_receipts": primary_concurrent_jobs,
              "concurrent_partner_receipts": partner_jobs,
              "isolated_nonoverlap_verified": True,
              "serialized_policy_evidence": serialized,
              "source_closure": {"path": str(closure), "sha256": cfg["source_archive_sha256"]},
              "binary": {"path": str(binary), "sha256": cfg["binary_sha256"]},
              "build_receipts": build_receipts,
              "limitations": ["nvidia-smi cannot prove the absence of activity shorter than the sampling interval",
                              "the t interval is based on independent Scheduler-job samples and is conservative for small n",
                              "status passed qualifies only this measurement environment and case; it is not a kernel performance claim"]}
    return result


def _save_analysis(args: argparse.Namespace, result: dict[str, Any]) -> None:
    if not args.output:
        return
    out = Path(os.path.abspath(args.output))
    _no_symlink_components(out.parent, allow_missing=True)
    _mkdir_safe(out.parent)
    _atomic_json(out, result)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    """Return an explicit inconclusive record for incomplete evidence joins."""
    try:
        result = _analyze_validated(args)
    except Exception as exc:
        cfg = None
        config_sha = None
        try:
            cfg, config_sha = _load_config(Path(args.config))
        except Exception:
            pass
        result = {"schema": RESULT_SCHEMA, "status": "inconclusive", "analysis_complete": False,
                  "reason": f"evidence validation failed: {exc}", "config_sha256": config_sha,
                  "device_uuid": cfg.get("primary_uuid") if cfg else None,
                  "primary_uuid": cfg.get("primary_uuid") if cfg else None,
                  "source_archive_sha256": cfg.get("source_archive_sha256") if cfg else None,
                  "binary_sha256": cfg.get("binary_sha256") if cfg else None,
                  "workload_sha256": cfg.get("workload_sha256") if cfg else None,
                  "isolated_receipts": [], "concurrent_receipts": [],
                  "concurrent_partner_receipts": [], "receipt_identities": {},
                  "limitations": ["Incomplete or mismatched evidence is not eligible for qualification"]}
    _save_analysis(args, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run", help="run one admitted isolated or concurrent sample")
    run.add_argument("--mode", choices=("isolated", "concurrent"), required=True)
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--binary", required=True, type=Path)
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--repetition", required=True, type=int)
    run.add_argument("--source-root", type=Path)
    run.add_argument("--shared-study-root", type=Path)
    run.add_argument("--barrier-id")
    run.add_argument("--barrier-timeout", type=int, default=45)
    analyze_p = sub.add_parser("analyze", help="host-only receipt-joined interference analysis")
    analyze_p.add_argument("--config", required=True, type=Path)
    analyze_p.add_argument("--receipts-root", required=True, type=Path)
    analyze_p.add_argument("--closures-root", required=True, type=Path)
    analyze_p.add_argument("--builds-root", required=True, type=Path)
    analyze_p.add_argument("--isolated", action="append", required=True, help="isolated run packet; repeat once per Scheduler job")
    analyze_p.add_argument("--concurrent", action="append", required=True, help="concurrent participant run packet; repeat for both devices and all repetitions")
    analyze_p.add_argument("--serialized-policy", type=Path, help="optional cgs.serialized-policy.v1 plus its job packets and receipts")
    analyze_p.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.action == "run":
            result = run_packet(args)
            print(json.dumps({"status": result.get("status"), "output": str(args.output_dir / "interference-run.json"),
                              "scheduler_job_id": result.get("scheduler_job_id"), "device_uuid": result.get("device_uuid")}, sort_keys=True))
            return 0 if result.get("status") == "ok" else 1
        result = analyze(args)
        print(json.dumps({"schema": result["schema"], "status": result["status"],
                          "output": str(args.output) if args.output else None,
                          "primary_uuid": result["primary_uuid"],
                          "median_ratio": result.get("measurement", {}).get("median_concurrent_over_isolated_ratio")}, sort_keys=True))
        return 0 if result["status"] in ("passed", "serialized") else 2
    except Exception as exc:
        print(f"interference: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
