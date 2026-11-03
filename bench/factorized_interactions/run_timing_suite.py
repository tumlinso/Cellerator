#!/usr/bin/env python3
"""Paired packet/direct timing after frozen correctness and sanitizers pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
VALIDATOR = ROOT / "bench" / "factorized_interactions" / "validate_runner.py"
sys.path.insert(0, str(ROOT / "tests" / "factorized_interactions"))
from workloads import performance_workload_ids  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--expected-uuid", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()

    job_id = os.environ.get("AUTOBENCH_SCHEDULER_JOB_ID", "").strip()
    if not job_id:
        parser.error("timing suite requires an Autobench Scheduler job context")
    if not args.expected_uuid.startswith("GPU-"):
        parser.error("--expected-uuid must be the exact Scheduler GPU UUID")
    if args.iterations < 10 or args.iterations > 256:
        parser.error("--iterations must be in the range 10..256")
    runner = args.runner.resolve(strict=True)
    manifest_path = args.build_manifest.resolve(strict=True)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    runner_hash = sha256_file(runner)
    expected_runner = (manifest_path.parent /
                       manifest["binaries"]["runner"]["path"]).resolve(strict=True)
    if runner != expected_runner or runner_hash != manifest["binaries"]["runner"]["sha256"]:
        parser.error("runner executable identity differs from the build manifest")
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    results = []
    comparisons = []
    for index, workload in enumerate(performance_workload_ids()):
        route_order = ("packet", "direct") if index % 2 == 0 else ("direct", "packet")
        pair = {}
        for route in route_order:
            case_dir = args.artifact_dir / f"{index:02d}-{route}-{workload}"
            case_dir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable, str(VALIDATOR),
                "--workload", workload,
                "--runner", shlex.quote(str(runner)),
                "--runner-binary", str(runner),
                "--build-manifest", str(manifest_path),
                "--route", route,
                "--activation-dtype", "fp16",
                "--parameter-dtype", "fp16",
                "--output-dtype", "fp16",
                "--gradient-mode", "overwrite",
                "--expected-uuid", args.expected_uuid,
                "--iterations", str(args.iterations),
                "--artifact-dir", str(case_dir),
            ]
            proc = subprocess.run(command, cwd=ROOT, text=True,
                                  capture_output=True, check=False)
            (case_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
            (case_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
            if proc.returncode:
                sys.stderr.write(proc.stderr)
                raise RuntimeError(f"timing validation failed for {route}/{workload}")
            summary = json.loads(proc.stdout)
            metrics = summary.get("metrics")
            if not summary.get("validated") or not isinstance(metrics, dict):
                raise RuntimeError(f"timing output missing validated metrics for {route}/{workload}")
            if metrics.get("scheduler_job_id") != job_id:
                raise RuntimeError("timing metrics do not match the Scheduler job ID")
            if metrics.get("device_uuid") != args.expected_uuid:
                raise RuntimeError("timing metrics do not match the explicitly selected GPU UUID")
            if metrics.get("selected_backend") != \
                    summary.get("native_execution", {}).get("selected_backend"):
                raise RuntimeError("timing metrics and dispatch metadata disagree on backend")
            record = {"workload": workload, "route": route,
                      "validation": summary, "metrics": metrics}
            pair[route] = metrics
            results.append(record)

        packet = pair["packet"]["phase_ms"]
        direct = pair["direct"]["phase_ms"]
        packet_backend = pair["packet"].get("selected_backend")
        direct_backend = pair["direct"].get("selected_backend")
        distinct_requested_routes = packet_backend == "packet" and direct_backend == "direct"
        comparisons.append({
            "workload": workload,
            "packet_first": route_order[0] == "packet",
            "packet_selected_backend": packet_backend,
            "direct_selected_backend": direct_backend,
            "distinct_packet_vs_direct_backends": distinct_requested_routes,
            "packet_over_direct_resident_step_p50": (
                direct["resident_forward_backward_p50"] /
                packet["resident_forward_backward_p50"]
                if distinct_requested_routes and packet["resident_forward_backward_p50"]
                else None),
            "packet_over_direct_first_train_step_p50": (
                direct["first_train_step_including_setup_p50"] /
                packet["first_train_step_including_setup_p50"]
                if distinct_requested_routes and packet["first_train_step_including_setup_p50"]
                else None),
        })

    result = {
        "schema": "cellerator.factorized_interactions.timing_suite.v1",
        "scheduler_job_id": job_id,
        "device_uuid": args.expected_uuid,
        "build_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_closure_sha256": manifest["source_closure_sha256"],
        "runner_binary_sha256": runner_hash,
        "timed": True,
        "iterations_per_route_case": args.iterations,
        "route_order_policy": "alternate packet-first and direct-first by workload index",
        "case_count": len(results),
        "results": results,
        "packet_over_direct_ratios": comparisons,
        "promotion_decision": "not made by this harness",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"suite": result["schema"], "case_count": len(results),
                      "workloads": len(comparisons), "timed": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
