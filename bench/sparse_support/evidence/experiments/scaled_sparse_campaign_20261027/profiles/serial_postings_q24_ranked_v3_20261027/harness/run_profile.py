#!/usr/bin/env python3
"""Collect source-bound Nsight Compute reports for the frozen q24 posting path.

This entrypoint is intended to run only inside an admitted Device Scheduler job.
It records one representative launch from each serial posting kernel; the NCU
reports are diagnostic and must not be used as throughput timings.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path("/home/tumlinson/Cellerator")
CAMPAIGN = ROOT / "bench/sparse_support/evidence/experiments/scaled_sparse_campaign_20261027"
RUN = CAMPAIGN / "profiles/serial_postings_q24_ranked_v3_20261027"
HARNESS = RUN / "harness"
REPORTS = RUN / "reports"
BINARY = HARNESS / "cellerator_sparse_support_cuda_indexing_bench"
SOURCE_ROOT = HARNESS / "source"
FIXTURE = Path("/tmp/cellerator_scaled_sparse_fixtures_887437fc/large-ranked")
NCU = Path("/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/profilers/12.9/Nsight_Compute/ncu")
EXPECTED_BINARY_SHA256 = "a628a22579a576d51f942f91d3c0f279ec7c371fdec36f0b454a956e7a44ae7f"
EXPECTED_MANIFEST_SHA256 = "674d5d4660befcb6ddbcd618b16b136203d3478ddc577d09a037e116bffd6e32"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if os.environ.get("CELLERATOR_SPARSE_SUPPORT_PROFILE_ADMITTED") != "1":
        raise SystemExit("NCU collection requires Device Scheduler admission")
    if sha256(BINARY) != EXPECTED_BINARY_SHA256:
        raise SystemExit("frozen profile binary identity mismatch")
    manifest = FIXTURE / "manifest.json"
    if sha256(manifest) != EXPECTED_MANIFEST_SHA256:
        raise SystemExit("large-ranked fixture manifest identity mismatch")

    ncu_version = subprocess.run([str(NCU), "--version"], check=True,
                                 capture_output=True, text=True).stdout.strip()
    ncu_help = subprocess.run([str(NCU), "--help"], check=True,
                              capture_output=True, text=True).stdout
    if "--source-folders" not in ncu_help or "--clock-control" not in ncu_help:
        raise SystemExit("selected NCU does not expose the verified options")

    base_args = [
        "--case-id", "large_ranked_v3_ncu_q24", "--edges", "262144",
        "--cells", "32", "--entities", "4096", "--fixture-dir", str(FIXTURE),
        "--warmup", "0", "--samples", "1", "--iterations", "1",
    ]
    kernels = ("query_counts_kernel", "query_scatter_kernel")
    records = []
    started = time.time()
    env = os.environ.copy()
    for kernel in kernels:
        tag = kernel.removesuffix("_kernel")
        report = REPORTS / f"full_{tag}_large_ranked_q24.ncu-rep"
        command = [
            str(NCU), "--set", "full", "--import-source", "yes",
            "--source-folders", str(SOURCE_ROOT), "--clock-control", "base",
            "-k", f"regex:{kernel}", "-c", "1", "-o", str(report),
            str(BINARY), *base_args,
        ]
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                                text=True, check=False)
        (REPORTS / f"{tag}.stdout.txt").write_text(result.stdout, encoding="utf-8")
        (REPORTS / f"{tag}.stderr.txt").write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0 or not report.is_file() or report.stat().st_size == 0:
            raise RuntimeError(f"NCU collection failed for {kernel}: {result.returncode}")
        records.append({
            "kernel": kernel, "command": command, "exit_code": result.returncode,
            "report_path": str(report), "report_bytes": report.stat().st_size,
            "report_sha256": sha256(report),
            "stdout_sha256": sha256(REPORTS / f"{tag}.stdout.txt"),
            "stderr_sha256": sha256(REPORTS / f"{tag}.stderr.txt"),
        })

    gpu_query = [
        "nvidia-smi", "--query-gpu=name,uuid,compute_cap,clocks.sm",
        "--format=csv,noheader",
    ]
    gpu = subprocess.run(gpu_query, cwd=ROOT, env=env, capture_output=True,
                         text=True, check=False)
    metadata = {
        "schema": "cellerator.sparse_support.serial_postings_ncu_profile.v1",
        "started_unix_seconds": started,
        "completed_unix_seconds": time.time(),
        "scheduler_job_id": os.environ.get("DEVICE_SCHEDULER_JOB_ID"),
        "scheduler_attempt_id": os.environ.get("DEVICE_SCHEDULER_ATTEMPT_ID"),
        "ncu_path": str(NCU), "ncu_sha256": sha256(NCU),
        "ncu_version": ncu_version,
        "clock_policy": "base",
        "target": "V100/SM70 large-ranked fixture, B32 E262144 N4096, q24 source postings per cell",
        "binary_sha256": sha256(BINARY),
        "fixture_manifest_sha256": sha256(manifest),
        "source_snapshot_manifest_sha256": sha256(CAMPAIGN / "frozen_baseline/manifest.json"),
        "profile_harness_source_sha256": sha256(HARNESS / "cuda_indexing_bench.cu"),
        "source_snapshot_root": str(SOURCE_ROOT),
        "commands": records,
        "gpu_query_exit_code": gpu.returncode,
        "gpu_query": gpu.stdout.strip(),
        "limitations": [
            "Profiler replay output is diagnostic, not throughput evidence.",
            "Volta does not support the PM-sampling timeline used on newer GPUs.",
            "This profile measures the frozen pre-fastpath serial baseline only.",
        ],
    }
    manifest_path = RUN / "collection_manifest.json"
    manifest_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    print(json.dumps({"profile_manifest": str(manifest_path),
                      "profile_manifest_sha256": sha256(manifest_path),
                      "reports": len(records)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
