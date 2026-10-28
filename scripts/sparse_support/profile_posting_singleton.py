#!/usr/bin/env python3
"""Collect NCU diagnostics for the singleton-query CUDA count and scatter.

Invoke only inside an admitted Device Scheduler job. This driver does not
select a GPU, build code, request resources, or launch an unmanaged kernel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ncu", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    ncu = args.ncu.expanduser().resolve()
    binary = args.binary.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    if not ncu.is_file() or not binary.is_file():
        parser.error("Nsight Compute or frozen candidate binary is missing")
    if (run_dir / "manifest.json").exists():
        parser.error(f"refusing to overwrite profile evidence in {run_dir}")

    harness = run_dir / "harness"
    reports = run_dir / "reports"
    harness.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    binary_copy = harness / binary.name
    binary_copy.write_bytes(binary.read_bytes())
    binary_copy.chmod(binary.stat().st_mode & 0o777)
    source = Path("/home/tumlinson/Cellerator/src/indexing/cuda_postings.cu")
    source_copy = harness / "cuda_postings.cu"
    source_copy.write_bytes(source.read_bytes())

    manifest: dict[str, object] = {
        "schema": "cellerator.sparse_support.ncu_profile_run.v1",
        "status": "running",
        "created_unix_ns": time.time_ns(),
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "ncu_path": str(ncu),
        "ncu_sha256": sha256(ncu),
        "ncu_version": subprocess.run([str(ncu), "--version"], check=True,
                                       text=True, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT).stdout.strip(),
        "binary_path": str(binary),
        "binary_sha256": sha256(binary),
        "profile_binary_path": str(binary_copy),
        "profile_binary_sha256": sha256(binary_copy),
        "source_path": str(source),
        "source_sha256": sha256(source),
        "profile_source_sha256": sha256(source_copy),
        "case": {
            "case_id": "source_skew_0_9",
            "edges": 2176,
            "cells": 64,
            "entities": 256,
            "gate_density": 0.5,
            "source_skew": 0.9,
            "dirty_fraction": 0.01,
            "seed": 20261027,
            "warmup": 1,
            "samples": 3,
            "iterations": 1,
        },
        "profiles": [],
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    for kernel in ("query_counts_kernel", "query_scatter_kernel"):
        output_base = reports / kernel
        command = [
            str(ncu), "--set", "full", "--import-source", "yes",
            "--source-folders",
            f"/home/tumlinson/Cellerator,{harness}",
            "--kernel-name", f"regex:{kernel}", "--launch-count", "1",
            "-o", str(output_base), str(binary_copy),
            "--case-id", "source_skew_0_9", "--edges", "2176",
            "--cells", "64", "--entities", "256", "--gate-density", "0.5",
            "--source-skew", "0.9", "--dirty-fraction", "0.01",
            "--seed", "20261027", "--warmup", "1", "--samples", "3",
            "--iterations", "1",
        ]
        completed = subprocess.run(command, check=False, text=True,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
        profile = {
            "kernel_filter": kernel,
            "command": command,
            "exit_code": completed.returncode,
            "output": completed.stdout,
            "expected_report": str(output_base.with_suffix(".ncu-rep")),
        }
        manifest["profiles"].append(profile)
        manifest["status"] = "running" if completed.returncode == 0 else "failed"
        manifest["updated_unix_ns"] = time.time_ns()
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    manifest["status"] = "complete" if all(
        profile["exit_code"] == 0 and Path(profile["expected_report"]).is_file()
        for profile in manifest["profiles"]
    ) else "failed"
    manifest["completed_unix_ns"] = time.time_ns()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
