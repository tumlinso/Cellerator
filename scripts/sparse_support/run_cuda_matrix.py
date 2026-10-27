#!/usr/bin/env python3
"""Run a bounded, already-built CUDA sparse-indexing matrix.

Run this driver only as the argv of an admitted Device Scheduler job. It does
not request a GPU, build code, select a device, or launch profilers itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
TARGET = "cellerator_sparse_support_cuda_indexing_bench"
SOURCE_INPUTS = [
    "CMakeLists.txt",
    "cmake/NativeFoundation.cmake",
    "bench/sparse_support/CMakeLists.txt",
    "bench/sparse_support/cuda_indexing_bench.cu",
    "include/Cellerator/compute/support/logical_support.hh",
    "src/compute/support/logical_support.cc",
    "include/Cellerator/compute/indexing/reverse_postings.hh",
    "include/Cellerator/compute/indexing/cuda_postings.cuh",
    "src/compute/indexing/reverse_postings.cc",
    "src/compute/indexing/cuda_postings.cu",
    "include/Cellerator/compute/gating/condition_program.hh",
    "include/Cellerator/compute/gating/cuda_worksets.cuh",
    "src/compute/gating/condition_program.cc",
    "src/compute/gating/cuda_worksets.cu",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(argv: list[str]) -> str:
    try:
        completed = subprocess.run(argv, check=True, text=True,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
        return completed.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unavailable"


def source_revision() -> str:
    return command_output(["git", "rev-parse", "HEAD"])


def get_matrix(matrix_name: str) -> list[tuple[str, dict[str, object]]]:
    anchor: dict[str, object] = {
        "edges": 2176, "cells": 64, "entities": 256,
        "gate_density": 0.5, "source_skew": 0.0, "dirty_fraction": 0.01,
    }
    if matrix_name == "anchor":
        return [("anchor_e2176_b64", anchor)]
    if matrix_name == "qualification":
        return [
            ("anchor_e2176_b64", anchor),
            ("batch_b1", {**anchor, "cells": 1}),
            ("batch_b512", {**anchor, "cells": 512}),
            ("scale_e65536_b1", {**anchor, "edges": 65536, "cells": 1}),
            ("source_skew_0_9", {**anchor, "source_skew": 0.9}),
            ("gate_density_0_1", {**anchor, "gate_density": 0.1}),
            ("gate_density_0_9", {**anchor, "gate_density": 0.9}),
            ("dirty_fraction_0_1", {**anchor, "dirty_fraction": 0.1}),
        ]
    raise ValueError(f"unsupported matrix: {matrix_name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matrix", choices=("anchor", "qualification"),
                        default="anchor")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args()
    if args.warmup < 0 or args.samples < 3 or args.iterations < 1:
        parser.error("warmup must be nonnegative, samples >= 3, iterations >= 1")

    binary = args.binary.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser.error(f"benchmark binary is not executable: {binary}")
    if output.exists():
        parser.error(f"refusing to overwrite evidence: {output}")
    missing = [path for path in SOURCE_INPUTS if not (ROOT / path).is_file()]
    if missing:
        parser.error(f"missing source inputs: {missing}")

    cases = get_matrix(args.matrix)
    commands: list[list[str]] = []
    for case_id, shape in cases:
        command = [
            str(binary), "--case-id", case_id,
            "--edges", str(shape["edges"]), "--cells", str(shape["cells"]),
            "--entities", str(shape["entities"]),
            "--gate-density", str(shape["gate_density"]),
            "--source-skew", str(shape["source_skew"]),
            "--dirty-fraction", str(shape["dirty_fraction"]),
            "--seed", str(args.seed), "--warmup", str(args.warmup),
            "--samples", str(args.samples), "--iterations", str(args.iterations),
        ]
        commands.append(command)

    build_dir = binary.parents[2]
    artifacts = {
        "benchmark_binary_sha256": sha256(binary),
        "sparse_support_host_library_sha256": sha256(
            build_dir / "src/compute/support/libcellerator_sparse_support.a"
        ),
        "sparse_support_cuda_library_sha256": sha256(
            build_dir / "libcellerator_sparse_support_cuda.a"
        ),
    }
    manifest = {
        "schema": "cellerator.sparse_support.cuda_run_manifest.v1",
        "scope": "device_resident_cuda_comparison_setup_separate",
        "timestamp_unix_ns": time.time_ns(),
        "git_head": source_revision(),
        "source_inputs_sha256": {
            path: sha256(ROOT / path) for path in SOURCE_INPUTS
        },
        **artifacts,
        "binary_path": str(binary),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "nvcc_version": command_output(["nvcc", "--version"]),
        "ncu_version": command_output(["ncu", "--version"]),
        "nsys_version": command_output(["nsys", "--version"]),
        "matrix": args.matrix,
        "cases": [case_id for case_id, _ in cases],
        "commands": commands,
        "warmup": args.warmup,
        "samples": args.samples,
        "iterations": args.iterations,
        "seed": args.seed,
        "timing_policy": (
            "CUDA events around steady-state device-resident calls; setup and H2D "
            "reported separately; raw samples retained; no host affinity control"
        ),
        "status": "running",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        destination.write(json.dumps(manifest, sort_keys=True) + "\n")
        destination.flush()
        for (case_id, _), command in zip(cases, commands, strict=True):
            print("+ " + " ".join(command), file=sys.stderr, flush=True)
            completed = subprocess.run(command, cwd=ROOT, check=True, text=True,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
            record = json.loads(completed.stdout)
            if record.get("case_id") != case_id \
                    or record.get("correctness") != "exact_host_oracle_pass":
                raise RuntimeError(f"case {case_id} failed its exact-oracle check")
            record["run_metadata"] = {
                "command": command,
                "benchmark_binary_sha256": artifacts["benchmark_binary_sha256"],
                "git_head": manifest["git_head"],
            }
            destination.write(json.dumps(record, sort_keys=True) + "\n")
            destination.flush()
        manifest["status"] = "complete"
        destination.write(json.dumps({"schema": manifest["schema"],
                                      "status": "complete",
                                      "completed_unix_ns": time.time_ns()},
                                     sort_keys=True) + "\n")
        destination.flush()

    print(f"wrote {len(cases)} exact CUDA cases to {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
