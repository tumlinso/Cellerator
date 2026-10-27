#!/usr/bin/env python3
"""Build and run the bounded, host-only sparse support indexing matrix."""

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
TARGET = "cellerator_sparse_support_indexing_bench"
SOURCE_INPUTS = [
    "cmake/CelleratorSparseSupport.cmake",
    "src/compute/support/CMakeLists.txt",
    "src/compute/support/logical_support.cc",
    "src/compute/indexing/reverse_postings.cc",
    "src/compute/gating/condition_program.cc",
    "include/Cellerator/compute/support/logical_support.hh",
    "include/Cellerator/compute/indexing/reverse_postings.hh",
    "include/Cellerator/compute/gating/condition_program.hh",
    "bench/sparse_support/CMakeLists.txt",
    "bench/sparse_support/indexing_bench.cc",
]


def run(command: list[str], *, cwd: Path = ROOT, capture: bool = False) -> str:
    print("+ " + " ".join(command), file=sys.stderr)
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    return completed.stdout if capture else ""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_revision() -> str:
    try:
        return run(["git", "rev-parse", "HEAD"], capture=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unavailable"


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unavailable"


def find_executable(build_dir: Path) -> Path:
    preferred = build_dir / "bench/sparse_support" / TARGET
    if preferred.is_file() and os.access(preferred, os.X_OK):
        return preferred
    candidates = [
        path for path in build_dir.rglob(TARGET)
        if path.is_file() and os.access(path, os.X_OK)
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one built {TARGET}, found {candidates}")
    return candidates[0]


def matrix(samples: int, target_edge_visits: int) -> list[tuple[str, list[str]]]:
    common = [
        "--entities", "256",
        "--query-terms", "1",
        "--match", "any",
        "--gate-density", "0.5",
        "--skew", "0",
        "--dirty-fraction", "0.01",
        "--samples", str(samples),
        "--target-edge-visits", str(target_edge_visits),
        "--seed", "20261027",
    ]

    def case(name: str, changes: list[str]) -> tuple[str, list[str]]:
        return name, common + changes

    return [
        case("anchor_e2176_b64", ["--edges", "2176", "--cells", "64"]),
        case("batch_b1", ["--edges", "2176", "--cells", "1"]),
        case("batch_b512", ["--edges", "2176", "--cells", "512"]),
        case("scale_e65536_b1", ["--edges", "65536", "--cells", "1"]),
        case("source_skew_0_9", ["--edges", "2176", "--cells", "1",
                                   "--skew", "0.9"]),
        case("gate_probability_0_1", ["--edges", "2176", "--cells", "64",
                                       "--gate-density", "0.1"]),
        case("gate_probability_0_9", ["--edges", "2176", "--cells", "64",
                                       "--gate-density", "0.9"]),
        case("dirty_fraction_0_1", ["--edges", "2176", "--cells", "64",
                                     "--dirty-fraction", "0.1"]),
        case("all_three_terms", ["--edges", "2176", "--cells", "16",
                                 "--query-terms", "3", "--match", "all"]),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path,
                        default=Path("/tmp/cellerator_sparse_support_cpu_bench_build"))
    parser.add_argument("--output", type=Path,
                        default=Path("/tmp/cellerator_sparse_support_cpu_indexing.jsonl"))
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--target-edge-visits", type=int, default=2_000_000)
    parser.add_argument("--skip-build", action="store_true",
                        help="run the existing benchmark binary without configuring/building")
    args = parser.parse_args()
    if args.samples < 3 or args.target_edge_visits < 1:
        parser.error("samples must be at least 3 and target-edge-visits must be positive")

    build_dir = args.build_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not args.skip_build:
        run([
            "cmake", "-S", str(ROOT), "-B", str(build_dir),
            "-DCELLERATOR_ENABLE_CUDA=OFF",
            "-DCELLERATOR_ENABLE_SPARSE_SUPPORT=ON",
            "-DCELLERATOR_BUILD_TESTS=OFF",
            "-DCELLERATOR_ENABLE_HARDWARE_PROBE=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
        ])
        run(["cmake", "--build", str(build_dir), "--target", TARGET,
             "--parallel", str(max(1, os.cpu_count() or 1))])

    executable = find_executable(build_dir)
    source = ROOT / "bench/sparse_support/indexing_bench.cc"
    manifest = {
        "schema": "cellerator.sparse_support.host_run_manifest.v1",
        "scope": "host_only_no_cuda",
        "timestamp_unix_ns": time.time_ns(),
        "git_head": source_revision(),
        "benchmark_source_sha256": sha256(source),
        "source_inputs_sha256": {
            rel: sha256(ROOT / rel) for rel in SOURCE_INPUTS
        },
        "benchmark_binary_sha256": sha256(executable),
        "linked_sparse_support_library_sha256": sha256(
            build_dir / "src/compute/support/libcellerator_sparse_support.a"
        ),
        "binary_path": str(executable),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_model": cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "cmake_version": run(["cmake", "--version"], capture=True).splitlines()[0],
        "build_type": "Release",
        "cuda_disabled": True,
        "samples": args.samples,
        "target_edge_visits_per_sample": args.target_edge_visits,
        "timing_policy": "steady_clock, 3 warmups, raw per-call samples; process affinity and CPU frequency were not controlled",
        "cases": [],
    }
    records: list[dict[str, object]] = []
    for name, case_args in matrix(args.samples, args.target_edge_visits):
        command = [str(executable), *case_args]
        raw = run(command, capture=True)
        record = json.loads(raw)
        if record.get("scope") != "host_only_no_cuda" \
                or record.get("correctness") != "exact_oracle_pass":
            raise RuntimeError(f"benchmark case {name} did not report exact host qualification")
        record["run_metadata"] = {
            "case": name,
            "command": command,
            "benchmark_source_sha256": manifest["benchmark_source_sha256"],
            "benchmark_binary_sha256": manifest["benchmark_binary_sha256"],
            "git_head": manifest["git_head"],
        }
        manifest["cases"].append(name)
        records.append(record)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as destination:
        destination.write(json.dumps(manifest, sort_keys=True) + "\n")
        for record in records:
            destination.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"wrote {len(records)} exact host-only cases to {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
