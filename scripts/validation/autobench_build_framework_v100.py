#!/usr/bin/env python3
"""Build finite SM70 smoke and performance targets from an Autobench snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(argv: list[str]) -> None:
    print("$", " ".join(argv), flush=True)
    subprocess.run(argv, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    source, build = args.source.resolve(strict=True), args.build.resolve()
    build.mkdir(parents=True, exist_ok=True)
    cmake_source = source / "planning/v100-framework-2026-10-29/autobench"
    cmake_build = build / "cmake"
    run(["cmake", "-S", str(cmake_source), "-B", str(cmake_build),
         f"-DCELLERATOR_SOURCE_DIR={source}",
         "-DCMAKE_CXX_COMPILER=/usr/bin/g++-12",
         "-DCMAKE_CUDA_COMPILER=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc",
         "-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12",
         "-DPython_EXECUTABLE=/home/tumlinson/Software/miniconda3/bin/python",
         "-Dpybind11_DIR=/home/tumlinson/Software/miniconda3/lib/python3.13/site-packages/pybind11/share/cmake/pybind11",
         "-DCMAKE_CUDA_ARCHITECTURES=70", "-DCMAKE_BUILD_TYPE=Release",
         "-DCMAKE_CUDA_FLAGS=-lineinfo"])
    targets = ["cellerator_framework_cuda", "cellerator_framework_parameter_transfer",
               "cellerator_framework_cuda_runtime_smoke",
               "cellerator_framework_cuda_dispatch_smoke",
               "cellerator_framework_cuda_quantization_smoke",
               "cellerator_framework_cuda_memory_smoke",
               "cellerator_framework_cuda_frontier_smoke",
               "cellerator_framework_cuda_attention_smoke",
               "cellerator_framework_cuda_grouped_linear_smoke",
               "cellerator_framework_cuda_packed_embedding_smoke",
               "cellerator_framework_cuda_reactions_smoke",
               "cellerator_framework_cuda_regulatory_gates_smoke",
               "cellerator_framework_cuda_parameter_transfer_smoke",
               "cellerator_modeling_cuda_python",
               "cellerator_v100_dispatch_benchmark",
               "cellerator_v100_memory_benchmark",
               "cellerator_v100_quantization_benchmark",
               "cellerator_v100_regulatory_benchmark",
               "cellerator_v100_grouped_linear_benchmark",
               "cellerator_v100_packed_embedding_candidate_smoke"]
    run(["cmake", "--build", str(cmake_build), "--parallel", str(args.jobs), *sum((["--target", t] for t in targets), [])])

    frontier_source = source / "planning/v100-framework-2026-10-29/frontier/candidate"
    frontier_build = build / "frontier-candidate"
    run(["cmake", "-S", str(frontier_source), "-B", str(frontier_build),
         f"-DCELLERATOR_SOURCE={source}",
         "-DCMAKE_CXX_COMPILER=/usr/bin/g++-12",
         "-DCMAKE_CUDA_COMPILER=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc",
         "-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12",
         "-DCMAKE_CUDA_ARCHITECTURES=70", "-DCMAKE_BUILD_TYPE=Release"])
    run(["cmake", "--build", str(frontier_build), "--parallel", str(args.jobs),
         "--target", "cellerator_frontier_candidate_benchmark", "cellerator_frontier_candidate_smoke"])

    files = [p for p in source.rglob("*") if p.is_file() and
             ("/src/modeling/cuda/" in p.as_posix() or
              "/include/Cellerator/modeling/cuda/" in p.as_posix() or
              "/include/Cellerator/execution/conditional/" in p.as_posix() or
              "/src/execution/conditional/" in p.as_posix() or
              "/tests/framework/cuda/" in p.as_posix() or
             "/python/cellerator/" in p.as_posix() or
              "/planning/v100-framework-2026-10-29/autobench/" in p.as_posix() or
              p.as_posix().endswith("/planning/v100-framework-2026-10-29/dispatch/dispatch_benchmark.cu") or
              p.as_posix().endswith("/planning/v100-framework-2026-10-29/memory/benchmark.cu") or
              p.as_posix().endswith("/planning/v100-framework-2026-10-29/quantization/decode_bench.cu") or
              p.as_posix().endswith("/planning/v100-framework-2026-10-29/regulatory/regulatory_gate_comparison.cu") or
              p.as_posix().endswith("/planning/v100-framework-2026-10-29/grouped_linear/benchmark/benchmark.cu") or
              "/planning/v100-framework-2026-10-29/frontier/candidate/" in p.as_posix() and p.suffix in {".cu", ".hh", ".h", ".txt"} or
              p.name in {"CelleratorFrameworkCUDA.cmake", "CelleratorConditional.cmake", "autobench_build_framework_v100.py", "autobench_run_framework_v100.py"})]
    binaries = [cmake_build / f"cellerator_framework_cuda_{name}_smoke"
                for name in ("runtime", "dispatch", "quantization", "memory", "frontier",
                             "attention", "grouped_linear", "packed_embedding", "reactions",
                             "regulatory_gates", "parameter_transfer")]
    binaries += [cmake_build / f"cellerator_v100_{name}_benchmark"
                 for name in ("dispatch", "memory", "quantization", "regulatory", "grouped_linear")]
    binaries.append(cmake_build / "cellerator_v100_packed_embedding_candidate_smoke")
    binaries.append(cmake_build / "_modeling_cuda.cpython-313-x86_64-linux-gnu.so")
    binaries += [frontier_build / f"cellerator_frontier_candidate_{name}"
                 for name in ("benchmark", "smoke")]
    metadata = {
        "schema": "cellerator.v100.framework.autobench.build.v1",
        "architecture": "sm_70",
        "source_files": {p.relative_to(source).as_posix(): sha256(p) for p in sorted(files)},
        "binaries": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                     for p in binaries},
    }
    (build / "build-metadata.json").write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"binary_count": len(binaries), "build_metadata": str(build / "build-metadata.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
