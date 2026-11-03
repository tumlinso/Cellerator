#!/usr/bin/env python3
"""Configure and build only the conventional ATen CUDA benchmark target."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


CUDA_ROOT = Path("/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9")
TORCH_CMAKE = Path(
    "/home/tumlinson/Software/miniconda3/lib/python3.13/site-packages/torch/share/cmake/Torch"
)


def run(command: list[str], cwd: Path) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    build = args.build.resolve()
    if not any(
        tuple(source.parts[index:index + 3]) == (".autobench", "state", "snapshots")
        for index in range(len(source.parts) - 2)
    ):
        raise SystemExit("native-data builds must consume an immutable Autobench snapshot")
    if not (CUDA_ROOT / "bin/nvcc").is_file() or not TORCH_CMAKE.joinpath("TorchConfig.cmake").is_file():
        raise SystemExit("pinned CUDA 12.9 / installed Torch CMake package is unavailable")
    build.mkdir(parents=True, exist_ok=True)
    cmake_build = build / "cmake"
    cmake_build.mkdir(parents=True, exist_ok=True)
    run([
        "cmake", "-S", str(source), "-B", str(cmake_build),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CXX_COMPILER=/usr/bin/g++-12",
        f"-DCMAKE_CUDA_COMPILER={CUDA_ROOT / 'bin/nvcc'}",
        "-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12",
        "-DCMAKE_CUDA_ARCHITECTURES=70",
        f"-DCMAKE_PREFIX_PATH={TORCH_CMAKE.parent}",
        f"-DTorch_DIR={TORCH_CMAKE}",
        "-DCELLERATOR_ENABLE_CUDA=ON",
        "-DCELLERATOR_ENABLE_HARDWARE_PROBE=OFF",
        "-DCELLERATOR_AUTO_DETECT_CUDA_ARCHITECTURES=OFF",
        "-DCELLERATOR_BUILD_TESTS=OFF",
        "-DCELLERATOR_BUILD_NATIVE_DATA_TESTS=OFF",
        "-DCELLERATOR_BUILD_NATIVE_DATA_BENCHMARKS=ON",
        "-DCELLERATOR_ENABLE_PYTHON=OFF",
        "-DCELLERATOR_ENABLE_TORCH=OFF",
    ], source)
    run([
        "cmake", "--build", str(cmake_build),
        "--target", "cellerator_native_data_aten_bench", "--parallel", "2",
    ], source)
    binary = cmake_build / "bench/native_data/cellerator_native_data_aten_bench"
    if not binary.is_file():
        raise SystemExit(f"baseline target build succeeded but binary is missing: {binary}")
    print(f"baseline_binary={binary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
