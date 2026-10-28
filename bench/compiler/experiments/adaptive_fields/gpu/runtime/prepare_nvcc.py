#!/usr/bin/env python3
"""CPU-only preparation of the E7 NVCC fatbinary from the shared region."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
DEFAULT_CUDA = Path("/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9")
SOURCE = HERE / "region.cu"
DRIVER = HERE / "nvrtc_compare.cu"
LIBDIR_PARTS = ("targets/x86_64-linux/lib",)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-root", type=Path, default=DEFAULT_CUDA)
    parser.add_argument("--output-dir", type=Path, default=HERE / "artifacts")
    parser.add_argument("--arch", default="sm_70")
    args = parser.parse_args()
    cuda_root = args.cuda_root.resolve(strict=True)
    nvcc = cuda_root / "bin/nvcc"
    if not nvcc.is_file():
        raise RuntimeError(f"NVCC is missing at {nvcc}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fatbin = args.output_dir / "cellerator_region.fatbin"
    command = [str(nvcc), "--fatbin", f"-arch={args.arch}", "--std=c++17",
               str(SOURCE), "-o", str(fatbin)]
    started = time.perf_counter()
    compiled = subprocess.run(command, text=True, capture_output=True, check=False)
    compile_ms = (time.perf_counter() - started) * 1000.0
    if compiled.returncode:
        raise RuntimeError(
            f"NVCC failed ({compiled.returncode})\n{compiled.stdout}\n{compiled.stderr}")
    version = subprocess.run([str(nvcc), "--version"], text=True,
                             capture_output=True, check=True)
    libdir = cuda_root / LIBDIR_PARTS[0]
    executable = args.output_dir / "nvrtc_compare"
    runner_command = [str(nvcc), "--std=c++17", f"-arch={args.arch}",
                      str(DRIVER), "-L", str(libdir), "-lnvrtc", "-lcuda",
                      "-Xlinker", "-rpath", "-Xlinker", str(libdir),
                      "-o", str(executable)]
    runner_start = time.perf_counter()
    runner_build = subprocess.run(runner_command, text=True, capture_output=True,
                                  check=False)
    runner_compile_ms = (time.perf_counter() - runner_start) * 1000.0
    if runner_build.returncode:
        raise RuntimeError(
            f"runner build failed ({runner_build.returncode})\n"
            f"{runner_build.stdout}\n{runner_build.stderr}")
    manifest = {
        "schema": "cellerator.adaptive_fields.e7_nvcc_artifact.v1",
        "gpu_execution_performed": False,
        "source": str(SOURCE),
        "source_sha256": digest(SOURCE),
        "compiler": str(nvcc),
        "compiler_sha256": digest(nvcc),
        "compiler_version": version.stdout.strip(),
        "arguments": command,
        "architecture": args.arch,
        "compile_wall_ms": compile_ms,
        "runner_compile_wall_ms": runner_compile_ms,
        "runner_command": runner_command,
        "runner_executable": str(executable.resolve()),
        "runner_sha256": digest(executable),
        "launch_wrapper": str(HERE / "run_nvrtc_compare.py"),
        "launch_wrapper_sha256": digest(HERE / "run_nvrtc_compare.py"),
        "artifact": str(fatbin.resolve()),
        "artifact_sha256": digest(fatbin),
        "artifact_bytes": fatbin.stat().st_size,
        "cuda_root": str(cuda_root),
        "driver_source": str(DRIVER),
        "driver_source_sha256": digest(DRIVER),
        "platform": platform.platform(),
    }
    path = args.output_dir / "nvcc_artifact_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(path), "fatbin_sha256": manifest["artifact_sha256"],
                      "compile_wall_ms": compile_ms, "gpu_execution_performed": False},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
