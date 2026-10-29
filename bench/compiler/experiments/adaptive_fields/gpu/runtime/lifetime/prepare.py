#!/usr/bin/env python3
"""CPU-only compile and identity capture for the E6 GPU lifetime runner."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time


HERE = Path(__file__).resolve().parent
DEFAULT_CUDA = Path("/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-root", type=Path, default=DEFAULT_CUDA)
    parser.add_argument("--output-dir", type=Path, default=HERE / "artifacts")
    args = parser.parse_args()
    cuda_root = args.cuda_root.resolve(strict=True)
    nvcc = cuda_root / "bin/nvcc"
    libdir = cuda_root / "targets/x86_64-linux/lib"
    source = HERE / "lifetime_runner.cu"
    kernel_source = HERE / "lifetime_kernels.cu"
    if not nvcc.is_file() or not (libdir / "libnvrtc.so").is_file():
        raise RuntimeError("CUDA 12.9 NVCC or NVRTC library is missing")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    executable = args.output_dir / "lifetime_runner"
    command = [str(nvcc), "--std=c++17", "-arch=sm_70", str(source),
               "-L", str(libdir), "-lnvrtc", "-lcuda", "-Xlinker", "-rpath",
               "-Xlinker", str(libdir), "-o", str(executable)]
    started = time.perf_counter()
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    compile_ms = (time.perf_counter() - started) * 1000.0
    if result.returncode:
        raise RuntimeError(f"nvcc runner build failed:\n{result.stdout}\n{result.stderr}")
    version = subprocess.run([str(nvcc), "--version"], text=True,
                             capture_output=True, check=True)
    manifest = {
        "schema": "cellerator.adaptive_fields.e6_gpu_lifetime_build.v1",
        "gpu_execution_performed": False,
        "cuda_root": str(cuda_root),
        "nvcc": str(nvcc),
        "nvcc_sha256": digest(nvcc),
        "nvcc_version": version.stdout.strip(),
        "nvrtc_library": str((libdir / "libnvrtc.so").resolve()),
        "nvrtc_library_sha256": digest(libdir / "libnvrtc.so"),
        "driver_source": str(source),
        "driver_source_sha256": digest(source),
        "kernel_source": str(kernel_source),
        "kernel_source_sha256": digest(kernel_source),
        "launch_wrapper": str(HERE / "run.py"),
        "launch_wrapper_sha256": digest(HERE / "run.py"),
        "sanitizer_plan_sha256": digest(HERE / "sanitizer_commands.json"),
        "runner_command": command,
        "runner_compile_wall_ms": compile_ms,
        "runner": str(executable.resolve()),
        "runner_sha256": digest(executable),
        "runner_bytes": executable.stat().st_size,
        "platform": platform.platform(),
    }
    output = args.output_dir / "build_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(output), "runner_sha256": manifest["runner_sha256"],
                      "compile_wall_ms": compile_ms, "gpu_execution_performed": False},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
