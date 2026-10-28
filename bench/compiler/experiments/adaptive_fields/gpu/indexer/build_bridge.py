#!/usr/bin/env python3
"""CPU-only shared-library build for the adaptive-fields indexer bridge."""
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
CUDA_DEFAULT = Path("/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9")
SOURCES = (
    HERE / "indexer_bridge.cu",
    ROOT / "src/compute/indexing/cuda_postings.cu",
    ROOT / "src/compute/indexing/reverse_postings.cc",
    ROOT / "src/compute/support/logical_support.cc",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str]) -> tuple[float, str, str]:
    start = time.perf_counter()
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    elapsed = (time.perf_counter() - start) * 1000.0
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {command}\n"
                           f"{result.stdout}\n{result.stderr}")
    return elapsed, result.stdout, result.stderr


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-root", type=Path, default=CUDA_DEFAULT)
    parser.add_argument("--cxx", default="/usr/bin/g++")
    parser.add_argument("--output-dir", type=Path, default=HERE / "artifacts")
    parser.add_argument("--arch", default="sm_70")
    args = parser.parse_args()
    cuda_root = args.cuda_root.resolve(strict=True)
    nvcc = cuda_root / "bin/nvcc"
    include = ROOT / "include"
    libdir = cuda_root / "targets/x86_64-linux/lib"
    if not nvcc.is_file() or not Path(args.cxx).is_file():
        raise RuntimeError("NVCC or C++ compiler is missing")
    for source in SOURCES:
        if not source.is_file():
            raise RuntimeError(f"missing build input {source}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    object_paths = []
    commands = []
    compile_times = []
    for index, source in enumerate(SOURCES):
        obj = args.output_dir / f"indexer_{index}.o"
        if source.suffix == ".cu":
            command = [str(nvcc), "-std=c++17", f"-arch={args.arch}",
                       "-Xcompiler", "-fPIC", "-I", str(include),
                       "-c", str(source), "-o", str(obj)]
        else:
            command = [args.cxx, "-std=c++17", "-fPIC", "-I", str(include),
                       "-c", str(source), "-o", str(obj)]
        elapsed, _, _ = run(command)
        object_paths.append(obj)
        commands.append(command)
        compile_times.append(elapsed)
    library = args.output_dir / "libadaptive_fields_indexer.so"
    link_command = [str(nvcc), "--shared", "-arch=" + args.arch,
                    *(str(item) for item in object_paths),
                    "-L", str(libdir), "-Xlinker", "-rpath", "-Xlinker",
                    str(libdir), "-o", str(library)]
    link_ms, _, _ = run(link_command)
    nm = subprocess.run(["nm", "-D", "--defined-only", str(library)],
                        text=True, capture_output=True, check=True)
    required_symbols = ("af_index_create_v1", "af_index_prepare_ordered_postings_v1",
                        "af_index_capacity_v1",
                        "af_index_run_v1", "af_index_destroy_v1",
                        "af_index_last_error_v1")
    missing = [symbol for symbol in required_symbols if symbol not in nm.stdout]
    if missing:
        raise RuntimeError(f"shared library omits C ABI symbols: {missing}")
    version = subprocess.run([str(nvcc), "--version"], text=True,
                             capture_output=True, check=True)
    manifest = {
        "schema": "cellerator.adaptive_fields.indexer_bridge_build.v1",
        "gpu_execution_performed": False,
        "repository_head": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True, capture_output=True, check=True).stdout.strip(),
        "sources": {str(path.relative_to(ROOT)): sha256(path) for path in SOURCES},
        "header_sha256": sha256(HERE / "indexer_bridge.h"),
        "ctypes_binding_sha256": sha256(HERE / "ctypes_binding.py"),
        "nvcc": str(nvcc),
        "nvcc_sha256": sha256(nvcc),
        "nvcc_version": version.stdout.strip(),
        "architecture": args.arch,
        "compile_commands": commands,
        "compile_wall_ms": compile_times,
        "link_command": link_command,
        "link_wall_ms": link_ms,
        "library": str(library.resolve()),
        "library_sha256": sha256(library),
        "library_bytes": library.stat().st_size,
        "platform": platform.platform(),
        "required_symbols": list(required_symbols),
    }
    manifest_path = args.output_dir / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest_path),
                      "library_sha256": manifest["library_sha256"],
                      "library_bytes": manifest["library_bytes"],
                      "gpu_execution_performed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
