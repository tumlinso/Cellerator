#!/usr/bin/env python3
"""Configure and build the bounded native arithmetic/moments probe targets.

This helper is safe for workflow gates: it never runs either executable or
cleans the caller's build directory. Use a build directory outside the source
checkout so generated files do not enter the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Sequence

TARGETS = ("ceNativeArithmeticTest", "ceNativeNeighborhoodMoments")
TARGET_BINARIES = {
    "ceNativeArithmeticTest": Path("src/compute/operation/native_numeric/ceNativeArithmeticTest"),
    "ceNativeNeighborhoodMoments": Path("ceNativeNeighborhoodMoments"),
}
SOURCE_PATHS = (
    "CMakeLists.txt",
    "cmake/NativeFoundation.cmake",
    "cmake/SemanticSpineV1.cmake",
    "include/Cellerator/compute/operation/native_numeric/device_linear.hh",
    "src/compute/operation/native_numeric/CMakeLists.txt",
    "src/compute/operation/native_numeric/device_linear.cu",
    "src/compute/operation/native_numeric/host_relation.cc",
    "include/Cellerator/compute/operation/prepared_relation.hh",
    "include/Cellerator/compute/operation/relation_semantics.hh",
    "src/compute/operation/prepared_relation.cu",
    "src/compute/candidate/sparse/project.cu",
    "examples/semantic_spine_v1/CMakeLists.txt",
    "tests/native_numeric/device_arithmetic.cc",
    "tests/native_numeric/build_probe.py",
)
EXAMPLE_ROOT = "examples/native_neighborhood_moments"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def version_output(command: Sequence[str]) -> str:
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"
    return (result.stdout + result.stderr).strip()


def run_step(command: Sequence[str], log_path: Path) -> dict[str, object]:
    result = subprocess.run(command, text=True, capture_output=True)
    log_path.write_text(
        f"$ {' '.join(command)}\n\n"
        f"[stdout]\n{result.stdout}\n[stderr]\n{result.stderr}",
        encoding="utf-8",
    )
    return {
        "argv": list(command),
        "returncode": result.returncode,
        "log": str(log_path),
    }


def file_record(path: Path, root: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(root)), "sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", required=True, type=Path, help="isolated CMake build directory")
    parser.add_argument("--baseplane-source", required=True, type=Path)
    parser.add_argument("--cuda-root", required=True, type=Path)
    parser.add_argument("--cxx", default="/usr/bin/g++-12")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--arch", default="70")
    args = parser.parse_args()

    source_root = Path(__file__).resolve().parents[2]
    build_dir = args.build.expanduser().resolve()
    baseplane_root = args.baseplane_source.expanduser().resolve()
    cuda_root = args.cuda_root.expanduser().resolve()
    cxx = shutil.which(args.cxx) if not Path(args.cxx).is_absolute() else args.cxx
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    if not cxx or not Path(cxx).is_file():
        parser.error(f"C++ compiler not found: {args.cxx}")
    nvcc = cuda_root / "bin" / "nvcc"
    if not nvcc.is_file():
        parser.error(f"CUDA compiler not found under --cuda-root: {nvcc}")
    if not (baseplane_root / "CMakeLists.txt").is_file():
        parser.error(f"Baseplane source tree is missing CMakeLists.txt: {baseplane_root}")
    if build_dir == source_root or source_root in build_dir.parents:
        parser.error("--build must be outside the Cellerator source checkout")
    if build_dir == baseplane_root or baseplane_root in build_dir.parents:
        parser.error("--build must be outside the Baseplane source checkout")

    sources = [source_root / item for item in SOURCE_PATHS]
    example_dir = source_root / EXAMPLE_ROOT
    if example_dir.is_dir():
        sources.extend(sorted(path for path in example_dir.rglob("*") if path.is_file()))
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        parser.error("missing probe source inputs:\n" + "\n".join(missing))

    cmake = shutil.which("cmake")
    if cmake is None:
        parser.error("cmake was not found on PATH")
    build_dir.mkdir(parents=True, exist_ok=True)

    configure = [
        cmake,
        "-S",
        str(source_root),
        "-B",
        str(build_dir),
        f"-DCMAKE_CXX_COMPILER={cxx}",
        f"-DCMAKE_CUDA_COMPILER={nvcc}",
        f"-DCMAKE_CUDA_HOST_COMPILER={cxx}",
        f"-DCMAKE_CUDA_ARCHITECTURES={args.arch}",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        f"-DCUDAToolkit_ROOT={cuda_root}",
        f"-DBASEPLANE_SOURCE_DIR={baseplane_root}",
        "-DCELLERATOR_ENABLE_CUDA=ON",
        "-DCELLERATOR_BUILD_SEMANTIC_SPINE_V1=ON",
        "-DCELLERATOR_BUILD_TESTS=OFF",
        "-DCELLERATOR_ENABLE_TORCH_MODELS=OFF",
        "-DCELLERATOR_ENABLE_CELLSHARD=OFF",
        "-DCELLERATOR_ENABLE_HARDWARE_PROBE=OFF",
        "-DCELLERATOR_AUTO_DETECT_CUDA_ARCHITECTURES=OFF",
        "-DCELLERATOR_BUILD_NATIVE_FOUNDATION_TESTS=OFF",
        "-DCELLERATOR_ENABLE_CUDA_LINEINFO=OFF",
    ]
    build = [cmake, "--build", str(build_dir), "--target", *TARGETS, "--parallel", str(args.jobs)]

    configure_result = run_step(configure, build_dir / "ce_native_probe_configure.log")
    build_result = None
    if configure_result["returncode"] == 0:
        build_result = run_step(build, build_dir / "ce_native_probe_build.log")

    manifest_path = build_dir / "ce_native_probe_build_manifest.json"
    cache = build_dir / "CMakeCache.txt"
    compile_commands = build_dir / "compile_commands.json"
    build_ok = build_result is not None and build_result["returncode"] == 0
    binary_paths = {target: build_dir / relative for target, relative in TARGET_BINARIES.items()}
    missing_binaries = [str(path) for path in binary_paths.values() if not path.is_file()]
    binary_records = [
        {
            "target": target,
            "path": str(path),
            "sha256": sha256(path) if build_ok and path.is_file() else None,
        }
        for target, path in binary_paths.items()
    ]

    configure_ok = configure_result["returncode"] == 0
    complete_build = configure_ok and build_ok and not missing_binaries
    source_file_hashes = {str(path.relative_to(source_root)): sha256(path) for path in sources}
    native_binary = binary_paths["ceNativeNeighborhoodMoments"]
    arithmetic_binary = binary_paths["ceNativeArithmeticTest"]
    configure_log = Path(configure_result["log"])
    build_log = Path(build_result["log"]) if build_result is not None else None
    cmake_version = version_output([cmake, "--version"])
    cxx_version = version_output([cxx, "--version"])
    cuda_version = version_output([str(nvcc), "--version"])
    baseplane_head = version_output(["git", "-C", str(baseplane_root), "rev-parse", "HEAD"])
    metadata = {
        "schema_version": 1,
        "status": "built" if complete_build else "failed",
        "source_root": str(source_root),
        "source_head": version_output(["git", "-C", str(source_root), "rev-parse", "HEAD"]),
        "baseplane_head": baseplane_head,
        "build_dir": str(build_dir),
        "targets": list(TARGETS),
        "architecture": f"sm_{args.arch}",
        "cuda_root": str(cuda_root),
        "configuration": {
            "cuda": "ON",
            "semantic_spine_v1": "ON",
            "tests": "OFF",
            "torch_models": "OFF",
            "cellshard": "OFF",
            "hardware_probe": "OFF",
            "auto_detect_cuda_architectures": "OFF",
            "native_foundation_tests": "OFF",
            "cuda_lineinfo": "OFF",
            "cuda_architectures": args.arch,
            "baseplane_source": str(baseplane_root),
        },
        "toolchain": {
            "cmake": cmake_version,
            "cxx": str(Path(cxx).resolve()),
            "cxx_version": cxx_version,
            "nvcc": str(nvcc.resolve()),
            "cuda_version": cuda_version,
        },
        "cmake": cmake_version,
        "cxx": cxx_version,
        "nvcc": cuda_version,
        "source_file_sha256": source_file_hashes,
        "source_sha256": [file_record(path, source_root) for path in sources],
        "cmake_cache_sha256": sha256(cache) if cache.is_file() else None,
        "compile_commands_sha256": sha256(compile_commands) if compile_commands.is_file() else None,
        "native_binary_sha256": sha256(native_binary) if complete_build else None,
        "arithmetic_binary_sha256": sha256(arithmetic_binary) if complete_build else None,
        "configure_log": str(configure_log),
        "build_log": str(build_log) if build_log else None,
        "missing_binaries": missing_binaries,
        "cmake_cache": file_record(cache, build_dir) if cache.is_file() else None,
        "binaries": binary_records,
        "steps": {"configure": configure_result, "build": build_result},
        "executables_run": False,
    }
    manifest_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Build status: {metadata['status']}")
    print(f"Manifest: {manifest_path}")
    if not configure_ok:
        return int(configure_result["returncode"])
    assert build_result is not None
    if build_result["returncode"] != 0:
        return int(build_result["returncode"])
    return 0 if complete_build else 2


if __name__ == "__main__":
    raise SystemExit(main())
