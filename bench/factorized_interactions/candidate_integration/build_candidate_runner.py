#!/usr/bin/env python3
"""Build a source-bound native candidate runner without running CUDA kernels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


DEFAULT_CUDA_ROOT = Path(
    "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9")
DEFAULT_NVCC = DEFAULT_CUDA_ROOT / "bin" / "nvcc"
DEFAULT_CXX = Path("/usr/bin/g++-12")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest(root: Path) -> tuple[list[dict], str]:
    """Use the same path/size/content digest as build_runner.py."""
    entries: list[dict] = []
    digest = hashlib.sha256()
    for current, dirs, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for directory in dirs:
            if (current_path / directory).is_symlink():
                raise RuntimeError(
                    f"source snapshot contains a symlink directory: {current_path / directory}")
        for filename in files:
            path = current_path / filename
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"source snapshot contains a nonregular file: {path}")
            relative = path.relative_to(root).as_posix()
            content_hash = sha256_file(path)
            size = path.stat().st_size
            entries.append({"path": relative, "bytes": size, "sha256": content_hash})
    entries.sort(key=lambda entry: entry["path"])
    for entry in entries:
        encoded = entry["path"].encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(entry["bytes"].to_bytes(8, "little"))
        digest.update(bytes.fromhex(entry["sha256"]))
    return entries, digest.hexdigest()


def run(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    print(json.dumps({"argv": command}, sort_keys=True), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def require_outside(path: Path, root: Path, name: str) -> None:
    if path.is_relative_to(root):
        raise RuntimeError(f"{name} must be outside the immutable source snapshot")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True,
                        help="immutable candidate source snapshot")
    parser.add_argument("--build", type=Path, required=True,
                        help="fresh output directory outside --source")
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--cuda-root", type=Path, default=DEFAULT_CUDA_ROOT)
    parser.add_argument("--nvcc", type=Path, default=DEFAULT_NVCC)
    parser.add_argument("--cxx", type=Path, default=DEFAULT_CXX)
    parser.add_argument("--cuda-host-cxx", type=Path, default=Path("/usr/bin/g++-12"))
    parser.add_argument("--configure-only", action="store_true",
                        help="configure the exact target without compiling it")
    args = parser.parse_args()

    source = args.source.resolve(strict=True)
    if not source.is_dir():
        parser.error("--source must name a directory")
    build = args.build.resolve()
    require_outside(build, source, "--build")
    if args.parallel < 1:
        parser.error("--parallel must be positive")
    cuda_root = args.cuda_root.resolve(strict=True)
    nvcc = args.nvcc.resolve(strict=True)
    cxx = args.cxx.resolve(strict=True)
    cuda_host_cxx = args.cuda_host_cxx.resolve(strict=True)
    for path, label in ((nvcc, "nvcc"), (cxx, "C++ compiler"),
                        (cuda_host_cxx, "CUDA host compiler")):
        if not path.is_file():
            parser.error(f"{label} is not a file: {path}")

    integration_dir = source / "bench/factorized_interactions/candidate_integration"
    for relative in ("CMakeLists.txt", "candidate_runner.cu",
                     "PROVENANCE.json"):
        if not (integration_dir / relative).is_file():
            parser.error(f"candidate source snapshot lacks candidate_integration/{relative}")
    if build.exists() and any(build.iterdir()):
        parser.error("--build must be a new or empty directory")
    build.mkdir(parents=True, exist_ok=True)
    native_build = build / "native_build"
    native_build.mkdir(exist_ok=True)
    cmake_build = native_build / "cmake"
    attach_hook = native_build / "attach_candidate_runner.cmake"
    files_before, source_hash = source_manifest(source)

    provenance_path = integration_dir / "PROVENANCE.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("schema") != \
            "cellerator.factorized_interactions.candidate_runner_provenance.v1":
        parser.error("candidate runner provenance has an unknown schema")
    baseline_path = source / "bench/factorized_interactions/native_runner.cu"
    original_baseline_hash = provenance.get("base_runner_sha256")
    if not isinstance(original_baseline_hash, str) or len(original_baseline_hash) != 64:
        parser.error("candidate runner provenance must retain the original base_runner_sha256")
    try:
        bytes.fromhex(original_baseline_hash)
    except ValueError:
        parser.error("candidate runner base_runner_sha256 is not hexadecimal")
    snapshot_baseline_hash = sha256_file(baseline_path)
    declared_snapshot_hash = provenance.get("snapshot_baseline_sha256")
    if declared_snapshot_hash is None:
        if snapshot_baseline_hash != original_baseline_hash:
            parser.error(
                "snapshot native_runner.cu differs from its original fork; "
                "PROVENANCE.json must declare snapshot_baseline_sha256")
        declared_snapshot_hash = original_baseline_hash
    if declared_snapshot_hash != snapshot_baseline_hash:
        parser.error("snapshot_baseline_sha256 does not match the source snapshot")
    if provenance.get("derived_from") != "bench/factorized_interactions/native_runner.cu":
        parser.error("candidate runner provenance names an unexpected derivation source")
    if provenance.get("candidate_runner") != \
            "bench/factorized_interactions/candidate_integration/candidate_runner.cu":
        parser.error("candidate runner provenance names a different runner source")
    candidate_runner_source = (integration_dir / "candidate_runner.cu").read_text(
        encoding="utf-8")
    ancestry_line = (
        f"// Explicit fork of native_runner.cu SHA256 {original_baseline_hash};")
    if not candidate_runner_source.startswith(ancestry_line):
        parser.error("candidate runner source does not identify the original base fork")

    environment = os.environ.copy()
    environment["PATH"] = str(cuda_root / "bin") + os.pathsep + environment.get("PATH", "")
    environment["CUDA_HOME"] = str(cuda_root)
    environment["CUDA_PATH"] = str(cuda_root)
    environment["CUDA_NVCC_EXECUTABLE"] = str(nvcc)

    attach_hook_text = r'''# Generated by build_candidate_runner.py from the source-bound helper.
set(CELLERATOR_ENABLE_CUDA ON CACHE STRING "Build CUDA native component" FORCE)
set(CELLERATOR_ENABLE_FRAMEWORK_CUDA ON CACHE BOOL
    "Expose the separately selectable factorized interaction component" FORCE)
set(CELLERATOR_ENABLE_HARDWARE_PROBE OFF CACHE BOOL
    "Do not probe a device while configuring" FORCE)
set(CELLERATOR_AUTO_DETECT_CUDA_ARCHITECTURES OFF CACHE BOOL
    "Use the explicit SM70 target" FORCE)
set(CELLERATOR_ENABLE_TORCH OFF CACHE BOOL "Keep candidate native-only" FORCE)
set(CELLERATOR_ENABLE_PYTHON OFF CACHE BOOL "Keep candidate native-only" FORCE)
set(CELLERATOR_ENABLE_TOKENIZATION OFF CACHE BOOL "Keep candidate native-only" FORCE)
set(CELLERATOR_ENABLE_NN OFF CACHE BOOL "Keep candidate native-only" FORCE)
set(CELLERATOR_ENABLE_NN_CUDA OFF CACHE BOOL "Keep candidate native-only" FORCE)
set(CELLERATOR_BUILD_TESTS OFF CACHE BOOL "Do not add repository tests" FORCE)
set(CELLERATOR_BUILD_NATIVE_DATA_TESTS OFF CACHE BOOL "Do not add repository tests" FORCE)
set(CELLERATOR_BUILD_FACTORIZED_INTERACTION_VALIDATOR OFF CACHE BOOL
    "Use the isolated candidate runner" FORCE)
set(CELLERATOR_BUILD_FACTORIZED_INTERACTION_EXAMPLE OFF CACHE BOOL
    "Do not build unrelated examples" FORCE)
set(CELLERATOR_BUILD_FACTORIZED_INTERACTION_TESTS OFF CACHE BOOL
    "Do not add repository tests" FORCE)

function(_cellerator_attach_factorized_candidate_runner)
    include(
        "${CELLERATOR_SOURCE_ROOT}/bench/factorized_interactions/candidate_integration/CMakeLists.txt")
endfunction()

if(PROJECT_NAME STREQUAL "Cellerator" AND
   CMAKE_SOURCE_DIR STREQUAL CELLERATOR_SOURCE_ROOT AND
   NOT CELLERATOR_CANDIDATE_RUNNER_ATTACHED)
    set(CELLERATOR_CANDIDATE_RUNNER_ATTACHED TRUE CACHE INTERNAL
        "Candidate runner CMake attachment was scheduled" FORCE)
    cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}"
        CALL _cellerator_attach_factorized_candidate_runner)
endif()
'''
    attach_hook.write_text(attach_hook_text, encoding="utf-8")

    configure = [
        "cmake", "-S", str(source), "-B", str(cmake_build),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_CXX_COMPILER={cxx}",
        f"-DCMAKE_CUDA_COMPILER={nvcc}",
        f"-DCMAKE_CUDA_HOST_COMPILER={cuda_host_cxx}",
        f"-DCUDAToolkit_ROOT={cuda_root}",
        f"-DCUDA_TOOLKIT_ROOT_DIR={cuda_root}",
        f"-DCUDA_NVCC_EXECUTABLE={nvcc}",
        f"-DCMAKE_CUDA_ARCHITECTURES=70",
        f"-DCELLERATOR_SOURCE_ROOT={source}",
        f"-DBASEPLANE_SOURCE_DIR={source / 'external' / 'Baseplane'}",
        f"-DCMAKE_PROJECT_INCLUDE={attach_hook}",
    ]
    run(configure, source, environment)
    if args.configure_only:
        print(json.dumps({
            "configure_only": True,
            "source": str(source),
            "source_closure_sha256": source_hash,
            "build": str(build),
            "cuda_root": str(cuda_root),
            "cuda_compiler": str(nvcc),
        }, sort_keys=True))
        return 0

    build_command = [
        "cmake", "--build", str(cmake_build), "--parallel", str(args.parallel),
        "--target", "cellerator_factorized_interaction_candidate_runner",
        "cellerator_factorized_interaction_baseline_runner",
    ]
    run(build_command, source, environment)

    runner = cmake_build / "cellerator_factorized_interaction_candidate_runner"
    if not runner.is_file():
        raise RuntimeError(f"candidate build did not produce the runner: {runner}")
    baseline_runner = cmake_build / "cellerator_factorized_interaction_baseline_runner"
    if not baseline_runner.is_file():
        raise RuntimeError(f"candidate build did not produce the baseline runner: {baseline_runner}")
    native_libraries = list(cmake_build.rglob("libcellerator_factorized_interaction_cuda.a"))
    if len(native_libraries) != 1:
        raise RuntimeError(
            "expected exactly one in-tree factorized interaction library, found "
            f"{len(native_libraries)}")

    files_after, source_hash_after = source_manifest(source)
    if source_hash_after != source_hash or files_after != files_before:
        raise RuntimeError("candidate source snapshot changed during the build")

    def binary_record(path: Path) -> dict:
        return {
            "path": path.relative_to(native_build).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    metadata = {
        "schema": "cellerator.factorized_interactions.build.v1",
        "source_closure_sha256": source_hash,
        "source_files": files_after,
        "architecture": "sm_70",
        "cuda_compiler": str(nvcc),
        "cuda_root": str(cuda_root),
        "cxx_compiler": str(cxx),
        "cuda_host_compiler": str(cuda_host_cxx),
        "fast_math": False,
        "candidate_runner_provenance": {
            **provenance,
            "snapshot_baseline_sha256": snapshot_baseline_hash,
        },
        "candidate_cmake_hook_sha256": sha256_file(attach_hook),
        "configure_argv": configure,
        "build_argv": build_command,
        "binaries": {
            "runner": binary_record(runner),
            "factorized_interaction_cuda": binary_record(native_libraries[0]),
        },
    }
    manifest_path = native_build / "factorized_interactions_build_manifest.json"
    manifest_path.write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n",
                             encoding="utf-8")
    baseline_metadata = {
        **metadata,
        "binaries": {
            "runner": binary_record(baseline_runner),
            "factorized_interaction_cuda": binary_record(native_libraries[0]),
        },
        "candidate_runner": {
            "manifest_path": manifest_path.relative_to(native_build).as_posix(),
            "manifest_sha256": sha256_file(manifest_path),
            "binary_path": runner.relative_to(native_build).as_posix(),
            "binary_sha256": sha256_file(runner),
        },
    }
    baseline_manifest = native_build / "factorized_interactions_baseline_build_manifest.json"
    baseline_manifest.write_text(json.dumps(baseline_metadata, sort_keys=True, indent=2) + "\n",
                                 encoding="utf-8")
    print(json.dumps({
        "build_manifest": str(manifest_path),
        "build_manifest_sha256": sha256_file(manifest_path),
        "source_closure_sha256": source_hash,
        "runner": str(runner),
        "runner_sha256": metadata["binaries"]["runner"]["sha256"],
        "baseline_runner": str(baseline_runner),
        "baseline_runner_sha256": sha256_file(baseline_runner),
        "baseline_build_manifest": str(baseline_manifest),
        "baseline_build_manifest_sha256": sha256_file(baseline_manifest),
        "native_library_sha256": metadata["binaries"]["factorized_interaction_cuda"]["sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError,
            RuntimeError) as exc:
        print(f"candidate build failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

