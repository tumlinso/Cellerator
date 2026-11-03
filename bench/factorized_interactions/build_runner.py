#!/usr/bin/env python3
"""Build the native component from one immutable Autobench source snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


CUDA_ROOT = Path("/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9")
NVCC = CUDA_ROOT / "bin" / "nvcc"
CUDA_INCLUDE = CUDA_ROOT / "targets" / "x86_64-linux" / "include"
PYTHON = Path("/home/tumlinson/Software/miniconda3/bin/python3").resolve()
TARGETS = (
    "cellerator_factorized_interaction_runner",
    "cellerator_factorized_interaction_training",
    "cellerator_factorized_interaction_cuda_contract_test",
    "cellerator_factorized_interaction_preparation_test",
    "cellerator_factorized_interaction_pattern_test",
    "cellerator_nn_python",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest(root: Path) -> tuple[list[dict], str]:
    entries = []
    digest = hashlib.sha256()
    files = sorted((path for path in root.rglob("*") if path.is_file()),
                   key=lambda path: path.relative_to(root).as_posix())
    for path in files:
        if path.is_symlink():
            raise RuntimeError(f"source snapshot unexpectedly contains a symlink: {path}")
        relative = path.relative_to(root).as_posix()
        content_hash = sha256_file(path)
        size = path.stat().st_size
        entry = {"path": relative, "bytes": size, "sha256": content_hash}
        entries.append(entry)
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(size.to_bytes(8, "little"))
        digest.update(bytes.fromhex(content_hash))
    return entries, digest.hexdigest()


def run(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    print(json.dumps({"argv": command}, sort_keys=True), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--configure-only", action="store_true",
                        help="configure the exact build but do not compile targets")
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    build = args.build.resolve()
    if not PYTHON.is_file():
        raise FileNotFoundError(f"the configured Python development environment is missing: {PYTHON}")
    build.mkdir(parents=True, exist_ok=True)
    cmake_build = build / "cmake"
    pybind11_probe = subprocess.run(
        [str(PYTHON), "-m", "pybind11", "--cmakedir"],
        check=True, capture_output=True, text=True)
    pybind11_cmake_dir = Path(pybind11_probe.stdout.strip()).resolve(strict=True)
    if not (pybind11_cmake_dir / "pybind11Config.cmake").is_file():
        raise FileNotFoundError(
            f"the selected Python's pybind11 CMake package is missing: {pybind11_cmake_dir}")
    environment = os.environ.copy()
    environment["PATH"] = str(CUDA_ROOT / "bin") + os.pathsep + environment.get("PATH", "")
    environment["CUDA_HOME"] = str(CUDA_ROOT)
    environment["CUDA_PATH"] = str(CUDA_ROOT)
    environment["CUDA_NVCC_EXECUTABLE"] = str(NVCC)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    configure = [
        "cmake", "-S", str(source), "-B", str(cmake_build),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CXX_COMPILER=/usr/bin/c++",
        f"-DCMAKE_CUDA_COMPILER={NVCC}",
        "-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12",
        f"-DCUDAToolkit_ROOT={CUDA_ROOT}",
        f"-DCUDA_TOOLKIT_ROOT_DIR={CUDA_ROOT}",
        f"-DCUDA_NVCC_EXECUTABLE={NVCC}",
        f"-DCUDA_CUDART_LIBRARY={CUDA_ROOT / 'lib64' / 'libcudart.so'}",
        f"-DBASEPLANE_SOURCE_DIR={source / 'external' / 'Baseplane'}",
        f"-DPython_EXECUTABLE={PYTHON}",
        f"-Dpybind11_DIR={pybind11_cmake_dir}",
        "-DCMAKE_CUDA_ARCHITECTURES=70",
        "-DCELLERATOR_ENABLE_CUDA=ON",
        "-DCELLERATOR_ENABLE_HARDWARE_PROBE=OFF",
        "-DCELLERATOR_AUTO_DETECT_CUDA_ARCHITECTURES=OFF",
        "-DCELLERATOR_ENABLE_FRAMEWORK_CUDA=OFF",
        "-DCELLERATOR_ENABLE_MODELING=ON",
        "-DCELLERATOR_ENABLE_SPARSE_SUPPORT=ON",
        "-DCELLERATOR_ENABLE_FRAMEWORK_CUDA=ON",
        "-DCELLERATOR_ENABLE_TORCH=OFF",
        "-DCELLERATOR_ENABLE_PYTHON=ON",
        "-DCELLERATOR_BUILD_PRODUCT2=ON",
        "-DCELLERATOR_ENABLE_NN=ON",
        "-DCELLERATOR_ENABLE_NN_CUDA=ON",
        "-DCELLERATOR_ENABLE_TOKENIZATION=OFF",
        "-DCELLERATOR_BUILD_TESTS=OFF",
        "-DCELLERATOR_BUILD_FACTORIZED_INTERACTION_VALIDATOR=ON",
        "-DCELLERATOR_BUILD_FACTORIZED_INTERACTION_EXAMPLE=ON",
        "-DCELLERATOR_BUILD_FACTORIZED_INTERACTION_TESTS=ON",
    ]
    run(configure, source, environment)
    if args.configure_only:
        print(json.dumps({
            "configure_only": True,
            "source": str(source),
            "build": str(build),
            "cuda_root": str(CUDA_ROOT),
            "cuda_compiler": str(NVCC),
            "python_executable": str(PYTHON),
            "pybind11_cmake_dir": str(pybind11_cmake_dir),
        }, sort_keys=True))
        return 0
    build_command = ["cmake", "--build", str(cmake_build), "--parallel",
                     str(args.parallel), "--target", *TARGETS]
    run(build_command, source, environment)

    runner = cmake_build / "factorized_interactions_validator" / \
        "cellerator_factorized_interaction_runner"
    binaries = {
        "runner": runner,
        "training_example": cmake_build / "cellerator_factorized_interaction_training",
        "host_contract_test": cmake_build / "cellerator_factorized_interaction_cuda_contract_test",
        "preparation_test": cmake_build / "cellerator_factorized_interaction_preparation_test",
        "pattern_binding_test": cmake_build / "cellerator_factorized_interaction_pattern_test",
    }
    missing = [str(path) for path in binaries.values() if not path.is_file()]
    if missing:
        raise RuntimeError("build completed without expected artifacts: " + ", ".join(missing))

    extensions = sorted(cmake_build.glob("_nn_native*"))
    if len(extensions) != 1 or not extensions[0].is_file():
        raise RuntimeError(
            "Torch-enabled build did not produce exactly one _nn_native extension")
    python_package = build / "python"
    python_package.mkdir(parents=True, exist_ok=False)
    package_source = source / "python" / "cellerator"
    package_destination = python_package / "cellerator"
    shutil.copytree(
        package_source, package_destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        symlinks=False)
    # The immutable source snapshot marks its package directories read-only;
    # copytree preserves that mode. Make this generated build-output directory
    # writable before installing the freshly built extension into it.
    package_destination.chmod(package_destination.stat().st_mode | 0o200)
    shutil.copy2(extensions[0], package_destination / extensions[0].name)
    binaries["python_extension"] = extensions[0]

    candidate_builder = source / "bench" / "factorized_interactions" / \
        "candidate_integration" / "build_candidate_runner.py"
    if not candidate_builder.is_file():
        raise FileNotFoundError(
            "the captured source closure omits the candidate runner builder")
    candidate_build = build / "candidate_native"
    run([str(PYTHON), str(candidate_builder), "--source", str(source),
         "--build", str(candidate_build), "--parallel", str(args.parallel)],
        source, environment)
    candidate_manifest = candidate_build / "native_build" / \
        "factorized_interactions_build_manifest.json"
    candidate_metadata = json.loads(candidate_manifest.read_text(encoding="utf-8"))
    if candidate_metadata.get("schema") != \
            "cellerator.factorized_interactions.build.v1":
        raise RuntimeError("candidate build emitted an unknown manifest schema")
    files, source_hash = source_manifest(source)
    if candidate_metadata.get("source_closure_sha256") != \
            source_hash:
        raise RuntimeError("candidate binary was built from a different source closure")
    candidate_entry = candidate_metadata.get("binaries", {}).get("runner")
    if not isinstance(candidate_entry, dict):
        raise RuntimeError("candidate build manifest omits its runner binary")
    candidate_runner = (candidate_manifest.parent /
                        candidate_entry["path"]).resolve(strict=True)
    if sha256_file(candidate_runner) != candidate_entry.get("sha256"):
        raise RuntimeError("candidate runner differs from its build manifest")

    python_probe = subprocess.run(
        [str(PYTHON), "-c",
         "import json,sys,torch; print(json.dumps({'executable':sys.executable,"
         "'version':sys.version,'torch_version':torch.__version__}))"],
        check=True, capture_output=True, text=True)
    python_metadata = json.loads(python_probe.stdout)
    staged_manifest = source.parent / f"{source.name}-manifest.json"
    if staged_manifest.is_file():
        staged = json.loads(staged_manifest.read_text(encoding="utf-8"))
        if staged.get("schema") != "cellerator.factorized_interactions.source_closure.v1":
            raise RuntimeError("staged source manifest has an unknown schema")
        if staged.get("source_closure_sha256") != source_hash or \
                staged.get("files") != files:
            raise RuntimeError("staged source closure differs from its immutable manifest")
    metadata = {
        "schema": "cellerator.factorized_interactions.build.v1",
        "source_closure_sha256": source_hash,
        "source_files": files,
        "architecture": "sm_70",
        "cuda_compiler": str(NVCC),
        "cuda_include": str(CUDA_INCLUDE),
        "python": {
            **python_metadata,
            "executable_sha256": sha256_file(PYTHON),
            "pybind11_cmake_dir": str(pybind11_cmake_dir),
        },
        "python_package_root": str(python_package.relative_to(build)),
        "candidate_runner": {
            "manifest_path": str(candidate_manifest.relative_to(build)),
            "manifest_sha256": sha256_file(candidate_manifest),
            "binary_path": str(candidate_runner.relative_to(build)),
            "binary_sha256": candidate_entry["sha256"],
        },
        "configure_argv": configure,
        "build_argv": build_command,
        "binaries": {
            name: {"path": str(path.relative_to(build)),
                   "bytes": path.stat().st_size,
                   "sha256": sha256_file(path)}
            for name, path in binaries.items()
        },
    }
    manifest_path = build / "factorized_interactions_build_manifest.json"
    manifest_path.write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n",
                             encoding="utf-8")
    print(json.dumps({
        "build_manifest": str(manifest_path),
        "build_manifest_sha256": sha256_file(manifest_path),
        "source_closure_sha256": source_hash,
        "runner_sha256": metadata["binaries"]["runner"]["sha256"],
        "python_extension_sha256": metadata["binaries"]["python_extension"]["sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
