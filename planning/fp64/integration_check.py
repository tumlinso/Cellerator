#!/usr/bin/env python3
"""Rebuild combined Cellerator consumers and qualify them with FP64 gates.

CPU configuration/build/install steps run before any controller request. Runtime
checks each acquire their own CUDA controller lease; this script never nests
leases and never cleans an output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CUDA_ROOT = Path("/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9")
GPU_UUID = "GPU-6c1cac7f-a360-0aef-ba98-2828bfd1db1a"
PYTHON = Path("/home/tumlinson/Software/venvs/cellerator-ml2-py313-cu126/bin/python")
FP64_PYTHON = Path("/home/tumlinson/Software/miniconda3/bin/python3")
TORCH_DIR = PYTHON.parent.parent / "lib/python3.13/site-packages/torch/share/cmake/Torch"
BASEPLANE = Path("/home/tumlinson/Baseplane")
CONTROLLER = Path.home() / ".agents/skills/cuda/scripts/cuda_controller.py"

COMBINED_TARGETS = (
    "cellerator_torch_adapter_compile_tests",
    "cellerator_python_extensions",
    "cellerator_indexed_mechanism_binding_test",
)
COMBINED_BINARIES = (
    "cellerator_indexed_mechanism_binding_test",
    "bindings/torch/cellerator_torch_native_views_test",
    "bindings/torch/cellerator_torch_program_ops_test",
    "bindings/torch/cellerator_torch_autograd_ops_test",
)
SOURCE_ROOTS = (
    "bindings", "python/cellerator", "include/Cellerator",
    "src", "cmake", "modules",
    "tests/bindings", "tests/native_numeric", "tests/fp64",
    "examples/native_neighborhood_moments", "CMakeLists.txt",
    "planning/fp64/integration_check.py",
    "planning/fp64/integration-runtime.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_output(*args: str) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(ROOT), *args], check=True,
                                text=True, capture_output=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_hashes() -> dict[str, str]:
    paths: set[Path] = set()
    for item in SOURCE_ROOTS:
        path = ROOT / item
        if path.is_file():
            paths.add(path)
        elif path.is_dir():
            paths.update(p for p in path.rglob("*") if p.is_file() and
                         not any(part in {"__pycache__", ".git"} for part in p.parts))
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(paths)}


def run_step(records: list[dict[str, Any]], output: Path, label: str,
             argv: list[str], cwd: Path = ROOT) -> dict[str, Any]:
    started = time.monotonic()
    result = subprocess.run(argv, cwd=cwd, text=True, capture_output=True, check=False)
    stdout_path = output / f"{label}.stdout.txt"
    stderr_path = output / f"{label}.stderr.txt"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    record = {
        "label": label, "argv": argv, "cwd": str(cwd),
        "returncode": result.returncode, "elapsed_seconds": time.monotonic() - started,
        "stdout_path": str(stdout_path), "stderr_path": str(stderr_path),
        "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
    }
    records.append(record)
    if result.returncode:
        raise RuntimeError(f"{label} failed with exit status {result.returncode}; see {stderr_path}")
    return record


def cmake_combined(build: Path, install: Path, logs: Path,
                   records: list[dict[str, Any]]) -> list[Path]:
    cmake = shutil.which("cmake")
    if not cmake:
        raise RuntimeError("cmake was not found on PATH")
    nvcc = CUDA_ROOT / "bin/nvcc"
    if not nvcc.is_file():
        raise RuntimeError(f"CUDA 12.9 nvcc is unavailable: {nvcc}")
    if not PYTHON.is_file() or not TORCH_DIR.is_dir():
        raise RuntimeError(f"binding Python/Torch toolchain unavailable: {PYTHON} / {TORCH_DIR}")
    configure = [
        cmake, "-S", str(ROOT), "-B", str(build),
        "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_INSTALL_PREFIX={install}",
        "-DCMAKE_CXX_COMPILER=/usr/bin/g++-12", f"-DCMAKE_CUDA_COMPILER={nvcc}",
        f"-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12", "-DCMAKE_CUDA_ARCHITECTURES=70",
        f"-DCUDAToolkit_ROOT={CUDA_ROOT}", f"-DPython_EXECUTABLE={PYTHON}",
        f"-DTorch_DIR={TORCH_DIR}", "-DCELLERATOR_ENABLE_CUDA=ON",
        "-DCELLERATOR_ENABLE_PYTHON=ON",
        "-DCELLERATOR_ENABLE_TORCH=ON", "-DCELLERATOR_BUILD_TESTS=ON",
    ]
    run_step(records, logs, "combined_configure", configure)
    run_step(records, logs, "combined_build", [cmake, "--build", str(build),
             "--parallel", "4", "--target", *COMBINED_TARGETS])
    for component in ("Python", "Torch"):
        run_step(records, logs, f"install_{component.lower()}",
                 [cmake, "--install", str(build), "--component", component])
    binaries = [build / relative for relative in COMBINED_BINARIES]
    missing = [str(path) for path in binaries if not path.is_file()]
    if missing:
        raise RuntimeError("combined build omitted runtime fixture(s): " + ", ".join(missing))
    return binaries


def build_native(build: Path, logs: Path, records: list[dict[str, Any]]) -> list[Path]:
    run_step(records, logs, "native_probe_build", [
        str(PYTHON), "-B", str(ROOT / "tests/native_numeric/build_probe.py"),
        "--build", str(build), "--baseplane-source", str(BASEPLANE),
        "--cuda-root", str(CUDA_ROOT), "--cxx", "/usr/bin/g++-12", "--jobs", "4", "--arch", "70",
    ])
    manifest_path = build / "ce_native_probe_build_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"native probe manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "built":
        raise RuntimeError(f"native probe build did not pass: {manifest.get('status')!r}")
    binary_paths = [build / "src/compute/operation/native_numeric/ceNativeArithmeticTest",
                    build / "ceNativeNeighborhoodMoments"]
    missing = [str(path) for path in binary_paths if not path.is_file()]
    if missing:
        raise RuntimeError("native probe build omitted binary(s): " + ", ".join(missing))
    return binary_paths


def controller_spec(build: Path, install: Path, native_build: Path, output: Path) -> dict[str, Any]:
    runtime = ROOT / "planning/fp64/integration-runtime.py"
    return {
        "schema_version": 1,
        "project_root": str(ROOT), "command_cwd": str(ROOT),
        "campaign_id": "cellerator-fp64-combined-integration",
        "recipe": "baseline", "paths": list(SOURCE_ROOTS),
        "argv": [str(PYTHON), "-B", str(runtime), "--build", str(build),
                 "--package-root", str(install), "--native-build", str(native_build),
                 "--output", str(output)],
        "timeout": 3600,
        "resources": {"gpus": 1, "gpu_uuids": [GPU_UUID], "cpu_threads": 4},
        "toolchain": {"root": str(CUDA_ROOT)},
    }


def run_combined_controller(spec: dict[str, Any], output: Path,
                            records: list[dict[str, Any]]) -> dict[str, Any]:
    if not CONTROLLER.is_file():
        raise RuntimeError(f"CUDA controller is unavailable: {CONTROLLER}")
    controller_dir = output / "controller"
    controller_dir.mkdir(parents=True, exist_ok=True)
    spec_path = controller_dir / "combined.spec.json"
    write_json(spec_path, spec)
    argv = [str(PYTHON), str(CONTROLLER), "run", "--spec", str(spec_path), "--json"]
    started = time.monotonic()
    result = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True, check=False)
    stdout_path, stderr_path = controller_dir / "combined.stdout.txt", controller_dir / "combined.stderr.txt"
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    try:
        receipt = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        receipt = {"parse_error": str(error), "stdout_tail": result.stdout[-4000:]}
    record = {
        "label": "combined_runtime_controller", "argv": argv, "cwd": str(ROOT),
        "returncode": result.returncode, "elapsed_seconds": time.monotonic() - started,
        "spec_path": str(spec_path), "stdout_path": str(stdout_path), "stderr_path": str(stderr_path),
        "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
        "controller_receipt": receipt,
    }
    records.append(record)
    if result.returncode or not isinstance(receipt, dict) or not receipt.get("ok"):
        raise RuntimeError(f"combined runtime controller failed; inspect {stdout_path} and {stderr_path}")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("build", "all"), default="all",
                        help="build performs CPU-only preparation; all also runs controller qualification")
    parser.add_argument("--combined-build", type=Path, default=Path("/tmp/cellerator-fp64-integration-combined"))
    parser.add_argument("--install-prefix", type=Path, default=Path("/tmp/cellerator-fp64-integration-install"))
    parser.add_argument("--native-build", type=Path, default=Path("/tmp/cellerator-fp64-integration-native"))
    parser.add_argument("--fp64-build", type=Path, default=Path("/tmp/cellerator-fp64-integration-fp64"))
    parser.add_argument("--output", type=Path, default=Path("planning/fp64/integration/evidence/combined"))
    args = parser.parse_args()
    resolve = lambda path: path.expanduser().resolve() if path.is_absolute() else (ROOT / path).resolve()
    combined_build, install = resolve(args.combined_build), resolve(args.install_prefix)
    native_build, fp64_build, output = resolve(args.native_build), resolve(args.fp64_build), resolve(args.output)
    if not PYTHON.is_file():
        parser.error(f"configured Python executable does not exist: {PYTHON}")
    if not FP64_PYTHON.is_file():
        parser.error(f"NumPy/SciPy FP64 Python executable does not exist: {FP64_PYTHON}")
    if any(path == ROOT or ROOT in path.parents for path in (combined_build, install, native_build, fp64_build)):
        parser.error("all build and install paths must remain outside the source checkout")
    logs = output / "build"
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "integration.json"
    record: dict[str, Any] = {
        "schema": "cellerator.fp64.combined-integration.v1", "status": "running",
        "source_root": str(ROOT), "source_head": git_output("rev-parse", "HEAD"),
        "source_sha256": source_hashes(), "gpu_uuid": GPU_UUID,
        "cuda_root": str(CUDA_ROOT), "architecture": "sm_70", "cpu_threads": 4,
        "build_paths": {"combined": str(combined_build), "install": str(install),
                        "native": str(native_build), "fp64": str(fp64_build)},
        "steps": [], "binary_sha256": {}, "error": None,
    }

    def save() -> None:
        record["source_sha256_final"] = source_hashes()
        write_json(receipt_path, record)

    try:
        combined_binaries = cmake_combined(combined_build, install, logs, record["steps"])
        native_binaries = build_native(native_build, logs, record["steps"])
        probe_manifest = native_build / "ce_native_probe_build_manifest.json"
        record["native_probe_build_manifest"] = json.loads(
            probe_manifest.read_text(encoding="utf-8"))
        record["binary_sha256"] = {str(path): sha256(path) for path in combined_binaries + native_binaries}
        save()
        if args.phase == "all":
            run_step(record["steps"], logs, "fp64_python_dependencies", [
                str(FP64_PYTHON), "-B", "-c",
                "import numpy, scipy; print('numpy=' + numpy.__version__); print('scipy=' + scipy.__version__)",
            ])
            spec = controller_spec(combined_build, install, native_build, output / "runtime")
            record["combined_controller_spec"] = spec
            run_combined_controller(spec, output, record["steps"])
            fp64_output = output.parent / "fp64"
            fp64_output.mkdir(parents=True, exist_ok=True)
            run_step(record["steps"], logs, "fp64_run_checks_all", [
                str(FP64_PYTHON), "-B", str(ROOT / "tests/fp64/run_checks.py"),
                "--phase", "all", "--build", str(fp64_build), "--gpu-uuid", GPU_UUID,
                "--output", str(fp64_output),
            ])
            fp64_result = json.loads((fp64_output / "run_checks.json").read_text(encoding="utf-8"))
            if fp64_result.get("status") != "passed":
                raise RuntimeError(f"FP64 all-phase runner reported {fp64_result.get('status')!r}")
            record["fp64_run_checks"] = str(fp64_output / "run_checks.json")
            record["fp64_source_sha256"] = fp64_result.get("source_sha256", {})
            record["fp64_phases"] = fp64_result.get("phases", [])
        final_hashes = source_hashes()
        changed = sorted(path for path, digest in record["source_sha256"].items()
                         if final_hashes.get(path) != digest)
        new_sources = sorted(set(final_hashes) - set(record["source_sha256"]))
        if changed or new_sources:
            record["source_hash_changes"] = {"changed": changed, "added": new_sources}
            raise RuntimeError("source inputs changed during integration run")
        record["status"] = "passed" if args.phase == "all" else "built"
        save()
        print(json.dumps({"status": record["status"], "receipt": str(receipt_path)}, sort_keys=True))
        return 0
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as error:
        record["status"] = "failed"
        record["error"] = str(error)
        save()
        print(f"Combined FP64 integration failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
