#!/usr/bin/env python3
"""Build and run the FP64 qualification gates through the CUDA controller."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any


PHASES = ("oracle", "native", "sanitize", "costs")
WITNESSES = ("fp64_existing_adapter_test", "fp64_existing_forward_test",
             "fp64_existing_transpose_test", "fp64_existing_lifecycle_test",
             "fp64_existing_overhead_test", "fp64_admission_test")
CUDA_ROOT = Path("/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9")
CONTROLLER = Path.home() / ".agents/skills/cuda/scripts/cuda_controller.py"
SOURCE_PATHS = (
    "tests/fp64/CMakeLists.txt",
    "tests/fp64/admission_test.cu",
    "tests/fp64/compare_scipy.py",
    "tests/fp64/cost_bench.cu",
    "tests/fp64/fp64_qualification.cu",
    "tests/fp64/reference/project_f32_baseline.cu",
    "tests/fp64/measure_costs.py",
    "tests/fp64/run_checks.py",
    "include/Cellerator/compute/operation/prepared_relation.hh",
    "src/compute/operation/prepared_relation.cu",
    "src/compute/operation/relation_semantics.cc",
    "include/Cellerator/compute/operation/device_elementwise.hh",
    "src/compute/operation/device_elementwise.cuh",
    "include/Cellerator/compute/candidate/sparse/project.hh",
    "src/compute/candidate/sparse/project.cu",
    "include/Cellerator/compute/operation/relation_semantics.hh",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes(root: Path) -> dict[str, str]:
    return {name: sha256(root / name) for name in SOURCE_PATHS if (root / name).is_file()}


def binary_hashes(paths: list[Path]) -> dict[str, str]:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError("required build output is missing: " + ", ".join(missing))
    return {str(path.resolve()): sha256(path) for path in paths}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class GateRun:
    def __init__(self, root: Path, build: Path, output: Path, gpu_uuid: str) -> None:
        self.root = root
        self.build = build
        self.output = output
        self.gpu_uuid = gpu_uuid
        self.records: list[dict[str, Any]] = []
        self.output.mkdir(parents=True, exist_ok=True)

    def save(self, status: str, error: str | None = None) -> None:
        write_json(self.output / "run_checks.json", {
            "schema": "cellerator.fp64.gates.v1",
            "status": status,
            "source_root": str(self.root),
            "source_head": self.git_head(),
            "build": str(self.build),
            "gpu_uuid": self.gpu_uuid,
            "source_sha256": source_hashes(self.root),
            "phases": self.records,
            "error": error,
        })

    def git_head(self) -> str | None:
        try:
            result = subprocess.run(["git", "-C", str(self.root), "rev-parse", "HEAD"],
                                    check=True, capture_output=True, text=True)
            return result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    def record_process(self, label: str, argv: list[str], *, cwd: Path,
                       env: dict[str, str] | None = None) -> dict[str, Any]:
        started = time.monotonic()
        result = subprocess.run(argv, cwd=cwd, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        elapsed = time.monotonic() - started
        record: dict[str, Any] = {
            "label": label, "argv": argv, "cwd": str(cwd), "returncode": result.returncode,
            "elapsed_seconds": elapsed,
            "stdout_path": str(self.output / f"{label}.stdout.txt"),
            "stderr_path": str(self.output / f"{label}.stderr.txt"),
            "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
        }
        (self.output / f"{label}.stdout.txt").write_text(result.stdout, encoding="utf-8")
        (self.output / f"{label}.stderr.txt").write_text(result.stderr, encoding="utf-8")
        self.records.append(record)
        if result.returncode:
            raise RuntimeError(f"{label} failed with exit status {result.returncode}; see {record['stderr_path']}")
        return record

    def ensure_build(self, targets: list[str]) -> None:
        cmake = shutil.which("cmake")
        if not cmake:
            raise RuntimeError("cmake is required")
        cache = self.build / "CMakeCache.txt"
        if not cache.is_file():
            nvcc = CUDA_ROOT / "bin/nvcc"
            if not nvcc.is_file():
                raise RuntimeError(f"CUDA 12.9 nvcc is unavailable: {nvcc}")
            configure = [
                cmake, "-S", str(self.root / "tests/fp64"), "-B", str(self.build),
                f"-DCELLERATOR_SOURCE_DIR={self.root}", "-DCMAKE_BUILD_TYPE=Release",
                f"-DCUDAToolkit_ROOT={CUDA_ROOT}", f"-DCMAKE_CUDA_COMPILER={nvcc}",
                "-DCMAKE_CUDA_ARCHITECTURES=70",
            ]
            self.record_process("configure", configure, cwd=self.root)
        command = [cmake, "--build", str(self.build), "--parallel", "4", "--target", *targets]
        self.record_process("build_" + "_".join(targets), command, cwd=self.root)

    def controller_spec(self, phase: str, argv: list[str], *, recipe: str = "baseline",
                        require_sanitizer: bool = False) -> dict[str, Any]:
        spec: dict[str, Any] = {
            "schema_version": 1,
            "project_root": str(self.root),
            "command_cwd": str(self.root),
            "campaign_id": f"cellerator-fp64-{phase}",
            "recipe": recipe,
            "paths": list(SOURCE_PATHS),
            "argv": argv,
            "timeout": 3600,
            "resources": {"gpus": 1, "gpu_uuids": [self.gpu_uuid], "cpu_threads": 4},
            "toolchain": {"root": str(CUDA_ROOT),
                          "require_sanitizer": require_sanitizer or recipe == "compute-sanitizer"},
        }
        return spec

    def run_controller(self, phase: str, spec: dict[str, Any], binary_paths: list[Path]) -> None:
        if not CONTROLLER.is_file():
            raise RuntimeError(f"CUDA controller unavailable: {CONTROLLER}")
        controller_dir = self.output / "controller"
        controller_dir.mkdir(parents=True, exist_ok=True)
        spec_path = controller_dir / f"{phase}.spec.json"
        write_json(spec_path, spec)
        argv = [sys.executable, str(CONTROLLER), "run", "--spec", str(spec_path), "--json"]
        child_env = None
        if spec.get("recipe") == "compute-sanitizer":
            sanitizer = CUDA_ROOT / "compute-sanitizer" / "compute-sanitizer"
            if not sanitizer.is_file() or not os.access(sanitizer, os.X_OK):
                raise RuntimeError(f"CUDA compute-sanitizer executable unavailable: {sanitizer}")
            child_env = os.environ.copy()
            child_env["COMPUTE_SANITIZER_BIN"] = str(sanitizer)
        started = time.monotonic()
        result = subprocess.run(argv, cwd=self.root, env=child_env, text=True,
                                capture_output=True, check=False)
        elapsed = time.monotonic() - started
        stdout_path = controller_dir / f"{phase}.stdout.txt"
        stderr_path = controller_dir / f"{phase}.stderr.txt"
        stdout_path.write_text(result.stdout, encoding="utf-8")
        stderr_path.write_text(result.stderr, encoding="utf-8")
        try:
            controller_receipt = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            controller_receipt = {"parse_error": str(error), "stdout": result.stdout[-4000:]}
        if not isinstance(controller_receipt, dict):
            controller_receipt = {"parse_error": "controller JSON receipt is not an object",
                                  "value": controller_receipt}
        sanitizer_validation = None
        if phase.startswith("sanitize_"):
            sanitizer_validation = self.validate_sanitizer_receipt(phase, controller_receipt)
        metadata = {
            "phase": phase, "argv": argv, "spec_path": str(spec_path),
            "returncode": result.returncode, "elapsed_seconds": elapsed,
            "stdout_path": str(stdout_path), "stderr_path": str(stderr_path),
            "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
            "spec": spec, "source_sha256": source_hashes(self.root),
            "binary_sha256": binary_hashes(binary_paths),
            "controller_receipt": controller_receipt,
        }
        if sanitizer_validation is not None:
            metadata["sanitizer_validation"] = sanitizer_validation
        receipt_path = controller_dir / f"{phase}.json"
        write_json(receipt_path, metadata)
        controller_ok = bool(controller_receipt.get("ok", False))
        if sanitizer_validation is not None:
            controller_ok = controller_ok and sanitizer_validation["passed"]
        self.records.append({"phase": phase, "receipt": str(receipt_path),
                             "returncode": result.returncode,
                             "controller_ok": controller_ok})
        self.save("running")
        if result.returncode or not controller_ok:
            raise RuntimeError(f"CUDA controller gate failed for {phase}; see {receipt_path}")

    def validate_sanitizer_receipt(self, phase: str, receipt: dict[str, Any]) -> dict[str, Any]:
        """The controller wrapper returns zero after classification; verify its real log."""
        stdout_path = Path(str(receipt.get("stdout_path", "")))
        run_dir = stdout_path.parent / "sanitizer" / "run"
        raw_log = run_dir / "raw.log"
        summary_json = run_dir / "summary.json"
        evidence_dir = self.output / "sanitizer"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        saved_log = evidence_dir / f"{phase}.raw.log"
        saved_summary = evidence_dir / f"{phase}.summary.json"
        errors: list[str] = []
        if not raw_log.is_file():
            errors.append("controller sanitizer raw log is missing")
        else:
            shutil.copyfile(raw_log, saved_log)
        summary: dict[str, Any] = {}
        if not summary_json.is_file():
            errors.append("controller sanitizer summary JSON is missing")
        else:
            try:
                loaded = json.loads(summary_json.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    summary = loaded
                else:
                    errors.append("controller sanitizer summary is not a JSON object")
            except (OSError, json.JSONDecodeError) as error:
                errors.append(f"controller sanitizer summary cannot be read: {error}")
            if summary:
                shutil.copyfile(summary_json, saved_summary)
        raw_text = saved_log.read_text(encoding="utf-8", errors="replace") if saved_log.is_file() else ""
        error_counts = [int(count) for count in re.findall(
            r"ERROR SUMMARY:\s*(\d+)\s+errors?", raw_text, flags=re.IGNORECASE)]
        if not error_counts:
            errors.append("sanitizer raw log has no ERROR SUMMARY line")
        elif any(count != 0 for count in error_counts):
            errors.append(f"sanitizer reported nonzero ERROR SUMMARY values: {error_counts}")
        if re.search(r"(?:=========\s*)?(?:ERROR|INTERNAL ERROR):", raw_text, flags=re.IGNORECASE):
            errors.append("sanitizer raw log contains an instrumentation error")
        if summary.get("error_count") != 0:
            errors.append(f"sanitizer classifier error_count is not zero: {summary.get('error_count')!r}")
        if summary.get("exit_status") != 0:
            errors.append(f"sanitized child exit status is not zero: {summary.get('exit_status')!r}")
        return {
            "passed": not errors,
            "errors": errors,
            "error_summary_values": error_counts,
            "child_exit_status": summary.get("exit_status"),
            "classifier_status": summary.get("status"),
            "raw_log": str(saved_log) if saved_log.is_file() else None,
            "raw_log_sha256": sha256(saved_log) if saved_log.is_file() else None,
            "summary": str(saved_summary) if saved_summary.is_file() else None,
            "summary_sha256": sha256(saved_summary) if saved_summary.is_file() else None,
        }

    def phase_oracle(self) -> None:
        binary = self.build / "fp64_qualification"
        self.ensure_build(["fp64_qualification"])
        receipt = self.output / "scipy.json"
        script = self.root / "tests/fp64/compare_scipy.py"
        argv = [sys.executable, str(script), "--binary", str(binary), "--receipt", str(receipt)]
        self.run_controller("oracle", self.controller_spec("oracle", argv), [binary])

    def phase_native(self, *, under_controller: bool = False) -> None:
        targets = list(WITNESSES)
        binaries = [self.build / target for target in targets]
        if not under_controller:
            self.ensure_build(targets)
            argv = [sys.executable, str(self.root / "tests/fp64/run_checks.py"),
                    "--phase", "native", "--under-controller", "--build", str(self.build),
                    "--gpu-uuid", self.gpu_uuid, "--output", str(self.output)]
            self.run_controller("native", self.controller_spec("native", argv), binaries)
            return
        lease_path = os.environ.get("TODO_GPU_LEASE_RECEIPT")
        if os.environ.get("CUDA_BENCHMARK_FOREGROUND_INTENT_HELD") != "1" or not lease_path:
            raise RuntimeError("native witnesses may run only inside the foreground CUDA controller lease")
        try:
            lease = json.loads(Path(lease_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"cannot read foreground CUDA lease receipt: {error}") from error
        resource_ids = lease.get("resource_ids", []) if isinstance(lease, dict) else []
        if f"accelerator:{self.gpu_uuid}" not in resource_ids:
            raise RuntimeError("foreground CUDA lease does not cover the requested GPU UUID")
        missing = [str(path) for path in binaries if not path.is_file()]
        if missing:
            raise RuntimeError("native controller phase has missing prebuilt binaries: " + ", ".join(missing))
        records = []
        for target, binary in zip(WITNESSES, binaries):
            name = target.removeprefix("fp64_existing_").removesuffix("_test")
            started = time.monotonic()
            result = subprocess.run([str(binary)], cwd=self.root, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            stdout_path = self.output / "native" / f"{name}.stdout.txt"
            stderr_path = self.output / "native" / f"{name}.stderr.txt"
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            stdout_path.write_text(result.stdout, encoding="utf-8")
            stderr_path.write_text(result.stderr, encoding="utf-8")
            record = {
                "name": name, "argv": [str(binary)], "returncode": result.returncode,
                "elapsed_seconds": time.monotonic() - started,
                "binary_sha256": sha256(binary), "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
                "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
            }
            records.append(record)
            write_json(self.output / "native" / "native.json", {
                "schema": "cellerator.fp64.native.v1", "status": "running",
                "source_sha256": source_hashes(self.root), "tests": records,
            })
            if result.returncode:
                raise RuntimeError(f"native witness {name} failed with exit status {result.returncode}")
        write_json(self.output / "native" / "native.json", {
            "schema": "cellerator.fp64.native.v1", "status": "passed",
            "source_sha256": source_hashes(self.root), "tests": records,
        })

    def phase_sanitize(self) -> None:
        binaries = [self.build / "fp64_qualification", self.build / "fp64_admission_test"]
        self.ensure_build([path.name for path in binaries])
        for name, binary in zip(("qualification", "admission"), binaries):
            argv = [str(binary), "--dump-jsonl"] if name == "qualification" else [str(binary)]
            spec = self.controller_spec(f"sanitize_{name}", argv, recipe="compute-sanitizer")
            self.run_controller(f"sanitize_{name}", spec, [binary])

    def phase_costs(self) -> None:
        binary = self.build / "fp64_cost_bench"
        self.ensure_build(["fp64_cost_bench"])
        baseline = self.output / "baseline" / "libhistorical_csr_baseline.so"
        baseline.parent.mkdir(parents=True, exist_ok=True)
        script = self.root / "tests/fp64/measure_costs.py"
        self.record_process("build_historical_baseline", [
            sys.executable, str(script), "--build-baseline-only", "--binary", str(binary),
            "--baseline-library", str(baseline),
        ], cwd=self.root)
        receipt = self.output / "costs.json"
        argv = [sys.executable, str(script), "--binary", str(binary),
                "--baseline-library", str(baseline), "--receipt", str(receipt)]
        self.run_controller("costs", self.controller_spec("costs", argv), [binary, baseline])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=(*PHASES, "all"), required=True)
    parser.add_argument("--build", type=Path, default=Path("/tmp/cellerator-fp64-build"))
    parser.add_argument("--gpu-uuid", default="GPU-6c1cac7f-a360-0aef-ba98-2828bfd1db1a")
    parser.add_argument("--output", type=Path, default=Path("planning/fp64/evidence"))
    parser.add_argument("--under-controller", action="store_true",
                        help="internal native-test mode; run only inside the controller-held GPU lease")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    build = args.build.expanduser().resolve()
    output = args.output.expanduser()
    if not output.is_absolute():
        output = (root / output).resolve()
    if args.under_controller and args.phase != "native":
        parser.error("--under-controller is supported only with --phase native")

    runner = GateRun(root, build, output, args.gpu_uuid)
    selected = PHASES if args.phase == "all" else (args.phase,)
    try:
        if args.under_controller:
            runner.phase_native(under_controller=True)
        else:
            for phase in selected:
                getattr(runner, f"phase_{phase}")()
        runner.save("passed")
        print(json.dumps({"status": "passed", "phase": args.phase,
                          "receipt": str(output / "run_checks.json")}, sort_keys=True))
        return 0
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
        runner.save("failed", str(error))
        print(f"FP64 gate failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
