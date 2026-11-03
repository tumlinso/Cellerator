#!/usr/bin/env python3
"""Bounded, serial Autobench/Scheduler correctness suite for both CUDA routes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
VALIDATOR = ROOT / "bench" / "factorized_interactions" / "validate_runner.py"


def task(case: str, activation: str = "fp16", parameter: str = "fp16",
         output: str = "fp16", gradient: str = "overwrite", repeat: bool = False):
    return {"workload": case, "activation": activation, "parameter": parameter,
            "output": output, "gradient": gradient, "repeat_vjp": repeat}


COMMON = (
    task("correctness_ordered_tied_zero", gradient="overwrite", repeat=True),
    task("correctness_ordered_tied_zero", gradient="accumulate", repeat=True),
    task("correctness_ordered_tied_zero", "fp16", "fp32", "fp32"),
    task("correctness_ordered_tied_zero", "fp32", "fp32", "fp16"),
    task("correctness_ordered_role_context_variants"),
    task("correctness_fma_product_residual_cancellation",
         "fp32", "fp32", "fp32"),
    task("correctness_context_cancellation_middle", "fp32", "fp32", "fp32"),
    task("correctness_context_cancellation_tail", "fp32", "fp32", "fp32"),
    task("correctness_saturated_gates"),
    task("correctness_empty_mechanisms"),
    task("correctness_empty_cells"),
    task("correctness_empty_channels"),
    task("packet_a4_m32_u128"),
)

TRAINING_DEVICE = 0
TRAINING_ITERATIONS = 200


def _run_task(native_runner: Path, expected_uuid: str, route: str,
              specification: dict, artifact_dir: Path,
              build_manifest: Path, environment: dict[str, str]) -> dict:
    command = [
        sys.executable, str(VALIDATOR),
        "--workload", specification["workload"],
        "--runner", shlex.quote(str(native_runner)),
        "--route", route,
        "--activation-dtype", specification["activation"],
        "--parameter-dtype", specification["parameter"],
        "--output-dtype", specification["output"],
        "--gradient-mode", specification["gradient"],
        "--expected-uuid", expected_uuid,
        "--runner-binary", str(native_runner),
        "--build-manifest", str(build_manifest),
        "--artifact-dir", str(artifact_dir),
    ]
    if specification["repeat_vjp"]:
        command.append("--repeat-vjp")
    result = subprocess.run(command, cwd=artifact_dir, env=environment,
                            capture_output=True, text=True)
    record = {
        "workload": specification["workload"],
        "route": route,
        "activation_dtype": specification["activation"],
        "parameter_dtype": specification["parameter"],
        "output_dtype": specification["output"],
        "gradient_mode": specification["gradient"],
        "repeat_vjp": specification["repeat_vjp"],
        "exit_code": result.returncode,
    }
    (artifact_dir / "stdout.jsonl").write_text(result.stdout, encoding="utf-8")
    (artifact_dir / "stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"native correctness failed for {route}/{specification['workload']}; "
            f"stderr captured in {artifact_dir / 'stderr.log'}")
    record["validation"] = json.loads(result.stdout)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--expected-uuid", required=True)
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.artifact_dir = args.artifact_dir.resolve()
    args.output = args.output.resolve()

    job_id = os.environ.get("AUTOBENCH_SCHEDULER_JOB_ID", "").strip()
    if not job_id:
        parser.error("CUDA suite requires an Autobench worker Scheduler job context")
    if not args.expected_uuid.startswith("GPU-"):
        parser.error("--expected-uuid must be the exact GPU Scheduler UUID")
    runner = args.runner.resolve(strict=True)
    build_manifest_path = args.build_manifest.resolve(strict=True)
    manifest_bytes = build_manifest_path.read_bytes()
    build_manifest = json.loads(manifest_bytes)
    if build_manifest.get("schema") != "cellerator.factorized_interactions.build.v1":
        parser.error("--build-manifest has an unknown schema")
    expected_runner = (build_manifest_path.parent /
                       build_manifest["binaries"]["runner"]["path"]).resolve(strict=True)
    runner_sha256 = hashlib.sha256(runner.read_bytes()).hexdigest()
    if runner != expected_runner or runner_sha256 != build_manifest["binaries"]["runner"]["sha256"]:
        parser.error("runner executable identity differs from its build manifest")
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    packet_scratch = args.artifact_dir / "scratch"
    packet_scratch.mkdir(parents=True, exist_ok=True)
    temporary = packet_scratch / "temp"
    temporary.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["CUDA_CACHE_PATH"] = str(packet_scratch / "cuda-cache")
    environment["TMPDIR"] = str(temporary)
    environment["TEMP"] = str(temporary)
    environment["TMP"] = str(temporary)

    results = []
    for route in ("packet", "direct"):
        for index, specification in enumerate(COMMON):
            case_dir = args.artifact_dir / f"{index:02d}-{route}-{specification['workload']}"
            case_dir.mkdir(parents=True, exist_ok=True)
            results.append(_run_task(runner, args.expected_uuid, route,
                                     specification, case_dir, build_manifest_path,
                                     environment))

    high_fanin_cases = (
        ("direct", task("correctness_high_fanin_100k", "fp16", "fp16", "fp32")),
        ("packet", task("correctness_packet_high_fanin_fp32_100k",
                        "fp16", "fp32", "fp32")),
        ("packet", task("correctness_packet_high_fanin_1m",
                        "fp16", "fp16", "fp32")),
    )
    for index, (route, specification) in enumerate(high_fanin_cases, start=len(results)):
        case_dir = args.artifact_dir / f"{index:02d}-{route}-{specification['workload']}"
        case_dir.mkdir(parents=True, exist_ok=True)
        results.append(_run_task(runner, args.expected_uuid, route,
                                 specification, case_dir, build_manifest_path,
                                 environment))

    training_entry = build_manifest.get("binaries", {}).get("training_example")
    if not training_entry:
        parser.error("the build manifest omits the native C++ training example")
    training_binary = (build_manifest_path.parent / training_entry["path"]).resolve(strict=True)
    if hashlib.sha256(training_binary.read_bytes()).hexdigest() != training_entry["sha256"]:
        parser.error("training executable differs from the source-bound build manifest")
    training_command = [str(training_binary), str(TRAINING_DEVICE),
                        str(TRAINING_ITERATIONS)]
    training_dir = args.artifact_dir / "native-training-example"
    training_dir.mkdir(parents=True, exist_ok=True)
    training = subprocess.run(training_command, cwd=training_dir,
                              env=environment, capture_output=True,
                              text=True, check=False)
    (training_dir / "stdout.log").write_text(training.stdout, encoding="utf-8")
    (training_dir / "stderr.log").write_text(training.stderr, encoding="utf-8")
    if training.returncode != 0:
        raise RuntimeError(
            f"native training example failed with status {training.returncode}; "
            f"see {training_dir}")
    trained = re.search(
        r"steps=(\d+)\s+output=([^\s]+)\s+alpha=([^\s]+)\s+"
        r"bias=([^\s]+)\s+gamma=([^\s]+)", training.stdout)
    if not trained or int(trained.group(1)) != TRAINING_ITERATIONS:
        raise RuntimeError("training example did not report the requested update count")
    final_values = [float(value) for value in trained.groups()[1:]]
    if not all(math.isfinite(value) for value in final_values):
        raise RuntimeError("training example reported a nonfinite output or parameter")
    x0, x1, context = (float(np.float16(value)) for value in (0.75, 0.60, 0.50))
    initial_alpha, initial_bias, initial_gamma = (
        float(np.float16(value)) for value in (0.80, -0.20, 0.90))
    initial_gate = 1.0 / (1.0 + math.exp(-(initial_bias + initial_gamma * context)))
    initial_output = float(np.float16(initial_alpha * initial_gate * x0 * x1))
    final_output, final_alpha, final_bias, final_gamma = final_values
    target = 0.25
    if abs(final_output - target) >= abs(initial_output - target):
        raise RuntimeError("repeated C++ parameter updates did not improve the example loss")
    if (final_alpha, final_bias, final_gamma) == (
            initial_alpha, initial_bias, initial_gamma):
        raise RuntimeError("repeated C++ training steps did not change parameters")

    package_root = build_manifest.get("python_package_root")
    if not package_root:
        parser.error("the build manifest omits the staged Python package")
    python_metadata = build_manifest.get("python", {})
    python_executable = python_metadata.get("executable")
    if not python_executable:
        parser.error("the build manifest omits the selected Torch Python interpreter")
    python_suite_path = (ROOT / "bench" / "factorized_interactions" /
                         "run_prepared_torch_suite.py")
    python_suite_output = args.artifact_dir / "prepared-torch-suite.json"
    python_suite_dir = args.artifact_dir / "prepared-torch"
    python_suite_dir.mkdir(parents=True, exist_ok=True)
    python_suite = subprocess.run([
        python_executable, str(python_suite_path),
        "--package-root", str(build_manifest_path.parent / package_root),
        "--build-manifest", str(build_manifest_path),
        "--expected-python", python_executable,
        "--expected-uuid", args.expected_uuid,
        "--output", str(python_suite_output),
    ], cwd=python_suite_dir, env=environment,
        capture_output=True, text=True, check=False)
    (args.artifact_dir / "prepared-torch-suite.stdout.log").write_text(
        python_suite.stdout, encoding="utf-8")
    (args.artifact_dir / "prepared-torch-suite.stderr.log").write_text(
        python_suite.stderr, encoding="utf-8")
    if python_suite.returncode != 0:
        raise RuntimeError(
            "prepared ATen/Python lifecycle suite failed; see "
            f"{args.artifact_dir / 'prepared-torch-suite.stderr.log'}")
    python_summary = json.loads(python_suite_output.read_text(encoding="utf-8"))
    if python_summary.get("schema") != \
            "cellerator.factorized_interactions.prepared_torch_runtime.v1":
        raise RuntimeError("prepared ATen/Python suite returned an unknown summary schema")

    summary = {
        "schema": "cellerator.factorized_interactions.correctness_suite.v1",
        "scheduler_job_id": job_id,
        "expected_device_uuid": args.expected_uuid,
        "build_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_closure_sha256": build_manifest["source_closure_sha256"],
        "runner_binary_sha256": runner_sha256,
        "timed": False,
        "case_count": len(results),
        "passed": len(results),
        "native_training_example": {
            "argv": training_command,
            "exit_code": training.returncode,
            "device_ordinal": TRAINING_DEVICE,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "iterations": TRAINING_ITERATIONS,
            "stdout": training.stdout.strip(),
            "initial_output": initial_output,
            "target": target,
            "final_abs_error": abs(final_output - target),
            "binary_sha256": training_entry["sha256"],
            "timed": False,
        },
        "prepared_torch_suite": python_summary,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"suite": summary["schema"], "case_count": len(results),
                      "passed": len(results), "timed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
