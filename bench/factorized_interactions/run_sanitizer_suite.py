#!/usr/bin/env python3
"""Serial full-VJP sanitizer matrix for the fixed native correctness corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = ROOT / "bench" / "factorized_interactions" / "validate_runner.py"
COMPUTE_SANITIZER = "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer"

CASES = (
    ("correctness_ordered_tied_zero", "fp16", "fp16", "fp16"),
    ("correctness_context_cancellation_middle", "fp32", "fp32", "fp32"),
    ("correctness_context_cancellation_tail", "fp32", "fp32", "fp32"),
    ("correctness_fma_product_residual_cancellation", "fp32", "fp32", "fp32"),
    ("a4_m256_u32_fanin64_tied", "fp16", "fp16", "fp16"),
    ("correctness_empty_mechanisms", "fp16", "fp16", "fp16"),
)
CANDIDATE_CASES = (
    ("direct_unique_output", "packet_a4_m32_u16", "fp16", "fp16", "fp32"),
    ("minimal_direct", "correctness_ordered_tied_zero", "fp16", "fp16", "fp16"),
)
TOOLS = ("memcheck", "racecheck", "initcheck", "synccheck")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--expected-uuid", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.artifact_dir = args.artifact_dir.resolve()
    args.output = args.output.resolve()

    job_id = os.environ.get("AUTOBENCH_SCHEDULER_JOB_ID", "").strip()
    if not job_id:
        parser.error("sanitizer suite requires an Autobench Scheduler job context")
    if not args.expected_uuid.startswith("GPU-"):
        parser.error("--expected-uuid must be the exact Scheduler GPU UUID")
    runner = args.runner.resolve(strict=True)
    manifest_path = args.build_manifest.resolve(strict=True)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    runner_hash = sha256_file(runner)
    expected_runner = (manifest_path.parent /
                       manifest["binaries"]["runner"]["path"]).resolve(strict=True)
    if runner != expected_runner or runner_hash != manifest["binaries"]["runner"]["sha256"]:
        parser.error("runner executable identity differs from the build manifest")
    candidate_identity = manifest.get("candidate_runner")
    if not isinstance(candidate_identity, dict):
        parser.error("the source-bound build manifest omits its candidate runner")
    candidate_runner = (manifest_path.parent /
                        candidate_identity.get("binary_path", "")).resolve(strict=True)
    candidate_manifest_path = (manifest_path.parent /
                               candidate_identity.get("manifest_path", "")).resolve(strict=True)
    candidate_manifest_bytes = candidate_manifest_path.read_bytes()
    candidate_manifest = json.loads(candidate_manifest_bytes)
    candidate_hash = sha256_file(candidate_runner)
    expected_candidate = (candidate_manifest_path.parent /
                          candidate_manifest["binaries"]["runner"]["path"]).resolve(strict=True)
    if candidate_manifest.get("schema") != \
            "cellerator.factorized_interactions.build.v1":
        parser.error("candidate build manifest has an unknown schema")
    if candidate_manifest.get("source_closure_sha256") != \
            manifest.get("source_closure_sha256"):
        parser.error("candidate and baseline builds use different source closures")
    if hashlib.sha256(candidate_manifest_bytes).hexdigest() != \
            candidate_identity.get("manifest_sha256"):
        parser.error("candidate build manifest differs from the parent build manifest")
    if candidate_runner != expected_candidate or candidate_hash != \
            candidate_manifest["binaries"]["runner"]["sha256"] or \
            candidate_hash != candidate_identity.get("binary_sha256"):
        parser.error("candidate runner differs from its source-bound build manifest")
    sanitizer = Path(COMPUTE_SANITIZER).resolve(strict=True)
    version = subprocess.run([str(sanitizer), "--version"], text=True,
                             capture_output=True, check=True)
    args.artifact_dir = args.artifact_dir.resolve()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    packet_scratch = args.artifact_dir / "scratch"
    packet_scratch.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["CUDA_CACHE_PATH"] = str(packet_scratch / "cuda-cache")
    temporary = packet_scratch / "temp"
    temporary.mkdir(parents=True, exist_ok=True)
    environment["TMPDIR"] = str(temporary)
    environment["TEMP"] = str(temporary)
    environment["TMP"] = str(temporary)

    results = []
    for tool in TOOLS:
        for route in ("packet", "direct"):
            for workload, activation, parameter, output in CASES:
                case_dir = args.artifact_dir / f"{tool}-{route}-{workload}"
                case_dir.mkdir(parents=True, exist_ok=True)
                runner_command = shlex.join([
                    str(sanitizer), "--tool", tool, "--error-exitcode", "99",
                    *( ["--leak-check", "full"] if tool == "memcheck" else [] ),
                    str(runner),
                ])
                command = [
                    sys.executable, str(VALIDATOR),
                    "--workload", workload,
                    "--runner", runner_command,
                    "--runner-binary", str(runner),
                    "--build-manifest", str(manifest_path),
                    "--route", route,
                    "--activation-dtype", activation,
                    "--parameter-dtype", parameter,
                    "--output-dtype", output,
                    "--gradient-mode", "overwrite",
                    "--expected-uuid", args.expected_uuid,
                    "--artifact-dir", str(case_dir),
                ]
                proc = subprocess.run(command, cwd=case_dir, env=environment,
                                      text=True, capture_output=True, check=False)
                (case_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
                (case_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
                if proc.returncode:
                    sys.stderr.write(proc.stderr)
                    raise RuntimeError(
                        f"{tool} failed for {route}/{workload}; see {case_dir}")
                validation = json.loads(proc.stdout)
                if not validation.get("validated"):
                    raise RuntimeError(f"{tool} did not produce a validated result")
                results.append({
                    "tool": tool,
                    "route": route,
                    "workload": workload,
                    "exit_code": proc.returncode,
                    "validation": validation,
                })

        for candidate, workload, activation, parameter, output in CANDIDATE_CASES:
            case_dir = args.artifact_dir / f"{tool}-candidate-{candidate}-{workload}"
            case_dir.mkdir(parents=True, exist_ok=True)
            runner_command = shlex.join([
                str(sanitizer), "--tool", tool, "--error-exitcode", "99",
                *( ["--leak-check", "full"] if tool == "memcheck" else [] ),
                str(candidate_runner), "--candidate", candidate,
            ])
            command = [
                sys.executable, str(VALIDATOR),
                "--workload", workload,
                "--runner", runner_command,
                "--runner-binary", str(candidate_runner),
                "--build-manifest", str(candidate_manifest_path),
                "--route", "direct",
                "--activation-dtype", activation,
                "--parameter-dtype", parameter,
                "--output-dtype", output,
                "--gradient-mode", "overwrite",
                "--expected-uuid", args.expected_uuid,
                "--artifact-dir", str(case_dir),
            ]
            proc = subprocess.run(command, cwd=case_dir, env=environment,
                                  text=True, capture_output=True, check=False)
            (case_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
            (case_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
            if proc.returncode:
                sys.stderr.write(proc.stderr)
                raise RuntimeError(
                    f"{tool} failed for {candidate}/{workload}; see {case_dir}")
            validation = json.loads(proc.stdout)
            native = validation.get("native_execution") or {}
            if not validation.get("validated") or native.get("candidate") != candidate:
                raise RuntimeError(
                    f"{candidate} metadata did not match for {tool}/{workload}")
            if candidate == "direct_unique_output":
                if native.get("actual_implementation") != "direct_unique_output" or \
                        native.get("structural_layout") != "canonical_direct" or \
                        native.get("selected_backend") != "direct":
                    raise RuntimeError(
                        f"unique-output specialization was not selected for {tool}/{workload}")
            elif native.get("structural_layout") != "canonical_direct" or \
                    native.get("selected_backend") != "direct":
                raise RuntimeError(
                    f"minimal-direct layout was not selected for {tool}/{workload}")
            results.append({
                "tool": tool,
                "route": "direct",
                "candidate": candidate,
                "workload": workload,
                "exit_code": proc.returncode,
                "validation": validation,
            })

    result = {
        "schema": "cellerator.factorized_interactions.sanitizer_suite.v1",
        "scheduler_job_id": job_id,
        "device_uuid": args.expected_uuid,
        "compute_sanitizer": str(sanitizer),
        "compute_sanitizer_version": (version.stdout + version.stderr).strip(),
        "build_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_closure_sha256": manifest["source_closure_sha256"],
        "runner_binary_sha256": runner_hash,
        "candidate_build_manifest_sha256": hashlib.sha256(
            candidate_manifest_bytes).hexdigest(),
        "candidate_runner_binary_sha256": candidate_hash,
        "timed": False,
        "case_count": len(results),
        "passed": len(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"suite": result["schema"], "case_count": len(results),
                      "passed": len(results), "timed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
