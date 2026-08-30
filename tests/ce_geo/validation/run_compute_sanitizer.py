#!/usr/bin/env python3
"""Build and sanitize the bounded CE-GEO CUDA validation campaign."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass


SANITIZER = pathlib.Path(
    "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/"
    "compute-sanitizer/compute-sanitizer"
)


@dataclass(frozen=True)
class Target:
    name: str
    coverage: tuple[str, ...]
    tools: tuple[str, ...]


TARGETS = (
    Target("projectionEnumerationTest",
           ("device_views", "cpe2_rebind"), ("memcheck", "initcheck")),
    Target("ceGeoSm70ValuePackTest",
           ("value_pack", "generation_reuse"),
           ("memcheck", "racecheck", "initcheck", "synccheck")),
    Target("ceGeoSm70N64HybridTest",
           ("mma", "residual", "epilogue"),
           ("memcheck", "racecheck", "initcheck", "synccheck")),
    Target("ceGeoSegmentNormalizeTest",
           ("segments",),
           ("memcheck", "racecheck", "initcheck", "synccheck")),
    Target("ceGeoSm70TransposeRelationApplyTest",
           ("transpose",), ("memcheck", "initcheck")),
    Target("ceGeoSm70ContractOnSupportTest",
           ("contraction",), ("memcheck", "initcheck")),
    Target("ceGeoGradientValidationTest",
           ("gradients", "transpose", "contraction", "segments"),
           ("memcheck", "racecheck", "initcheck", "synccheck")),
)

REQUIRED_COVERAGE = {
    "device_views",
    "cpe2_rebind",
    "value_pack",
    "mma",
    "residual",
    "segments",
    "transpose",
    "contraction",
    "gradients",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], cwd: pathlib.Path,
        timeout: int) -> tuple[subprocess.CompletedProcess[bytes], float]:
    start = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
    )
    return completed, time.monotonic() - start


def text(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")


def command_record(command: list[str], completed: subprocess.CompletedProcess[bytes],
                   elapsed: float) -> dict[str, object]:
    return {
        "argv": command,
        "returncode": completed.returncode,
        "elapsed_seconds": round(elapsed, 6),
        "stdout": text(completed.stdout),
        "stderr": text(completed.stderr),
        "stdout_sha256": sha256_bytes(completed.stdout),
        "stderr_sha256": sha256_bytes(completed.stderr),
    }


def git_head(repo_root: pathlib.Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(f"git rev-parse failed: {completed.stderr.strip()}")
    value = completed.stdout.strip()
    if len(value) != 40:
        raise RuntimeError("git rev-parse returned an invalid commit")
    return value


def fail(command: list[str], completed: subprocess.CompletedProcess[bytes]) -> None:
    print(f"CE-GEO sanitizer command failed: {command}", file=sys.stderr)
    if completed.stdout:
        print(text(completed.stdout), file=sys.stderr, end="")
    if completed.stderr:
        print(text(completed.stderr), file=sys.stderr, end="")
    raise RuntimeError(f"command exited {completed.returncode}")


def build_command(target: Target, repo_root: pathlib.Path,
                  build: pathlib.Path, jobs: int) -> list[str]:
    direct_source: pathlib.Path | None = None
    direct_libraries: tuple[str, ...] = ()
    if target.name == "projectionEnumerationTest":
        direct_source = (
            repo_root / "tests/ce_geo/persistence/projection_enumeration_test.cu"
        )
        direct_libraries = (
            "libcellerator_opaque_execution_artifact.a",
            "libcellpack_execution_image_v2.a",
            "libcellpack_persistent_packing_payload.a",
        )
    elif target.name == "ceGeoGradientValidationTest":
        direct_source = (
            repo_root / "tests/ce_geo/validation/gradient_validation_test.cu"
        )
        direct_libraries = (
            "libcellerator_relation_algebra.a",
            "libcellerator_operation_core.a",
            "libcellerator_runtime.a",
        )
    if direct_source is not None:
        return [
            "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc",
            "-std=c++17",
            "-arch=sm_70",
            "-Wno-deprecated-gpu-targets",
            "-Xcompiler=-Wall,-Wextra,-Werror",
            "-ccbin",
            "/usr/bin/g++",
            f"-I{repo_root / 'include'}",
            f"-I{build / 'generated'}",
            str(direct_source),
            "-o",
            str(build / target.name),
            *(str(build / library) for library in direct_libraries),
        ]
    return ["cmake", "--build", str(build), "--target",
            target.name, "-j", str(jobs)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    build = args.build if args.build.is_absolute() else repo_root / args.build
    output = args.output if args.output.is_absolute() else repo_root / args.output

    try:
        if not SANITIZER.is_file() or not os.access(SANITIZER, os.X_OK):
            raise RuntimeError(f"CUDA 12.9 compute-sanitizer is unavailable: {SANITIZER}")
        if not build.is_dir():
            raise RuntimeError(f"build directory does not exist: {build}")
        coverage = {item for target in TARGETS for item in target.coverage}
        if not REQUIRED_COVERAGE.issubset(coverage):
            missing = sorted(REQUIRED_COVERAGE - coverage)
            raise RuntimeError(f"sanitizer campaign lacks required coverage: {missing}")

        version, _ = run([str(SANITIZER), "--version"], repo_root, 30)
        if version.returncode != 0:
            fail([str(SANITIZER), "--version"], version)
        build_records: list[dict[str, object]] = []
        jobs = max(1, os.cpu_count() or 1)
        for target in TARGETS:
            command = build_command(target, repo_root, build, jobs)
            completed, elapsed = run(command, repo_root, 600)
            record = command_record(command, completed, elapsed)
            record["target"] = target.name
            build_records.append(record)
            if completed.returncode != 0:
                fail(command, completed)
            binary = build / target.name
            if not binary.is_file() or not os.access(binary, os.X_OK):
                raise RuntimeError(f"build did not produce executable: {binary}")

        run_records: list[dict[str, object]] = []
        for target in TARGETS:
            binary = (build / target.name).resolve()
            binary_digest = sha256_file(binary)
            for tool in target.tools:
                command = [str(SANITIZER), "--tool", tool,
                           "--error-exitcode", "99", str(binary)]
                completed, elapsed = run(command, repo_root, 300)
                record = command_record(command, completed, elapsed)
                record.update({
                    "target": target.name,
                    "tool": tool,
                    "coverage": list(target.coverage),
                    "binary": str(binary.relative_to(repo_root)),
                    "binary_sha256": binary_digest,
                    "validated": completed.returncode == 0,
                })
                run_records.append(record)
                if completed.returncode != 0:
                    fail(command, completed)

        report = {
            "schema_version": 1,
            "campaign": "CE-GEO",
            "task_id": "CE-GEO-108",
            "validated": True,
            "source_commit": git_head(repo_root),
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "tool": {
                "path": str(SANITIZER),
                "version_stdout": text(version.stdout).strip(),
                "version_stderr": text(version.stderr).strip(),
            },
            "environment": {
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "build_directory": str(build.relative_to(repo_root)),
                "parallel_build_jobs": jobs,
            },
            "required_coverage": sorted(REQUIRED_COVERAGE),
            "observed_coverage": sorted(coverage),
            "builds": build_records,
            "runs": run_records,
            "summary": {
                "target_count": len(TARGETS),
                "sanitizer_run_count": len(run_records),
                "passed_run_count": sum(
                    bool(record["validated"]) for record in run_records
                ),
                "tools": sorted({tool for target in TARGETS for tool in target.tools}),
            },
        }
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=output, prefix="campaign.",
                suffix=".tmp", delete=False) as stream:
            temporary = pathlib.Path(stream.name)
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
        temporary.replace(output / "campaign.json")
        print(
            "CE-GEO compute-sanitizer campaign passed: "
            f"targets={len(TARGETS)} runs={len(run_records)} "
            f"output={output / 'campaign.json'}"
        )
        return 0
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"CE-GEO compute-sanitizer campaign failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
