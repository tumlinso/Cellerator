#!/usr/bin/env python3
"""Run and record the complete CE-GEO Volta acceptance campaign."""

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
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[3]
NVCC = pathlib.Path(
    "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc"
)
SANITIZER_OUTPUT = pathlib.Path("/tmp/ce_geo_full_volta_sanitizer")
MAX_CAPTURE_BYTES = 16384


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")


def tail(value: bytes) -> str:
    return decode(value[-MAX_CAPTURE_BYTES:])


def run(command: list[str], timeout: int) -> dict[str, Any]:
    print("[CE-GEO-109] " + " ".join(command), flush=True)
    start = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
    )
    elapsed = time.monotonic() - start
    if completed.stdout:
        print(decode(completed.stdout), end="")
    if completed.stderr:
        print(decode(completed.stderr), end="", file=sys.stderr)
    record: dict[str, Any] = {
        "argv": command,
        "returncode": completed.returncode,
        "elapsed_seconds": round(elapsed, 6),
        "stdout_bytes": len(completed.stdout),
        "stderr_bytes": len(completed.stderr),
        "stdout_sha256": sha256_bytes(completed.stdout),
        "stderr_sha256": sha256_bytes(completed.stderr),
        "stdout_tail": tail(completed.stdout),
        "stderr_tail": tail(completed.stderr),
    }
    if completed.returncode != 0:
        raise RuntimeError(
            f"command exited {completed.returncode}: {json.dumps(command)}"
        )
    return record


def git_output(arguments: list[str]) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout


def allowed_projection_dirt(path: str) -> bool:
    return (
        path == "AGENTS.md"
        or path in {"todo-status.md", "todos.md"}
        or path.startswith(".todo-orchestrator/")
        or path.startswith("todos/")
    )


def contamination() -> dict[str, Any]:
    raw = git_output(["status", "--porcelain=v1", "--untracked-files=all"])
    entries = []
    disallowed = []
    for line in raw.splitlines():
        if len(line) < 4:
            raise RuntimeError(f"malformed git status entry: {line!r}")
        path = line[3:].split(" -> ")[-1]
        entry = {"status": line[:2], "path": path,
                 "allowed_projection_dirt": allowed_projection_dirt(path)}
        entries.append(entry)
        if not entry["allowed_projection_dirt"]:
            disallowed.append(entry)
    if disallowed:
        raise RuntimeError(
            "non-projection worktree contamination: "
            + json.dumps(disallowed, sort_keys=True)
        )
    return {"porcelain": raw, "entries": entries,
            "disallowed_entries": disallowed}


def direct_validation_commands(build: pathlib.Path) -> list[list[str]]:
    cxx = "/usr/bin/g++"
    common_host = [cxx, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                   "-Iinclude"]
    common_cuda = [str(NVCC), "-std=c++17", "-arch=sm_70",
                   "-Wno-deprecated-gpu-targets",
                   "-Xcompiler=-Wall,-Wextra,-Werror", "-ccbin", cxx,
                   "-Iinclude", "-Ibuild/generated"]
    semantic = build / "ceGeoSemanticPropertyTest"
    physical = build / "ceGeoPhysicalCoverPropertyTest"
    numerical = build / "ceGeoNumericalRefereeTest"
    negative = build / "ceGeoFoundationNegativeTest"
    gradient = build / "ceGeoGradientValidationTest"
    hot_path = build / "ceGeoHotPathTest"
    return [
        [*common_host, "tests/ce_geo/validation/semantic_property_test.cc",
         "-o", str(semantic),
         "build/src/geometry/libcellerator_semantic_geometry_v1.a"],
        [str(semantic)],
        [*common_host, "tests/ce_geo/validation/physical_cover_property_test.cc",
         "-o", str(physical), "build/libcellerator_architecture_provider.a"],
        [str(physical)],
        [*common_host, "tests/ce_geo/validation/numerical_referee_test.cc",
         "-o", str(numerical)],
        [str(numerical)],
        [*common_cuda, "tests/ce_geo/validation/foundation_negative_test.cu",
         "-o", str(negative), "build/libcellerator_architecture_provider.a",
         "build/libcellerator_operation_core.a", "build/libcellerator_runtime.a"],
        [str(negative)],
        [*common_cuda, "tests/ce_geo/validation/gradient_validation_test.cu",
         "src/compute/candidate/segment/normalize.cu", "-o", str(gradient)],
        [str(gradient)],
        [*common_cuda, "tests/ce_geo/validation/hot_path_test.cu",
         "-o", str(hot_path),
         "build/libcellerator_executable_program.a",
         "build/libcellerator_projection_activation.a",
         "build/libcellpack_execution_image_v2.a",
         "build/libcellerator_cusparse_csr_candidate.a",
         "build/libcellerator_preparation_factory.a",
         "build/libcellerator_builtin_candidate_catalog.a",
         "build/libcellerator_csr_fallback_candidate.a",
         "build/libcellerator_row_masked_n1_candidate.a",
         "build/src/geometry/libcellpack_feature_weighted_row_reduction_cuda.a",
         "build/libcellerator_feature_major_small_n_candidate.a",
         "build/libcellerator_transpose_backward_candidate.a",
         "build/libcellerator_native_training_slice.a",
         "build/libcellerator_transpose_projection.a",
         "build/libcellerator_feature_major_projection.a",
         "build/libcellpack_persistent_packing_payload.a",
         "build/libcellpack_feature_weighted_row_reduction.a",
         "build/libcellpack_apply_plan.a",
         "build/src/geometry/libcellpack_warp_tiles.a",
         "build/libcellpack_local_cell_ordering.a",
         "build/libcellerator_planner.a",
         "build/libcellerator_operation_core.a",
         "build/libcellerator_runtime.a",
         "build/libcellpack_semantic_geometry.a",
         "build/libcellpack_alternating_refinement.a",
         "build/libcellpack_record_statistical_validation.a",
         "build/libcellpack_statistical_validation.a",
         "build/src/geometry/libcellpack.a",
         "build/libcellerator_compute_sampling.a",
         "-L/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/math_libs/12.9/lib64",
         "-lcusparse", "-lcublas", "-lcublasLt"],
        [str(hot_path)],
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", required=True, type=int)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    output = args.output if args.output.is_absolute() else ROOT / args.output
    build = ROOT / "build"

    try:
        if not NVCC.is_file() or not os.access(NVCC, os.X_OK):
            raise RuntimeError(f"required CUDA compiler is unavailable: {NVCC}")
        before = contamination()
        source_commit = git_output(["rev-parse", "HEAD"]).strip()
        source_tree = git_output(["rev-parse", "HEAD^{tree}"]).strip()

        metadata_commands = [
            ["nvidia-smi", "--query-gpu=name,uuid,compute_cap,driver_version,"
             "pci.bus_id,memory.total", "--format=csv,noheader,nounits"],
            [str(NVCC), "--version"],
            ["cmake", "--version"],
            ["/usr/bin/g++", "--version"],
            [sys.executable, "--version"],
        ]
        metadata = [run(command, 60) for command in metadata_commands]

        commands = [
            ["cmake", "-S", ".", "-B", "build",
             "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=70",
             "-DCELLERATOR_ENABLE_TORCH_MODELS=OFF"],
            [sys.executable,
             "tests/ce_geo/validation/baseline_golden_regression.py",
             "--build", "build"],
            ["cmake", "--build", "build", "-j", str(args.jobs)],
            [sys.executable, "tests/ce_geo/run_build_matrix.py",
             "--jobs", str(args.jobs)],
            [sys.executable, "tests/ce_geo/run_foundation_suite.py",
             "--build", "build"],
            [sys.executable, "tests/ce_geo/run_sm70_vertical_suite.py",
             "--build", "build"],
            [sys.executable, "tests/ce_geo/run_biology_suite.py",
             "--build", "build"],
            *direct_validation_commands(build),
            [sys.executable, "tests/ce_geo/validation/test_static_contracts.py"],
            [sys.executable, "scripts/ce_geo/check_static_contracts.py",
             "--manifest", "tests/ce_geo/validation/owned_production_paths.json"],
            [sys.executable, "tests/ce_geo/validation/run_compute_sanitizer.py",
             "--build", "build", "--output", str(SANITIZER_OUTPUT)],
        ]
        records = []
        for command in commands:
            timeout = 3600 if command[0] in {"cmake", sys.executable} else 1200
            records.append(run(command, timeout))

        sanitizer_report = SANITIZER_OUTPUT / "campaign.json"
        if not sanitizer_report.is_file():
            raise RuntimeError("sanitizer controller did not produce campaign.json")
        sanitizer = json.loads(sanitizer_report.read_text(encoding="utf-8"))
        if not sanitizer.get("validated"):
            raise RuntimeError("sanitizer controller report is not validated")
        after = contamination()

        report = {
            "schema_version": 1,
            "campaign": "CE-GEO",
            "task_id": "CE-GEO-109",
            "validated": True,
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source": {
                "commit": source_commit,
                "tree": source_tree,
                "owned_manifest_sha256": sha256_file(
                    ROOT / "tests/ce_geo/validation/owned_production_paths.json"
                ),
                "sanitizer_evidence_sha256": sha256_file(
                    ROOT / "bench/ce_geo/evidence/sanitizer/campaign.json"
                ),
            },
            "hardware_and_toolchain": metadata,
            "contamination": {"before": before, "after": after},
            "commands": records,
            "sanitizer": {
                "report": str(sanitizer_report),
                "sha256": sha256_file(sanitizer_report),
                "summary": sanitizer.get("summary"),
            },
            "coverage": {
                "normal_release_sm70_build": True,
                "torch_compatibility_build": True,
                "frozen_baseline": True,
                "portable_foundation": True,
                "sm70_vertical": True,
                "integrated_biology": True,
                "negative_contracts": True,
                "semantic_and_physical_properties": True,
                "numerical_referee_and_gradients": True,
                "sealed_hot_path": True,
                "static_contracts": True,
                "compute_sanitizer": True,
            },
            "summary": {
                "metadata_command_count": len(metadata),
                "validation_command_count": len(records),
                "passed_validation_command_count": sum(
                    record["returncode"] == 0 for record in records
                ),
            },
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=output.parent,
            prefix=output.name + ".", suffix=".tmp", delete=False,
        ) as stream:
            temporary = pathlib.Path(stream.name)
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
        temporary.replace(output)
        print(
            "CE-GEO full Volta acceptance passed: "
            f"commands={len(records)} output={output}"
        )
        return 0
    except (OSError, RuntimeError, subprocess.TimeoutExpired,
            json.JSONDecodeError) as error:
        print(f"CE-GEO full Volta acceptance failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
