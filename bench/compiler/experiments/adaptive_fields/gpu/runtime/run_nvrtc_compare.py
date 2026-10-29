#!/usr/bin/env python3
"""Fail-closed scheduler entrypoint for the E7 NVRTC comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


HERE = Path(__file__).resolve().parent
DEFAULT_CUDA = Path("/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=HERE / "artifacts/nvcc_artifact_manifest.json")
    parser.add_argument("--output", type=Path, default=HERE / "artifacts/nvrtc_run_result.json")
    parser.add_argument("--cuda-root", type=Path, default=DEFAULT_CUDA)
    args = parser.parse_args()
    if os.environ.get("CELLERATOR_SPARSE_SUPPORT_BENCH_ADMITTED") != "1":
        raise RuntimeError("GPU execution requires the parent-owned scheduler admission")
    visible_device = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible_device.startswith(("GPU-", "MIG-")) or "," in visible_device:
        raise RuntimeError("exactly one UUID-routed CUDA_VISIBLE_DEVICES value is required")
    run_key = os.environ.get("ADAPTIVE_FIELDS_RUN_KEY")
    if not run_key:
        raise RuntimeError("parent JobSpec correlation key is required")

    manifest = json.loads(args.manifest.read_text())
    source = HERE / "region.cu"
    driver_source = HERE / "nvrtc_compare.cu"
    fatbin = Path(manifest["artifact"])
    executable = Path(manifest["runner_executable"])
    for label, path, expected in (
        ("source", source, manifest["source_sha256"]),
        ("driver source", driver_source, manifest["driver_source_sha256"]),
        ("NVCC artifact", fatbin, manifest["artifact_sha256"]),
        ("runner executable", executable, manifest["runner_sha256"]),
        ("launch wrapper", Path(__file__).resolve(), manifest["launch_wrapper_sha256"]),
    ):
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"{label} hash changed: {actual} != {expected}")

    cuda_root = args.cuda_root.resolve(strict=True)
    nvrtc_library = cuda_root / "targets/x86_64-linux/lib/libnvrtc.so"
    if not nvrtc_library.is_file():
        raise RuntimeError(f"NVRTC shared library is missing: {nvrtc_library}")
    environment = os.environ.copy()
    libdir = str(nvrtc_library.parent)
    environment["LD_LIBRARY_PATH"] = libdir + os.pathsep + environment.get("LD_LIBRARY_PATH", "")
    temporary_output = args.output.with_suffix(".runner.json")
    command = [str(executable), str(source), str(fatbin),
               str(temporary_output), "compute_70"]
    process = subprocess.run(command, text=True, capture_output=True,
                             env=environment, check=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if process.returncode:
        failure = {
            "schema": "cellerator.adaptive_fields.e7_nvrtc_run_failure.v1",
            "returncode": process.returncode,
            "command": command,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "scheduler_correlation_key": run_key,
            "visible_device_uuid": visible_device,
            "scheduler_authority": "external JobSpec, admission, terminal and cleanup receipts",
        }
        args.output.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        return process.returncode
    result = json.loads(temporary_output.read_text())
    result.update({
        "command": command,
        "manifest_sha256": sha256(args.manifest),
        "region_source_sha256": sha256(source),
        "nvcc_fatbin_sha256": sha256(fatbin),
        "runner_sha256": sha256(executable),
        "launch_wrapper_sha256": sha256(Path(__file__).resolve()),
        "nvrtc_library": str(nvrtc_library.resolve()),
        "nvrtc_library_sha256": sha256(nvrtc_library),
        "scheduler": {
            "correlation_key": run_key,
            "visible_device_uuid": visible_device,
            "authority": "external JobSpec, admission, terminal and cleanup receipts",
            "marker_role": "accidental-run guard only; not authorization or isolation",
        },
        "platform": platform.platform(),
        "python": sys.version,
        "nvcc_preparation_compile_ms": manifest["compile_wall_ms"],
        "nvcc_preparation_command": manifest["arguments"],
        "precompiled_execution_baseline": "NVCC fatbinary loaded in the same driver context as NVRTC PTX",
    })
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary_output.unlink(missing_ok=True)
    print(json.dumps({"output": str(args.output),
                      "nvrtc_compile_ms": result["cold_nvrtc_compile_ms"],
                      "paired_count": sum(len(row) for row in result["samples"])},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
