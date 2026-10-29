#!/usr/bin/env python3
"""Scheduler-only launcher and source identity wrapper for E6 lifetime tests."""
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


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=HERE / "artifacts/build_manifest.json")
    parser.add_argument("--output", type=Path, default=HERE / "artifacts/lifetime_result.json")
    parser.add_argument("--cuda-root", type=Path, default=DEFAULT_CUDA)
    args = parser.parse_args()
    if os.environ.get("CELLERATOR_SPARSE_SUPPORT_BENCH_ADMITTED") != "1":
        raise RuntimeError("GPU execution requires root-owned scheduler admission")
    visible_device = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible_device.startswith(("GPU-", "MIG-")) or "," in visible_device:
        raise RuntimeError("exactly one UUID-routed CUDA_VISIBLE_DEVICES value is required")
    run_key = os.environ.get("ADAPTIVE_FIELDS_RUN_KEY")
    if not run_key:
        raise RuntimeError("parent JobSpec correlation key is required")

    manifest = json.loads(args.manifest.read_text())
    runner = Path(manifest["runner"])
    source = HERE / "lifetime_runner.cu"
    kernel_source = HERE / "lifetime_kernels.cu"
    for label, path, expected in (
        ("runner", runner, manifest["runner_sha256"]),
        ("driver source", source, manifest["driver_source_sha256"]),
        ("kernel source", kernel_source, manifest["kernel_source_sha256"]),
        ("launch wrapper", Path(__file__).resolve(), manifest["launch_wrapper_sha256"]),
        ("sanitizer plan", HERE / "sanitizer_commands.json", manifest["sanitizer_plan_sha256"]),
    ):
        actual = digest(path)
        if actual != expected:
            raise RuntimeError(f"{label} source or binary changed: {actual} != {expected}")
    cuda_root = args.cuda_root.resolve(strict=True)
    libdir = cuda_root / "targets/x86_64-linux/lib"
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = str(libdir) + os.pathsep + environment.get("LD_LIBRARY_PATH", "")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = args.output.with_suffix(".runner.json")
    command = [str(runner), str(kernel_source), str(raw_output)]
    proc = subprocess.run(command, text=True, capture_output=True,
                          env=environment, check=False)
    if proc.returncode:
        failure = {
            "schema": "cellerator.adaptive_fields.e6_gpu_lifetime_failure.v1",
            "returncode": proc.returncode,
            "command": command,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "scheduler": {"correlation_key": run_key,
                          "visible_device_uuid": visible_device,
                          "authority": "external JobSpec/admission/terminal/cleanup receipts"},
        }
        args.output.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        return proc.returncode
    result = json.loads(raw_output.read_text())
    result.update({
        "command": command,
        "manifest_sha256": digest(args.manifest),
        "runner_sha256": digest(runner),
        "driver_source_sha256": digest(source),
        "kernel_source_sha256": digest(kernel_source),
        "launch_wrapper_sha256": digest(Path(__file__).resolve()),
        "sanitizer_plan_sha256": digest(HERE / "sanitizer_commands.json"),
        "nvrtc_library": manifest["nvrtc_library"],
        "nvrtc_library_sha256": digest(Path(manifest["nvrtc_library"])),
        "scheduler": {"correlation_key": run_key,
                      "visible_device_uuid": visible_device,
                      "authority": "external JobSpec, admission, terminal and cleanup receipts",
                      "marker_role": "accidental-run guard only; not authorization or isolation"},
        "platform": platform.platform(),
        "python": sys.version,
    })
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    raw_output.unlink(missing_ok=True)
    print(json.dumps({"output": str(args.output),
                      "checks": ["replacement", "tape", "failure", "cancel",
                                 "dedupe", "stale", "value_compatible"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
