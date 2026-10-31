#!/usr/bin/env python3
"""Trusted KDA adapter for the packed-embedding SM70 comparison."""
from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(os.environ.get("CELLERATOR_KDA_SOURCE_ROOT", Path(__file__).resolve().parents[2])).resolve()
KDA_DIR = ROOT / "planning/v100-framework-2026-10-29/kda/packed_embedding"
NVCC = Path("/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc")
SOURCE_FILES = (
    "src/modeling/cuda/packed_embedding.cu",
    "include/Cellerator/modeling/cuda/packed_embedding.hh",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path: str, value: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def invocation() -> tuple[str, dict, str]:
    if len(sys.argv) != 4:
        raise RuntimeError("expected OPERATION REQUEST RESULT")
    op, request_path, result_path = sys.argv[1:]
    request = read(request_path)
    return op, request, result_path


def assert_admitted() -> str:
    job = os.environ.get("DEVICE_SCHEDULER_JOB_ID")
    attempt = os.environ.get("DEVICE_SCHEDULER_ATTEMPT_ID")
    if not job or not attempt:
        raise RuntimeError("GPU evaluation requires scheduler job and attempt IDs from an admitted Autobench packet")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    entries = [item.strip() for item in visible.split(",") if item.strip()]
    if len(entries) != 1 or not entries[0].startswith("GPU-"):
        raise RuntimeError("admitted evaluation must be bound to exactly one scheduler-issued GPU UUID")
    return job


def compile_target_probe() -> None:
    result = subprocess.run([str(NVCC), "--list-gpu-code"], check=True, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if "sm_70" not in result.stdout.split():
        raise RuntimeError("selected nvcc does not report sm_70 support")
    sys.stdout.write(result.stdout)


def gpu_environment(request: dict) -> dict:
    job_id = assert_admitted()
    visible = os.environ["CUDA_VISIBLE_DEVICES"]
    attempt_id = os.environ["DEVICE_SCHEDULER_ATTEMPT_ID"]
    visible_uuid = visible.strip()

    query = subprocess.run([
        "nvidia-smi", "--query-gpu=uuid,name,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    selected = None
    for line in query.stdout.splitlines():
        columns = [part.strip() for part in line.split(",", 3)]
        if len(columns) == 4 and columns[0] == visible_uuid:
            selected = columns
            break
    if selected is None:
        raise RuntimeError(f"scheduler-bound UUID {visible_uuid!r} is absent from nvidia-smi inventory")
    library = ctypes.util.find_library("cudart")
    if not library:
        raise RuntimeError("CUDA runtime library was not discoverable")
    cudart = ctypes.CDLL(library)
    runtime_version = ctypes.c_int()
    if cudart.cudaRuntimeGetVersion(ctypes.byref(runtime_version)) != 0:
        raise RuntimeError("cudaRuntimeGetVersion failed")
    device_count = ctypes.c_int()
    if cudart.cudaGetDeviceCount(ctypes.byref(device_count)) != 0 or device_count.value != 1:
        raise RuntimeError("CUDA runtime does not expose exactly one scheduler-bound device")
    uuid_bytes = (ctypes.c_ubyte * 16)()
    cudart.cudaDeviceGetUuid.argtypes = [ctypes.c_void_p, ctypes.c_int]
    if cudart.cudaDeviceGetUuid(ctypes.byref(uuid_bytes), 0) != 0:
        raise RuntimeError("cudaDeviceGetUuid failed for visible CUDA device zero")
    import uuid
    device_uuid = "GPU-" + str(uuid.UUID(bytes=bytes(uuid_bytes))).lower()
    if device_uuid != visible_uuid:
        raise RuntimeError(f"CUDA visible device UUID {device_uuid!r} differs from scheduler binding {visible_uuid!r}")
    return {
        "kind": "cuda",
        "architecture": "sm_" + selected[3].replace(".", ""),
        "device_uuid": device_uuid,
        "device": {"name": selected[1], "uuid": device_uuid},
        "driver_version": selected[2],
        "runtime_version": str(runtime_version.value),
        "tools": {"nvcc": str(NVCC), "autobench_job_id": job_id,
                  "autobench_attempt_id": attempt_id,
                  "cuda_visible_devices": visible},
    }


def build(request: dict) -> list[dict[str, str]]:
    closure = read(str(KDA_DIR / "base-source-manifest.json"))
    for rel, expected in closure["files"].items():
        if digest(ROOT / rel) != expected:
            raise RuntimeError(f"shared build source changed after experiment setup: {rel}")
    candidate = request["candidate"]
    source_dir = Path(candidate["source_dir"]).resolve(strict=True)
    build_dir = Path(request["build"]["directory"]).resolve()
    for rel in SOURCE_FILES:
        if digest(source_dir / rel) != digest(ROOT / rel):
            raise RuntimeError(f"candidate snapshot differs from build source: {rel}")
    cmake_build = build_dir / "cmake"
    args = [
        "cmake", "-S", str(KDA_DIR), "-B", str(cmake_build),
        f"-DCELLERATOR_SOURCE_DIR={ROOT}",
        f"-DKDA_CANDIDATE_SOURCE_DIR={source_dir}",
        f"-DKDA_WARP_BACKWARD={'ON' if candidate['id'] == 'warp_backward' else 'OFF'}",
        "-DCMAKE_CXX_COMPILER=/usr/bin/x86_64-linux-gnu-g++-12",
        f"-DCMAKE_CUDA_COMPILER={NVCC}",
        "-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/x86_64-linux-gnu-g++-12",
        "-DCMAKE_CUDA_ARCHITECTURES=70", "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CUDA_FLAGS=-lineinfo",
    ]
    subprocess.run(args, check=True)
    subprocess.run(["cmake", "--build", str(cmake_build), "--parallel", "4",
                    "--target", "cellerator_kda_packed_embedding_smoke"], check=True)
    for rel, expected in closure["files"].items():
        if digest(ROOT / rel) != expected:
            raise RuntimeError(f"shared build source changed during candidate build: {rel}")
    binary = cmake_build / "cellerator_kda_packed_embedding_smoke"
    metadata = build_dir / "kda-build-metadata.json"
    metadata.write_text(json.dumps({
        "candidate_id": candidate["id"], "source_snapshot_sha256": candidate["snapshot_sha256"],
        "binary_sha256": digest(binary), "architecture": "sm_70",
        "warp_backward": candidate["id"] == "warp_backward",
        "common_source_root": str(ROOT), "source_files": {rel: digest(ROOT / rel) for rel in SOURCE_FILES},
        "trusted_smoke_sha256": digest(ROOT / "tests/framework/cuda/packed_embedding_smoke.cu"),
        "build_argv": args,
    }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return [
        {"path": "cmake/cellerator_kda_packed_embedding_smoke", "sha256": digest(binary)},
        {"path": metadata.name, "sha256": digest(metadata)},
    ]


def binary_for(request: dict) -> Path:
    return Path(request["candidate"]["build"]["directory"]) / "cmake/cellerator_kda_packed_embedding_smoke"


def validate(request: dict) -> list[dict]:
    assert_admitted()
    candidate = request["candidate"]
    # The trusted smoke's --performance path also runs its CPU oracle over the
    # exact timed D=128/16,384-occurrence fixture before it emits that timing.
    process = subprocess.run([str(binary_for(request)), "--performance"], check=True, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if "packed embedding oracle smoke passed" not in process.stdout:
        raise RuntimeError("independent packed-embedding oracle smoke did not report completion")
    if len([line for line in process.stdout.splitlines()
            if "occurrences=16384" in line and "dim=128" in line and "dtype=fp32" in line]) != 1:
        raise RuntimeError("trusted correctness smoke did not validate the exact timed fixture")
    return [{"case_id": case["id"], "workload_sha256": case["workload_sha256"], "status": "pass"}
            for case in request["cases"]]


def benchmark(request: dict) -> list[float]:
    assert_admitted()
    process = subprocess.run([str(binary_for(request)), "--performance"], check=True, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = [line for line in process.stdout.splitlines()
             if "occurrences=16384" in line and "dim=128" in line and "dtype=fp32" in line]
    if len(lines) != 1:
        raise RuntimeError("expected exactly one resident 16,384-occurrence FP32 timing row")
    match = re.search(r"backward_ms=([0-9.eE+-]+)", lines[0])
    if not match:
        raise RuntimeError("packed-embedding timing row omitted backward_ms")
    value = float(match.group(1))
    if not value > 0:
        raise RuntimeError("packed-embedding backward measurement is not positive")
    return [value]


def main() -> int:
    op, request, result_path = invocation()
    envelope = {"schema_version": 1, "request_id": request["request_id"], "status": "ok"}
    if op == "compiler-target":
        compile_target_probe()
        return 0
    if op == "probe":
        envelope["environment"] = gpu_environment(request)
    elif op == "build":
        envelope["artifacts"] = build(request)
    elif op == "validate":
        envelope["cases"] = validate(request)
    elif op == "benchmark":
        case, timing = request["case"], request["timing"]
        envelope.update({
            "case_id": case["id"], "workload_sha256": case["workload_sha256"],
            "metric": timing["metric"], "unit": timing["unit"],
            "timing_scope": timing["scope"], "warmup": timing["warmup"],
            "repetitions": timing["repetitions"], "samples": benchmark(request),
        })
    else:
        raise RuntimeError(f"unknown operation: {op}")
    write(result_path, envelope)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        request_path = sys.argv[2] if len(sys.argv) > 2 else ""
        result_path = sys.argv[3] if len(sys.argv) > 3 else ""
        request_id = None
        try:
            request_id = read(request_path).get("request_id")
        except Exception:
            pass
        if result_path and request_id:
            write(result_path, {"schema_version": 1, "request_id": request_id,
                                "status": "error", "error": f"{type(exc).__name__}: {exc}"})
        print(f"KDA packed-embedding adapter error: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
