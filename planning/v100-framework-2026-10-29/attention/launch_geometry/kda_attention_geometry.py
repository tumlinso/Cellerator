#!/usr/bin/env python3
"""Trusted build, correctness, and timing adapter for attention launch geometry."""
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


ROOT = Path(os.environ.get("CELLERATOR_KDA_SOURCE_ROOT", Path(__file__).resolve().parents[4])).resolve()
KDA_DIR = ROOT / "planning/v100-framework-2026-10-29/attention/launch_geometry"
NVCC = Path("/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc")
GCC = Path("/usr/bin/x86_64-linux-gnu-g++-12")
SOURCE_FILE = "src/modeling/cuda/attention.cu"
SMOKE_FILE = "tests/framework/cuda/attention_smoke.cu"
DRIVER_FILE = "planning/v100-framework-2026-10-29/attention/launch_geometry/attention_geometry_driver.cu"
TASK_SOURCE_MANIFEST = "task-source-manifest.json"


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
    operation, request_path, result_path = sys.argv[1:]
    return operation, read(request_path), result_path


def assert_admitted() -> str:
    job = os.environ.get("DEVICE_SCHEDULER_JOB_ID")
    attempt = os.environ.get("DEVICE_SCHEDULER_ATTEMPT_ID")
    if not job or not attempt:
        raise RuntimeError("GPU evaluation requires scheduler and attempt IDs from an admitted Autobench packet")
    visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if len(visible) != 1 or not visible[0].startswith("GPU-"):
        raise RuntimeError("admitted evaluation must bind exactly one scheduler-issued GPU UUID")
    return job


def compiler_target_probe() -> None:
    result = subprocess.run([str(NVCC), "--list-gpu-code"], check=True, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if "sm_70" not in result.stdout.split():
        raise RuntimeError("selected nvcc does not report sm_70 support")
    sys.stdout.write(result.stdout)


def gpu_environment() -> dict:
    job_id = assert_admitted()
    visible = os.environ["CUDA_VISIBLE_DEVICES"].strip()
    attempt_id = os.environ["DEVICE_SCHEDULER_ATTEMPT_ID"]
    query = subprocess.run([
        "nvidia-smi", "--query-gpu=uuid,name,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    selected = None
    for line in query.stdout.splitlines():
        columns = [part.strip() for part in line.split(",", 3)]
        if len(columns) == 4 and columns[0] == visible:
            selected = columns
            break
    if selected is None:
        raise RuntimeError(f"scheduler-bound UUID {visible!r} is absent from nvidia-smi inventory")
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
    if device_uuid != visible:
        raise RuntimeError(f"CUDA visible UUID {device_uuid!r} differs from scheduler binding {visible!r}")
    return {
        "kind": "cuda", "architecture": "sm_" + selected[3].replace(".", ""),
        "device_uuid": device_uuid, "device": {"name": selected[1], "uuid": device_uuid},
        "driver_version": selected[2], "runtime_version": str(runtime_version.value),
        "tools": {"nvcc": str(NVCC), "autobench_job_id": job_id,
                  "autobench_attempt_id": attempt_id, "cuda_visible_devices": visible},
    }


def assert_frozen_sources() -> dict:
    task_manifest = read(str(KDA_DIR / TASK_SOURCE_MANIFEST))
    actual_root = Path(task_manifest["source_root"]).resolve(strict=True)
    for relative, expected in task_manifest["files"].items():
        if digest(actual_root / relative) != expected or digest(ROOT / relative) != expected:
            raise RuntimeError(f"shared source differs from the frozen attention task snapshot: {relative}")
    baseline_manifest = read(str(KDA_DIR / "base-source-manifest.json"))
    for relative, expected in baseline_manifest["files"].items():
        if digest(actual_root / relative) != expected or digest(ROOT / relative) != expected:
            raise RuntimeError(f"common V100 source closure differs from its frozen manifest: {relative}")
    return baseline_manifest


def build(request: dict) -> list[dict[str, str]]:
    baseline_manifest = assert_frozen_sources()
    candidate = request["candidate"]
    candidate_id = candidate["id"]
    if candidate_id not in {"baseline", "one_cta_per_row"}:
        raise RuntimeError(f"unexpected candidate id: {candidate_id}")
    candidate_dir = Path(candidate["source_dir"]).resolve(strict=True)
    reference = KDA_DIR / "candidate-sources" / candidate_id / SOURCE_FILE
    if digest(candidate_dir / SOURCE_FILE) != digest(reference):
        raise RuntimeError("registered attention source differs from the declared frozen candidate")
    if candidate_id == "baseline" and digest(candidate_dir / SOURCE_FILE) != baseline_manifest["files"][SOURCE_FILE]:
        raise RuntimeError("baseline attention source differs from the common source manifest")

    build_dir = Path(request["build"]["directory"]).resolve()
    cmake_build = build_dir / "cmake"
    args = [
        "cmake", "-S", str(KDA_DIR), "-B", str(cmake_build),
        f"-DCELLERATOR_SOURCE_DIR={ROOT}", f"-DKDA_CANDIDATE_SOURCE_DIR={candidate_dir}",
        f"-DCMAKE_CXX_COMPILER={GCC}", f"-DCMAKE_CUDA_COMPILER={NVCC}",
        f"-DCMAKE_CUDA_HOST_COMPILER={GCC}", "-DCMAKE_CUDA_ARCHITECTURES=70",
        "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_FLAGS=-lineinfo",
    ]
    subprocess.run(args, check=True)
    subprocess.run([
        "cmake", "--build", str(cmake_build), "--parallel", "1", "--target",
        "cellerator_kda_attention_smoke", "cellerator_kda_attention_geometry_driver",
    ], check=True)
    assert_frozen_sources()
    artifacts = []
    for relative in ("cellerator_kda_attention_smoke", "cellerator_kda_attention_geometry_driver"):
        path = cmake_build / relative
        artifacts.append({"path": f"cmake/{relative}", "sha256": digest(path)})
    metadata = build_dir / "kda-build-metadata.json"
    metadata.write_text(json.dumps({
        "candidate_id": candidate_id, "candidate_snapshot_sha256": candidate["snapshot_sha256"],
        "attention_source_sha256": digest(candidate_dir / SOURCE_FILE),
        "binary_sha256": {entry["path"]: entry["sha256"] for entry in artifacts},
        "architecture": "sm_70", "common_source_root": str(ROOT),
        "task_source_manifest_sha256": digest(KDA_DIR / TASK_SOURCE_MANIFEST),
        "baseline_source_manifest_sha256": digest(KDA_DIR / "base-source-manifest.json"),
        "trusted_smoke_sha256": digest(ROOT / SMOKE_FILE),
        "trusted_driver_sha256": digest(KDA_DIR / "attention_geometry_driver.cu"),
        "build_argv": args,
    }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    artifacts.append({"path": metadata.name, "sha256": digest(metadata)})
    return artifacts


def binary_for(request: dict, name: str) -> Path:
    return Path(request["candidate"]["build"]["directory"]) / f"cmake/{name}"


def validate(request: dict) -> list[dict]:
    assert_admitted()
    smoke = subprocess.run([str(binary_for(request, "cellerator_kda_attention_smoke")), "--device"],
                           check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    required = (
        "FP32 staged forward and VJP passed", "FP32 fused forward and VJP passed",
        "FP16 staged forward and VJP passed", "FP16 fused forward and VJP passed",
        "fused block-stride case passed (Q=65536; degree=1; V=1)",
    )
    if any(marker not in smoke.stdout for marker in required):
        raise RuntimeError("full trusted attention smoke did not report every FP16/FP32 route and block-stride result")
    sys.stdout.write(smoke.stdout)
    fixture = subprocess.run([
        str(binary_for(request, "cellerator_kda_attention_geometry_driver")), "--validate",
    ], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if "attention geometry independent FP64 forward oracle passed" not in fixture.stdout:
        raise RuntimeError("exact-timing independent FP64 oracle did not report completion")
    sys.stdout.write(fixture.stdout)
    return [{"case_id": case["id"], "workload_sha256": case["workload_sha256"], "status": "pass"}
            for case in request["cases"]]


def benchmark(request: dict) -> list[float]:
    assert_admitted()
    case_id = request["case"]["id"]
    process = subprocess.run([
        str(binary_for(request, "cellerator_kda_attention_geometry_driver")),
        "--benchmark", case_id,
    ], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    sys.stdout.write(process.stdout)
    rows = [line for line in process.stdout.splitlines() if line.startswith(f"case={case_id} samples_us=")]
    if len(rows) != 1:
        raise RuntimeError("expected exactly one exact-fixture event timing row")
    match = re.search(
        r"samples_us=([0-9.,eE+-]+)\s+warmup_per_sample=4\s+iterations_per_sample=25\s+dtype=fp32 Q=512 N=96 D=32 V=32 edges=(4096|5920)",
        rows[0])
    if not match:
        raise RuntimeError("timing output does not match the declared FP32 Q=512 fixture and sampling contract")
    values = [float(item) / 1000.0 for item in match.group(1).split(",")]
    if len(values) != 7 or any(not (value > 0 and value < float("inf")) for value in values):
        raise RuntimeError("expected seven positive finite CUDA-event samples")
    return values


def main() -> int:
    operation, request, result_path = invocation()
    envelope = {"schema_version": 1, "request_id": request["request_id"], "status": "ok"}
    if operation == "compiler-target":
        compiler_target_probe()
        return 0
    if operation == "probe":
        envelope["environment"] = gpu_environment()
    elif operation == "build":
        envelope["artifacts"] = build(request)
    elif operation == "validate":
        envelope["cases"] = validate(request)
    elif operation == "benchmark":
        case, timing = request["case"], request["timing"]
        envelope.update({
            "case_id": case["id"], "workload_sha256": case["workload_sha256"],
            "metric": timing["metric"], "unit": timing["unit"],
            "timing_scope": timing["scope"], "warmup": timing["warmup"],
            "repetitions": timing["repetitions"], "samples": benchmark(request),
        })
    else:
        raise RuntimeError(f"unknown operation: {operation}")
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
        print(f"KDA attention geometry adapter error: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
