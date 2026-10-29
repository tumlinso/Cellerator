#!/usr/bin/env python3
"""KDA task adapter for the frozen sparse-support backend comparison.

KDA candidate snapshots contain a runtime backend selector (SIMT or CE-GEO).
The trusted benchmark driver runs both paths in paired order for each request;
this adapter returns the selected candidate's raw training-step samples to the
KDA engine. The candidate code never controls references, fixtures, validation,
or timing policy.
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
import traceback


REPO = Path(__file__).resolve().parents[2]
DRIVER = REPO / "bench/sparse_support/packed_training_benchmark.py"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_mode(candidate: dict, build_directory: str | None = None) -> str:
    source = Path(candidate["source_dir"]).resolve(strict=True)
    source_mode = read_json(source / "backend.json").get("backend")
    if source_mode not in {"simt", "packed"} or source_mode != candidate.get("id"):
        raise ValueError("candidate ID and backend.json mode do not agree")
    build = candidate.get("build", {})
    directory = build.get("directory", build_directory)
    if directory is not None:
        artifact = Path(directory).resolve(strict=True) / "backend.json"
        if not artifact.is_file() or read_json(artifact).get("backend") != source_mode:
            raise ValueError("prepared build backend selector differs from candidate snapshot")
    return source_mode


def run_driver(request_path: Path, *, case: str, warmup: int, rounds: int,
               iterations: int, setup_pairs: int, seed: int,
               fixture_root: str | None = None) -> dict:
    output_dir = request_path.parent / (request_path.stem + f"-{case}-driver")
    command = [sys.executable, str(DRIVER), "--output-dir", str(output_dir),
               "--case", case, "--warmup", str(warmup),
               "--rounds", str(rounds), "--iterations", str(iterations),
               "--setup-pairs", str(setup_pairs),
               "--seed", str(seed)]
    if fixture_root:
        command.extend(["--fixture-root", fixture_root])
    proc = subprocess.run(command, cwd=REPO, text=True, capture_output=True,
                          env=os.environ.copy(), check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"benchmark driver failed ({proc.returncode})\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    result = read_json(output_dir / "results.json")
    config = read_json(output_dir / "run_config.json")
    return {"results": result, "config": config,
            "output_dir": str(output_dir), "stdout": proc.stdout.strip()}


def op_probe(request: dict) -> dict:
    import torch
    from cellerator.torch import _torch

    if os.environ.get("CELLERATOR_SPARSE_SUPPORT_BENCH_ADMITTED") != "1":
        raise RuntimeError("KDA probe must execute inside the admitted Device Scheduler job")
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    extension = Path(_torch.__file__).resolve()
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True,
    )
    gpu_rows = []
    for line in smi.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", maxsplit=1)]
        if len(fields) == 2:
            gpu_rows.append({"uuid": fields[0], "driver_version": fields[1]})
    visible = [part.strip() for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
               if part.strip()]
    visible_token = visible[device] if device < len(visible) else None
    property_uuid = getattr(props, "uuid", None)
    selected_uuid = None
    if visible_token and visible_token.startswith(("GPU-", "MIG-")):
        # Device Scheduler binds the job to the granted UUID through this
        # variable. Prefer it over CUDA's optional DeviceProperties UUID,
        # whose formatting differs across PyTorch/CUDA builds.
        selected_uuid = visible_token
    elif property_uuid:
        selected_uuid = str(property_uuid)
    elif visible_token and visible_token.isdigit() and int(visible_token) < len(gpu_rows):
        selected_uuid = gpu_rows[int(visible_token)]["uuid"]
    if selected_uuid is None and not visible and len(gpu_rows) == 1:
        selected_uuid = gpu_rows[0]["uuid"]
    selected_row = next((row for row in gpu_rows if row["uuid"] == selected_uuid), None)
    if selected_row is None:
        raise RuntimeError(
            "could not identify scheduled CUDA device UUID: "
            f"visible_token={visible_token!r}, property_uuid={property_uuid!r}, "
            f"nvidia_smi_uuids={[row['uuid'] for row in gpu_rows]!r}"
        )
    runtime_version = str(torch.version.cuda or "")
    if not runtime_version:
        raise RuntimeError("PyTorch did not report its CUDA runtime version")
    environment = {
        "kind": "cuda",
        "architecture": f"sm_{props.major}{props.minor}",
        "compute_capability": f"{props.major}.{props.minor}",
        "device_uuid": selected_row["uuid"],
        "driver_version": selected_row["driver_version"],
        "runtime_version": runtime_version,
        "device": {"index": device, "name": props.name,
                   "total_memory_bytes": props.total_memory,
                   "uuid": selected_row["uuid"]},
        "tools": {
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "torch_cuda_runtime": runtime_version,
            "cellerator_torch_extension_sha256": sha256_file(extension),
        },
    }
    return {
        "schema_version": 1, "request_id": request["request_id"], "status": "ok",
        "environment": environment,
        "python": sys.executable, "python_prefix": sys.prefix,
        "torch_extension": str(extension),
    }


def op_build(request: dict) -> dict:
    candidate = request["candidate"]
    mode = candidate_mode(candidate)
    source = Path(candidate["source_dir"]).resolve(strict=True) / "backend.json"
    build_dir = Path(request["build"]["directory"]).resolve(strict=True)
    target = build_dir / "backend.json"
    shutil.copyfile(source, target)
    if read_json(target).get("backend") != mode:
        raise RuntimeError("candidate selector changed while preparing its frozen build artifact")
    return {
        "schema_version": 1, "request_id": request["request_id"], "status": "ok",
        "artifacts": [{"path": "backend.json", "sha256": sha256_file(target)}],
        "backend": mode, "build_kind": "frozen_binary_runtime_selector",
    }


def checks_pass(case_result: dict) -> bool:
    checks = case_result["checks"]
    return bool(
        checks["inference_output_allclose_atol_5e-3_rtol_5e-3"]
        and checks["training_parameter_gradient_allclose_atol_6e-3_rtol_6e-3"]
        and checks["loss_close_abs_5e-4_rel_5e-3"]
        and all(value > 0 for value in
                checks["derivative_only_probability_grad_nonzero"].values())
        and checks["remaining_read_tickets"] == 0)


def op_validate(request: dict, request_path: Path) -> dict:
    if os.environ.get("CELLERATOR_SPARSE_SUPPORT_BENCH_ADMITTED") != "1":
        raise RuntimeError("KDA validation must execute inside the admitted Device Scheduler job")
    mode = candidate_mode(request["candidate"])
    rows = []
    for case in request["cases"]:
        result = run_driver(request_path, case=case["id"], warmup=1,
                            rounds=1, iterations=1,
                            setup_pairs=1,
                            seed=int(case["input"]["seed"]),
                            fixture_root=case["input"].get("fixture_root"))
        case_result = result["results"][case["id"]]
        rows.append({
            "case_id": case["id"], "workload_sha256": case["workload_sha256"],
            "status": "pass" if checks_pass(case_result) else "fail",
            "backend_candidate": mode,
            "workset_digest": case_result["case"]["workset_digest"],
            "frozen_input_digest": case_result["case"]["frozen_input_digest"],
            "derivative_only_probability_grad_nonzero":
                case_result["checks"]["derivative_only_probability_grad_nonzero"],
        })
    status = "ok" if all(row["status"] == "pass" for row in rows) else "error"
    return {"schema_version": 1, "request_id": request["request_id"],
            "status": status, "cases": rows}


def op_benchmark(request: dict, request_path: Path) -> dict:
    if os.environ.get("CELLERATOR_SPARSE_SUPPORT_BENCH_ADMITTED") != "1":
        raise RuntimeError("KDA benchmark must execute inside the admitted Device Scheduler job")
    mode = candidate_mode(request["candidate"])
    case = request["case"]
    timing = request["timing"]
    result = run_driver(
        request_path, case=case["id"], warmup=int(timing["warmup"]),
        rounds=int(timing["repetitions"]),
        iterations=int(timing.get("iterations", 2)),
        setup_pairs=int(timing.get("setup_pairs", 5)),
        seed=int(case["input"]["seed"]),
        fixture_root=case["input"].get("fixture_root"))
    case_result = result["results"][case["id"]]
    if not checks_pass(case_result):
        raise RuntimeError("benchmark numerical checks did not pass")
    samples = [float(item["synchronized_wall_ms_per_step"])
               for item in case_result["phases"]["training"][mode]["raw_samples"]]
    if len(samples) != int(timing["repetitions"]):
        raise RuntimeError("selected backend returned the wrong sample count")
    return {
        "schema_version": 1, "request_id": request["request_id"], "status": "ok",
        "case_id": case["id"], "workload_sha256": case["workload_sha256"],
        "timing_scope": timing["scope"], "warmup": timing["warmup"],
        "repetitions": timing["repetitions"], "metric": timing["metric"],
        "unit": timing["unit"], "samples": samples,
        "backend_candidate": mode,
        "diagnostic_paired_ratios": case_result["phases"]["training"]["paired"],
        "driver_output_dir": result["output_dir"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("probe", "build", "validate", "benchmark"))
    parser.add_argument("request", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    request = read_json(args.request)
    try:
        if args.operation == "probe":
            response = op_probe(request)
        elif args.operation == "build":
            response = op_build(request)
        elif args.operation == "validate":
            response = op_validate(request, args.request.resolve())
        else:
            response = op_benchmark(request, args.request.resolve())
    except Exception as exc:
        response = {"schema_version": 1,
                    "request_id": request.get("request_id"),
                    "status": "error", "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()}
        write_json(args.result, response)
        print(json.dumps(response, sort_keys=True))
        return 1
    write_json(args.result, response)
    print(json.dumps({"request_id": response["request_id"],
                      "status": response["status"], "operation": args.operation},
                     sort_keys=True))
    return 0 if response.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
