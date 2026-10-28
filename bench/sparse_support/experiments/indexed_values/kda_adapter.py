#!/usr/bin/env python3
"""Trusted KDA adapter for the isolated indexed edge-value comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent
DRIVER = HERE / "probe.py"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def strategy(request, *, verify_build=True):
    source = Path(request["candidate"]["source_dir"]).resolve(strict=True)
    selected = read_json(source / "backend.json").get("strategy")
    if selected not in ("full_E", "indexed_U"):
        raise ValueError("candidate backend.json has unknown strategy")
    build = request.get("build", {}).get("directory")
    if build and verify_build:
        staged = read_json(Path(build) / "backend.json").get("strategy")
        if staged != selected:
            raise ValueError("prepared artifact does not match selected strategy")
    return selected


def run_probe(request_path, request, *, validation=False):
    timing = request.get("timing", {})
    case = request.get("case") or request.get("cases", [])[0]
    output_dir = Path(request_path).parent / (
        Path(request_path).stem + "-indexed-values-" + case["id"])
    output = output_dir / "probe.json"
    package_root = os.environ.get("CELLERATOR_SPARSE_PACKAGE_ROOT")
    if not package_root:
        raise RuntimeError("CELLERATOR_SPARSE_PACKAGE_ROOT must identify frozen Torch v12 validator")
    command = [sys.executable, str(DRIVER), "--package-root", package_root,
               "--output", str(output), "--cases", case["id"],
               "--seed", str(int(case["input"].get("seed", 20261027))),
               "--warmup", str(1 if validation else int(timing.get("warmup", 2))),
               "--repetitions", str(1 if validation else int(timing.get("repetitions", 5)))]
    fixture_root = case.get("input", {}).get("fixture_root")
    if fixture_root:
        command.extend(["--fixture-root", fixture_root])
    env = os.environ.copy()
    env["CELLERATOR_SPARSE_SUPPORT_BENCH_ADMITTED"] = "1"
    proc = subprocess.run(command, cwd=HERE.parents[3], env=env, text=True,
                          capture_output=True, check=False)
    if proc.returncode:
        raise RuntimeError(f"probe failed ({proc.returncode})\n{proc.stdout}\n{proc.stderr}")
    return read_json(output), output_dir


def probe(request):
    import torch
    from cellerator.torch import _torch

    if os.environ.get("CELLERATOR_SPARSE_SUPPORT_BENCH_ADMITTED") != "1":
        raise RuntimeError("KDA probe requires scheduler admission")
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    inventory = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True)
    gpu_rows = []
    for line in inventory.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", maxsplit=1)]
        if len(fields) == 2:
            gpu_rows.append({"uuid": fields[0], "driver_version": fields[1]})
    uuids = [row["uuid"] for row in gpu_rows]
    visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
               if item.strip()]
    visible_token = visible[device] if device < len(visible) else None
    props_uuid = getattr(props, "uuid", None)
    if visible_token and visible_token.startswith(("GPU-", "MIG-")):
        device_uuid = visible_token
    elif props_uuid:
        device_uuid = str(props_uuid)
    elif visible_token and visible_token.isdigit() and int(visible_token) < len(uuids):
        device_uuid = uuids[int(visible_token)]
    elif not visible and len(uuids) == 1:
        device_uuid = uuids[0]
    else:
        raise RuntimeError("could not determine scheduler-admitted GPU UUID")
    selected_row = next((row for row in gpu_rows if row["uuid"] == device_uuid), None)
    if selected_row is None:
        raise RuntimeError("scheduler-admitted GPU UUID is absent from driver inventory")
    return {
        "schema_version": 1, "request_id": request["request_id"], "status": "ok",
        "environment": {
            "kind": "cuda", "architecture": f"sm_{props.major}{props.minor}",
            "compute_capability": f"{props.major}.{props.minor}",
            "device_uuid": device_uuid,
            "driver_version": selected_row["driver_version"],
            "runtime_version": str(torch.version.cuda),
            "device": {"index": device, "name": props.name,
                       "total_memory_bytes": props.total_memory,
                       "uuid": device_uuid},
            "tools": {"python": sys.version.split()[0], "torch": str(torch.__version__),
                      "extension_sha256": sha256(_torch.__file__)},
        },
    }


def build(request):
    mode = strategy(request, verify_build=False)
    source = Path(request["candidate"]["source_dir"]).resolve(strict=True) / "backend.json"
    directory = Path(request["build"]["directory"]).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "backend.json"
    shutil.copyfile(source, target)
    return {"schema_version": 1, "request_id": request["request_id"],
            "status": "ok", "backend": mode,
            "artifacts": [{"path": "backend.json", "sha256": sha256(target)}]}


def validate(request, request_path):
    mode = strategy(request)
    rows = []
    for case_input in request["cases"]:
        case_request = dict(request)
        case_request["case"] = case_input
        result, _ = run_probe(request_path, case_request, validation=True)
        case = result["cases"][0]
        passed = bool(case["correctness"]["matched_full_torch_vjp"]
                      and case["correctness"]["parameter_vjp_max_abs"] <= 2e-5
                      and case["correctness"]["output_max_abs"] <= 2e-5)
        rows.append({"case_id": case_input["id"],
                     "workload_sha256": case_input["workload_sha256"],
                     "status": "pass" if passed else "fail",
                     "candidate_strategy": mode,
                     "workset_digest": case["workset_digest"]})
    passed_all = all(row["status"] == "pass" for row in rows)
    return {"schema_version": 1, "request_id": request["request_id"],
            "status": "ok" if passed_all else "error", "cases": rows}


def benchmark(request, request_path):
    mode = strategy(request)
    result, output_dir = run_probe(request_path, request)
    case_result = result["cases"][0]
    timing = case_result["complete_step"]["full_E" if mode == "full_E" else "indexed_U"]
    expected = int(request["timing"]["repetitions"])
    samples = timing["samples_ms"]
    if len(samples) != expected:
        raise RuntimeError("probe returned a different sample count than requested")
    return {
        "schema_version": 1, "request_id": request["request_id"], "status": "ok",
        "case_id": request["case"]["id"],
        "workload_sha256": request["case"]["workload_sha256"],
        "timing_scope": request["timing"]["scope"],
        "warmup": request["timing"]["warmup"], "repetitions": expected,
        "metric": request["timing"]["metric"], "unit": request["timing"]["unit"],
        "samples": samples, "candidate_strategy": mode,
        "paired_diagnostic": case_result["complete_step"],
        "probe_output_dir": str(output_dir),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("probe", "build", "validate", "benchmark"))
    parser.add_argument("request", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    request = read_json(args.request)
    try:
        if args.operation == "probe":
            result = probe(request)
        elif args.operation == "build":
            result = build(request)
        elif args.operation == "validate":
            result = validate(request, args.request)
        else:
            result = benchmark(request, args.request)
    except Exception as error:
        result = {"schema_version": 1, "request_id": request.get("request_id"),
                  "status": "error", "error": f"{type(error).__name__}: {error}"}
    write_json(args.result, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
