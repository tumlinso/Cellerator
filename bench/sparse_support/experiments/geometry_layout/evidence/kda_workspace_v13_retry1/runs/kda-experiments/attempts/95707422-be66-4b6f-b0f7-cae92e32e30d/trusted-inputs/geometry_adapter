#!/usr/bin/env python3
"""KDA adapter for measured CE-GEO physical-layout candidates."""
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


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PYTHON = "/home/tumlinson/Software/venvs/cellerator-ml2-py313-cu126/bin/python"
DRIVER = HERE / "geometry_training_benchmark.py"
PACKAGE_ROOT = Path(os.environ.get(
    "CELLERATOR_SPARSE_PACKAGE_ROOT",
    str(REPO / "build-sparse-support/evidence/torch-v12-validator/package"),
))
IMAGE_ROOT = HERE / "evidence/prepared_images"
VALID_VARIANTS = {"simt": "canonical", "canonical": "canonical",
                  "reordered": "reordered", "hybrid8": "hybrid8",
                  "hybrid16": "hybrid16"}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_candidate(candidate: dict, build_directory: str | None = None) -> tuple[str, str]:
    source_dir = Path(candidate["source_dir"]).resolve(strict=True)
    config = read_json(source_dir / "geometry.json")
    candidate_id = candidate["id"]
    if config.get("candidate") != candidate_id or candidate_id not in VALID_VARIANTS:
        raise ValueError("geometry candidate ID and candidate config do not agree")
    variant = config.get("image_variant")
    if variant != VALID_VARIANTS[candidate_id]:
        raise ValueError("candidate configuration selects an unexpected geometry image")
    build = candidate.get("build", {})
    directory = build.get("directory", build_directory)
    if directory is not None:
        artifact = Path(directory).resolve(strict=True) / "geometry.json"
        built = read_json(artifact)
        if built != config:
            raise ValueError("built geometry selector differs from its source snapshot")
    return candidate_id, variant


def case_image_path(case_id: str, variant: str) -> Path:
    case_dir = {"small": "small", "large_ranked": "large-ranked",
                "event_local": "event_local"}[case_id]
    return IMAGE_ROOT / case_dir / f"{variant}.cegeo"


def op_probe(request: dict) -> dict:
    import torch
    from cellerator.torch import _torch

    if os.environ.get("CELLERATOR_SPARSE_SUPPORT_BENCH_ADMITTED") != "1":
        raise RuntimeError("geometry probe requires Device Scheduler admission")
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True,
    )
    visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
               if item.strip()]
    token = visible[device] if device < len(visible) else None
    rows = [dict(zip(("uuid", "driver_version"),
                     (part.strip() for part in line.split(",", maxsplit=1))))
            for line in smi.stdout.splitlines() if "," in line]
    selected = next((row for row in rows if row["uuid"] == token), None)
    if selected is None and len(rows) == 1:
        selected = rows[0]
    if selected is None:
        raise RuntimeError(f"could not bind scheduled GPU identity: token={token!r}")
    return {
        "schema_version": 1, "request_id": request["request_id"], "status": "ok",
        "environment": {
            "kind": "cuda", "architecture": f"sm_{props.major}{props.minor}",
            "device": {"index": device, "name": props.name,
                       "total_memory_bytes": props.total_memory,
                       "uuid": selected["uuid"]},
            "device_uuid": selected["uuid"],
            "driver_version": selected["driver_version"],
            "runtime_version": str(torch.version.cuda),
            "tools": {"python": sys.version.split()[0], "torch": str(torch.__version__),
                      "torch_extension_sha256": sha256_file(Path(_torch.__file__).resolve())},
        },
    }


def op_build(request: dict) -> dict:
    candidate_id, variant = selected_candidate(request["candidate"])
    build_dir = Path(request["build"]["directory"]).resolve(strict=True)
    source = Path(request["candidate"]["source_dir"]).resolve(strict=True) / "geometry.json"
    output = build_dir / "geometry.json"
    shutil.copyfile(source, output)
    manifest = read_json(IMAGE_ROOT / "prepared_images.json")
    image_hashes = {}
    for case_id in ("small", "large_ranked", "event_local"):
        image = case_image_path(case_id, variant)
        image_hashes[case_id] = sha256_file(image)
        expected = manifest["cases"][
            {"small": "small", "large_ranked": "large-ranked",
             "event_local": "event_local"}[case_id]]["candidates"][variant]["image_sha256"]
        if image_hashes[case_id] != expected:
            raise RuntimeError(f"prepared image hash mismatch for {case_id}/{variant}")
    artifact_manifest = build_dir / "geometry_image_hashes.json"
    write_json(artifact_manifest, image_hashes)
    return {
        "schema_version": 1, "request_id": request["request_id"], "status": "ok",
        "artifacts": [
            {"path": output.name, "sha256": sha256_file(output)},
            {"path": artifact_manifest.name, "sha256": sha256_file(artifact_manifest)},
        ],
        "candidate": candidate_id, "image_variant": variant,
        "image_sha256_by_case": image_hashes,
        "frozen_torch_package": str(PACKAGE_ROOT.resolve()),
    }


def run_driver(request_path: Path, case_id: str, candidate_id: str,
               *, warmup: int, rounds: int, iterations: int, setup_pairs: int,
               seed: int) -> dict:
    output_path = request_path.parent / (
        f"{request_path.stem}-{case_id}-{candidate_id}-driver.json")
    command = [PYTHON, str(DRIVER), "--case", case_id,
               "--candidate", candidate_id, "--output", str(output_path),
               "--warmup", str(warmup), "--rounds", str(rounds),
               "--iterations", str(iterations), "--setup-pairs", str(setup_pairs),
               "--seed", str(seed)]
    proc = subprocess.run(command, cwd=REPO, text=True, capture_output=True,
                          env=os.environ.copy(), check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"geometry benchmark failed ({proc.returncode})\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    return {"result": read_json(output_path), "output": str(output_path),
            "stdout": proc.stdout.strip()}


def checks_pass(result: dict) -> bool:
    checks = result["checks"]
    return bool(
        checks["inference_output_allclose_atol_5e-3_rtol_5e-3"]
        and checks["training_parameter_gradient_allclose_atol_6e-3_rtol_6e-3"]
        and checks["loss_close_abs_5e-4_rel_5e-3"]
        and all(value > 0 for value in
                checks["derivative_only_probability_grad_nonzero"].values())
        and checks["remaining_read_tickets"] == 0)


def op_validate(request: dict, request_path: Path) -> dict:
    if os.environ.get("CELLERATOR_SPARSE_SUPPORT_BENCH_ADMITTED") != "1":
        raise RuntimeError("geometry validation requires Device Scheduler admission")
    candidate_id, _ = selected_candidate(request["candidate"])
    rows = []
    for case in request["cases"]:
        run = run_driver(request_path, case["id"], candidate_id,
                         warmup=1, rounds=1, iterations=1, setup_pairs=1,
                         seed=int(case["input"]["seed"]))
        result = run["result"]
        expected_manifest = case["input"].get("fixture_manifest_sha256")
        if result.get("fixture_manifest_sha256") != expected_manifest:
            raise RuntimeError(f"fixture manifest mismatch for {case['id']}")
        rows.append({
            "case_id": case["id"], "workload_sha256": case["workload_sha256"],
            "status": "pass" if checks_pass(result) else "fail",
            "workset_digest": result["case"]["workset_digest"],
            "frozen_input_digest": result["case"]["frozen_input_digest"],
            "geometry_image": result["case"]["geometry_image"],
            "checks": result["checks"], "driver_output": run["output"],
        })
    status = "ok" if all(row["status"] == "pass" for row in rows) else "error"
    return {"schema_version": 1, "request_id": request["request_id"],
            "status": status, "cases": rows}


def op_benchmark(request: dict, request_path: Path) -> dict:
    if os.environ.get("CELLERATOR_SPARSE_SUPPORT_BENCH_ADMITTED") != "1":
        raise RuntimeError("geometry benchmark requires Device Scheduler admission")
    candidate_id, _ = selected_candidate(request["candidate"])
    case = request["case"]
    timing = request["timing"]
    run = run_driver(
        request_path, case["id"], candidate_id,
        warmup=int(timing["warmup"]), rounds=int(timing["repetitions"]),
        iterations=int(timing.get("iterations", 2)),
        setup_pairs=int(timing.get("setup_pairs", 5)),
        seed=int(case["input"]["seed"]))
    result = run["result"]
    expected_manifest = case["input"].get("fixture_manifest_sha256")
    if result.get("fixture_manifest_sha256") != expected_manifest:
        raise RuntimeError(f"fixture manifest mismatch for {case['id']}")
    if not checks_pass(result):
        raise RuntimeError("Torch output/gradient oracle failed before timing acceptance")
    phase = result["phases"]["training"]["simt" if candidate_id == "simt" else "packed"]
    samples = [float(item["synchronized_wall_ms_per_step"])
               for item in phase["raw_samples"]]
    if len(samples) != int(timing["repetitions"]):
        raise RuntimeError("geometry benchmark returned the wrong sample count")
    return {
        "schema_version": 1, "request_id": request["request_id"], "status": "ok",
        "case_id": case["id"], "workload_sha256": case["workload_sha256"],
        "timing_scope": timing["scope"], "warmup": timing["warmup"],
        "repetitions": timing["repetitions"], "metric": timing["metric"],
        "unit": timing["unit"], "samples": samples,
        "candidate": candidate_id,
        "diagnostic_paired_ratios": result["phases"]["training"]["paired"],
        "geometry_image": result["case"]["geometry_image"],
        "setup": result["setup"],
        "phase_medians": {
            phase_name: {
                path: result["phases"][phase_name][path]["synchronized_wall_ms_per_step"]["median"]
                for path in ("simt", "packed")
            }
            for phase_name in ("inference", "training",
                               "global_edge_value_refresh_fwd_plus_ones_vjp")
        },
        "checks": result["checks"],
        "driver_output": run["output"],
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
        response = {"schema_version": 1, "request_id": request.get("request_id"),
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
