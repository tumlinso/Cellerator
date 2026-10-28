#!/usr/bin/env python3
"""KDA adapter for the frozen serial versus ordered-posting CUDA comparison."""
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

from kda_adapter import op_probe


REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "bench/sparse_support/run_cuda_indexing_bench.py"


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


def candidate_binary(candidate: dict, build_dir: str | None = None) -> tuple[Path, str]:
    source = Path(candidate["source_dir"]).resolve(strict=True)
    build = candidate.get("build", {})
    directory = build.get("directory", build_dir)
    spec_path = (Path(directory).resolve(strict=True) / "backend.json"
                 if directory is not None else source / "backend.json")
    spec = read_json(spec_path)
    if spec.get("candidate") != candidate.get("id"):
        raise ValueError("candidate ID and frozen binary selector do not agree")
    binary = Path(spec["binary_path"]).resolve(strict=True)
    expected = spec["binary_sha256"]
    actual = sha256_file(binary)
    if actual != expected:
        raise ValueError(f"candidate binary hash mismatch: expected {expected}, got {actual}")
    return binary, actual


def run_driver(request_path: Path, candidate: dict, case: dict, *, warmup: int,
               samples: int, iterations: int) -> dict:
    binary, binary_sha = candidate_binary(candidate)
    case_input = case["input"]
    output_dir = request_path.parent / f"{request_path.stem}-{case['id']}-native-driver"
    output_path = output_dir / "result.json"
    command = [
        sys.executable, str(RUNNER), "--binary", str(binary),
        "--fixture-dir", str(Path(case_input["fixture_root"]) / case_input["profile"]),
        "--case-id", case["id"], "--edges", str(case_input["edges"]),
        "--cells", str(case_input["cells"]), "--entities", str(case_input["entities"]),
        "--warmup", str(warmup), "--samples", str(samples),
        "--iterations", str(iterations), "--output", str(output_path),
    ]
    proc = subprocess.run(command, cwd=REPO, text=True, capture_output=True,
                          env=os.environ.copy(), check=False)
    if proc.returncode != 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
        (output_dir / "stderr.txt").write_text(proc.stderr, encoding="utf-8")
        raise RuntimeError(f"native driver failed for {case['id']} ({proc.returncode})\n{proc.stderr}")
    result = read_json(output_path)
    if result.get("correctness") != "exact_host_oracle_pass":
        raise RuntimeError(f"exact host oracle did not pass for {case['id']}")
    if result.get("input_identity", {}).get("benchmark_binary_sha256") != binary_sha:
        raise RuntimeError("native driver result binary identity does not match candidate")
    expected_fixture = case_input["fixture_manifest_sha256"]
    if result.get("input_identity", {}).get("fixture_manifest_sha256") != expected_fixture:
        raise RuntimeError(f"fixture manifest identity mismatch for {case['id']}")
    return result


def op_build(request: dict) -> dict:
    candidate = request["candidate"]
    source = Path(candidate["source_dir"]).resolve(strict=True) / "backend.json"
    spec = read_json(source)
    binary = Path(spec["binary_path"]).resolve(strict=True)
    actual = sha256_file(binary)
    if actual != spec["binary_sha256"]:
        raise RuntimeError("candidate binary changed before preparation")
    build_dir = Path(request["build"]["directory"]).resolve(strict=True)
    destination = build_dir / "backend.json"
    shutil.copyfile(source, destination)
    return {
        "schema_version": 1, "request_id": request["request_id"], "status": "ok",
        "artifacts": [{"path": "backend.json", "sha256": sha256_file(destination)}],
        "binary_sha256": actual,
    }


def op_validate(request: dict, request_path: Path) -> dict:
    if os.environ.get("CELLERATOR_SPARSE_SUPPORT_BENCH_ADMITTED") != "1":
        raise RuntimeError("KDA native validation requires Device Scheduler admission")
    rows = []
    for case in request["cases"]:
        result = run_driver(request_path, request["candidate"], case,
                            warmup=1, samples=3, iterations=1)
        timing = result["timings_ms"].get("cuda_complete_postings_bitmap_workset", {})
        if len(timing.get("samples_ms", ())) != 3:
            raise RuntimeError(f"validation samples missing for {case['id']}")
        rows.append({
            "case_id": case["id"], "workload_sha256": case["workload_sha256"],
            "status": "pass", "correctness": result["correctness"],
            "posting_candidates": result["counts"]["derivative_candidates"],
            "complete_postings_sample_count": len(timing["samples_ms"]),
        })
    return {"schema_version": 1, "request_id": request["request_id"],
            "status": "ok", "cases": rows}


def op_benchmark(request: dict, request_path: Path) -> dict:
    if os.environ.get("CELLERATOR_SPARSE_SUPPORT_BENCH_ADMITTED") != "1":
        raise RuntimeError("KDA native benchmark requires Device Scheduler admission")
    case = request["case"]
    timing = request["timing"]
    result = run_driver(request_path, request["candidate"], case,
                        warmup=int(timing["warmup"]),
                        samples=int(timing["repetitions"]),
                        iterations=int(timing["iterations"]))
    metric = timing["metric"]
    samples = [float(value) for value in result["timings_ms"][metric]["samples_ms"]]
    if len(samples) != int(timing["repetitions"]):
        raise RuntimeError(f"wrong sample count for {case['id']}")
    return {
        "schema_version": 1, "request_id": request["request_id"], "status": "ok",
        "case_id": case["id"], "workload_sha256": case["workload_sha256"],
        "timing_scope": timing["scope"], "warmup": timing["warmup"],
        "repetitions": timing["repetitions"], "metric": metric,
        "unit": timing["unit"], "samples": samples,
        "selected_candidate": request["candidate"]["id"],
        "diagnostics": {
            "complete_rescan_ms": result["timings_ms"]["cuda_complete_rescan_bitmap_workset"],
            "discovery_postings_ms": result["timings_ms"]["cuda_reverse_posting_pair_discovery"],
            "setup_ms": result["host_setup_ms"],
            "memory_bytes": result["memory_bytes"],
            "counts": result["counts"],
            "input_identity": result["input_identity"],
        },
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
