#!/usr/bin/env python3
"""Paired, hash-bound comparison of frozen sparse-indexing CUDA binaries.

Run only as the argv of an admitted Device Scheduler job. This driver does not
select a device, build code, request a GPU, or launch a profiler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time


CASES: dict[str, dict[str, object]] = {
    "anchor_e2176_b64": {
        "edges": 2176, "cells": 64, "entities": 256,
        "gate_density": 0.5, "source_skew": 0.0, "dirty_fraction": 0.01,
    },
    "source_skew_0_9": {
        "edges": 2176, "cells": 64, "entities": 256,
        "gate_density": 0.5, "source_skew": 0.9, "dirty_fraction": 0.01,
    },
    "scale_e65536_b1": {
        "edges": 65536, "cells": 1, "entities": 256,
        "gate_density": 0.5, "source_skew": 0.0, "dirty_fraction": 0.01,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binary_record(binary: Path, label: str, case_id: str,
                  shape: dict[str, object], seed: int, warmup: int,
                  samples: int, iterations: int) -> tuple[dict[str, object], list[str]]:
    command = [
        str(binary), "--case-id", case_id,
        "--edges", str(shape["edges"]), "--cells", str(shape["cells"]),
        "--entities", str(shape["entities"]),
        "--gate-density", str(shape["gate_density"]),
        "--source-skew", str(shape["source_skew"]),
        "--dirty-fraction", str(shape["dirty_fraction"]),
        "--seed", str(seed), "--warmup", str(warmup),
        "--samples", str(samples), "--iterations", str(iterations),
    ]
    completed = subprocess.run(command, check=True, text=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    record = json.loads(completed.stdout)
    if record.get("case_id") != case_id \
            or record.get("correctness") != "exact_host_oracle_pass":
        raise RuntimeError(f"{label}/{case_id} failed its exact host oracle")
    record["comparison"] = {
        "variant": label,
        "binary_path": str(binary),
        "binary_sha256": sha256(binary),
        "command": command,
    }
    return record, command


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args()

    control = args.control.expanduser().resolve()
    candidate = args.candidate.expanduser().resolve()
    output = args.output.expanduser().resolve()
    for label, binary in (("control", control), ("candidate", candidate)):
        if not binary.is_file() or not binary.stat().st_mode & 0o111:
            parser.error(f"{label} binary is missing or not executable: {binary}")
    if control == candidate or sha256(control) == sha256(candidate):
        parser.error("control and candidate must be distinct binary snapshots")
    if output.exists():
        parser.error(f"refusing to overwrite evidence: {output}")
    if args.warmup < 0 or args.samples < 3 or args.iterations < 1 or args.rounds < 3:
        parser.error("warmup >= 0, samples >= 3, iterations >= 1, rounds >= 3 required")

    control_manifest = control.parent / "manifest.json"
    candidate_manifest = candidate.parent / "manifest.json"
    if not control_manifest.is_file() or not candidate_manifest.is_file():
        parser.error("each frozen binary directory must contain its source/hash manifest")

    manifest = {
        "schema": "cellerator.sparse_support.cuda_variant_comparison.v1",
        "status": "running",
        "timestamp_unix_ns": time.time_ns(),
        "scope": "exact_host_oracle_paired_cuda_event_comparison_setup_separate",
        "control_manifest_sha256": sha256(control_manifest),
        "candidate_manifest_sha256": sha256(candidate_manifest),
        "comparison_driver_path": str(Path(__file__).resolve()),
        "comparison_driver_sha256": sha256(Path(__file__).resolve()),
        "control_binary_sha256": sha256(control),
        "candidate_binary_sha256": sha256(candidate),
        "control_binary_path": str(control),
        "candidate_binary_path": str(candidate),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "warmup": args.warmup,
        "samples_per_process": args.samples,
        "iterations_per_sample": args.iterations,
        "paired_rounds": args.rounds,
        "seed": args.seed,
        "cases": CASES,
        "order_policy": "per case AB, BA alternating by round to reduce order bias",
        "timing_policy": "embedded CUDA-event device timings; process/setup costs reported separately",
        "records": [],
        "summary": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        destination.write(json.dumps(manifest, sort_keys=True) + "\n")
        destination.flush()
        for case_id, shape in CASES.items():
            paired: list[tuple[dict[str, object], dict[str, object]]] = []
            for round_id in range(args.rounds):
                order = (("control", control), ("candidate", candidate))
                if round_id % 2:
                    order = (order[1], order[0])
                run: dict[str, dict[str, object]] = {}
                for label, binary in order:
                    print(f"{case_id} round={round_id} variant={label}",
                          file=sys.stderr, flush=True)
                    record, _ = binary_record(binary, label, case_id, shape,
                        args.seed, args.warmup, args.samples, args.iterations)
                    run[label] = record
                if run["control"]["fixture"]["digest"] != run["candidate"]["fixture"]["digest"]:
                    raise RuntimeError(f"{case_id}: candidate fixture digest differs")
                if run["control"]["counts"] != run["candidate"]["counts"]:
                    raise RuntimeError(f"{case_id}: candidate work counts differ")
                pair = {
                    "case_id": case_id,
                    "round": round_id,
                    "fixture_digest": run["control"]["fixture"]["digest"],
                    "control": run["control"],
                    "candidate": run["candidate"],
                }
                paired.append((run["control"], run["candidate"]))
                manifest["records"].append(pair)
                destination.write(json.dumps({"type": "pair", **pair}, sort_keys=True) + "\n")
                destination.flush()

            metric_names = sorted(paired[0][0]["timings_ms"].keys())
            case_summary: dict[str, object] = {"fixture_digest": paired[0][0]["fixture"]["digest"], "metrics_ms": {}}
            for metric in metric_names:
                control_values = [float(row[0]["timings_ms"][metric]["median_ms"])
                                  for row in paired]
                candidate_values = [float(row[1]["timings_ms"][metric]["median_ms"])
                                    for row in paired]
                ratios = [a / b if b else float("inf")
                          for a, b in zip(control_values, candidate_values, strict=True)]
                case_summary["metrics_ms"][metric] = {
                    "control_round_medians": control_values,
                    "candidate_round_medians": candidate_values,
                    "paired_control_over_candidate": ratios,
                    "control_median": median(control_values),
                    "candidate_median": median(candidate_values),
                    "median_paired_ratio": median(ratios),
                }
            reuse_names = sorted(paired[0][0]["reuse_amortization_ms"].keys())
            case_summary["reuse_amortization_ms"] = {}
            for metric in reuse_names:
                control_values = [float(row[0]["reuse_amortization_ms"][metric]) for row in paired]
                candidate_values = [float(row[1]["reuse_amortization_ms"][metric]) for row in paired]
                ratios = [a / b if b else float("inf")
                          for a, b in zip(control_values, candidate_values, strict=True)]
                case_summary["reuse_amortization_ms"][metric] = {
                    "control_round_values": control_values,
                    "candidate_round_values": candidate_values,
                    "paired_control_over_candidate": ratios,
                    "control_median": median(control_values),
                    "candidate_median": median(candidate_values),
                    "median_paired_ratio": median(ratios),
                }
            manifest["summary"][case_id] = case_summary
            destination.write(json.dumps({"type": "case_summary", "case_id": case_id,
                                         **case_summary}, sort_keys=True) + "\n")
            destination.flush()

        manifest["status"] = "complete"
        destination.write(json.dumps({"type": "manifest_final", "status": "complete",
                                     "completed_unix_ns": time.time_ns(),
                                     "summary": manifest["summary"]}, sort_keys=True) + "\n")
        destination.flush()
    print(f"wrote exact paired comparisons to {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
