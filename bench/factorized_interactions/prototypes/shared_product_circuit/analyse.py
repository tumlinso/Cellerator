#!/usr/bin/env python3
"""CPU-only analysis of the seven original CUDA measurement pairs.

Use KDA's experiment-engine statistics directly. This command launches no GPU
work, replays no measurements and makes no production selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyse(summary: dict, kda_root: Path) -> dict:
    if summary.get("kind") != "paired_measurement" or any(
        summary.get(key) is not True for key in ("gpu_executed", "validated", "timed")
    ):
        raise ValueError("requires a numerically accepted GPU measurement summary")
    if not (summary["case_sha256"] == summary["case_sha256_before"]
            == summary["case_sha256_after"]):
        raise ValueError("timed case identity changed")
    if not (summary["build_identity"] == summary["build_identity_before"]
            == summary["build_identity_after"]):
        raise ValueError("timed build identity changed")
    measurements = summary["measurements"]
    if (measurements["block_count"], measurements["iterations_per_block"],
        measurements["warmup_iterations_per_arm"]) != (7, 30, 10):
        raise ValueError("requires the declared seven-pair schedule")
    candidates = ("unfactored", "circuit", "native_direct")
    rows = measurements["raw_measurements"]
    if len(rows) != 21:
        raise ValueError("expected exactly 21 original block observations")
    samples = {candidate: [] for candidate in candidates}
    backends = {candidate: set() for candidate in candidates}
    for block in range(7):
        block_rows = sorted((row for row in rows if row["block"] == block),
                            key=lambda row: row["position"])
        if len(block_rows) != 3 or [row["position"] for row in block_rows] != [0, 1, 2]:
            raise ValueError("incomplete or duplicate paired block")
        order = [row["candidate"] for row in block_rows]
        pair = ["unfactored", "circuit"] if block % 2 == 0 else ["circuit", "unfactored"]
        expected = pair.copy()
        expected.insert(block % 3, "native_direct")
        if order != expected or order != measurements["actual_order"][block]:
            raise ValueError("paired arm order or descriptive rotation changed")
        for row in block_rows:
            value = row["host_mean_ms"]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0
                    or row["iterations"] != 30
                    or not math.isclose(value, row["host_block_ms"] / 30,
                                        rel_tol=1e-9, abs_tol=1e-12)):
                raise ValueError("invalid raw block time")
            samples[row["candidate"]].append(float(value))
            backends[row["candidate"]].add(row["selected_backend"])
    sys.path.insert(0, str(kda_root.resolve()))
    from kda_experiments import selection

    module_path = Path(selection.__file__).resolve()
    module_before = digest(module_path)
    case_id = summary["case_id"]
    cases = [{"id": case_id, "weight": 1.0}]
    rounds = {name: [{case_id: [sample]} for sample in samples[name]]
              for name in ("unfactored", "circuit")}
    ranking = selection.rank(cases, rounds, "unfactored", "minimize")
    interval = selection.confirm(
        cases, rounds["unfactored"], rounds["circuit"], "minimize",
        min_improvement=0.0, max_regression=0.0, bootstrap_samples=10000,
        seed=20261103, confidence=0.95,
    )
    if digest(module_path) != module_before:
        raise ValueError("KDA statistics implementation changed during analysis")
    return {
        "schema": "cellerator.shared_product_circuit.analysis.v1",
        "scope": "Exploratory analysis of seven original paired CUDA blocks; no fresh measurements",
        "production_selection": False,
        "receipt_acceptance": "Original execution and cleanup require separate operator/verifier acceptance",
        "scheduler_context": summary["scheduler_context"],
        "case_sha256": summary["case_sha256"],
        "build_identity": summary["build_identity"],
        "raw_host_mean_ms": samples,
        "median_host_mean_ms": {key: statistics.median(value) for key, value in samples.items()},
        "selected_backends": {key: sorted(value) for key, value in backends.items()},
        "paired_unfactored_over_circuit_ratios": [
            a / b for a, b in zip(samples["unfactored"], samples["circuit"])],
        "kda_primary_ranking": ranking,
        "kda_exploratory_paired_interval": interval,
        "kda_selection_source": str(module_path),
        "kda_selection_sha256": module_before,
        "bootstrap": {"samples": 10000, "seed": 20261103, "confidence": 0.95},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-summary", type=Path, required=True)
    parser.add_argument("--kda-root", type=Path, default=Path("/home/tumlinson/kda"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = digest(args.measurement_summary)
    result = analyse(json.loads(args.measurement_summary.read_text()), args.kda_root)
    if digest(args.measurement_summary) != before:
        raise ValueError("measurement summary changed during analysis")
    result["measurement_summary_path"] = str(args.measurement_summary.resolve())
    result["measurement_summary_sha256"] = before
    result["analysis_source_sha256"] = digest(Path(__file__))
    with args.output.open("x") as output:
        json.dump(result, output, sort_keys=True, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({"analysis": str(args.output), "sha256": digest(args.output)}))


if __name__ == "__main__":
    main()
