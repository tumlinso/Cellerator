#!/usr/bin/env python3
"""Aggregate the frozen native ordered-postings KDA attempt."""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE / "kda/workspace_v2"
RUN_ID = "c43195bc-b656-43b1-a752-e78e9b0e6bd2"
ATTEMPT = WORKSPACE / "runs/kda-experiments/attempts" / RUN_ID
EXPECTED_BINARY = {
    "serial_frozen": "a628a22579a576d51f942f91d3c0f279ec7c371fdec36f0b454a956e7a44ae7f",
    "ordered_fastpath": "e08c17f5142363ada4d5de298d146570efeff1669c2327d9e7f3bc681502e393",
}
EXPECTED_FIXTURE = {
    "small_v3": "c33facc18f330739411c1f293609c237689685bd4003099ca79b331fded144be",
    "large_ranked_v3": "674d5d4660befcb6ddbcd618b16b136203d3478ddc577d09a037e116bffd6e32",
    "large_events_v3": "910a4c742b795a2a827cb21e62c510a3bc5039e6878ae75d4826fc0d33234be8",
}
CASES = ("small_v3", "large_ranked_v3", "large_events_v3")
CANDIDATES = ("serial_frozen", "ordered_fastpath")


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def distribution(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "median": statistics.median(values),
        "p10": quantile(values, 0.1),
        "p90": quantile(values, 0.9),
    }


def main() -> None:
    worker_summary = json.loads((ATTEMPT / "worker-summary.json").read_text())
    final_summary = json.loads((ATTEMPT / "summary.json").read_text())
    collected: dict[str, dict[str, list[dict]]] = {
        case: {candidate: [] for candidate in CANDIDATES} for case in CASES
    }

    for candidate in CANDIDATES:
        for phase in ("exploration", "confirmation"):
            pattern = f"stages/{candidate}/{phase}/round-*/**/requests/*-native-driver/result.json"
            for result_path in ATTEMPT.glob(pattern):
                result = json.loads(result_path.read_text())
                case = result["case_id"]
                if case not in collected:
                    raise ValueError(f"unexpected case in driver result: {case}")
                if result["correctness"] != "exact_host_oracle_pass":
                    raise ValueError(f"exact host oracle did not pass: {result_path}")
                identity = result["input_identity"]
                if identity["benchmark_binary_sha256"] != EXPECTED_BINARY[candidate]:
                    raise ValueError(f"candidate binary identity mismatch: {result_path}")
                if identity["fixture_manifest_sha256"] != EXPECTED_FIXTURE[case]:
                    raise ValueError(f"fixture identity mismatch: {result_path}")
                if identity["fixture_generator_sha256"] != "887437fc3418e06c25e515487891a3224f8dd56493b689bf41bc70e66ae5a76b":
                    raise ValueError(f"fixture generator identity mismatch: {result_path}")
                if result["counts"]["warmup"] != 5 or result["counts"]["samples"] != 21:
                    raise ValueError(f"timing policy mismatch: {result_path}")
                if result["counts"]["iterations_per_sample"] != 3:
                    raise ValueError(f"iteration count mismatch: {result_path}")
                rel = result_path.relative_to(ATTEMPT / "stages").parts
                if rel[1] != phase or not rel[2].startswith("round-"):
                    raise ValueError(f"malformed stage path: {result_path}")
                result["_phase_round"] = f"{phase}/{rel[2]}"
                collected[case][candidate].append(result)

    cases: list[dict] = []
    for case in CASES:
        candidate_results: dict[str, dict] = {}
        for candidate in CANDIDATES:
            runs = collected[case][candidate]
            if len(runs) != 4:
                raise ValueError(f"expected four KDA rounds for {case}/{candidate}, got {len(runs)}")
            metrics: dict[str, list[float]] = {}
            round_metrics: dict[str, dict[str, list[float]]] = {}
            setup: dict[str, list[float]] = {}
            memory: dict[str, list[int]] = {}
            for run in runs:
                round_key = run["_phase_round"]
                round_metrics[round_key] = {}
                for name, report in run["timings_ms"].items():
                    values = [float(value) for value in report.get("samples_ms", [])]
                    if not values and "median_ms" in report:
                        values = [float(report["median_ms"])]
                    metrics.setdefault(name, []).extend(values)
                    round_metrics[round_key][name] = values
                for name, value in run["host_setup_ms"].items():
                    setup.setdefault(name, []).append(float(value))
                for name, value in run["memory_bytes"].items():
                    memory.setdefault(name, []).append(int(value))

            candidate_results[candidate] = {
                "rounds": {
                    key: {
                        name: statistics.median(values)
                        for name, values in metric_values.items()
                    }
                    for key, metric_values in sorted(round_metrics.items())
                },
                "timings_ms": {
                    name: distribution(values)
                    for name, values in sorted(metrics.items())
                },
                "host_setup_ms": {
                    name: statistics.median(values)
                    for name, values in sorted(setup.items())
                },
                "memory_bytes": {
                    name: int(statistics.median(values))
                    for name, values in sorted(memory.items())
                },
                "reuse_amortization_ms": {
                    name: statistics.median(values)
                    for name, values in sorted(
                        (metric, [float(run["reuse_amortization_ms"][metric]) for run in runs])
                        for metric in runs[0].get("reuse_amortization_ms", {})
                    )
                },
                "counts": runs[-1]["counts"],
                "fixture_manifest_sha256": runs[-1]["input_identity"]["fixture_manifest_sha256"],
                "binary_sha256": runs[-1]["input_identity"]["benchmark_binary_sha256"],
            }

        serial = candidate_results["serial_frozen"]["timings_ms"]
        fast = candidate_results["ordered_fastpath"]["timings_ms"]
        paired_ratio = (
            serial["cuda_complete_postings_bitmap_workset"]["median"]
            / fast["cuda_complete_postings_bitmap_workset"]["median"]
        )
        fast_scan_ratio = (
            fast["cuda_complete_rescan_bitmap_workset"]["median"]
            / fast["cuda_complete_postings_bitmap_workset"]["median"]
        )
        cases.append({
            "case_id": case,
            "candidates": candidate_results,
            "pooled_serial_over_fastpath_ratio": paired_ratio,
            "fastpath_rescan_over_postings_ratio": fast_scan_ratio,
            "kda_per_case_decision": final_summary["decision"]["per_case"][case],
        })

    output = {
        "schema": "cellerator.native_ordered_postings_kda_aggregate.v1",
        "kda_run_id": RUN_ID,
        "kda_task_id": final_summary["task_id"],
        "kda_task_contract_sha256": final_summary["task_sha256"],
        "comparison_group": final_summary["comparison_group"],
        "environment": final_summary["environment"],
        "kda_decision": final_summary["decision"],
        "promotion_status": final_summary["promotion_status"],
        "promoted_within_kda_workspace": final_summary["promoted"],
        "worker_summary_status": worker_summary["status"],
        "cases": cases,
    }
    output_path = HERE / "analysis.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"analysis_path": str(output_path), "cases": len(cases)}, sort_keys=True))


if __name__ == "__main__":
    main()
