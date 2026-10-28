#!/usr/bin/env python3
"""Verify and aggregate the source-bound v13 SIMT/CE-GEO KDA attempt."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE / "workspace"
RUN_ID = "c70c526c-3452-40fd-9fca-a4c6066d7f68"
ATTEMPT = WORKSPACE / "runs/kda-experiments/attempts" / RUN_ID
CASE_IDS = ("event_local", "ranked_batch", "large_ranked", "large_events")
CANDIDATES = ("simt", "packed")
PHASES = ("inference", "training", "global_edge_value_refresh_fwd_plus_ones_vjp")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qtile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * probability
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def dist(values: list[float | int]) -> dict[str, float | int]:
    data = [float(value) for value in values]
    return {
        "n": len(data),
        "min": min(data),
        "median": statistics.median(data),
        "p10": qtile(data, 0.10),
        "p90": qtile(data, 0.90),
        "max": max(data),
        "mean": statistics.fmean(data),
        "stdev_sample": statistics.stdev(data) if len(data) > 1 else 0.0,
    }


def unique_file(pattern: str) -> Path:
    matches = list(ATTEMPT.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one file for {pattern}, found {len(matches)}")
    return matches[0]


def main() -> None:
    summary = json.loads((ATTEMPT / "summary.json").read_text())
    if summary["status"] != "complete" or summary["benchmark_failures"]:
        raise ValueError("KDA attempt is not a clean completed comparison")
    if len(summary["benchmark_records"]) != len(CASE_IDS) * len(CANDIDATES) * 4:
        raise ValueError("unexpected benchmark record count")

    raw: dict[str, dict[str, dict[str, list[float]]]] = {
        case: {candidate: {} for candidate in CANDIDATES} for case in CASE_IDS
    }
    confirmation_training: dict[str, dict[str, list[float]]] = {
        case: {candidate: [] for candidate in CANDIDATES} for case in CASE_IDS
    }
    confirmation_round_medians: dict[str, dict[str, dict[int, float]]] = {
        case: {candidate: {} for candidate in CANDIDATES} for case in CASE_IDS
    }
    setup_values: dict[str, dict[str, dict[str, list[float]]]] = {
        case: {**{candidate: {} for candidate in CANDIDATES}, "setup": {}, "break_even": {}}
        for case in CASE_IDS
    }
    case_metadata: dict[str, dict] = {}
    workset_digest: dict[str, str] = {}
    frozen_input_digest: dict[str, str] = {}
    driver_outputs: dict[str, list[str]] = {case: [] for case in CASE_IDS}
    preflight_estimates: dict[str, dict] = {}

    for record in summary["benchmark_records"]:
        candidate, case, round_index = (
            record["candidate_id"], record["case_id"], record["round"])
        if candidate not in CANDIDATES or case not in CASE_IDS:
            raise ValueError(f"unexpected candidate/case: {record}")
        request_id = record["request_id"]
        phase = record["phase"]
        response_path = unique_file(
            f"stages/{candidate}/{phase}/round-{round_index:03d}/**/responses/{request_id}.json")
        if sha256_file(response_path) != record["response_sha256"]:
            raise ValueError(f"KDA response digest mismatch: {response_path}")
        response = json.loads(response_path.read_text())
        if response.get("status") != "ok" or response.get("case_id") != case:
            raise ValueError(f"non-success or case mismatch: {response_path}")
        if response.get("backend_candidate") != candidate:
            raise ValueError(f"candidate mismatch: {response_path}")

        driver_dir = Path(response["driver_output_dir"])
        result_path = driver_dir / "results.json"
        driver_result = json.loads(result_path.read_text())
        case_result = driver_result[case]
        for check_name, passed in case_result["checks"].items():
            if check_name in {
                "inference_output_allclose_atol_5e-3_rtol_5e-3",
                "training_parameter_gradient_allclose_atol_6e-3_rtol_6e-3",
                "loss_close_abs_5e-4_rel_5e-3",
            } and not passed:
                raise ValueError(f"numerical check failed: {case}/{candidate}/{check_name}")
        if case_result["checks"]["remaining_read_tickets"] != 0:
            raise ValueError(f"unreleased read tickets: {case}/{candidate}")
        if not all(case_result["checks"]["derivative_only_probability_grad_nonzero"].values()):
            raise ValueError(f"derivative-only STE gradient missing: {case}/{candidate}")

        selected_training_samples = [
            float(item["synchronized_wall_ms_per_step"])
            for item in case_result["phases"]["training"][candidate]["raw_samples"]
        ]
        engine_samples = [float(value) for value in response["samples"]]
        if len(engine_samples) != 21 or engine_samples != selected_training_samples:
            raise ValueError(f"engine samples differ from driver samples: {response_path}")
        raw[case][candidate].setdefault("training_wall_ms", []).extend(engine_samples)
        if phase == "confirmation":
            confirmation_training[case][candidate].extend(engine_samples)
            confirmation_round_medians[case][candidate][round_index] = statistics.median(
                engine_samples)

        digest = case_result["case"]["workset_digest"]
        input_digest = case_result["case"]["frozen_input_digest"]
        if case in workset_digest and workset_digest[case] != digest:
            raise ValueError(f"workset digest changed across paired candidates: {case}")
        if case in frozen_input_digest and frozen_input_digest[case] != input_digest:
            raise ValueError(f"frozen input digest changed across paired candidates: {case}")
        workset_digest[case] = digest
        frozen_input_digest[case] = input_digest

        for phase_name in PHASES:
            selected = case_result["phases"][phase_name][candidate]
            for metric_name, field in (
                ("wall_ms", "synchronized_wall_ms_per_step"),
                ("cuda_event_ms", "cuda_event_ms_per_step"),
            ):
                raw[case][candidate].setdefault(f"{phase_name}.{metric_name}", []).extend(
                    float(item[field]) for item in selected["raw_samples"])
            raw[case][candidate].setdefault(
                f"{phase_name}.peak_allocated_delta_bytes", []).extend(
                    float(item["peak_allocated_delta_bytes"])
                    for item in selected["raw_samples"])

        setup = case_result["setup"]
        for mode in CANDIDATES:
            setup_values[case][mode].setdefault("module_setup_wall_ms", []).extend(
                float(item["wall_ms"]) for item in setup[mode]["raw_samples"])
            setup_values[case][mode].setdefault("module_setup_allocated_delta_bytes", []).extend(
                float(item["allocated_delta_bytes"])
                for item in setup[mode]["raw_samples"])
        for metric in (
            "common_cuda_torch_initialization_ms",
            "fixture_bundle_load_ms",
            "support_spec_prepare_ms",
            "fixture_owner_prepare_ms",
            "shared_workset_and_input_prep_ms",
            "hard_gate_freeze_and_primal_workset_ms",
            "shared_geometry_compile_host_ms",
            "packed_extra_module_setup_ms_median",
            "packed_extra_cold_setup_ms_including_geometry_compile",
        ):
            setup_values[case]["setup"].setdefault(metric, []).append(float(setup[metric]))
        for metric, info in setup["break_even"].items():
            setup_values[case]["break_even"].setdefault(metric, []).append(info)

        if case not in case_metadata:
            case_metadata[case] = case_result["case"]
            if "gpu_memory_preflight" in case_result["case"]:
                preflight_estimates[case] = case_result["case"]["gpu_memory_preflight"]
        elif case_metadata[case]["fixture_bundle_identity"] != case_result["case"].get("fixture_bundle_identity"):
            raise ValueError(f"fixture bundle identity changed in {case}")
        driver_outputs[case].append(str(result_path))

    case_results = []
    for case in CASE_IDS:
        candidates = {}
        for candidate in CANDIDATES:
            metrics = {key: dist(values) for key, values in sorted(raw[case][candidate].items())}
            setup = {
                mode: {key: dist(values) for key, values in sorted(values_by_name.items())}
                for mode, values_by_name in setup_values[case].items()
                if mode in CANDIDATES
            }
            setup["measured_prep_medians_ms"] = {
                key: statistics.median(values)
                for key, values in sorted(setup_values[case]["setup"].items())
            }
            setup["break_even_reports"] = setup_values[case]["break_even"]
            candidates[candidate] = {
                "metrics": metrics,
                "confirmation_training_wall_ms": dist(confirmation_training[case][candidate]),
                "confirmation_round_medians_ms": {
                    str(round_id): value
                    for round_id, value in sorted(confirmation_round_medians[case][candidate].items())
                },
                "setup": setup,
            }

        ratios = {}
        for metric in (
            "training_wall_ms", "inference.wall_ms",
            "global_edge_value_refresh_fwd_plus_ones_vjp.wall_ms",
        ):
            simt = candidates["simt"]["metrics"][metric]["median"]
            packed = candidates["packed"]["metrics"][metric]["median"]
            ratios[f"simt_over_packed_{metric}"] = simt / packed
        decision = summary["decision"]["per_case"][case]
        case_results.append({
            "case_id": case,
            "fixture_metadata": case_metadata[case],
            "workset_digest": workset_digest[case],
            "frozen_input_digest": frozen_input_digest[case],
            "candidates": candidates,
            "simt_over_packed_median_ratios": ratios,
            "kda_decision": decision,
            "driver_results_json_paths": driver_outputs[case],
            "reported_memory_preflight": preflight_estimates.get(case),
        })

    identity = json.loads((HERE / "package_identity.json").read_text())
    output = {
        "schema": "cellerator.sparse_support.torch_v13_complete_cost_kda.v1",
        "run_id": RUN_ID,
        "task_id": summary["task_id"],
        "task_sha256": summary["task_sha256"],
        "comparison_group": summary["comparison_group"],
        "environment": summary["environment"],
        "trusted_source_identities": summary["trusted_inputs"],
        "v13_package_identity": identity,
        "kda_decision": summary["decision"],
        "promotion_status": summary["promotion_status"],
        "promoted_within_kda_workspace": summary["promoted"],
        "post_run_host_memory_preflight_fix": {
            "driver_sha256_used_by_kda_run": next(
                item["sha256"] for item in summary["trusted_inputs"]
                if item["id"] == "benchmark_driver"),
            "current_benchmark_driver_sha256": sha256_file(
                HERE.parents[3] / "packed_training_benchmark.py"),
            "host_regression_test_sha256": sha256_file(
                HERE.parents[3] / "test_packed_training_preflight.py"),
            "host_regression_test_status": "passed",
            "scope": "host memory-estimate bookkeeping only; performance run used frozen driver above",
        },
        "benchmark_records": len(summary["benchmark_records"]),
        "samples_per_candidate_case": 84,
        "timing_policy": json.loads((HERE / "task.json").read_text())["timing"],
        "cases": case_results,
    }
    path = HERE / "analysis.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"analysis_path": str(path), "cases": len(case_results),
                      "records": len(summary["benchmark_records"])}, sort_keys=True))


if __name__ == "__main__":
    main()
