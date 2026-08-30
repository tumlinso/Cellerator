#!/usr/bin/env python3
"""Finalize CE-GEO Volta microarchitecture evidence without new measurement."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MICRO = ROOT / "bench/ce_geo/evidence/micro"
FORWARD = ROOT / "bench/ce_geo/evidence/sm70_forward_complete_cost.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} has no object records")
    return rows


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def linear_fit(points: list[tuple[float, float]]) -> tuple[float, float]:
    if len(points) < 2:
        raise ValueError("cost-surface fit requires at least two points")
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    if denominator == 0.0:
        raise ValueError("cost-surface fit has no independent-variable range")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator
    return mean_y - slope * mean_x, slope


def require_finite_positive(value: object, label: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} is not finite and positive")
    return result


def width_surfaces(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    measurements = [row for row in rows if row.get("record_type") == "measurement"]
    if len(measurements) != 324:
        raise ValueError("width/reuse sweep must contain all 324 measurements")
    if not all(row.get("correctness_passed") is True for row in measurements):
        raise ValueError("width/reuse correctness is incomplete")
    widths = sorted({int(row["N"]) for row in measurements})
    dimensions = sorted({int(row["D"]) for row in measurements})
    reuses = sorted({int(row["reuse"]) for row in measurements})
    if widths != [1, 4, 8, 16, 32, 64, 128, 256, 512]:
        raise ValueError("width sweep is incomplete")
    if dimensions != [16, 32, 64, 128, 256, 512]:
        raise ValueError("dimension sweep is incomplete")
    if reuses != [1, 4, 16, 64, 256, 1024]:
        raise ValueError("reuse sweep is incomplete")

    surfaces: list[dict[str, Any]] = []
    held_out_errors: list[float] = []
    for width in widths:
        by_dimension: dict[int, dict[str, Any]] = {}
        for dimension in dimensions:
            records = [row for row in measurements
                       if int(row["N"]) == width and int(row["D"]) == dimension]
            if len(records) != len(reuses):
                raise ValueError(f"incomplete surface at N={width}, D={dimension}")
            steady_values = {float(row["steady_wall_ns"]) for row in records}
            if len(steady_values) != 1:
                raise ValueError("steady timing changed across algebraic reuse points")
            by_dimension[dimension] = {
                "steady_wall_ns": next(iter(steady_values)),
                "host_structure_ns": float(records[0]["host_structure_ns"]),
                "structure_upload_ns": float(records[0]["structure_upload_ns"]),
                "complete_ns_by_reuse": {
                    str(int(row["reuse"])): float(row["complete_ns"])
                    for row in sorted(records, key=lambda item: int(item["reuse"]))
                },
                "estimated_effective_bandwidth_gbps": float(
                    records[0]["estimated_effective_bandwidth_gbps"]),
                "mad_percent": float(records[0]["mad_percent"]),
                "max_abs_error": float(records[0]["max_abs_error"]),
            }
        training = [(float(dimension), by_dimension[dimension]["steady_wall_ns"])
                    for dimension in dimensions if dimension < 512]
        intercept, slope = linear_fit(training)
        actual = float(by_dimension[512]["steady_wall_ns"])
        predicted = intercept + slope * 512.0
        absolute_percentage_error = abs(predicted - actual) * 100.0 / actual
        held_out_errors.append(absolute_percentage_error)
        surfaces.append({
            "N": width,
            "model": "steady_wall_ns = intercept_ns + ns_per_destination * D",
            "training_dimensions": [16, 32, 64, 128, 256],
            "intercept_ns": intercept,
            "ns_per_destination": slope,
            "held_out": {
                "D": 512,
                "actual_steady_wall_ns": actual,
                "predicted_steady_wall_ns": predicted,
                "absolute_percentage_error": absolute_percentage_error,
                "is_structural_holdout": False,
                "note": "D=512 is a dimensional timing holdout from the same deterministic synthetic organization, not biological held-out evidence.",
            },
            "measured_surface": {str(key): value for key, value in by_dimension.items()},
        })
    return surfaces, {
        "mean_absolute_percentage_error": sum(held_out_errors) / len(held_out_errors),
        "maximum_absolute_percentage_error": max(held_out_errors),
    }


def break_even(forward: list[dict[str, Any]]) -> dict[str, Any]:
    measurements = [row for row in forward if row.get("record_type") == "measurement"]
    if len(measurements) != 2 or not all(row.get("correctness_passed") is True
                                         for row in measurements):
        raise ValueError("integrated forward evidence is incomplete")
    phases = measurements[0]["phases_ns"]
    hybrid_cold = sum(float(phases[key]) for key in (
        "semantic_search", "refinement", "projection_construction",
        "structure_upload"))
    sparse_cold = sum(float(phases[key]) for key in (
        "semantic_search", "refinement", "structure_upload", "sparse_prepare"))
    hybrid_steady = float(phases["hybrid_dynamic_pack_execute_epilogue_order_sync_d2h"])
    sparse_steady = float(phases["sparse_dynamic_upload_execute_sync_d2h"])
    denominator = hybrid_steady - sparse_steady
    crossing = None if denominator <= 0.0 else (sparse_cold - hybrid_cold) / denominator
    return {
        "equation": "steady_ns + cold_ns / reuse",
        "hybrid": {"cold_ns": hybrid_cold, "steady_ns": hybrid_steady},
        "cusparse_spmm_n64": {"cold_ns": sparse_cold, "steady_ns": sparse_steady},
        "break_even_reuse": crossing,
        "interpretation": (
            "hybrid is lower complete cost below the crossing; cuSPARSE has lower "
            "steady cost and becomes lower beyond it"
            if crossing is not None else
            "no positive finite crossing is implied by the measured phase model"
        ),
        "measured_checks": [
            {"reuse": int(row["reuse"]),
             "hybrid_complete_ns": float(row["hybrid_complete_ns"]),
             "sparse_complete_ns": float(row["sparse_complete_ns"])}
            for row in sorted(measurements, key=lambda item: int(item["reuse"]))
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    inputs = {
        "value_pack_residual": MICRO / "value_pack_residual.jsonl",
        "n64_output_owned": MICRO / "n64_output_owned.jsonl",
        "width_reuse": MICRO / "width_reuse_sweep.jsonl",
        "integrated_hybrid_forward": FORWARD,
    }
    evidence = {name: load_jsonl(path) for name, path in inputs.items()}
    surfaces, held_out = width_surfaces(evidence["width_reuse"])
    crossing = break_even(evidence["integrated_hybrid_forward"])

    width_provenance = evidence["width_reuse"][0]
    profiler = next((row for row in evidence["width_reuse"]
                     if row.get("record_type") == "profiler"), None)
    if profiler is None:
        raise ValueError("width/reuse profiler availability record is missing")
    profiler_capture = {
        "available": bool(profiler.get("profiler_available")),
        "returncode": profiler.get("profiler_returncode"),
        "metrics": profiler.get("metrics", {}),
        "missing_metrics": profiler.get("missing_metrics", []),
        "limitation": profiler.get("limitation"),
        "diagnostic": profiler.get("profiler_diagnostic"),
        "cuda_runtime_attributes": width_provenance.get("kernel_resources", {}),
        "policy": "Unavailable hardware counters remain unavailable; analytical traffic estimates are not relabeled as profiler measurements.",
    }

    value_pack = evidence["value_pack_residual"]
    n64 = evidence["n64_output_owned"]
    forward = evidence["integrated_hybrid_forward"]
    candidate_dispositions = [
        {
            "candidate": "integrated_hybrid_mma_residual_n64",
            "organization": "exact MMA plus row-owned residual",
            "widths": [64],
            "disposition": "validated",
            "basis": "host-wall complete-cost evidence beats cuSPARSE at measured reuse 1 and 16 with exact correctness",
        },
        {
            "candidate": "output_owned_mma_n64",
            "organization": "output-owned dense 16x16 MMA fragments",
            "widths": [64],
            "disposition": "evaluated_not_promoted",
            "basis": n64[0].get("promotion_decision", {}).get("reason"),
        },
        {
            "candidate": "historical_atomic_mma_n64",
            "organization": "source-faithful atomic fragment baseline",
            "widths": [64],
            "disposition": "baseline_only",
            "basis": "production candidate entrypoint is private; retained only as historical comparison",
        },
        {
            "candidate": "csr_n1_repeated_n64",
            "organization": "production CSR N=1 repeated 64 times with remap",
            "widths": [64],
            "disposition": "baseline_only",
            "basis": "not a competitive N64 organization and not the strongest integrated sparse baseline",
        },
        {
            "candidate": "dense_cublas_n64",
            "organization": "legal cuBLAS dense tensor-op",
            "widths": [64],
            "disposition": "baseline_only",
            "basis": "dense-favorable comparison, not a sparse biological candidate",
        },
        {
            "candidate": "cusparse_spmm_n64",
            "organization": "cuSPARSE CSR SpMM float32",
            "widths": [64],
            "disposition": "baseline_only",
            "basis": "strongest measured integrated sparse fallback and predicted winner beyond break-even reuse",
        },
        {
            "candidate": "value_pack_residual_calibration",
            "organization": "isolated value pack and calibration-only residual phases",
            "widths": [64],
            "disposition": "partial_calibration",
            "basis": value_pack[0].get("limitations", []),
        },
    ]
    for surface in surfaces:
        candidate_dispositions.append({
            "candidate": f"row_owned_residual_n{surface['N']}",
            "organization": "deterministic synthetic row-owned degree-8 residual",
            "widths": [surface["N"]],
            "disposition": "evaluated_not_promoted",
            "basis": "microarchitecture cost-surface calibration only; no biological held-out organization or profiler cache/stall capture",
        })

    value_pack_record = value_pack[1]
    n64_record = n64[1]
    result = {
        "schema": "CELLERATOR-CE-GEO-FINAL-VOLTA-MICROARCHITECTURE/1",
        "task_id": "CE-GEO-115",
        "hardware_scope": "Tesla V100 sm_70",
        "disposition": "evidence_finalized",
        "promotion_authority": "candidate records only; this aggregate makes no new promotion",
        "source_evidence": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": digest(path),
                   "record_count": len(evidence[name])}
            for name, path in inputs.items()
        },
        "correctness": {
            "all_source_campaigns_passed": (
                value_pack_record.get("correctness_passed") is True
                and n64_record.get("correctness_passed") is True
                and all(row.get("correctness_passed") is True for row in forward[1:])
                and all(row.get("correctness_passed") is True
                        for row in evidence["width_reuse"][1:-1])
            ),
            "maximum_width_sweep_abs_error": max(
                float(row["max_abs_error"]) for row in evidence["width_reuse"]
                if row.get("record_type") == "measurement"),
            "integrated_hybrid_max_abs_error": max(
                float(row["max_abs_error"]) for row in forward
                if row.get("record_type") == "measurement"),
            "held_out_cost_model_error": held_out,
            "held_out_scope_limit": "D=512 timing holdout within one deterministic synthetic organization; not biological held-out correctness evidence.",
        },
        "cost_surfaces": {
            "width_dimension_reuse": surfaces,
            "integrated_hybrid_break_even": crossing,
            "value_pack_residual_scenarios": value_pack_record.get("scenarios", []),
            "n64_organization_scenarios": n64_record.get("scenarios", []),
        },
        "profiler_capture": profiler_capture,
        "complete_mechanism_accounting": {
            "integrated_hybrid": forward[0].get("methodology", {}),
            "width_reuse": width_provenance.get("methodology", {}),
            "value_pack_residual": value_pack_record.get("methodology", {}),
            "n64_output_owned": n64_record.get("methodology", {}),
            "included_integrated_phases": [
                "semantic search", "refinement", "projection construction",
                "structure upload", "backend preparation", "dynamic values and RHS upload",
                "value pack", "MMA", "residual", "epilogue", "output transfer",
                "explicit synchronization", "reuse amortization",
            ],
            "known_non_equivalences": [
                "CUDA-event microcalibrations are not treated as host-wall complete cost.",
                "Analytical byte traffic is not treated as a DRAM or cache profiler counter.",
                "Synthetic dimensional holdout is not treated as biological held-out evidence.",
            ],
        },
        "candidate_dispositions": candidate_dispositions,
        "limitations": [
            profiler_capture.get("limitation"),
            "CE-GEO-111 retained calibration-only phases and is not promotion-ready.",
            "CE-GEO-112 includes source-faithful historical and repeated-N1 baselines with documented non-equivalences.",
            "CE-GEO-114 is a synthetic row-owned residual surface, not an end-to-end biological campaign.",
            "Only the integrated N64 hybrid campaign supplies host-wall complete-cost comparison against cuSPARSE SpMM.",
        ],
    }
    if result["correctness"]["all_source_campaigns_passed"] is not True:
        raise ValueError("source correctness is incomplete")
    if not all(item.get("disposition") for item in candidate_dispositions):
        raise ValueError("a candidate disposition is missing")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8")
    print(json.dumps({
        "evidence_valid": 1,
        "candidate_count": len(candidate_dispositions),
        "surface_count": len(surfaces),
        "profiler_metrics_available": len(profiler_capture["metrics"]),
        "output": str(arguments.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
