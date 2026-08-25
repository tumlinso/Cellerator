#!/usr/bin/env python3
"""Validate the CE-ARCH-30 evidence package without running benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from trace_tool import load_json, validate_compact_trace


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "bench" / "architecture_evidence"
MANIFESTS = ROOT / "data" / "manifests" / "architecture_evidence"

REQUIRED_SOURCE_CLASSES = {
    "scRNA-seq",
    "scATAC-seq",
    "matched_multiome",
    "gene_regulatory_graph",
    "enhancer_promoter_relation",
    "developmental_trajectory",
    "cell_neighborhood_graph",
    "pathway_or_module_membership",
    "baseplane_motif_regulatory_relation",
    "sparse_learned_biological_network",
}
REQUIRED_N = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}
REQUIRED_NON_POWER_N = {3, 7, 24, 96}
REQUIRED_ADVERSARIAL_TAGS = {
    "uniform_random_sparsity",
    "no_reusable_modules",
    "global_hub_features",
    "isolated_rare_features",
    "severe_row_length_skew",
    "nearly_empty_masks",
    "tiny_partial_blocks",
    "dense_fragments_below_break_even",
    "structure_changes_too_frequently",
    "N_misaligned_to_matrix_fragments",
    "transpose_heavy_training",
    "one_shot_operation",
    "nearly_all_base_predicates",
    "nearly_no_base_predicates",
    "ambiguous_sequence_regions",
    "chunk_boundary_motif_hits",
    "high_cross_gpu_cut",
    "output_order_conversion_dominates",
}
REQUIRED_PHASES = {
    "host_preparation",
    "semantic_packing",
    "projection_construction",
    "h2d",
    "backend_prepare",
    "dynamic_input_pack",
    "kernel",
    "epilogue",
    "order_transform",
    "synchronization",
    "communication",
    "d2h",
    "end_to_end",
}
REQUIRED_BASELINES = {
    "current_cp_bp_direct",
    "current_csr",
    "sell_c_sigma",
    "bsr",
    "blocked_ell",
    "cusparse",
    "dense_cublas",
    "current_cpk1_load_execute",
    "canonical_order_pipeline",
    "persistent_execution_order_pipeline",
    "baseplane_materialized",
    "baseplane_fused",
}
REQUIRED_METRICS = {
    "ns_per_useful_biological_edge",
    "bytes_per_useful_biological_edge",
    "useful_interactions_per_dram_byte",
    "ns_per_expressed_gene",
    "ns_per_active_regulatory_interaction",
    "achieved_memory_bandwidth",
    "arithmetic_intensity",
    "warp_execution_efficiency",
    "branch_efficiency",
    "registers_per_thread",
    "shared_memory_per_block",
    "achieved_occupancy",
    "l1_reuse",
    "l2_reuse",
    "global_transaction_count",
    "metadata_bytes",
    "descriptor_lane_efficiency",
    "launch_count",
    "host_time",
    "persistent_memory_expansion",
    "transient_workspace",
    "preparation_break_even_reuse",
    "graph_capture_compatible",
}
REQUIRED_TIERS = {"smoke", "representative", "throughput", "adversarial", "deep_profile"}
REQUIRED_DIRECTIONS = {"forward", "transpose"}
REQUIRED_OBSERVABILITY = {
    "descriptor_lane_visits",
    "useful_interactions",
    "metadata_bytes",
    "value_bytes",
    "dense_rhs_bytes",
    "order_transform_bytes",
    "launch_count",
    "registers_per_thread",
    "shared_memory_per_block",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def unique_ids(items: list[dict[str, Any]], context: str) -> set[str]:
    ids = [str(item["id"]) for item in items]
    require(len(ids) == len(set(ids)), f"duplicate {context} ids")
    return set(ids)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sources(sources: dict[str, Any], verify_local: bool) -> set[str]:
    require(sources.get("orientation_contract") == "observations_by_features", "source orientation must be explicit")
    require(sources.get("quantitative_values_enabled") is False, "CE-ARCH-30 local sources must remain structure-only")
    source_ids = unique_ids(sources["sources"], "source")
    for source in sources["sources"]:
        require(source.get("structure_only") is True, f"source {source['id']} is not structure-only")
        require(len(source.get("sha256", "")) == 64, f"source {source['id']} lacks a SHA-256")
        require(source.get("required_provenance_missing"), f"source {source['id']} hides missing provenance")
        if verify_local:
            path = ROOT / source["local_path"]
            require(path.is_file(), f"local source missing: {path}")
            require(path.stat().st_size == int(source["bytes"]), f"source size changed: {source['id']}")
            require(sha256_file(path) == source["sha256"], f"source checksum changed: {source['id']}")
    recipe_classes = {item["source_class"] for item in sources["activation_recipes"]}
    require(REQUIRED_SOURCE_CLASSES - {"scRNA-seq"} <= recipe_classes, "activation recipes do not cover required source classes")
    return source_ids


def validate_workloads(workloads: dict[str, Any], source_ids: set[str]) -> set[str]:
    items = workloads["workloads"]
    workload_ids = unique_ids(items, "workload")
    regimes = workloads["operation_regimes"]
    require(REQUIRED_N <= set(regimes["required_power_of_two_n"]), "required power-of-two N regimes are incomplete")
    require(REQUIRED_NON_POWER_N <= set(regimes["required_non_power_of_two_n"]), "required non-power-of-two N regimes are incomplete")
    represented_classes = {item["source_class"] for item in items}
    require(REQUIRED_SOURCE_CLASSES <= represented_classes, "workloads do not cover every required biological source class")
    adversarial_tags = {
        tag
        for item in items
        for tag in item.get("adversarial_tags", [])
    }
    require(REQUIRED_ADVERSARIAL_TAGS <= adversarial_tags, "adversarial workload coverage is incomplete")
    for item in items:
        require("axes" in item, f"workload {item['id']} lacks axes")
        if "real_source" in item:
            require(item["real_source"]["source_id"] in source_ids, f"workload {item['id']} names an unknown source")
        if item.get("control_only"):
            require("synthetic" not in item and "real_source" not in item, f"control workload {item['id']} must not imply a trace")
        else:
            require(("synthetic" in item) != ("real_source" in item), f"workload {item['id']} must select exactly one trace source")
    return workload_ids


def validate_benchmark_contract(contract: dict[str, Any]) -> None:
    require(unique_ids(contract["phases"], "phase") == REQUIRED_PHASES, "benchmark phases differ from the complete accounting contract")
    require(unique_ids(contract["baseline_catalog"], "baseline") == REQUIRED_BASELINES, "baseline catalog is incomplete or unexpected")
    require(unique_ids(contract["required_metrics"], "metric") == REQUIRED_METRICS, "required metric catalog is incomplete or unexpected")
    require(contract.get("falsification_rules"), "falsification rules are missing")
    require(contract.get("stop_conditions"), "stop conditions are missing")


def validate_resources(resources: dict[str, Any]) -> None:
    tiers = resources["tiers"]
    require(unique_ids(tiers, "resource tier") == REQUIRED_TIERS, "resource tiers are incomplete")
    for tier in tiers:
        require(tier.get("correctness_required") is True, f"tier {tier['id']} does not require correctness")
        headroom = float(tier["minimum_free_memory_fraction"])
        require(0.0 < headroom < 1.0, f"tier {tier['id']} has invalid memory headroom")
    activity = resources["current_task_activity"]
    require(not any(activity.values()), "CE-ARCH-30 recorded a forbidden current lease or performance run")


def validate_watch_plan(watches: dict[str, Any]) -> None:
    require(watches.get("status") == "design_only_unarmed", "watch plan must remain unarmed")
    require(watches.get("controller_specs_to_arm_now") == [], "watch plan arms a nonexistent target")
    require(watches["historical_watch_policy"].get("preserve_all_existing_cp_bp_watches") is True, "historical CP-BP watches are not preserved")
    for family in watches["deferred_watch_families"]:
        require(family.get("blocked_by"), f"watch family {family['id']} has no activation checkpoint")
        require(str(family.get("activation_state", "")).startswith("blocked_"), f"watch family {family['id']} is accidentally active")


def validate_smoke_traces(workload_ids: set[str]) -> list[str]:
    paths = sorted((MANIFESTS / "smoke_traces").glob("*.json"))
    require(len(paths) >= 3, "at least three committed smoke traces are required")
    trace_ids = []
    for path in paths:
        trace = load_json(path)
        validate_compact_trace(trace)
        require(trace["trace_id"] in workload_ids, f"smoke trace {path.name} is not a workload")
        require(trace.get("value_semantics") == "structure_only", f"smoke trace {path.name} carries quantitative semantics")
        trace_ids.append(trace["trace_id"])
    require(len(trace_ids) == len(set(trace_ids)), "duplicate smoke trace ids")
    return trace_ids


def validate_representative_traces(sources: dict[str, Any]) -> list[str]:
    index = load_json(PACKAGE / "representative_trace_index.json")
    require(index.get("schema_version") == 1, "representative trace index schema differs")
    require(set(index.get("required_directions", [])) == REQUIRED_DIRECTIONS,
            "forward/transpose observability is incomplete")
    require(set(index.get("required_observability", [])) == REQUIRED_OBSERVABILITY,
            "runtime and hardware-pressure observability is incomplete")
    source_hashes = {item["id"]: item["sha256"] for item in sources["sources"]}
    trace_ids = unique_ids(index["traces"], "representative trace")
    require(len(trace_ids) >= 2, "two real representative traces are required")
    for item in index["traces"]:
        path = PACKAGE / item["path"]
        require(path.is_file(), f"representative trace missing: {path}")
        require(sha256_file(path) == item["file_sha256"],
                f"representative trace file checksum changed: {item['id']}")
        trace = load_json(path)
        validate_compact_trace(trace)
        require(trace["trace_id"] == item["id"],
                f"representative trace identity changed: {item['id']}")
        require(trace["value_semantics"] == "structure_only",
                f"representative trace carries quantitative values: {item['id']}")
        require(trace["payload_sha256"] == item["payload_sha256"],
                f"representative trace payload changed: {item['id']}")
        provenance = trace["provenance"]
        require(provenance["source_id"] == item["source_id"],
                f"representative trace source changed: {item['id']}")
        require(source_hashes[item["source_id"]] == item["source_sha256"]
                == provenance["source_sha256"],
                f"representative source checksum changed: {item['id']}")
        stats = trace["statistics"]
        features = item["planner_features"]
        observed = {
            "row_count": trace["row_count"],
            "feature_count": trace["column_count"],
            "nnz": stats["nnz"],
            "occupied_feature_count": stats["occupied_feature_count"],
            "maximum_feature_frequency": stats["maximum_feature_frequency"],
            "row_degree_min": stats["row_degree_min"],
            "row_degree_max": stats["row_degree_max"],
            "row_degree_mean": stats["row_degree_mean"],
            "block_width_32_occupied_row_blocks":
                stats["block_width_summaries"]["32"]["occupied_row_blocks"],
            "block_width_32_scalar_occupancy":
                stats["block_width_summaries"]["32"]["scalar_occupancy"],
        }
        require(features == observed,
                f"planner-ready features are stale: {item['id']}")
    return sorted(trace_ids)


def validate_package(verify_local_sources: bool = False) -> dict[str, Any]:
    sources = load_json(MANIFESTS / "sources.json")
    workloads = load_json(MANIFESTS / "workloads.json")
    source_ids = validate_sources(sources, verify_local_sources)
    workload_ids = validate_workloads(workloads, source_ids)
    validate_benchmark_contract(load_json(PACKAGE / "benchmark_contract.json"))
    validate_resources(load_json(PACKAGE / "resource_contracts.json"))
    validate_watch_plan(load_json(PACKAGE / "watch_plan.json"))
    trace_ids = validate_smoke_traces(workload_ids)
    representative_trace_ids = validate_representative_traces(sources)
    return {
        "status": "valid",
        "verified_local_sources": verify_local_sources,
        "source_count": len(source_ids),
        "workload_count": len(workload_ids),
        "smoke_trace_ids": trace_ids,
        "representative_trace_ids": representative_trace_ids,
        "performance_run": False,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--verify-local-sources", action="store_true")
    result.add_argument("--json", action="store_true")
    return result


def main() -> int:
    try:
        args = parser().parse_args()
        result = validate_package(args.verify_local_sources)
        if args.json:
            print(json.dumps(result, sort_keys=True))
        else:
            print(
                "valid CE-ARCH-30 evidence package: "
                f"{result['source_count']} sources, {result['workload_count']} workloads, "
                f"{len(result['smoke_trace_ids'])} smoke traces, "
                f"{len(result['representative_trace_ids'])} real representative traces; "
                "no performance run"
            )
        return 0
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"validate_evidence: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
