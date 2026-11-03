#!/usr/bin/env python3
"""Create a KDA task JSON after source/route/site inputs are explicit."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

from common import (
    AdapterError,
    BENCH,
    EXPERIMENTS,
    ROOT,
    digest_file,
    load_variant_map,
    write_json,
)


PROTOCOL_PATH = EXPERIMENTS / "protocol.json"
ADAPTER_PATH = EXPERIMENTS / "adapter.py"
COMMON_PATH = EXPERIMENTS / "common.py"
GUARD_PATH = EXPERIMENTS / "selection_guard.py"
VALIDATOR_PATH = BENCH / "validate_runner.py"
SANITIZER_SUITE_PATH = BENCH / "run_sanitizer_suite.py"
DTYPE_QUALIFICATION_PATH = EXPERIMENTS / "run_dtype_qualification.py"
PERFORMANCE_PACKET_PATH = EXPERIMENTS / "run_performance_packet.py"
REFERENCE_PATH = ROOT / "tests" / "factorized_interactions" / "reference.py"
REFERENCE_CACHE_PATH = ROOT / "tests" / "factorized_interactions" / "reference_cache.py"
WORKLOADS_PATH = ROOT / "tests" / "factorized_interactions" / "workloads.py"
CASE_IO_PATH = ROOT / "tests" / "factorized_interactions" / "case_io.py"
WORKLOAD_MANIFEST = BENCH / "workloads.json"
IN_JOB_ENTRYPOINT = EXPERIMENTS / "in_job_kda.py"
DEFAULT_KDA_ROOT = Path("/home/tumlinson/kda")


def _trusted(identifier: str, path: Path, role: str) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not (resolved.is_file() or resolved.is_dir()):
        raise AdapterError(f"trusted input is not a regular file/directory: {path}")
    return {"id": identifier, "path": str(resolved), "role": role}


def _direct_control_facts(workload_id: str) -> dict[str, Any]:
    """Derive exact direct-layout applicability from the generated records."""
    import numpy as np

    sys.path.insert(0, str(ROOT / "tests" / "factorized_interactions"))
    from workloads import make_workload  # noqa: E402

    problem, _ = make_workload(workload_id, np.float16, np.float16)
    mechanisms = problem.mechanisms
    output_counts = Counter(mechanism.target for mechanism in mechanisms)
    input_counts = Counter(participant for mechanism in mechanisms
                           for participant in mechanism.participants)
    context_counts = Counter(context for mechanism in mechanisms
                             for context in mechanism.context_ids)
    alpha_counts = Counter(mechanism.alpha_id for mechanism in mechanisms)
    bias_counts = Counter(mechanism.bias_id for mechanism in mechanisms)
    gamma_counts = Counter(gamma for mechanism in mechanisms
                           for gamma in mechanism.gamma_ids)
    maxima = {
        "max_output_mechanisms": max(output_counts.values(), default=0),
        "max_input_roles": max(input_counts.values(), default=0),
        "max_context_terms": max(context_counts.values(), default=0),
        "max_alpha_mechanisms": max(alpha_counts.values(), default=0),
        "max_bias_mechanisms": max(bias_counts.values(), default=0),
        "max_gamma_terms": max(gamma_counts.values(), default=0),
        "max_mechanism_terms": max((len(mechanism.context_ids)
                                     for mechanism in mechanisms), default=0),
    }
    cells = int(problem.x.shape[2])
    channels = int(problem.x.shape[1])
    mechanism_count = len(mechanisms)
    outputs = int(problem.target_count)
    limit = 64

    def over(count: int, repeat: int) -> bool:
        return count > 0 and repeat > limit // count

    active = mechanism_count > 0 and cells > 0 and channels > 0
    stable = active and (
        maxima["max_mechanism_terms"] > limit or
        maxima["max_output_mechanisms"] > limit or
        maxima["max_input_roles"] > limit or
        maxima["max_context_terms"] > limit or
        over(maxima["max_alpha_mechanisms"], math.ceil(cells / 4)) or
        over(maxima["max_bias_mechanisms"], math.ceil(cells / 32)) or
        over(maxima["max_gamma_terms"], math.ceil(cells / 32))
    )
    native_direct_backend = "stable" if stable else "direct"
    unique_output = bool(
        active and outputs > 0 and mechanism_count == outputs and
        maxima["max_output_mechanisms"] == 1 and
        native_direct_backend == "direct"
    )
    return {
        "workload_id": workload_id,
        "cells": cells,
        "channels": channels,
        "mechanisms": mechanism_count,
        "outputs": outputs,
        **maxima,
        "native_direct_backend_from_record_stats": native_direct_backend,
        "unique_output_direct_applicable": unique_output,
        "strongest_direct_candidate_argument":
            "direct_unique_output" if unique_output else "minimal_direct",
    }


def make_task(
    *, output: Path, variant_map_path: Path, site_policy_path: Path,
    nvcc: Path, expected_device_uuid: str,
    candidate_win_cases: list[str], max_seconds: float,
    task_id: str, case_ids: list[str] | None = None,
    kda_root: Path = DEFAULT_KDA_ROOT,
) -> dict[str, Any]:
    variant_map_path = variant_map_path.resolve(strict=True)
    variant_map = load_variant_map(variant_map_path)
    site_policy_path = site_policy_path.resolve(strict=True)
    nvcc = nvcc.resolve(strict=True)
    kda_root = kda_root.resolve(strict=True)
    if not site_policy_path.is_file() or site_policy_path.is_symlink():
        raise AdapterError("site policy must be a regular non-symlink file")
    if not nvcc.is_file() or nvcc.is_symlink():
        raise AdapterError("nvcc path must resolve to a regular non-symlink executable")
    if not re.fullmatch(
            r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
            expected_device_uuid):
        raise AdapterError("expected_device_uuid must be one explicit NVIDIA GPU UUID")
    if IN_JOB_ENTRYPOINT.is_symlink() or not IN_JOB_ENTRYPOINT.is_file():
        raise AdapterError("the trusted in-job KDA entrypoint is missing or non-regular")

    sys.path.insert(0, str(ROOT / "tests" / "factorized_interactions"))
    from workloads import (correctness_workload_ids,
                           performance_workload_ids,
                           workload_ids)  # noqa: E402

    catalog = list(performance_workload_ids())
    correctness_catalog = list(correctness_workload_ids())
    all_workloads = list(workload_ids())
    selected_cases = list(case_ids) if case_ids else catalog
    if len(selected_cases) != len(set(selected_cases)) or not selected_cases:
        raise AdapterError("case IDs must be a nonempty unique list")
    unknown = set(selected_cases) - set(catalog)
    if unknown:
        raise AdapterError(f"unknown fixed performance workload IDs: {sorted(unknown)}")
    if len(candidate_win_cases) != len(set(candidate_win_cases)):
        raise AdapterError("candidate-win case IDs must be unique")
    if not candidate_win_cases or not set(candidate_win_cases).issubset(selected_cases):
        raise AdapterError("at least one candidate-win case must be selected and included in the task")
    challenger_variant = variant_map["variants"][variant_map["comparison"]["challenger"]]
    if not isinstance(challenger_variant.get("win_backend"), str) or \
            not isinstance(challenger_variant.get("win_layout"), str):
        raise AdapterError("challenger variant must declare exact win_backend and win_layout identities")
    if not isinstance(challenger_variant.get("allowed_implementations"), list) or \
            not isinstance(challenger_variant.get("allowed_layouts"), list) or \
            challenger_variant["win_backend"] not in challenger_variant["allowed_implementations"] or \
            challenger_variant["win_layout"] not in challenger_variant["allowed_layouts"]:
        raise AdapterError("challenger win backend/layout must also be admitted by its allowed identity sets")

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    execution_contract = protocol.get("execution")
    if not isinstance(execution_contract, dict) or \
            execution_contract.get("kda_mode") != "direct" or \
            execution_contract.get("submit_nested_scheduler_job") is not False or \
            execution_contract.get("selection_requires_external_receipt_and_cleanup") is not True:
        raise AdapterError("protocol must bind direct KDA execution to the existing outer job")
    selection = protocol["root_selection"]
    timing = dict(protocol["timing"])
    comparison = variant_map["comparison"]
    variant_ids = [comparison["incumbent"], comparison["challenger"]]
    experiment_group = comparison.get("experiment_group", task_id.replace("-", "_"))

    cases = []
    direct_facts_by_case = {
        workload_id: _direct_control_facts(workload_id)
        for workload_id in catalog
    }
    for workload_id in selected_cases:
        cases.append({
            "id": workload_id,
            "weight": 1.0,
            "input": {"workload_id": workload_id},
            "parameters": {
                **protocol["fixed_precision"],
                "candidate_win_required": workload_id in candidate_win_cases,
                "direct_control": direct_facts_by_case[workload_id],
            },
        })

    trusted_inputs = [
        _trusted("variant_map", variant_map_path, "harness"),
        _trusted("protocol", PROTOCOL_PATH, "harness"),
        _trusted("task_adapter", ADAPTER_PATH, "harness"),
        _trusted("adapter_contracts", COMMON_PATH, "harness"),
        _trusted("root_selection_guard", GUARD_PATH, "harness"),
        _trusted("fixed_validator", VALIDATOR_PATH, "validator"),
        _trusted("strict_sanitizer_suite", SANITIZER_SUITE_PATH, "validator"),
        _trusted("dtype_qualification_driver", DTYPE_QUALIFICATION_PATH, "harness"),
        _trusted("performance_packet_driver", PERFORMANCE_PACKET_PATH, "harness"),
        _trusted("independent_reference", REFERENCE_PATH, "reference"),
        _trusted("reference_cache", REFERENCE_CACHE_PATH, "reference"),
        _trusted("fixed_workload_generator", WORKLOADS_PATH, "harness"),
        _trusted("workload_catalog", WORKLOAD_MANIFEST, "harness"),
        _trusted("native_wire_protocol", CASE_IO_PATH, "harness"),
        _trusted("candidate_build_helper", Path(variant_map["builder"]["trusted_script"]), "harness"),
        _trusted("site_execution_policy", site_policy_path, "toolchain"),
        _trusted("in_job_entrypoint", IN_JOB_ENTRYPOINT, "harness"),
    ]

    python = sys.executable
    adapter_prefix = [python, str(ADAPTER_PATH), "--variant-map", str(variant_map_path)]
    nvcc_probe = [str(nvcc), "--list-gpu-code"]
    root_policy = {
        "experiment_only": True,
        "minimum_paired_blocks_per_case": protocol["comparison"]["paired_blocks"],
        "minimum_confirmation_blocks_per_case": protocol["comparison"]["confirmation_blocks"],
        "required_candidate_win_point_speedup": selection["candidate_win_point_speedup_min"],
        "required_candidate_win_ci_low_exclusive": selection["candidate_win_ci_low_min_exclusive"],
        "maximum_supported_case_time_regression": selection["maximum_supported_case_time_regression"],
        "kda_max_regression_benefit_units": selection["kda_max_regression_benefit_units"],
        "order_policy": protocol["comparison"]["order_policy"],
    }
    task = {
        "schema_version": 1,
        "task_id": task_id,
        "cases": cases,
        "commands": {
            "build": {"argv": adapter_prefix + ["build", "{request}", "{result}"],
                      "cpu_only": True, "timeout_seconds": 7200},
            "validate": {"argv": adapter_prefix + ["validate", "{request}", "{result}"],
                         "timeout_seconds": 7200},
            "benchmark": {"argv": adapter_prefix + ["benchmark", "{request}", "{result}"],
                          "timeout_seconds": 7200},
        },
        "trusted_inputs": trusted_inputs,
        "numerical_requirements": {
            "law": "sigmoid(bias + sum_k gamma[k] * z[context[k],cell]) * alpha[row,channel] * ordered_product_r x[occurrence[r],channel,cell]",
            "finite_domain": "x, alpha, gamma, z in [-1,1]; bias in [-4,4]",
            "reference": "tests/factorized_interactions/reference.py evaluates quantized declared inputs in float64",
            "fp32_bound": "abs(error) <= 2e-6 + 2e-5 * sum_abs_reference_contributions per element",
            "fp16_output_bound": "correctly rounded float64 output within one local FP16 ULP plus the FP32 accumulation bound",
            "validator": "unchanged bench/factorized_interactions/validate_runner.py; all five native VJPs are checked",
            "auxiliary_correctness_workload_ids": correctness_catalog,
            "auxiliary_direct_control_candidate_argument": "minimal_direct",
            "candidate_variants_sha256": digest_file(variant_map_path),
            "root_selection_policy": root_policy,
        },
        "probe": {"argv": adapter_prefix + ["probe", "{request}", "{result}"],
                  "timeout_seconds": 120},
        "timing": {
            "metric": timing["metric"], "unit": timing["unit"],
            "direction": timing["direction"], "warmup": timing["warmup"],
            "repetitions": timing["repetitions"],
            "rounds": timing["rounds"],
            "confirmation_rounds": timing["confirmation_rounds"],
            "seed": timing["seed"],
            "scope": timing["scope"],
        },
        "promotion": {
            "min_improvement": 0.0,
            "max_regression": selection["kda_max_regression_benefit_units"],
            "confidence": protocol["comparison"]["confidence"],
            "bootstrap_samples": protocol["comparison"]["bootstrap_samples"],
        },
        "target": {"kind": "cuda", "architecture": "sm_70", "compiler_target": "sm_70",
                   "expected_device_uuid": expected_device_uuid},
        "toolchain": {
            "compiler_target_probe_argv": nvcc_probe,
            "nvcc": str(nvcc),
            "cuda_toolkit": "12.9",
            "host_cxx": "/usr/bin/g++-12",
        },
        "execution": {
            "mode": execution_contract["kda_mode"],
        },
        "budgets": {
            "max_candidates": 4,
            "max_build_jobs": 1,
            "max_seconds": max_seconds,
        },
        "factorized_interaction": {
            "provisional_only": True,
            "outer_execution_context_schema": execution_contract["outer_context_schema"],
            "outer_execution_context_file": "outer-execution-context.json",
            "candidate_ids": variant_ids,
            "strongest_direct_control_candidate_id": comparison["strongest_direct_control"],
            "same_source_runner_required": comparison["same_source_runner_required"],
            "variant_map_path": str(variant_map_path),
            "variant_map_sha256": digest_file(variant_map_path),
            "expected_device_uuid": expected_device_uuid,
            "in_job_entrypoint_path": str(IN_JOB_ENTRYPOINT.resolve(strict=True)),
            "in_job_entrypoint_sha256": digest_file(IN_JOB_ENTRYPOINT),
            "experiment_group": experiment_group,
            "runner_iterations_per_paired_block": int(variant_map["runner_iterations"]),
            "timed_performance_case_ids": selected_cases,
            "unmeasured_performance_case_ids_direct_fallback": sorted(set(catalog) - set(selected_cases)),
            "correctness_only_case_ids_direct_unmeasured": correctness_catalog,
            "all_fixed_workload_ids": all_workloads,
            "direct_control_facts_by_performance_case": direct_facts_by_case,
            "direct_control_mode_policy": "direct_unique_output iff the exact generated structure has M == output_count, max_output_mechanisms == 1, and the current direct backend applicability predicate selects direct; otherwise minimal_direct",
            "candidate_build_helper_sha256": digest_file(
                Path(variant_map["builder"]["trusted_script"])
            ),
            "candidate_build_helper_snapshot_path": variant_map["builder"][
                "snapshot_script_relative_path"],
            "native_metric_records_are_secondary_evidence": [
                "host_preparation", "device_layout_upload",
                "activation_and_cotangent_conversion", "parameter_initialization",
                "device_buffer_allocation_and_seed", "parameter_refresh",
                "forward", "backward_and_gradient_reductions", "memory_bytes",
            ],
        },
    }

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, task, exclusive=True)
    sys.path.insert(0, str(kda_root))
    try:
        from kda_experiments.contracts import load_task
    except ImportError as exc:
        output.unlink()
        raise AdapterError(f"cannot import KDA experiment contracts from {kda_root}") from exc
    try:
        load_task(output)
    except Exception:
        output.unlink()
        raise
    return task


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant-map", type=Path, required=True)
    parser.add_argument("--site-policy", type=Path, required=True)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--expected-uuid", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--candidate-win-case", action="append", required=True)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--max-seconds", type=float, required=True)
    parser.add_argument("--kda-root", type=Path, default=DEFAULT_KDA_ROOT)
    args = parser.parse_args(argv)
    try:
        task = make_task(
            output=args.output, variant_map_path=args.variant_map,
            site_policy_path=args.site_policy, nvcc=args.nvcc,
            expected_device_uuid=args.expected_uuid,
            candidate_win_cases=args.candidate_win_case,
            max_seconds=args.max_seconds, task_id=args.task_id,
            case_ids=args.case_id, kda_root=args.kda_root,
        )
    except Exception as exc:
        parser.error(f"cannot prepare KDA task: {exc}")
    print(json.dumps({"task_id": task["task_id"], "cases": len(task["cases"]),
                      "candidate_ids": task["factorized_interaction"]["candidate_ids"],
                      "promotion": task["promotion"], "execution": "direct-in-verified-outer-job",
                      "expected_device_uuid": task["factorized_interaction"]["expected_device_uuid"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
