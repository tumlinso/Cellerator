#!/usr/bin/env python3
"""KDA task commands for source-bound factorized-interaction candidates.

CUDA probe, correctness, and benchmark commands are intentionally admitted
only through KDA's configured launcher. Build mode is CPU-only in the KDA
sense: it compiles code but never launches a candidate kernel.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any, Callable, Mapping

from common import (
    AdapterError,
    BENCH,
    EXPERIMENTS,
    ROOT,
    attempt_dir_for_request,
    candidate_tree_identity,
    checked_build,
    digest_file,
    digest_json,
    find_build_record,
    load_variant_map,
    read_json,
    safe_relative,
    write_json,
    workspace_for_attempt,
    verify_regular_within,
)
from in_job_kda import (
    CONTEXT_FILE,
    SCHEMA as OUTER_CONTEXT_SCHEMA,
    TRUSTED_ID as IN_JOB_TRUSTED_ID,
    validate_context,
    validate_task,
)


VALIDATOR = BENCH / "validate_runner.py"
PAIR_CACHE_SCHEMA = "cellerator.factorized_interactions.kda_pair_block.v1"
DESCRIPTIVE_DIRECT_ID = "__descriptive_native_direct__"
REFERENCE_CACHE_DIR_NAME = "factorized-reference-cache"


def _reference_cache_directory(attempt_dir: Path) -> Path:
    """Use one shared cache under this KDA attempt's monitored output tree."""
    return attempt_dir.resolve(strict=True) / REFERENCE_CACHE_DIR_NAME


def _reference_cache_arguments(cache_dir: Path) -> list[str]:
    return ["--reference-cache-dir", str(cache_dir)]


def _read_request(path: Path) -> dict[str, Any]:
    request = read_json(path)
    if request.get("schema_version") != 1 or not isinstance(request.get("request_id"), str):
        raise AdapterError("request does not follow the KDA schema-version 1 envelope")
    return request


def _response(request: Mapping[str, Any], status: str, **fields: Any) -> dict[str, Any]:
    return {"schema_version": 1, "request_id": request["request_id"],
            "status": status, **fields}


def _require_admitted(request: Mapping[str, Any], request_path: Path) -> dict[str, Any]:
    """Recheck the borrowed outer Scheduler context for every CUDA task command."""
    target = request.get("declared_target", request.get("target"))
    if not isinstance(target, dict):
        raise AdapterError("KDA request has no declared CUDA target")
    expected_uuid = target.get("expected_device_uuid")
    if not isinstance(expected_uuid, str):
        raise AdapterError("KDA target omitted its task-bound expected GPU UUID")

    attempt_dir = attempt_dir_for_request(request_path)
    workspace = _workspace_from_attempt(attempt_dir)
    batch = read_json(attempt_dir / "batch-request.json")
    task_path_value = batch.get("task_path")
    if not isinstance(task_path_value, str) or not task_path_value:
        raise AdapterError("KDA batch omitted the task path needed to verify outer context")
    task_path = Path(task_path_value).resolve(strict=True)
    task = read_json(task_path)
    task_hash = digest_file(task_path)
    if batch.get("task_sha256") != task_hash:
        raise AdapterError("KDA task hash differs from the admitted batch request")
    task_target = task.get("target")
    if not isinstance(task_target, dict) or \
            task_target.get("expected_device_uuid") != expected_uuid:
        raise AdapterError("task target and KDA request expected UUID differ")
    factorized = task.get("factorized_interaction")
    if not isinstance(factorized, dict) or \
            factorized.get("expected_device_uuid") != expected_uuid or \
            factorized.get("outer_execution_context_schema") != OUTER_CONTEXT_SCHEMA or \
            factorized.get("outer_execution_context_file") != CONTEXT_FILE:
        raise AdapterError("task factorized-interaction outer-context contract differs from this adapter")
    trusted_entrypoints = [item for item in task.get("trusted_inputs", [])
                           if isinstance(item, dict) and item.get("id") == IN_JOB_TRUSTED_ID]
    entrypoint_path = (EXPERIMENTS / "in_job_kda.py").resolve(strict=True)
    if len(trusted_entrypoints) != 1 or trusted_entrypoints[0].get("path") != str(entrypoint_path):
        raise AdapterError("task does not trust this exact in-job entrypoint")
    entrypoint_hash = digest_file(entrypoint_path)
    if factorized.get("in_job_entrypoint_path") != str(entrypoint_path) or \
            factorized.get("in_job_entrypoint_sha256") != entrypoint_hash:
        raise AdapterError("task in-job entrypoint identity differs from the current trusted file")
    try:
        validate_task(task, task_path, expected_uuid, entrypoint_path)
        live = validate_context(
            os.environ,
            Path("/proc/self/cgroup").read_text(encoding="ascii"),
            expected_uuid,
        )
    except Exception as exc:
        raise AdapterError(f"outer Autobench/Scheduler context verification failed: {exc}") from exc

    context_path = workspace / CONTEXT_FILE
    if context_path.is_symlink() or not context_path.is_file():
        raise AdapterError("verified in-job KDA context record is missing or non-regular")
    context = read_json(context_path)
    expected_context = {
        **live,
        "task_sha256": task_hash,
        "entrypoint_sha256": entrypoint_hash,
    }
    if context.get("schema") != OUTER_CONTEXT_SCHEMA or \
            context.get("status") != "observed" or \
            any(context.get(key) != value for key, value in expected_context.items()) or \
            isinstance(context.get("pid"), bool) or not isinstance(context.get("pid"), int):
        raise AdapterError("outer context record differs from the live Scheduler job or trusted task")

    environment = request.get("environment")
    if environment is not None:
        if not isinstance(environment, dict):
            raise AdapterError("KDA request environment must be an object")
        for key, expected in (
            ("scheduler_job_id", live["scheduler_job_id"]),
            ("scheduler_attempt_id", live["scheduler_attempt_id"]),
            ("device_uuid", expected_uuid),
            ("outer_execution_context_sha256", digest_file(context_path)),
            ("outer_cgroup", live["cgroup"]),
            ("outer_unit_name", live["unit_name"]),
        ):
            if environment.get(key) != expected:
                raise AdapterError(f"KDA probe environment differs from live outer context: {key}")
    return {
        **context,
        "context_sha256": digest_file(context_path),
        "context_path": str(context_path.resolve()),
    }


def _run_json(argv: list[str], *, cwd: Path, label: str) -> dict[str, Any]:
    result = subprocess.run(argv, cwd=cwd, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        detail = result.stderr[-3000:]
        raise AdapterError(f"{label} failed with exit {result.returncode}: {detail}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise AdapterError(f"{label} did not emit its JSON summary: {exc}") from exc
    if not isinstance(payload, dict):
        raise AdapterError(f"{label} returned a non-object JSON summary")
    return payload


def _candidate_record(workspace: Path, candidate_id: str) -> dict[str, Any]:
    path = workspace / "candidates.jsonl"
    if path.is_symlink() or not path.is_file():
        raise AdapterError("KDA candidate registry is missing")
    matches: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if value.get("record_type") == "candidate" and value.get("candidate_id") == candidate_id:
            matches.append(value)
    if len(matches) != 1:
        raise AdapterError(f"expected one immutable KDA candidate record for {candidate_id!r}")
    return matches[0]


def _workspace_from_attempt(attempt_dir: Path) -> Path:
    return workspace_for_attempt(attempt_dir)


def _case_parameters(case: Mapping[str, Any], variant_map: Mapping[str, Any]) -> dict[str, Any]:
    source = case.get("parameters", {})
    if not isinstance(source, dict):
        raise AdapterError("case parameters must be a JSON object")
    precision = variant_map.get("precision", {})
    if not isinstance(precision, dict):
        raise AdapterError("variant map precision must be an object")
    return {
        "activation_dtype": source.get("activation_dtype", precision.get("activation_dtype", "fp16")),
        "parameter_dtype": source.get("parameter_dtype", precision.get("parameter_dtype", "fp16")),
        "output_dtype": source.get("output_dtype", precision.get("output_dtype", "fp16")),
        "gradient_mode": source.get("gradient_mode", precision.get("gradient_mode", "overwrite")),
    }


def _workload_id(case: Mapping[str, Any]) -> str:
    input_data = case.get("input")
    if not isinstance(input_data, dict):
        raise AdapterError("case input must identify a fixed workload")
    workload_id = input_data.get("workload_id")
    if not isinstance(workload_id, str) or not workload_id:
        raise AdapterError("case input must provide workload_id")
    return workload_id


def _candidate_builds(batch: Mapping[str, Any], workspace: Path,
                      variant_map: Mapping[str, Any]) -> dict[str, tuple[Path, Path, str]]:
    records = {candidate_id: _candidate_record(workspace, candidate_id)
               for candidate_id in (variant_map["comparison"]["incumbent"],
                                    variant_map["comparison"]["challenger"])}
    built: dict[str, tuple[Path, Path, str]] = {}
    for candidate_id, variant in variant_map["variants"].items():
        ref = find_build_record(batch, candidate_id)
        candidate_info = {
            "id": candidate_id,
            "snapshot_sha256": records[candidate_id].get("snapshot_sha256"),
        }
        built[candidate_id] = checked_build(candidate_info, ref, variant, records[candidate_id])
    incumbent = variant_map["comparison"]["incumbent"]
    challenger = variant_map["comparison"]["challenger"]
    if records[incumbent].get("snapshot_sha256") != records[challenger].get("snapshot_sha256"):
        raise AdapterError("paired route modes must use the same registered source snapshot")
    if built[incumbent][2] != built[challenger][2]:
        raise AdapterError("paired route modes must use the identical source-bound runner binary")
    incumbent_manifest = read_json(built[incumbent][1])
    challenger_manifest = read_json(built[challenger][1])
    if incumbent_manifest["binaries"]["factorized_interaction_cuda"]["sha256"] != \
            challenger_manifest["binaries"]["factorized_interaction_cuda"]["sha256"]:
        raise AdapterError("paired route modes must use the identical Cellerator CUDA library")
    return built


def _candidate_mode_identity(native: Mapping[str, Any], metrics: Mapping[str, Any],
                             variant: Mapping[str, Any], *,
                             candidate_argument: str,
                             direct_control_facts: Mapping[str, Any] | None = None
                             ) -> tuple[str, str]:
    selected = native.get("selected_backend")
    if selected not in variant["allowed_backends"]:
        raise AdapterError(f"native runner selected an unexpected backend: {selected!r}")
    implementation = native.get(
        "actual_implementation", metrics.get("actual_implementation", selected)
    )
    layout = native.get("structural_layout", metrics.get("structural_layout"))
    if not isinstance(implementation, str) or not implementation or \
            not isinstance(layout, str) or not layout:
        raise AdapterError("native runner omitted implementation/layout identity")
    if metrics.get("selected_backend", selected) != selected or \
            metrics.get("actual_implementation", implementation) != implementation or \
            metrics.get("structural_layout", layout) != layout:
        raise AdapterError("native timing and dispatch metadata disagree on backend identity")
    allowed_implementations = variant.get("allowed_implementations")
    if allowed_implementations is not None and implementation not in allowed_implementations:
        raise AdapterError(f"native runner selected an unexpected implementation: {implementation!r}")
    allowed_layouts = variant.get("allowed_layouts")
    if allowed_layouts is not None and layout not in allowed_layouts:
        raise AdapterError(f"native runner selected an unexpected structural layout: {layout!r}")
    candidate = native.get("candidate", metrics.get("candidate"))
    if candidate != candidate_argument:
        raise AdapterError("native runner did not execute the declared candidate mode")
    if direct_control_facts is not None:
        structural = native.get("structural_maxima")
        if not isinstance(structural, dict):
            raise AdapterError("native runner omitted prepared structural maxima")
        for name in ("cells", "channels", "mechanisms", "outputs"):
            if native.get(name) != direct_control_facts.get(name):
                raise AdapterError(f"native structural metadata differs from fixed workload for {name}")
        maxima_expected = {
            "output_mechanisms": direct_control_facts["max_output_mechanisms"],
            "input_roles": direct_control_facts["max_input_roles"],
            "context_terms": direct_control_facts["max_context_terms"],
            "alpha_mechanisms": direct_control_facts["max_alpha_mechanisms"],
            "bias_mechanisms": direct_control_facts["max_bias_mechanisms"],
            "gamma_terms": direct_control_facts["max_gamma_terms"],
            "mechanism_terms": direct_control_facts["max_mechanism_terms"],
        }
        if any(structural.get(name) != expected
               for name, expected in maxima_expected.items()):
            raise AdapterError("native prepared maxima differ from exact fixed workload records")
        expected_mode = direct_control_facts["strongest_direct_candidate_argument"]
        unique = direct_control_facts["unique_output_direct_applicable"] is True
        if candidate_argument == expected_mode == "direct_unique_output":
            if not unique or implementation != "direct_unique_output" or \
                    selected != "direct" or layout != "canonical_direct":
                raise AdapterError("unique-output control failed its exact record/backend applicability predicate")
        elif candidate_argument == expected_mode == "minimal_direct":
            if unique or implementation not in {"direct", "stable"}:
                raise AdapterError("minimal direct control disagrees with the exact unique-output predicate")
        elif candidate_argument == "native_direct":
            if layout != "full_packet" or implementation not in {"direct", "stable"}:
                raise AdapterError("descriptive native-direct arm did not use the prepared canonical path")
    return implementation, layout


def _candidate_argument_for_case(variant_map: Mapping[str, Any], candidate_id: str,
                                 case: Mapping[str, Any]) -> tuple[str, dict[str, Any] | None]:
    incumbent = variant_map["comparison"]["incumbent"]
    parameters = case.get("parameters")
    facts = parameters.get("direct_control") if isinstance(parameters, dict) else None
    if not isinstance(facts, dict):
        raise AdapterError("task case omitted the computed strongest-direct control contract")
    if candidate_id == incumbent:
        argument = facts.get("strongest_direct_candidate_argument")
        if argument not in {"direct_unique_output", "minimal_direct"}:
            raise AdapterError("task case has an invalid strongest-direct candidate mode")
        return argument, dict(facts)
    return variant_map["variants"][candidate_id]["candidate_argument"], dict(facts)


def _measurement_command(
    request: Mapping[str, Any], *, candidate_id: str, variant: Mapping[str, Any],
    runner_path: Path, manifest_path: Path, runner_hash: str, snapshot_hash: str,
    artifact_dir: Path, iterations: int, device_uuid: str,
    reference_cache_dir: Path,
    candidate_argument_override: str | None = None,
    direct_control_facts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    case = request.get("case")
    if not isinstance(case, dict):
        raise AdapterError("benchmark request has no case")
    values = _case_parameters(case, variant)
    selected_candidate_argument = candidate_argument_override
    if selected_candidate_argument is None:
        selected_candidate_argument, case_facts = _candidate_argument_for_case(
            variant_map, candidate_id, case,
        )
        if direct_control_facts is None:
            direct_control_facts = case_facts
    artifact_dir.mkdir(parents=True, exist_ok=True)
    runner_command = shlex.join([
        str(runner_path), "--candidate", selected_candidate_argument,
    ])
    argv = [
        sys.executable, str(VALIDATOR),
        "--workload", _workload_id(case),
        "--runner", runner_command,
        "--runner-binary", str(runner_path),
        "--build-manifest", str(manifest_path),
        "--route", variant["route"],
        "--activation-dtype", values["activation_dtype"],
        "--parameter-dtype", values["parameter_dtype"],
        "--output-dtype", values["output_dtype"],
        "--gradient-mode", values["gradient_mode"],
        "--expected-uuid", device_uuid,
        "--iterations", str(iterations),
        "--artifact-dir", str(artifact_dir),
        *_reference_cache_arguments(reference_cache_dir),
    ]
    summary = _run_json(argv, cwd=ROOT, label=f"validate/measure {candidate_id}/{case['id']}")
    if summary.get("validated") is not True or summary.get("timed") is not (iterations > 0):
        raise AdapterError("native runner validation did not complete under the frozen harness")
    native = summary.get("native_execution")
    metrics = summary.get("metrics")
    if not isinstance(native, dict) or not isinstance(metrics, dict):
        raise AdapterError("native runner omitted dispatch identity or phase metrics")
    if native.get("device_uuid") != device_uuid or metrics.get("device_uuid") != device_uuid:
        raise AdapterError("native runner used a different GPU UUID than the KDA probe")
    scheduler_job_id = os.environ.get("AUTOBENCH_SCHEDULER_JOB_ID", "").strip()
    if not scheduler_job_id or native.get("scheduler_job_id") != scheduler_job_id or \
            metrics.get("scheduler_job_id") != scheduler_job_id:
        raise AdapterError("native runner metrics do not match the admitted Scheduler job")
    selected_backend = native.get("selected_backend")
    if selected_backend not in {"packet", "direct", "stable"}:
        raise AdapterError(f"native runner reported an unknown backend: {selected_backend!r}")
    if metrics.get("selected_backend") != selected_backend:
        raise AdapterError("native metrics backend differs from native dispatch metadata")
    if native.get("requested_route") != variant["route"]:
        raise AdapterError("native runner executed a different requested route")
    actual_implementation, structural_layout = _candidate_mode_identity(
        native, metrics, variant, candidate_argument=selected_candidate_argument,
        direct_control_facts=direct_control_facts,
    )
    if summary.get("build_provenance", {}).get("runner_binary_sha256") != runner_hash:
        raise AdapterError("validator observed a runner binary different from the KDA artifact")
    metric_values = metrics.get("phase_ms")
    if not isinstance(metric_values, dict):
        raise AdapterError("native metrics omitted phase_ms")
    metric_name = request.get("timing", {}).get("metric")
    sample = metric_values.get(metric_name)
    if isinstance(sample, bool) or not isinstance(sample, (int, float)) or \
            not math.isfinite(float(sample)) or sample <= 0:
        raise AdapterError(f"primary timing metric {metric_name!r} is not positive and finite")
    raw_samples = metrics.get("samples_ms")
    if not isinstance(raw_samples, dict):
        raise AdapterError("native metrics omitted raw forward/VJP/refresh samples")
    for phase in ("forward", "backward", "forward_vjp", "parameter_refresh"):
        vector = raw_samples.get(phase)
        if not isinstance(vector, list) or len(vector) != iterations:
            raise AdapterError(f"native metric {phase!r} must retain all {iterations} iteration samples")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
               not math.isfinite(float(value)) or float(value) < 0.0 for value in vector):
            raise AdapterError(f"native metric {phase!r} contains an invalid raw sample")
    native_manifest = read_json(manifest_path)
    native_library_hash = native_manifest.get("binaries", {}).get(
        "factorized_interaction_cuda", {}).get("sha256")
    if not isinstance(native_library_hash, str) or len(native_library_hash) != 64:
        raise AdapterError("native build manifest omitted the linked library hash")
    return {
        "candidate_id": candidate_id,
        "candidate_argument": selected_candidate_argument,
        "requested_route": variant["route"],
        "accepted_route": variant.get("accepted_route", variant["route"]),
        "selected_backend": selected_backend,
        "actual_implementation": actual_implementation,
        "structural_layout": structural_layout,
        "sample": float(sample),
        "metrics": metrics,
        "native_execution": native,
        "build_provenance": summary.get("build_provenance"),
        "source_snapshot_sha256": snapshot_hash,
        "runner_binary_sha256": runner_hash,
        "native_build_manifest_sha256": digest_file(manifest_path),
        "native_library_sha256": native_library_hash,
        "direct_control_facts": dict(direct_control_facts)
        if direct_control_facts is not None else None,
        "validator_output": {
            "input_case_sha256": summary.get("input_case_sha256"),
            "contract_evidence": summary.get("contract_evidence"),
        },
        "artifact_dir": str(artifact_dir.resolve()),
    }


def _pair_key(request: Mapping[str, Any], batch: Mapping[str, Any],
              builds: Mapping[str, tuple[Path, Path, str]],
              variant_map: Mapping[str, Any]) -> str:
    case = request["case"]
    pair_ids = [variant_map["comparison"]["incumbent"],
                variant_map["comparison"]["challenger"]]
    outer_context = request.get("_outer_context", {})
    if not isinstance(outer_context, dict):
        raise AdapterError("pair key is missing its verified outer context")
    identity = {
        "run_id": batch.get("run_id"),
        "task_id": request.get("task_id"),
        "phase": request.get("phase"),
        "round": request.get("round"),
        "case_id": case.get("id"),
        "workload_sha256": case.get("workload_sha256"),
        "candidate_ids": pair_ids,
        "candidate_snapshots": {
            candidate_id: _candidate_record(_workspace_from_attempt(
                attempt_dir_for_request(Path(request["_request_path"]))), candidate_id)["snapshot_sha256"]
            for candidate_id in pair_ids
        },
        "runner_sha256": {candidate_id: builds[candidate_id][2] for candidate_id in pair_ids},
        "device_uuid": request.get("environment", {}).get("device_uuid"),
        "scheduler_job_id": os.environ.get("AUTOBENCH_SCHEDULER_JOB_ID", "").strip(),
        "scheduler_attempt_id": outer_context.get("scheduler_attempt_id"),
        "outer_context_sha256": outer_context.get("context_sha256"),
        "variant_map_sha256": request.get("_variant_map_sha256"),
        "metric": request.get("timing", {}).get("metric"),
    }
    return digest_json(identity)


def alternating_pair_order(incumbent: str, challenger: str,
                           phase: str, round_index: int) -> list[str]:
    """Return balanced per-case ordering across exploration and confirmation."""
    if phase not in {"exploration", "confirmation"} or round_index < 0:
        raise AdapterError("invalid phase or round for paired order")
    phase_offset = 0 if phase == "exploration" else 7
    return ([incumbent, challenger] if (round_index + phase_offset) % 2 == 0
            else [challenger, incumbent])


def get_or_measure_pair(
    block_path: Path, *, pair_key: str, identity: Mapping[str, Any],
    order: list[str], measure: Callable[[str], dict[str, Any]],
    descriptive_position: int,
) -> dict[str, Any]:
    """Persist one paired block once or return its exact cached measurements."""
    if len(order) != 2 or descriptive_position not in {0, 1, 2}:
        raise AdapterError("paired block requires two candidates and one descriptive-control position")
    execution_order = list(order)
    execution_order.insert(descriptive_position, DESCRIPTIVE_DIRECT_ID)
    identity_hash = digest_json(identity)
    if block_path.is_file():
        block = read_json(block_path)
        if block.get("pair_key") != pair_key or \
                block.get("pair_identity_sha256") != identity_hash or \
                block.get("order") != order or block.get("execution_order") != execution_order:
            raise AdapterError("cached paired block identity is stale or malformed")
        return block
    measurements = {candidate_id: measure(candidate_id) for candidate_id in execution_order}
    block = {
        "schema": PAIR_CACHE_SCHEMA,
        "pair_key": pair_key,
        "pair_identity_sha256": identity_hash,
        "identity": dict(identity),
        "order": list(order),
        "execution_order": execution_order,
        "measurements": measurements,
        "backend_pair_distinct": _implementation_id(measurements[order[0]]) !=
        _implementation_id(measurements[order[1]]),
    }
    write_json(block_path, block, exclusive=True)
    return block


def _implementation_id(measurement: Mapping[str, Any]) -> str | None:
    implementation = measurement.get("actual_implementation") or measurement.get("selected_backend")
    layout = measurement.get("structural_layout")
    return f"{implementation}/{layout}" if implementation and layout else implementation


def benchmark_response(request: Mapping[str, Any], candidate_id: str,
                       pair_key: str, block: Mapping[str, Any]) -> dict[str, Any]:
    case = request["case"]
    timing = request["timing"]
    measurement = block.get("measurements", {}).get(candidate_id)
    if not isinstance(measurement, dict):
        raise AdapterError("paired cache omits the requested candidate measurement")
    return _response(
        request, "ok", case_id=case["id"],
        workload_sha256=case["workload_sha256"],
        timing_scope=timing["scope"], warmup=timing["warmup"],
        repetitions=timing["repetitions"], metric=timing["metric"],
        unit=timing["unit"], samples=[measurement["sample"]],
        paired_block={
            "pair_key": pair_key, "phase": request["phase"],
            "round": request["round"], "order": block["order"],
            "execution_order": block["execution_order"],
            "candidate_ids": list(block["order"]),
            "backend_pair_distinct": block["backend_pair_distinct"],
            "selected_backend_by_candidate": {
                key: block["measurements"][key].get("selected_backend")
                for key in block["order"]
            },
            "actual_implementation_by_candidate": {
                key: _implementation_id(block["measurements"][key])
                for key in block["order"]
            },
            "descriptive_control": {
                "candidate_id": DESCRIPTIVE_DIRECT_ID,
                "candidate_argument": "native_direct",
                "position": block["execution_order"].index(DESCRIPTIVE_DIRECT_ID),
                "sample": block["measurements"][DESCRIPTIVE_DIRECT_ID]["sample"],
                "actual_implementation": block["measurements"][DESCRIPTIVE_DIRECT_ID][
                    "actual_implementation"],
                "structural_layout": block["measurements"][DESCRIPTIVE_DIRECT_ID][
                    "structural_layout"],
            },
            "outer_execution_context": {
                key: block.get("identity", {}).get(key)
                for key in ("outer_context_sha256", "scheduler_job_id",
                            "scheduler_attempt_id", "device_uuid", "outer_cgroup",
                            "outer_unit_name")
            },
        },
        candidate_measurement=measurement,
        descriptive_control_measurement=block["measurements"][DESCRIPTIVE_DIRECT_ID],
    )


def _measure_pair(request: Mapping[str, Any], request_path: Path,
                  variant_map: Mapping[str, Any], variant_map_hash: str,
                  run_measurement: Callable[..., dict[str, Any]]) -> dict[str, Any]:
    outer_context = _require_admitted(request, request_path)
    scheduler_job_id = outer_context["scheduler_job_id"]
    device_uuid = outer_context["device_uuid"]
    attempt_dir = attempt_dir_for_request(request_path)
    workspace = _workspace_from_attempt(attempt_dir)
    batch_path = attempt_dir / "batch-request.json"
    batch = read_json(batch_path)
    current_candidate = request.get("candidate")
    if not isinstance(current_candidate, dict):
        raise AdapterError("benchmark request has no candidate")
    comparison = variant_map["comparison"]
    incumbent = comparison["incumbent"]
    challenger = comparison["challenger"]
    candidate_ids = {incumbent, challenger}
    if current_candidate.get("id") not in candidate_ids:
        raise AdapterError("KDA request candidate is outside the fixed pair")
    if batch.get("incumbent_id") != incumbent or set(batch.get("candidate_ids", [])) != {challenger}:
        raise AdapterError("KDA batch is not the required single-challenger comparison")
    if request.get("phase") not in {"exploration", "confirmation"}:
        raise AdapterError("unknown KDA benchmark phase")
    round_index = request.get("round")
    if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index < 0:
        raise AdapterError("KDA benchmark round must be a zero-based integer")
    case = request.get("case")
    if not isinstance(case, dict) or not isinstance(case.get("workload_sha256"), str):
        raise AdapterError("KDA benchmark case is malformed")
    builds = _candidate_builds(batch, workspace, variant_map)
    pair_key = _pair_key(
        {**request, "_request_path": str(request_path),
         "_variant_map_sha256": variant_map_hash,
         "_outer_context": outer_context}, batch, builds, variant_map,
    )
    block_path = attempt_dir / "factorized-paired-measurements" / f"{pair_key}.json"
    order = alternating_pair_order(incumbent, challenger, request["phase"], round_index)
    pair_identity = {
        "run_id": batch.get("run_id"), "task_id": request.get("task_id"),
        "phase": request["phase"], "round": round_index,
        "case_id": case["id"], "workload_sha256": case["workload_sha256"],
        "candidate_ids": [incumbent, challenger],
        "runner_sha256": {candidate_id: builds[candidate_id][2]
                           for candidate_id in (incumbent, challenger)},
        "descriptive_control": {
            "candidate_argument": "native_direct",
            "runner_sha256": builds[incumbent][2],
        },
        "source_snapshot_sha256": {
            candidate_id: _candidate_record(workspace, candidate_id)["snapshot_sha256"]
            for candidate_id in (incumbent, challenger)},
        "device_uuid": device_uuid, "scheduler_job_id": scheduler_job_id,
        "scheduler_attempt_id": outer_context["scheduler_attempt_id"],
        "outer_context_sha256": outer_context["context_sha256"],
        "outer_cgroup": outer_context["cgroup"],
        "outer_unit_name": outer_context["unit_name"],
        "variant_map_sha256": variant_map_hash,
        "metric": request["timing"]["metric"],
    }
    phase_offset = 0 if request["phase"] == "exploration" else 7
    descriptive_position = (round_index + phase_offset) % 3

    def measure(measurement_id: str) -> dict[str, Any]:
        descriptive = measurement_id == DESCRIPTIVE_DIRECT_ID
        candidate_build_id = incumbent if descriptive else measurement_id
        variant = variant_map["variants"][candidate_build_id]
        runner_path, manifest_path, runner_hash = builds[candidate_build_id]
        candidate_record = _candidate_record(workspace, candidate_build_id)
        candidate_argument_override = "native_direct" if descriptive else None
        control_facts = None
        if descriptive:
            params = case.get("parameters", {})
            control_facts = params.get("direct_control") if isinstance(params, dict) else None
        elif candidate_build_id != incumbent:
            params = case.get("parameters", {})
            control_facts = params.get("direct_control") if isinstance(params, dict) else None
        else:
            params = case.get("parameters", {})
            control_facts = params.get("direct_control") if isinstance(params, dict) else None
        case_component = hashlib.sha256(str(case["id"]).encode("utf-8")).hexdigest()[:20]
        artifact_dir = (
            attempt_dir / "factorized-paired-measurements" / pair_key /
            measurement_id / case_component
        )
        return run_measurement(
            request, candidate_id=measurement_id, variant=variant,
            runner_path=runner_path, manifest_path=manifest_path,
            runner_hash=runner_hash,
            snapshot_hash=candidate_record["snapshot_sha256"],
            artifact_dir=artifact_dir,
            iterations=int(variant_map["runner_iterations"]),
            device_uuid=device_uuid,
            reference_cache_dir=_reference_cache_directory(attempt_dir),
            candidate_argument_override=candidate_argument_override,
            direct_control_facts=control_facts,
        )

    block = get_or_measure_pair(
        block_path, pair_key=pair_key, identity=pair_identity,
        order=order, measure=measure,
        descriptive_position=descriptive_position,
    )
    return benchmark_response(request, current_candidate["id"], pair_key, block)


def _validate_candidate(request: Mapping[str, Any], request_path: Path,
                        variant_map: Mapping[str, Any]) -> dict[str, Any]:
    outer_context = _require_admitted(request, request_path)
    attempt_dir = attempt_dir_for_request(request_path)
    workspace = _workspace_from_attempt(attempt_dir)
    batch = read_json(attempt_dir / "batch-request.json")
    candidate = request.get("candidate")
    if not isinstance(candidate, dict) or candidate.get("id") not in variant_map["variants"]:
        raise AdapterError("validation request candidate is not in the variant map")
    candidate_id = candidate["id"]
    variant = variant_map["variants"][candidate_id]
    # Bind the candidate request to the engine's registry before executing it.
    registered = _candidate_record(workspace, candidate_id)
    if registered.get("snapshot_sha256") != candidate.get("snapshot_sha256"):
        raise AdapterError("validation source hash differs from the registered KDA candidate")
    runner_path, manifest_path, runner_hash = checked_build(
        candidate, candidate, variant, registered,
    )
    cases = request.get("cases")
    if not isinstance(cases, list) or not cases:
        raise AdapterError("validation request must include fixed cases")
    device_uuid = outer_context["device_uuid"]
    reference_cache_dir = _reference_cache_directory(attempt_dir)
    rows: list[dict[str, Any]] = []
    for case in cases:
        case_id = case.get("id") if isinstance(case, dict) else None
        if not isinstance(case_id, str):
            raise AdapterError("validation case has no string ID")
        case_component = hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:20]
        artifact_dir = attempt_dir / "factorized-validation" / candidate_id / case_component
        values = _case_parameters(case, variant_map)
        candidate_argument, direct_control_facts = _candidate_argument_for_case(
            variant_map, candidate_id, case,
        )
        runner_command = shlex.join([
            str(runner_path), "--candidate", candidate_argument,
        ])
        argv = [
            sys.executable, str(VALIDATOR), "--workload", _workload_id(case),
            "--runner", runner_command, "--runner-binary", str(runner_path),
            "--build-manifest", str(manifest_path), "--route", variant["route"],
            "--activation-dtype", values["activation_dtype"],
            "--parameter-dtype", values["parameter_dtype"],
            "--output-dtype", values["output_dtype"],
            "--gradient-mode", values["gradient_mode"],
            "--expected-uuid", device_uuid, "--iterations", "0",
            "--artifact-dir", str(artifact_dir),
            *_reference_cache_arguments(reference_cache_dir),
        ]
        summary = _run_json(argv, cwd=ROOT, label=f"validate {candidate_id}/{case_id}")
        native = summary.get("native_execution")
        passed = summary.get("validated") is True and isinstance(native, dict)
        backend = native.get("selected_backend") if isinstance(native, dict) else None
        try:
            actual_implementation, structural_layout = _candidate_mode_identity(
                native if isinstance(native, dict) else {},
                summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {},
                variant, candidate_argument=candidate_argument,
                direct_control_facts=direct_control_facts,
            )
        except AdapterError:
            actual_implementation, structural_layout = None, None
            passed = False
        passed = passed and native.get("requested_route") == variant["route"]
        rows.append({
            "case_id": case_id,
            "workload_sha256": case["workload_sha256"],
            "status": "pass" if passed else "fail",
            "backend_identity": {
                "candidate_id": candidate_id, "requested_route": variant["route"],
                "selected_backend": backend, "runner_binary_sha256": runner_hash,
                "actual_implementation": actual_implementation,
                "structural_layout": structural_layout,
                "candidate_argument": candidate_argument,
                "source_snapshot_sha256": candidate["snapshot_sha256"],
            },
            "validator_input_case_sha256": summary.get("input_case_sha256"),
            "contract_evidence": summary.get("contract_evidence"),
        })

    auxiliary_ids = request.get("numerical_requirements", {}).get(
        "auxiliary_correctness_workload_ids", []
    )
    if not isinstance(auxiliary_ids, list) or any(not isinstance(value, str) for value in auxiliary_ids):
        raise AdapterError("task has an invalid auxiliary correctness workload list")
    if auxiliary_ids:
        sys.path.insert(0, str(ROOT / "tests" / "factorized_interactions"))
        from workloads import correctness_workload_ids  # noqa: E402

        frozen_ids = list(correctness_workload_ids())
        if auxiliary_ids != frozen_ids:
            raise AdapterError("task auxiliary correctness list differs from the frozen full corpus")
    auxiliary_rows: list[dict[str, Any]] = []
    direct_control_candidate = candidate_id == variant_map["comparison"]["incumbent"]
    if direct_control_candidate:
        # Use the same record-derived predicate as task preparation. In particular,
        # unique-output applicability is a property of the fixed records and the
        # selected direct backend, not a workload name or a fan-in label.
        sys.path.insert(0, str(EXPERIMENTS))
        from prepare_task import _direct_control_facts  # noqa: E402

    for workload_id in auxiliary_ids:
        case_component = hashlib.sha256(workload_id.encode("utf-8")).hexdigest()[:20]
        artifact_dir = attempt_dir / "factorized-validation" / candidate_id / "auxiliary" / case_component
        activation_dtype = "fp32" if workload_id == "correctness_fma_product_residual_cancellation" else "fp16"
        parameter_dtype = (
            "fp32" if workload_id in {
                "correctness_fma_product_residual_cancellation",
                "correctness_packet_high_fanin_fp32_100k",
            } else "fp16"
        )
        direct_control_facts = None
        if direct_control_candidate:
            direct_control_facts = _direct_control_facts(workload_id)
            candidate_argument = direct_control_facts[
                "strongest_direct_candidate_argument"
            ]
        else:
            candidate_argument = variant["candidate_argument"]
        runner_command = shlex.join([
            str(runner_path), "--candidate", candidate_argument,
        ])
        argv = [
            sys.executable, str(VALIDATOR), "--workload", workload_id,
            "--runner", runner_command, "--runner-binary", str(runner_path),
            "--build-manifest", str(manifest_path), "--route", variant["route"],
            "--activation-dtype", activation_dtype,
            "--parameter-dtype", parameter_dtype,
            "--output-dtype", "fp16", "--gradient-mode", "overwrite",
            "--repeat-vjp", "--expected-uuid", device_uuid,
            "--iterations", "0", "--artifact-dir", str(artifact_dir),
            *_reference_cache_arguments(reference_cache_dir),
        ]
        try:
            summary = _run_json(
                argv, cwd=ROOT,
                label=f"auxiliary correctness {candidate_id}/{workload_id}",
            )
            native = summary.get("native_execution")
            implementation, layout = _candidate_mode_identity(
                native if isinstance(native, dict) else {}, {}, variant,
                candidate_argument=candidate_argument,
                direct_control_facts=direct_control_facts,
            )
            passed = summary.get("validated") is True and \
                isinstance(native, dict) and native.get("requested_route") == variant["route"]
            auxiliary_rows.append({
                "workload_id": workload_id,
                "status": "pass" if passed else "fail",
                "activation_dtype": activation_dtype,
                "parameter_dtype": parameter_dtype,
                "output_dtype": "fp16",
                "repeat_vjp": True,
                "candidate_argument": candidate_argument,
                "direct_control_facts": direct_control_facts,
                "selected_backend": native.get("selected_backend") if isinstance(native, dict) else None,
                "actual_implementation": implementation,
                "structural_layout": layout,
                "runner_binary_sha256": runner_hash,
                "input_case_sha256": summary.get("input_case_sha256"),
                "contract_evidence": summary.get("contract_evidence"),
            })
        except Exception as exc:
            auxiliary_rows.append({
                "workload_id": workload_id, "status": "fail",
                "error": f"{type(exc).__name__}: {exc}",
                "runner_binary_sha256": runner_hash,
            })
    auxiliary_ok = all(row["status"] == "pass" for row in auxiliary_rows)
    main_ok = all(row["status"] == "pass" for row in rows)
    response_status = "ok" if auxiliary_ok and main_ok else "error"
    return _response(
        request, response_status, cases=rows, candidate_id=candidate_id,
        outer_execution_context={key: outer_context[key] for key in (
            "schema", "status", "scheduler_job_id", "scheduler_attempt_id",
            "device_uuid", "cgroup", "unit_name", "task_sha256",
            "entrypoint_sha256", "pid", "context_sha256")},
        auxiliary_correctness={
            "scope": "all frozen correctness-only workloads; checked in the admitted validator",
            "status": "pass" if auxiliary_ok else "fail",
            "cases": auxiliary_rows,
        },
        reason=None if response_status == "ok" else
        "fixed candidate cases or the auxiliary full correctness corpus failed",
    )


def _build_candidate(request: Mapping[str, Any],
                     variant_map: Mapping[str, Any]) -> dict[str, Any]:
    candidate = request.get("candidate")
    build_info = request.get("build")
    if not isinstance(candidate, dict) or not isinstance(build_info, dict):
        raise AdapterError("build request is missing candidate or output directory")
    source = Path(candidate.get("source_dir", "")).resolve(strict=True)
    before_hash, before_entries = candidate_tree_identity(source)
    if before_hash != candidate.get("snapshot_sha256"):
        raise AdapterError("KDA candidate snapshot changed before build")
    if not (source / "CMakeLists.txt").is_file() or \
            not (source / "bench" / "factorized_interactions" / "CMakeLists.txt").is_file():
        raise AdapterError("candidate is not a complete Cellerator source tree")
    output_arg = Path(build_info.get("directory", ""))
    if output_arg.is_symlink():
        raise AdapterError("KDA build output directory may not be a symlink")
    output = output_arg.resolve(strict=True)
    if not output.is_dir():
        raise AdapterError("KDA build output directory is missing or a symlink")
    builder = variant_map["builder"]
    trusted_script = Path(builder["trusted_script"]).resolve(strict=True)
    snapshot_script = verify_regular_within(
        source, builder["snapshot_script_relative_path"],
        "candidate snapshot build helper",
    )
    trusted_script_hash = digest_file(trusted_script)
    if digest_file(snapshot_script) != trusted_script_hash:
        raise AdapterError("snapshot build helper differs from the task-trusted helper")
    candidate_build = output / "native_candidate"
    if candidate_build.exists() or candidate_build.is_symlink():
        raise AdapterError("candidate native build directory must be fresh")
    builder_arguments = [
        item.replace("{source}", str(source)).replace("{build}", str(candidate_build))
        for item in builder["argv"]
    ]
    argv = [sys.executable, str(trusted_script), *builder_arguments]
    result = subprocess.run(argv, cwd=source, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise AdapterError(f"trusted candidate build failed: {result.stderr[-4000:]}")
    manifest_path = verify_regular_within(
        output, builder["manifest_relative_path"], "native build manifest",
    )
    manifest = read_json(manifest_path)
    after_hash, after_entries = candidate_tree_identity(source)
    if after_hash != before_hash or after_entries != before_entries:
        raise AdapterError("candidate source snapshot changed during the native build")

    def validate_source_manifest(native_manifest: Mapping[str, Any], label: str) -> str:
        rows = native_manifest.get("source_files")
        if not isinstance(rows, list) or not rows:
            raise AdapterError(f"{label} omitted its complete native source manifest")
        digest = hashlib.sha256()
        normalized: list[dict[str, str]] = []
        previous_path: str | None = None
        for row in rows:
            if not isinstance(row, dict):
                raise AdapterError(f"{label} source manifest contains a malformed entry")
            relative_value = row.get("path")
            relative = safe_relative(relative_value, f"{label} source path")
            relative_name = relative.as_posix()
            if relative_name != relative_value or \
                    (previous_path is not None and relative_name <= previous_path):
                raise AdapterError(f"{label} source paths are not normalized, unique, and sorted")
            previous_path = relative_name
            byte_count = row.get("bytes")
            file_hash = row.get("sha256")
            if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0 or \
                    not isinstance(file_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", file_hash):
                raise AdapterError(f"{label} source entry has an invalid size or digest")
            source_file = verify_regular_within(source, relative_name, f"{label} source file")
            if source_file.stat().st_size != byte_count or digest_file(source_file) != file_hash:
                raise AdapterError(f"{label} source file differs from the immutable KDA snapshot")
            encoded_path = relative_name.encode("utf-8")
            digest.update(len(encoded_path).to_bytes(8, "little"))
            digest.update(encoded_path)
            digest.update(byte_count.to_bytes(8, "little"))
            digest.update(bytes.fromhex(file_hash))
            normalized.append({"path": relative_name, "sha256": file_hash})
        if normalized != before_entries:
            raise AdapterError(f"{label} source files differ from the KDA candidate snapshot")
        source_closure = digest.hexdigest()
        if native_manifest.get("source_closure_sha256") != source_closure:
            raise AdapterError(f"{label} source closure digest differs from its source files")
        return source_closure

    candidate_source_closure = validate_source_manifest(manifest, "candidate build manifest")
    runner_info = manifest.get("binaries", {}).get("runner")
    if not isinstance(runner_info, dict) or not isinstance(runner_info.get("path"), str):
        raise AdapterError("candidate native build omitted its runner binary")
    runner_relative = safe_relative(runner_info["path"], "native runner path")
    runner_path = (manifest_path.parent / Path(*runner_relative.parts)).resolve(strict=True)
    if not runner_path.is_relative_to(output) or runner_path.is_symlink() or not runner_path.is_file():
        raise AdapterError("candidate runner output is outside the KDA build directory")
    runner_hash = digest_file(runner_path)
    if runner_info.get("sha256") != runner_hash:
        raise AdapterError("candidate runner hash differs from its native build manifest")
    library_info = manifest.get("binaries", {}).get("factorized_interaction_cuda")
    if not isinstance(library_info, dict) or not isinstance(library_info.get("path"), str):
        raise AdapterError("candidate native build omitted the Cellerator CUDA library")
    library_relative = safe_relative(library_info["path"], "native CUDA library path")
    library_path = (manifest_path.parent / Path(*library_relative.parts)).resolve(strict=True)
    if not library_path.is_relative_to(output) or library_path.is_symlink() or not library_path.is_file():
        raise AdapterError("candidate CUDA library is outside the KDA build directory")
    library_hash = digest_file(library_path)
    if library_info.get("sha256") != library_hash:
        raise AdapterError("candidate CUDA library hash differs from its native build manifest")

    baseline_manifest_path = verify_regular_within(
        manifest_path.parent, "factorized_interactions_baseline_build_manifest.json",
        "baseline native build manifest",
    )
    baseline_manifest = read_json(baseline_manifest_path)
    if baseline_manifest.get("schema") != "cellerator.factorized_interactions.build.v1":
        raise AdapterError("baseline native build manifest schema is unknown")
    baseline_source_closure = validate_source_manifest(
        baseline_manifest, "baseline build manifest",
    )
    if baseline_source_closure != candidate_source_closure:
        raise AdapterError("baseline and candidate manifests use different source closures")
    if baseline_manifest.get("architecture") != manifest.get("architecture") or \
            baseline_manifest.get("fast_math") != manifest.get("fast_math"):
        raise AdapterError("baseline and candidate build modes differ")

    baseline_binaries = baseline_manifest.get("binaries")
    baseline_runner_info = baseline_binaries.get("runner") if isinstance(baseline_binaries, dict) else None
    if not isinstance(baseline_runner_info, dict) or \
            not isinstance(baseline_runner_info.get("path"), str):
        raise AdapterError("baseline native build omitted its runner binary")
    baseline_runner_relative = safe_relative(
        baseline_runner_info["path"], "baseline runner path",
    )
    baseline_runner_input = (baseline_manifest_path.parent /
                             Path(*baseline_runner_relative.parts))
    if baseline_runner_input.is_symlink():
        raise AdapterError("baseline runner output may not be a symbolic link")
    baseline_runner_path = baseline_runner_input.resolve(strict=True)
    if not baseline_runner_path.is_relative_to(output) or \
            not baseline_runner_path.is_file() or baseline_runner_path == runner_path:
        raise AdapterError("baseline runner output is missing, aliases the candidate, or escapes the KDA build")
    baseline_runner_hash = digest_file(baseline_runner_path)
    if baseline_runner_info.get("sha256") != baseline_runner_hash or \
            baseline_runner_info.get("bytes") != baseline_runner_path.stat().st_size:
        raise AdapterError("baseline runner hash or size differs from its native build manifest")

    baseline_library_info = (baseline_binaries.get("factorized_interaction_cuda")
                             if isinstance(baseline_binaries, dict) else None)
    if not isinstance(baseline_library_info, dict) or \
            baseline_library_info.get("sha256") != library_hash or \
            baseline_library_info.get("path") != library_info.get("path"):
        raise AdapterError("baseline and candidate manifests do not reference the same native library")

    candidate_link = baseline_manifest.get("candidate_runner")
    if not isinstance(candidate_link, dict) or \
            not isinstance(candidate_link.get("manifest_path"), str) or \
            not isinstance(candidate_link.get("binary_path"), str):
        raise AdapterError("baseline manifest omitted its candidate manifest and runner links")
    linked_candidate_manifest = verify_regular_within(
        baseline_manifest_path.parent, candidate_link["manifest_path"],
        "baseline-linked candidate manifest",
    )
    linked_candidate_runner = verify_regular_within(
        baseline_manifest_path.parent, candidate_link["binary_path"],
        "baseline-linked candidate runner",
    )
    candidate_manifest_hash = digest_file(manifest_path)
    if linked_candidate_manifest != manifest_path or \
            candidate_link.get("manifest_sha256") != candidate_manifest_hash:
        raise AdapterError("baseline manifest candidate-manifest link is stale or mismatched")
    if linked_candidate_runner != runner_path or \
            candidate_link.get("binary_sha256") != runner_hash:
        raise AdapterError("baseline manifest candidate-runner link is stale or mismatched")

    manifest_rel = manifest_path.relative_to(output).as_posix()
    runner_rel = runner_path.relative_to(output).as_posix()
    library_rel = library_path.relative_to(output).as_posix()
    baseline_manifest_rel = baseline_manifest_path.relative_to(output).as_posix()
    baseline_runner_rel = baseline_runner_path.relative_to(output).as_posix()
    baseline_manifest_hash = digest_file(baseline_manifest_path)
    artifacts = [
        {"path": manifest_rel, "sha256": digest_file(manifest_path)},
        {"path": runner_rel, "sha256": runner_hash},
        {"path": library_rel, "sha256": library_hash},
        {"path": baseline_manifest_rel, "sha256": baseline_manifest_hash},
        {"path": baseline_runner_rel, "sha256": baseline_runner_hash},
    ]
    return _response(
        request, "ok", artifacts=artifacts,
        provenance={
            "candidate_snapshot_sha256": before_hash,
            "native_source_closure_sha256": manifest.get("source_closure_sha256"),
            "native_build_manifest_sha256": digest_file(manifest_path),
            "runner_binary_sha256": runner_hash,
            "native_library_sha256": library_hash,
            "baseline_native_build_manifest_path": baseline_manifest_rel,
            "baseline_native_build_manifest_sha256": baseline_manifest_hash,
            "baseline_runner_binary_path": baseline_runner_rel,
            "baseline_runner_binary_sha256": baseline_runner_hash,
            "baseline_source_closure_sha256": baseline_manifest.get("source_closure_sha256"),
            "baseline_candidate_manifest_link_sha256": candidate_link["manifest_sha256"],
            "baseline_candidate_runner_link_sha256": candidate_link["binary_sha256"],
            "trusted_build_script_sha256": trusted_script_hash,
            "snapshot_build_script_sha256": digest_file(snapshot_script),
            "build_command": argv,
            "source_files": before_entries,
        },
    )


def _probe(request: Mapping[str, Any], request_path: Path) -> dict[str, Any]:
    outer_context = _require_admitted(request, request_path)
    expected_uuid = outer_context["device_uuid"]
    nvidia_smi = os.environ.get("CELLERATOR_NVIDIA_SMI", "nvidia-smi")
    nvcc_value = request.get("toolchain", {}).get("nvcc")
    if not isinstance(nvcc_value, str) or not nvcc_value:
        raise AdapterError("task toolchain must declare the selected nvcc path")
    query = subprocess.run(
        [nvidia_smi, "-i", expected_uuid,
         "--query-gpu=uuid,compute_cap,driver_version,name",
         "--format=csv,noheader"], text=True, capture_output=True, check=False,
    )
    if query.returncode != 0:
        raise AdapterError(f"nvidia-smi query failed: {query.stderr[-2000:]}")
    rows = list(csv.reader(line for line in query.stdout.splitlines() if line.strip()))
    if len(rows) != 1 or len(rows[0]) != 4:
        raise AdapterError("explicit Scheduler GPU UUID did not resolve to exactly one device")
    uuid, compute_cap, driver_version, name = (field.strip() for field in rows[0])
    if uuid != expected_uuid or not re.fullmatch(r"[0-9]+\.[0-9]+", compute_cap):
        raise AdapterError("nvidia-smi returned a device other than the explicitly granted UUID")
    architecture = "sm_" + compute_cap.replace(".", "")
    nvcc = subprocess.run([nvcc_value, "--version"], text=True,
                          capture_output=True, check=False)
    if nvcc.returncode != 0:
        raise AdapterError(f"nvcc version probe failed: {nvcc.stderr[-2000:]}")
    version_match = re.search(r"release\s+([0-9.]+)", nvcc.stdout)
    if not version_match:
        raise AdapterError("nvcc version output has no release token")
    runtime_version = f"CUDA toolkit/runtime target {version_match.group(1)}"
    environment = {
        "kind": "cuda", "architecture": architecture, "device_uuid": uuid,
        "driver_version": driver_version, "runtime_version": runtime_version,
        "device": {"uuid": uuid, "name": name, "compute_capability": compute_cap},
        "scheduler_job_id": outer_context["scheduler_job_id"],
        "scheduler_attempt_id": outer_context["scheduler_attempt_id"],
        "outer_execution_context_sha256": outer_context["context_sha256"],
        "outer_cgroup": outer_context["cgroup"],
        "outer_unit_name": outer_context["unit_name"],
        "tools": {"nvidia-smi": nvidia_smi, "nvcc": nvcc.stdout.strip()},
    }
    return _response(request, "ok", environment=environment)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-map", type=Path, required=True)
    parser.add_argument("operation", choices=("build", "validate", "benchmark", "probe"))
    parser.add_argument("request", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args(argv)
    request = _read_request(args.request)
    variant_map = load_variant_map(args.variant_map.resolve(strict=True))
    try:
        if args.operation == "build":
            if request.get("operation") != "build":
                raise AdapterError("build command received a different operation")
            response = _build_candidate(request, variant_map)
        elif args.operation == "validate":
            if request.get("operation") != "validate":
                raise AdapterError("validation command received a different operation")
            response = _validate_candidate(request, args.request, variant_map)
        elif args.operation == "benchmark":
            if request.get("operation") != "benchmark":
                raise AdapterError("benchmark command received a different operation")
            response = _measure_pair(
                request, args.request, variant_map, digest_file(args.variant_map),
                _measurement_command,
            )
        else:
            if request.get("operation") != "probe":
                raise AdapterError("probe command received a different operation")
            response = _probe(request, args.request)
    except Exception as exc:
        response = _response(request, "error", error=f"{type(exc).__name__}: {exc}")
        write_json(args.result, response)
        print(json.dumps({"status": "error", "error": response["error"]}, sort_keys=True), file=sys.stderr)
        return 1
    write_json(args.result, response)
    print(json.dumps({"status": response["status"], "operation": args.operation}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
