#!/usr/bin/env python3
"""Apply the fixed project gates to a completed, provisional KDA run.

This reads KDA's selected decision and raw paired response artifacts. It does
not re-rank candidates, write a KDA preferred pointer, or change production
routing. The output is conditional on external Scheduler receipt/cleanup
verification, or a retained rejection record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping

from common import (
    AdapterError,
    BENCH,
    checked_build,
    digest_file,
    digest_json,
    load_variant_map,
    read_json,
    write_json,
)
from adapter import (  # noqa: E402
    DESCRIPTIVE_DIRECT_ID,
    _candidate_mode_identity,
    _implementation_id,
    alternating_pair_order,
)
from in_job_kda import (
    CONTEXT_FILE,
    SCHEMA as OUTER_CONTEXT_SCHEMA,
    TRUSTED_ID as IN_JOB_TRUSTED_ID,
    scheduler_unit,
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _case_component(case_id: str) -> str:
    return _sha256_text(case_id)[:20]


def _candidate_identities(workspace: Path, summary: Mapping[str, Any],
                          variant_map: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    builds = summary.get("builds")
    if not isinstance(builds, dict):
        raise AdapterError("KDA summary has no candidate build identities")
    for candidate_id in variant_map["variants"]:
        candidate_path = workspace / "candidates.jsonl"
        rows = []
        for line in candidate_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("record_type") == "candidate" and row.get("candidate_id") == candidate_id:
                rows.append(row)
        if len(rows) != 1:
            raise AdapterError(f"expected one registered candidate record for {candidate_id}")
        build = builds.get(candidate_id)
        if not isinstance(build, dict):
            raise AdapterError(f"KDA summary has no build identity for {candidate_id}")
        registered = rows[0]
        if not isinstance(build.get("manifest_path"), str) or \
                digest_file(Path(build["manifest_path"])) != build.get("manifest_sha256"):
            raise AdapterError(f"KDA build manifest identity is stale for {candidate_id}")
        runner_path, native_manifest_path, runner_hash = checked_build(
            {"id": candidate_id, "snapshot_sha256": registered.get("snapshot_sha256")},
            build,
            variant_map["variants"][candidate_id],
            registered,
        )
        native_manifest = read_json(native_manifest_path)
        library_hash = native_manifest["binaries"]["factorized_interaction_cuda"]["sha256"]
        result[candidate_id] = {
            "candidate_id": candidate_id,
            "snapshot_sha256": registered.get("snapshot_sha256"),
            "candidate_manifest_sha256": registered.get("manifest_sha256"),
            "build_manifest_sha256": build.get("manifest_sha256"),
            "build_fingerprint": build.get("fingerprint"),
            "artifacts": build.get("artifacts"),
            "native_build_manifest_sha256": digest_file(native_manifest_path),
            "runner_path_in_build": str(runner_path),
            "runner_binary_sha256": runner_hash,
            "native_library_sha256": library_hash,
            "native_source_closure_sha256": native_manifest.get("source_closure_sha256"),
            "architecture": native_manifest.get("architecture"),
        }
    return result


def _validate_auxiliary_corpus(summary: Mapping[str, Any],
                               variant_map: Mapping[str, Any],
                               identities: Mapping[str, Mapping[str, Any]],
                               task: Mapping[str, Any]) -> dict[str, Any]:
    ids = task.get("numerical_requirements", {}).get(
        "auxiliary_correctness_workload_ids", []
    )
    if not isinstance(ids, list) or not ids:
        raise AdapterError("task omitted the frozen auxiliary correctness corpus")
    validation = summary.get("validation")
    if not isinstance(validation, dict):
        return {"status": "fail", "failures": ["KDA summary has no candidate validation evidence"],
                "candidate_results": {}}
    results = {}
    failures: list[str] = []
    for candidate_id in variant_map["variants"]:
        candidate_result = validation.get(candidate_id)
        if not isinstance(candidate_result, dict) or candidate_result.get("status") != "pass":
            failures.append(f"candidate {candidate_id} did not pass KDA correctness validation")
            results[candidate_id] = {"status": "fail", "workload_ids": ids,
                                     "reason": candidate_result.get("reason")
                                     if isinstance(candidate_result, dict) else "missing validation"}
            continue
        response = candidate_result.get("response")
        aux = response.get("auxiliary_correctness") if isinstance(response, dict) else None
        if not isinstance(aux, dict) or aux.get("status") != "pass":
            failures.append(f"candidate {candidate_id} did not pass the full auxiliary correctness corpus")
            results[candidate_id] = {"status": "fail", "workload_ids": ids,
                                     "cases": aux.get("cases", []) if isinstance(aux, dict) else []}
            continue
        rows = aux.get("cases")
        if not isinstance(rows, list) or [row.get("workload_id") for row in rows
                                          if isinstance(row, dict)] != ids:
            failures.append(f"candidate {candidate_id} auxiliary corpus coverage is incomplete or reordered")
            results[candidate_id] = {"status": "fail", "workload_ids": ids,
                                     "cases": rows if isinstance(rows, list) else []}
            continue
        if any(row.get("status") != "pass" or
               row.get("runner_binary_sha256") != identities[candidate_id]["runner_binary_sha256"]
               for row in rows):
            failures.append(f"candidate {candidate_id} has failed or unbound auxiliary correctness results")
            results[candidate_id] = {"status": "fail", "workload_ids": ids, "cases": rows}
            continue
        if candidate_id == variant_map["comparison"]["incumbent"]:
            bad_mode_rows = [row.get("workload_id") for row in rows
                             if not isinstance(row.get("direct_control_facts"), dict) or
                             row.get("candidate_argument") != row.get(
                                 "direct_control_facts", {}).get(
                                     "strongest_direct_candidate_argument")]
            if bad_mode_rows:
                failures.append(
                    f"incumbent did not use the generated-record strongest direct mode for {bad_mode_rows}"
                )
                results[candidate_id] = {
                    "status": "fail", "workload_ids": ids, "cases": rows,
                    "invalid_direct_mode_workloads": bad_mode_rows,
                }
                continue
        else:
            bad_mode_rows = [row.get("workload_id") for row in rows
                             if row.get("candidate_argument") !=
                             variant_map["variants"][candidate_id]["candidate_argument"]]
            if bad_mode_rows:
                failures.append(f"challenger correctness mode changed for {bad_mode_rows}")
                results[candidate_id] = {
                    "status": "fail", "workload_ids": ids, "cases": rows,
                    "invalid_candidate_mode_workloads": bad_mode_rows,
                }
                continue
        results[candidate_id] = {
            "status": "pass", "workload_ids": ids,
            "runner_binary_sha256": identities[candidate_id]["runner_binary_sha256"],
            "cases": rows,
        }
    return {"status": "pass" if not failures else "fail",
            "failures": failures, "candidate_results": results}


def _outer_execution_context(workspace: Path, task: Mapping[str, Any],
                             task_path: Path, summary: Mapping[str, Any]
                             ) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate recorded in-job identity without treating it as a receipt."""
    failures: list[str] = []
    context_path = workspace / CONTEXT_FILE
    if context_path.is_symlink() or not context_path.is_file():
        return None, ["in-job outer execution context is missing or non-regular"]
    context = read_json(context_path)
    target = task.get("target")
    factorized = task.get("factorized_interaction")
    environment = summary.get("environment")
    trusted = task.get("trusted_inputs")
    entrypoints = [item for item in trusted if isinstance(item, dict) and
                   item.get("id") == IN_JOB_TRUSTED_ID] if isinstance(trusted, list) else []
    if not isinstance(target, dict) or not isinstance(factorized, dict) or \
            not isinstance(environment, dict) or len(entrypoints) != 1:
        return context, ["task or KDA summary omitted outer execution bindings"]
    expected_uuid = factorized.get("expected_device_uuid")
    if not isinstance(expected_uuid, str) or target.get("expected_device_uuid") != expected_uuid:
        failures.append("task target and factorized expected UUID disagree")
    if factorized.get("outer_execution_context_schema") != OUTER_CONTEXT_SCHEMA or \
            factorized.get("outer_execution_context_file") != CONTEXT_FILE:
        failures.append("task outer-context schema or file binding differs from this guard")
    entrypoint_path_value = entrypoints[0].get("path")
    if not isinstance(entrypoint_path_value, str):
        failures.append("task in-job entrypoint binding has no path")
        entrypoint_hash = None
    else:
        entrypoint_path = Path(entrypoint_path_value)
        if entrypoint_path.is_symlink() or not entrypoint_path.is_file():
            failures.append("trusted in-job entrypoint is missing or non-regular")
            entrypoint_hash = None
        else:
            entrypoint_hash = digest_file(entrypoint_path)
            if factorized.get("in_job_entrypoint_path") != str(entrypoint_path.resolve()):
                failures.append("task factorized entrypoint path differs from its trusted input")
            if factorized.get("in_job_entrypoint_sha256") != entrypoint_hash:
                failures.append("task factorized entrypoint hash differs from its trusted input")

    if context.get("schema") != OUTER_CONTEXT_SCHEMA or context.get("status") != "observed":
        failures.append("outer context record is not an observation from the trusted in-job entrypoint")
    if context.get("task_sha256") != digest_file(task_path):
        failures.append("outer context record is bound to a different KDA task")
    if entrypoint_hash is None or context.get("entrypoint_sha256") != entrypoint_hash:
        failures.append("outer context record is bound to a different in-job entrypoint")
    if context.get("device_uuid") != expected_uuid:
        failures.append("outer context GPU UUID differs from the task's expected UUID")
    attempt_id = context.get("scheduler_attempt_id")
    job_id = context.get("scheduler_job_id")
    unit_name = context.get("unit_name")
    if not isinstance(job_id, str) or not job_id or \
            not isinstance(attempt_id, str) or not attempt_id or \
            unit_name != scheduler_unit(attempt_id):
        failures.append("outer context Scheduler unit does not match its attempt ID")
    cgroup = context.get("cgroup")
    if not isinstance(cgroup, str) or not cgroup.startswith("/") or \
            not isinstance(unit_name, str) or unit_name not in cgroup:
        failures.append("outer context cgroup does not identify its Scheduler attempt unit")
    pid = context.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        failures.append("outer context has no valid in-job process ID")
    context_hash = digest_file(context_path)
    for key, expected in (
        ("device_uuid", expected_uuid),
        ("scheduler_job_id", context.get("scheduler_job_id")),
        ("scheduler_attempt_id", context.get("scheduler_attempt_id")),
        ("outer_execution_context_sha256", context_hash),
        ("outer_cgroup", context.get("cgroup")),
        ("outer_unit_name", context.get("unit_name")),
    ):
        if environment.get(key) != expected:
            failures.append(f"KDA probe environment differs from recorded outer context: {key}")
    evidence = {
        "schema": context.get("schema"), "status": context.get("status"),
        "scheduler_job_id": context.get("scheduler_job_id"),
        "scheduler_attempt_id": context.get("scheduler_attempt_id"),
        "device_uuid": context.get("device_uuid"), "cgroup": context.get("cgroup"),
        "unit_name": context.get("unit_name"), "task_sha256": context.get("task_sha256"),
        "entrypoint_sha256": context.get("entrypoint_sha256"), "pid": context.get("pid"),
        "context_sha256": context_hash,
        "context_path": str(context_path.resolve()),
        "verification_status": "pending_external_scheduler_receipt_and_cleanup",
    }
    return evidence, failures


def _responses_for_run(workspace: Path, attempt: Path,
                       summary: Mapping[str, Any],
                       variant_map: Mapping[str, Any],
                       task: Mapping[str, Any],
                       identities: Mapping[str, Mapping[str, Any]]) -> dict[tuple[str, str, int], dict[str, dict[str, Any]]]:
    expected_ids = set(variant_map["variants"])
    cases_by_id = {case["id"]: case for case in task["cases"]}
    expected_cases = set(cases_by_id)
    records = summary.get("benchmark_records")
    if not isinstance(records, list):
        raise AdapterError("KDA summary has no benchmark records")
    grouped: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise AdapterError("KDA benchmark record is malformed")
        candidate_id = record.get("candidate_id")
        case_id = record.get("case_id")
        phase = record.get("phase")
        round_index = record.get("round")
        if candidate_id not in expected_ids or case_id not in expected_cases or \
                phase not in {"exploration", "confirmation"} or \
                isinstance(round_index, bool) or not isinstance(round_index, int):
            raise AdapterError("KDA benchmark record is outside this task's candidate/case scope")
        key = (phase, case_id, round_index)
        if candidate_id in grouped.setdefault(key, {}):
            raise AdapterError(f"duplicate benchmark evidence for {candidate_id}/{case_id}/{phase}/{round_index}")
        response_path = (
            attempt / "stages" / candidate_id / phase / f"round-{round_index:03d}" /
            _case_component(case_id) / "responses" / f"{record.get('request_id')}.json"
        )
        if response_path.is_symlink() or not response_path.is_file():
            raise AdapterError(f"KDA retained response is missing: {response_path}")
        response_hash = digest_file(response_path)
        if response_hash != record.get("response_sha256"):
            raise AdapterError(f"KDA response hash differs for {candidate_id}/{case_id}/{phase}/{round_index}")
        response = read_json(response_path)
        if response.get("status") != "ok" or response.get("case_id") != case_id:
            raise AdapterError("KDA response is not a successful matching workload result")
        if not isinstance(response.get("samples"), list) or len(response["samples"]) != 1:
            raise AdapterError("paired block must return exactly one raw sample to the engine")
        pair = response.get("paired_block")
        measurement = response.get("candidate_measurement")
        if not isinstance(pair, dict) or not isinstance(measurement, dict):
            raise AdapterError("response omitted paired order or candidate metric evidence")
        environment = summary.get("environment")
        outer_pair = pair.get("outer_execution_context")
        if not isinstance(environment, dict) or not isinstance(outer_pair, dict):
            raise AdapterError("paired result omitted its bound outer Scheduler context")
        for pair_key, environment_key in (
            ("outer_context_sha256", "outer_execution_context_sha256"),
            ("scheduler_job_id", "scheduler_job_id"),
            ("scheduler_attempt_id", "scheduler_attempt_id"),
            ("device_uuid", "device_uuid"),
            ("outer_cgroup", "outer_cgroup"),
            ("outer_unit_name", "outer_unit_name"),
        ):
            if outer_pair.get(pair_key) != environment.get(environment_key):
                raise AdapterError(f"paired block outer context differs from KDA probe: {pair_key}")
        if measurement.get("candidate_id") != candidate_id:
            raise AdapterError("response candidate metric identity mismatch")
        variant = variant_map["variants"][candidate_id]
        identity = identities[candidate_id]
        if measurement.get("source_snapshot_sha256") != identity.get("snapshot_sha256") or \
                measurement.get("runner_binary_sha256") != identity.get("runner_binary_sha256") or \
                measurement.get("native_build_manifest_sha256") != identity.get("native_build_manifest_sha256") or \
                measurement.get("native_library_sha256") != identity.get("native_library_sha256"):
            raise AdapterError("measurement is not bound to the registered source and built binaries")
        case_parameters = cases_by_id[case_id].get("parameters", {})
        direct_facts = case_parameters.get("direct_control") if isinstance(case_parameters, dict) else None
        if not isinstance(direct_facts, dict):
            raise AdapterError("timed case omitted its exact generated-record direct-control facts")
        expected_argument = (
            direct_facts.get("strongest_direct_candidate_argument")
            if candidate_id == variant_map["comparison"]["incumbent"]
            else variant["candidate_argument"]
        )
        if expected_argument not in {"packet", "minimal_direct", "direct_unique_output"} or \
                measurement.get("candidate_argument") != expected_argument:
            raise AdapterError("measurement executed a mode other than the case's fixed strongest-direct map")
        implementation = measurement.get("actual_implementation")
        layout = measurement.get("structural_layout")
        selected_backend = measurement.get("selected_backend")
        if selected_backend not in variant["allowed_backends"]:
            raise AdapterError("measurement selected a backend outside the fixed candidate map")
        if variant.get("allowed_implementations") is not None and \
                implementation not in variant["allowed_implementations"]:
            raise AdapterError("measurement selected an implementation outside the fixed candidate map")
        if variant.get("allowed_layouts") is not None and layout not in variant["allowed_layouts"]:
            raise AdapterError("measurement selected a layout outside the fixed candidate map")
        native_execution = measurement.get("native_execution")
        metrics = measurement.get("metrics")
        provenance = measurement.get("build_provenance")
        if not isinstance(native_execution, dict) or not isinstance(metrics, dict) or \
                not isinstance(provenance, dict):
            raise AdapterError("measurement omitted native identity or metrics")
        for identity_field, expected in (("selected_backend", selected_backend),
                                         ("actual_implementation", implementation),
                                         ("structural_layout", layout)):
            if native_execution.get(identity_field) != expected or metrics.get(identity_field) != expected:
                raise AdapterError(f"native metadata and metrics disagree on {identity_field}")
        if native_execution.get("candidate") != expected_argument or \
                native_execution.get("requested_route") != variant["route"] or \
                measurement.get("requested_route") != variant["route"] or \
                provenance.get("runner_binary_sha256") != identity.get("runner_binary_sha256") or \
                provenance.get("source_closure_sha256") != identity.get("native_source_closure_sha256"):
            raise AdapterError("runner execution provenance does not match registered candidate binaries")
        _candidate_mode_identity(
            native_execution, metrics, variant,
            candidate_argument=expected_argument,
            direct_control_facts=direct_facts,
        )
        if measurement.get("direct_control_facts") != direct_facts:
            raise AdapterError("measurement omitted or changed the fixed direct-control facts")
        if measurement.get("sample") != response["samples"][0]:
            raise AdapterError("KDA sample differs from the measured primary host-wall value")

        # The descriptive native-direct arm is measured in the same order block,
        # but it is not part of KDA's paired score. Bind it to the same runner,
        # source, record facts, and actual backend identity for later inspection.
        descriptive = response.get("descriptive_control_measurement")
        marker = pair.get("descriptive_control")
        if not isinstance(descriptive, dict) or not isinstance(marker, dict):
            raise AdapterError("paired block omitted its descriptive native-direct control")
        direct_variant = variant_map["variants"][variant_map["comparison"]["incumbent"]]
        if descriptive.get("candidate_id") != DESCRIPTIVE_DIRECT_ID or \
                descriptive.get("candidate_argument") != "native_direct" or \
                descriptive.get("requested_route") != "direct" or \
                descriptive.get("source_snapshot_sha256") != identities[
                    variant_map["comparison"]["incumbent"]
                ].get("snapshot_sha256") or \
                descriptive.get("runner_binary_sha256") != identities[
                    variant_map["comparison"]["incumbent"]
                ].get("runner_binary_sha256") or \
                descriptive.get("native_library_sha256") != identities[
                    variant_map["comparison"]["incumbent"]
                ].get("native_library_sha256"):
            raise AdapterError("descriptive native-direct control is not bound to the incumbent source and binary")
        descriptive_native = descriptive.get("native_execution")
        descriptive_metrics = descriptive.get("metrics")
        descriptive_provenance = descriptive.get("build_provenance")
        if not isinstance(descriptive_native, dict) or not isinstance(descriptive_metrics, dict) or \
                not isinstance(descriptive_provenance, dict) or \
                descriptive_provenance.get("runner_binary_sha256") != identities[
                    variant_map["comparison"]["incumbent"]
                ].get("runner_binary_sha256") or \
                descriptive_provenance.get("source_closure_sha256") != identities[
                    variant_map["comparison"]["incumbent"]
                ].get("native_source_closure_sha256"):
            raise AdapterError("descriptive direct-control metadata omitted binary provenance")
        descriptive_impl, descriptive_layout = _candidate_mode_identity(
            descriptive_native, descriptive_metrics, direct_variant,
            candidate_argument="native_direct", direct_control_facts=direct_facts,
        )
        if descriptive.get("actual_implementation") != descriptive_impl or \
                descriptive.get("structural_layout") != descriptive_layout or \
                descriptive_native.get("candidate") != "native_direct" or \
                descriptive_native.get("requested_route") != "direct" or \
                marker.get("candidate_id") != DESCRIPTIVE_DIRECT_ID or \
                marker.get("candidate_argument") != "native_direct" or \
                marker.get("actual_implementation") != descriptive_impl or \
                marker.get("structural_layout") != descriptive_layout:
            raise AdapterError("descriptive direct-control identity differs from native dispatch")
        primary_metric = task.get("timing", {}).get("metric")
        descriptive_sample = descriptive.get("sample")
        if isinstance(descriptive_sample, bool) or not isinstance(descriptive_sample, (int, float)) or \
                not math.isfinite(float(descriptive_sample)) or float(descriptive_sample) <= 0 or \
                marker.get("sample") != descriptive_sample or \
                descriptive_metrics.get("phase_ms", {}).get(primary_metric) != descriptive_sample:
            raise AdapterError("descriptive control omitted the same positive primary host-wall sample")
        descriptive_raw = descriptive_metrics.get("samples_ms")
        iterations = task.get("factorized_interaction", {}).get(
            "runner_iterations_per_paired_block")
        if not isinstance(descriptive_raw, dict) or not isinstance(iterations, int) or any(
                not isinstance(descriptive_raw.get(name), list) or
                len(descriptive_raw[name]) != iterations or any(
                    isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(float(value)) or float(value) < 0.0
                    for value in descriptive_raw[name]
                ) for name in ("forward", "backward", "forward_vjp", "parameter_refresh")):
            raise AdapterError("descriptive control omitted complete secondary timing vectors")
        raw_vectors = metrics.get("samples_ms")
        iterations = task.get("factorized_interaction", {}).get(
            "runner_iterations_per_paired_block")
        if not isinstance(raw_vectors, dict) or not isinstance(iterations, int):
            raise AdapterError("measurement omitted raw metric vectors or the iteration count")
        for phase_name in ("forward", "backward", "forward_vjp", "parameter_refresh"):
            values = raw_vectors.get(phase_name)
            if not isinstance(values, list) or len(values) != iterations or any(
                    isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(float(value)) or float(value) < 0.0 for value in values):
                raise AdapterError(f"measurement raw {phase_name} samples are incomplete or invalid")
        grouped[key][candidate_id] = {
            "response": response,
            "response_sha256": response_hash,
            "paired_block": pair,
            "measurement": measurement,
            "sample": float(response["samples"][0]),
        }
    for key, rows in grouped.items():
        if set(rows) != expected_ids:
            continue
        first_id, second_id = sorted(expected_ids)
        desc_a = rows[first_id]["response"].get("descriptive_control_measurement")
        desc_b = rows[second_id]["response"].get("descriptive_control_measurement")
        if digest_json(desc_a) != digest_json(desc_b):
            raise AdapterError(f"paired responses disagree on descriptive control for {key}")
    return grouped


def _guard(
    *, task: Mapping[str, Any], task_path: Path, workspace: Path,
    attempt: Path, summary: Mapping[str, Any],
    variant_map: Mapping[str, Any], summary_path: Path,
) -> dict[str, Any]:
    comparison = variant_map["comparison"]
    incumbent = comparison["incumbent"]
    challenger = comparison["challenger"]
    root_policy = task.get("numerical_requirements", {}).get("root_selection_policy")
    if not isinstance(root_policy, dict) or root_policy.get("experiment_only") is not True:
        raise AdapterError("task does not contain the fixed experiment-only root gate")
    expected_policy = {
        "minimum_paired_blocks_per_case": 7,
        "minimum_confirmation_blocks_per_case": 7,
        "required_candidate_win_point_speedup": 1.10,
        "required_candidate_win_ci_low_exclusive": 1.0,
        "maximum_supported_case_time_regression": 0.05,
        "kda_max_regression_benefit_units": 1.0 - (1.0 / 1.05),
    }
    for key, expected in expected_policy.items():
        actual = root_policy.get(key)
        if isinstance(expected, float):
            if not isinstance(actual, (int, float)) or not math.isclose(
                    float(actual), expected, rel_tol=0.0, abs_tol=1e-15):
                raise AdapterError(f"task root policy differs from fixed criterion: {key}")
        elif actual != expected:
            raise AdapterError(f"task root policy differs from fixed criterion: {key}")
    if task.get("promotion", {}).get("min_improvement") != 0.0 or \
            not math.isclose(float(task["promotion"].get("max_regression", -1.0)),
                             expected_policy["kda_max_regression_benefit_units"],
                             rel_tol=0.0, abs_tol=1e-15) or \
            task["promotion"].get("confidence") != 0.95:
        raise AdapterError("KDA engine thresholds do not match the frozen root gate")
    if task.get("timing", {}).get("rounds") != 7 or \
            task["timing"].get("confirmation_rounds") != 7:
        raise AdapterError("KDA task does not provide the required seven paired blocks per phase")

    failures: list[str] = []
    decision = summary.get("decision")
    if not isinstance(decision, dict):
        decision = {"eligible": False, "reason": "KDA summary omitted a decision"}
        failures.append("KDA summary omitted a provisional engine decision")
    elif decision.get("eligible") is not True or decision.get("candidate_id") != challenger:
        failures.append("KDA's provisional engine decision did not select this challenger")
    if not isinstance(decision.get("ci_low"), (int, float)) or \
            not float(decision.get("ci_low", float("nan"))) > 1.0:
        failures.append("KDA aggregate paired 95% confidence lower bound does not exceed 1.0")
    if summary.get("status") != "complete" or summary.get("run_id") != attempt.name:
        raise AdapterError("KDA run summary is incomplete or has a mismatched run ID")
    if summary.get("task_sha256") != digest_file(task_path):
        raise AdapterError("KDA run used a different task JSON")
    if summary.get("trusted_inputs_unchanged") is not True or \
            summary.get("tools_unchanged") is not True:
        raise AdapterError("KDA run reports changed trusted inputs or tools")
    receipt = summary.get("execution_receipt")
    if not isinstance(receipt, dict) or receipt.get("status") != "ok" or \
            receipt.get("mode") != "direct":
        failures.append("KDA did not record successful completion through its direct engine mode")
    if task.get("execution", {}).get("mode") != "direct":
        raise AdapterError("CUDA task must use KDA direct mode inside the verified outer job")
    outer_context, outer_context_failures = _outer_execution_context(
        workspace, task, task_path, summary,
    )
    failures.extend(outer_context_failures)

    source_binary_identities = _candidate_identities(workspace, summary, variant_map)
    auxiliary_correctness = _validate_auxiliary_corpus(
        summary, variant_map, source_binary_identities, task,
    )
    failures.extend(auxiliary_correctness.get("failures", []))
    grouped = _responses_for_run(
        workspace, attempt, summary, variant_map, task, source_binary_identities,
    )
    min_blocks = int(task["timing"]["confirmation_rounds"])
    case_decisions = decision.get("per_case")
    if not isinstance(case_decisions, dict):
        case_decisions = {}
        failures.append("KDA confirmation decision has no per-case bootstrap evidence")
    regression_floor = 1.0 - float(task["promotion"]["max_regression"])
    win_threshold = float(root_policy["required_candidate_win_point_speedup"])
    ci_win_floor = float(root_policy["required_candidate_win_ci_low_exclusive"])

    per_case: dict[str, Any] = {}
    routes: dict[str, Any] = {}
    task_cases_by_id = {case["id"]: case for case in task["cases"]}
    for case in task["cases"]:
        case_failure_start = len(failures)
        case_id = case["id"]
        required_win = bool(case.get("parameters", {}).get("candidate_win_required"))
        confirmation_rows = []
        exploration_rows = []
        for phase, destination in (("exploration", exploration_rows),
                                   ("confirmation", confirmation_rows)):
            for round_index in range(int(task["timing"]["rounds"] if phase == "exploration"
                                         else task["timing"]["confirmation_rounds"])):
                rows = grouped.get((phase, case_id, round_index))
                if rows is None or set(rows) != {incumbent, challenger}:
                    failures.append(f"missing paired response {phase}/{case_id}/{round_index}")
                    continue
                block_a = rows[incumbent]["paired_block"]
                block_b = rows[challenger]["paired_block"]
                if block_a.get("pair_key") != block_b.get("pair_key") or \
                        block_a.get("order") != block_b.get("order"):
                    failures.append(f"candidate responses disagree on paired block {phase}/{case_id}/{round_index}")
                phase_offset = 0 if phase == "exploration" else 7
                expected_order = [incumbent, challenger] if (round_index + phase_offset) % 2 == 0 \
                    else [challenger, incumbent]
                if block_a.get("order") != expected_order:
                    failures.append(f"AB/BA order failed for {phase}/{case_id}/{round_index}")
                expected_description_position = (round_index + phase_offset) % 3
                expected_execution_order = list(expected_order)
                expected_execution_order.insert(expected_description_position, DESCRIPTIVE_DIRECT_ID)
                if block_a.get("execution_order") != expected_execution_order or \
                        block_a.get("descriptive_control", {}).get("position") != expected_description_position:
                    failures.append(f"descriptive-control order failed for {phase}/{case_id}/{round_index}")
                destination.append(rows)
        if len(confirmation_rows) != min_blocks:
            failures.append(f"confirmation block count is not seven for {case_id}")
            continue
        incumbent_samples = [rows[incumbent]["sample"] for rows in confirmation_rows]
        challenger_samples = [rows[challenger]["sample"] for rows in confirmation_rows]
        if any(not math.isfinite(value) or value <= 0 for value in incumbent_samples + challenger_samples):
            failures.append(f"non-finite or non-positive sample for {case_id}")
            continue
        point_speedup = statistics.median(incumbent_samples) / statistics.median(challenger_samples)
        decision_case = case_decisions.get(case_id)
        if not isinstance(decision_case, dict):
            failures.append(f"KDA omitted per-case decision for {case_id}")
            continue
        engine_score = float(decision_case.get("score", float("nan")))
        ci_low = float(decision_case.get("ci_low", float("nan")))
        if not math.isfinite(engine_score) or not math.isclose(
                point_speedup, engine_score, rel_tol=1e-10, abs_tol=1e-12):
            failures.append(f"recomputed point median differs from KDA for {case_id}")
        if not math.isfinite(ci_low) or ci_low < regression_floor:
            failures.append(f"per-case time regression exceeds 5% for {case_id}")

        incumbent_backends = sorted({
            rows[incumbent]["measurement"].get("actual_implementation") or
            rows[incumbent]["measurement"].get("selected_backend")
            for rows in confirmation_rows
        })
        challenger_backends = sorted({
            rows[challenger]["measurement"].get("actual_implementation") or
            rows[challenger]["measurement"].get("selected_backend")
            for rows in confirmation_rows
        })
        incumbent_layouts = sorted({
            rows[incumbent]["measurement"].get("structural_layout")
            for rows in confirmation_rows
        })
        challenger_layouts = sorted({
            rows[challenger]["measurement"].get("structural_layout")
            for rows in confirmation_rows
        })
        variant = variant_map["variants"][challenger]
        required_backend = variant.get("win_backend")
        required_layout = variant.get("win_layout")
        backend_win = (len(challenger_backends) == 1 and
                       challenger_backends[0] == required_backend and
                       len(challenger_layouts) == 1 and
                       challenger_layouts[0] == required_layout and
                       all(row[challenger]["paired_block"].get("backend_pair_distinct")
                           for row in confirmation_rows))
        if required_win:
            if not backend_win:
                failures.append(f"required win case {case_id} did not execute the two declared candidate backends")
            if point_speedup < win_threshold:
                failures.append(f"required win case {case_id} is below 1.10x point median")
            if ci_low <= ci_win_floor:
                failures.append(f"required win case {case_id} does not have a 95% lower bound above 1.0x")
        case_pass = len(failures) == case_failure_start and ci_low >= regression_floor
        accepted_route = (variant.get("accepted_route", variant["route"])
                          if required_win and case_pass else
                          variant_map["variants"][incumbent].get("accepted_route", "direct"))
        accepted_candidate_id = challenger if required_win and case_pass else incumbent
        accepted_variant = variant_map["variants"][accepted_candidate_id]
        accepted_argument = (
            case.get("parameters", {}).get("direct_control", {}).get(
                "strongest_direct_candidate_argument"
            ) if accepted_candidate_id == incumbent else
            accepted_variant["candidate_argument"]
        )
        routes[case_id] = {
            "route": accepted_route,
            "candidate_id": accepted_candidate_id,
            "candidate_argument": accepted_argument,
            "fallback_route": "direct",
            "candidate_win_required": required_win,
            "measured": True,
            "reason": "candidate met its declared per-case gates" if
                accepted_candidate_id == challenger else
                "direct control retained for this exact case",
        }
        per_case[case_id] = {
            "point_speedup": point_speedup,
            "engine_ci_low_speedup": ci_low,
            "engine_ci_high_speedup": decision_case.get("ci_high"),
            "max_time_regression_threshold_benefit": regression_floor,
            "candidate_win_point_threshold": win_threshold if required_win else None,
            "incumbent_backends": incumbent_backends,
            "incumbent_layouts": incumbent_layouts,
            "challenger_backends": challenger_backends,
            "challenger_layouts": challenger_layouts,
            "backend_pair_distinct_per_confirmation_block": [
                rows[incumbent]["paired_block"].get("backend_pair_distinct")
                for rows in confirmation_rows
            ],
            "confirmation_samples_ms": {
                incumbent: incumbent_samples,
                challenger: challenger_samples,
            },
            "exploration_samples_ms": {
                incumbent: [row[incumbent]["sample"] for row in exploration_rows],
                challenger: [row[challenger]["sample"] for row in exploration_rows],
            },
            "ancillary_metrics": {
                candidate_id: {
                    "exploration": _summarize_metrics(
                        rows[candidate_id]["measurement"]["metrics"]
                        for rows in exploration_rows
                    ),
                    "confirmation": _summarize_metrics(
                        rows[candidate_id]["measurement"]["metrics"]
                        for rows in confirmation_rows
                    ),
                }
                for candidate_id in (incumbent, challenger)
            },
            "descriptive_native_direct_control": {
                "candidate_argument": "native_direct",
                "primary_metric": task["timing"]["metric"],
                "exploration_primary_samples_ms": [
                    row[incumbent]["response"]["descriptive_control_measurement"]["sample"]
                    for row in exploration_rows
                ],
                "confirmation_primary_samples_ms": [
                    row[incumbent]["response"]["descriptive_control_measurement"]["sample"]
                    for row in confirmation_rows
                ],
                "exploration_backend_layouts": sorted({
                    _implementation_id(row[incumbent]["response"][
                        "descriptive_control_measurement"])
                    for row in exploration_rows
                }),
                "confirmation_backend_layouts": sorted({
                    _implementation_id(row[incumbent]["response"][
                        "descriptive_control_measurement"])
                    for row in confirmation_rows
                }),
                "ancillary_metrics": {
                    "exploration": _summarize_metrics(
                        row[incumbent]["response"]["descriptive_control_measurement"]["metrics"]
                        for row in exploration_rows
                    ),
                    "confirmation": _summarize_metrics(
                        row[incumbent]["response"]["descriptive_control_measurement"]["metrics"]
                        for row in confirmation_rows
                    ),
                },
            },
            "case_gates_passed": case_pass,
        }

    factorized = task.get("factorized_interaction", {})
    sys.path.insert(0, str(BENCH.parent.parent / "tests" / "factorized_interactions"))
    from workloads import (correctness_workload_ids,
                           performance_workload_ids,
                           workload_ids)  # noqa: E402

    complete_performance = list(performance_workload_ids())
    complete_correctness = list(correctness_workload_ids())
    all_fixed = list(workload_ids())
    timed_ids = [case["id"] for case in task["cases"]]
    if factorized.get("all_fixed_workload_ids") != all_fixed or \
            factorized.get("timed_performance_case_ids") != timed_ids or \
            factorized.get("correctness_only_case_ids_direct_unmeasured") != complete_correctness:
        raise AdapterError("task case metadata differs from the frozen workload catalog")
    expected_unmeasured = sorted(set(complete_performance) - set(timed_ids))
    if factorized.get("unmeasured_performance_case_ids_direct_fallback") != expected_unmeasured:
        raise AdapterError("task omitted performance cases without a direct fallback declaration")
    incumbent_variant = variant_map["variants"][incumbent]
    direct_facts = factorized.get("direct_control_facts_by_performance_case", {})
    for case_id in [case["id"] for case in task["cases"]]:
        routes.setdefault(case_id, {
            "route": "direct", "candidate_id": incumbent,
            "candidate_argument": task_cases_by_id.get(case_id, {}).get("parameters", {}).get(
                "direct_control", {}).get("strongest_direct_candidate_argument",
                incumbent_variant["candidate_argument"]),
            "fallback_route": "direct", "candidate_win_required": False,
            "measured": True, "reason": "incomplete paired evidence; retain direct control",
        })
    for case_id in expected_unmeasured:
        routes[case_id] = {
            "route": "direct", "candidate_id": incumbent,
            "candidate_argument": direct_facts.get(case_id, {}).get(
                "strongest_direct_candidate_argument", incumbent_variant["candidate_argument"]),
            "fallback_route": "direct", "candidate_win_required": False,
            "measured": False, "reason": "outside the declared timing regime",
        }
    for case_id in complete_correctness:
        auxiliary_facts = {}
        incumbent_aux = auxiliary_correctness.get("candidate_results", {}).get(
            incumbent, {}).get("cases", [])
        for row in incumbent_aux:
            if isinstance(row, dict) and row.get("workload_id") == case_id:
                facts = row.get("direct_control_facts")
                if isinstance(facts, dict):
                    auxiliary_facts = facts
                break
        routes[case_id] = {
            "route": "direct", "candidate_id": incumbent,
            "candidate_argument": auxiliary_facts.get(
                "strongest_direct_candidate_argument", incumbent_variant["candidate_argument"]),
            "fallback_route": "direct", "candidate_win_required": False,
            "measured": False,
            "reason": "correctness-only workload; validated but not timed for route selection",
        }
    for case_id, route in routes.items():
        route.setdefault("candidate_id", challenger if route["route"] != "direct" else incumbent)
        selected_variant = variant_map["variants"][route["candidate_id"]]
        route.setdefault("candidate_argument", selected_variant["candidate_argument"])
        route.setdefault("measured", case_id in timed_ids)
    root_pass = not failures and len(per_case) == len(task["cases"])
    return {
        "schema": "cellerator.factorized_interactions.conditional_selection.v1" if root_pass
        else "cellerator.factorized_interactions.rejected_selection.v1",
        "status": "conditional" if root_pass else "rejected",
        "gate_status": "pass" if root_pass else "rejected",
        "experiment_only": True,
        "selection_scope": "exact_benchmark_case_ids",
        "production_authority": "none; outer receipt/cleanup matching and root acceptance are pending",
        "outer_receipt_verification": "pending",
        "outer_execution_context": outer_context,
        "kda_internal_execution_receipt": receipt,
        "task_id": task["task_id"],
        "task_sha256": digest_file(task_path),
        "run_id": summary["run_id"],
        "kda_summary_sha256": digest_file(summary_path),
        "comparison_group": summary.get("comparison_group"),
        "provisional_engine_preferred_candidate": summary.get("preferred", {}).get("candidate_id"),
        "engine_decision": decision,
        "root_guard": {
            "minimum_point_speedup": win_threshold,
            "minimum_ci_low_speedup_exclusive": ci_win_floor,
            "maximum_time_regression": 0.05,
            "minimum_paired_blocks_per_phase": 7,
            "paired_order_verified": not any("AB/BA order" in reason for reason in failures),
            "all_supported_case_regressions_within_limit": not any("regression" in reason for reason in failures),
            "candidate_win_cases_passed": not any("required win case" in reason for reason in failures),
            "failures": failures,
        },
        "incumbent_id": incumbent,
        "candidate_id": challenger,
        "source_binary_identities": source_binary_identities,
        "strongest_direct_control": {
            "candidate_id": incumbent,
            "identity": source_binary_identities[incumbent],
            "qualification_basis": "passed this run's source-bound full correctness validation; paired performance control",
        },
        "auxiliary_full_correctness_validation": auxiliary_correctness,
        "routes_by_case": routes,
        "per_case_evidence": per_case,
        "source_manifest_sha256": digest_file(
            Path(task["factorized_interaction"]["variant_map_path"])
        ),
        "production_route_changed": False,
    }


def _summarize_metrics(records) -> dict[str, Any]:
    values: dict[str, list[float]] = {}
    phase_samples: dict[str, list[float]] = {}
    backend_ids: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        if isinstance(record.get("selected_backend"), str):
            backend_ids.add(record["selected_backend"])
        phases = record.get("phase_ms")
        if isinstance(phases, dict):
            for key, value in phases.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    values.setdefault(key, []).append(float(value))
                    phase_samples.setdefault(key, []).append(float(value))
    raw_vectors: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        metrics = record.get("metrics", record)
        if not isinstance(metrics, dict):
            continue
        vectors = metrics.get("samples_ms")
        if isinstance(vectors, dict):
            raw_vectors.append({"block_index": index, **vectors})
    return {
        "summary": "median across paired blocks; setup and conversion remain separate from the primary resident metric",
        "parameter_refresh_scope": "host wall time for staging and synchronously completing three parameter uploads; same payloads are reuploaded and no optimizer update occurs",
        "backend_ids": sorted(backend_ids),
        "phase_ms_medians": {key: statistics.median(series) for key, series in sorted(values.items())},
        "phase_ms_by_block": phase_samples,
        "raw_samples_ms_by_block": raw_vectors,
    }


def guard_run(task_path: Path, workspace: Path, run_id: str,
              variant_map_path: Path, output_path: Path) -> dict[str, Any]:
    task_path = task_path.resolve(strict=True)
    workspace = workspace.resolve(strict=True)
    attempt = workspace / "runs" / "kda-experiments" / "attempts" / run_id
    summary_path = attempt / "summary.json"
    if summary_path.is_symlink() or not summary_path.is_file():
        raise AdapterError("completed KDA summary is missing")
    summary = read_json(summary_path)
    task = read_json(task_path)
    variant_map_path = variant_map_path.resolve(strict=True)
    if task.get("factorized_interaction", {}).get("variant_map_sha256") != digest_file(variant_map_path):
        raise AdapterError("variant map differs from the exact task-bound bytes")
    variant_map = load_variant_map(variant_map_path)
    if output_path.resolve().is_relative_to(workspace):
        raise AdapterError("selection evidence output must live outside the provisional KDA workspace")
    result = _guard(
        task=task, task_path=task_path, workspace=workspace, attempt=attempt,
        summary=summary, variant_map=variant_map, summary_path=summary_path,
    )
    write_json(output_path, result, exclusive=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--variant-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = guard_run(args.task, args.workspace, args.run_id,
                           args.variant_map, args.output)
    except Exception as exc:
        parser.error(f"selection guard rejected the KDA run: {exc}")
    print(json.dumps({"status": result["status"], "run_id": result["run_id"],
                      "candidate_id": result["candidate_id"],
                      "production_route_changed": False,
                      "output": str(args.output.resolve())}, sort_keys=True))
    return 0 if result["status"] == "conditional" else 3


if __name__ == "__main__":
    raise SystemExit(main())
