#!/usr/bin/env python3
"""Summarize complete costs and resource fields from qualified study packets.

The exporter trusts the qualification export for receipt, source-closure,
binary, input, correctness and cleanup joins. It verifies each referenced raw
campaign record hash again, preserves packet identity, and never schedules or
profiles work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


STUDY = Path(__file__).resolve().parents[1]
DEFAULT_QUALIFICATION = STUDY / "analysis/qualification.json"
TIME_FIELDS = ("recurrent_complete_horizon_ms", "output_recovery_ms")
IDENTITY_FIELDS = (
    "case", "gate", "body", "precision", "condition_update", "input",
    "seed", "reuse_horizon", "parameter_update",
    "nominal_input_change_percent", "applied_input_change_percent",
    "workload_sha256", "tile", "stages",
)
MEMORY_FIELDS = (
    "workspace_bytes", "condition_device_bytes", "library_owned_device_bytes",
    "library_pinned_host_bytes", "net_device_allocated_bytes", "device_total_bytes",
)
COLD_COMPONENT_FIELDS = (
    "replay_input_preparation_ms", "host_layout_preparation_ms",
    "cold_device_preparation_ms",
)
EXCLUDED_DIAGNOSTIC_FIELDS = (
    "host_fixture_ms", "replay_reference_recompute_ms",
    "host_layout_validation_ms", "validation_ms",
)
TEACHER_FIELDS = (
    "teacher_uniform_soft_hard_mean", "teacher_workload_weighted_soft_hard_mean",
    "teacher_observed_rows", "teacher_validation_threshold_crossings",
    "teacher_fit_every_step",
)


def fail(message: str) -> None:
    raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        fail(f"expected JSON object: {path}")
    return value


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        fail(f"{label} must be a finite number")
    return float(value)


def finite_list(value: Any, label: str, *, allow_empty: bool = False) -> list[float]:
    if not isinstance(value, list) or (not value and not allow_empty):
        fail(f"{label} must be a {'possibly empty ' if allow_empty else ''}list of finite numbers")
    return [finite_number(item, f"{label}[{index}]") for index, item in enumerate(value)]


def median(values: list[float], label: str) -> float:
    if not values:
        fail(f"cannot summarize empty {label}")
    return float(statistics.median(values))


def quantile_linear(values: list[float], fraction: float) -> float:
    """Linear percentile without adding a numpy dependency."""
    ordered = sorted(values)
    if not ordered:
        fail("cannot take a percentile of empty values")
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def round_value(raw: dict[str, Any], packet: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    require_identity = {
        "backend": record.get("backend"),
        "device_uuid": packet.get("device_uuid"),
    }
    for field, expected in require_identity.items():
        if raw.get(field) != expected:
            fail(f"raw record {record.get('path')} has mismatched {field}")
    if raw.get("status") != "ok" or raw.get("accuracy_accepted") is not True:
        fail(f"raw record {record.get('path')} is not a successful accepted campaign")
    if raw.get("validation_scope") != "first-and-last-complete-steps":
        fail(f"raw record {record.get('path')} has unexpected validation scope")

    raw_identity = {field: raw.get(field) for field in IDENTITY_FIELDS}
    if any(value is None for value in raw_identity.values()):
        fail(f"raw record {record.get('path')} lacks a scenario identity field")
    if raw.get("schema") != "cgs.campaign.v1":
        fail(f"raw record {record.get('path')} has unsupported campaign schema")
    horizon = raw.get("reuse_horizon")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
        fail("reuse_horizon must be a positive integer")

    cold = finite_number(raw.get("cold_execution_preparation_ms"), "cold_execution_preparation_ms")
    cold_parts = {field: finite_number(raw.get(field), field) for field in COLD_COMPONENT_FIELDS}
    if any(value < 0 for value in [cold, *cold_parts.values()]) or not math.isclose(cold, sum(cold_parts.values()), rel_tol=1e-9, abs_tol=1e-8):
        fail("cold preparation does not match its recorded operational components")
    excluded = {field: finite_number(raw.get(field), field) for field in EXCLUDED_DIAGNOSTIC_FIELDS}
    if any(value < 0 for value in excluded.values()):
        fail("excluded diagnostic measurements cannot be negative")
    replay_bytes = {}
    for field in ("replay_payload_bytes_hashed", "replay_payload_bytes_read"):
        value = raw.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            fail(f"{field} must be a nonnegative byte count")
        replay_bytes[field] = value
    recurrent = finite_list(raw.get("recurrent_complete_horizon_ms"), "recurrent_complete_horizon_ms")
    recovery = finite_list(raw.get("output_recovery_ms"), "output_recovery_ms")
    if len(recurrent) != len(recovery):
        fail("recurrent and output-recovery samples must be paired")
    if any(value < 0 for value in [cold, *recurrent, *recovery]):
        fail("complete-cost components cannot be negative")

    complete_us = [1000.0 * (cold + r + output) for r, output in zip(recurrent, recovery)]
    timing = {
        "cold_execution_preparation_ms": cold,
        "recurrent_samples_ms": recurrent,
        "output_recovery_samples_ms": recovery,
        "complete_operational_samples_us": complete_us,
        "complete_operational_round_median_us": median(complete_us, "complete operational samples"),
        "recurrent_horizon_round_median_ms": median(recurrent, "recurrent horizon samples"),
        "output_recovery_round_median_ms": median(recovery, "output recovery samples"),
        "recurrent_per_evaluation_round_median_ms": median(recurrent, "recurrent horizon samples") / horizon,
        "linear_model_intercept_round_ms": cold + median(recovery, "output recovery samples"),
    }

    phases: dict[str, Any] = {}
    for key in ("phase_probe_host_ms", "phase_probe_device_ms"):
        phase_map = raw.get(key)
        if not isinstance(phase_map, dict):
            fail(f"{key} must be an object")
        clean = {}
        for phase, values in sorted(phase_map.items()):
            clean[phase] = finite_list(values, f"{key}.{phase}", allow_empty=True)
        phases[key] = clean

    memory = {}
    for key in MEMORY_FIELDS:
        value = raw.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            fail(f"{key} must be a nonnegative integer byte count")
        memory[key] = value

    # The runner emits zero-filled teacher fields for non-teacher routes.
    # Treat them as not applicable instead of reporting fake zero change.
    teacher = ({field: raw.get(field) for field in TEACHER_FIELDS}
               if raw.get("gate") == "teacher" else {field: None for field in TEACHER_FIELDS})
    for key in ("teacher_uniform_soft_hard_mean", "teacher_workload_weighted_soft_hard_mean"):
        if teacher[key] is not None:
            value = finite_number(teacher[key], key)
            if value < 0 or value > 1:
                fail(f"{key} must be between 0 and 1")
            teacher[key] = value
    rows = teacher["teacher_observed_rows"]
    if rows is not None and (not isinstance(rows, list) or any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in rows)):
        fail("teacher_observed_rows must be a list of nonnegative counts")
    crossings = teacher["teacher_validation_threshold_crossings"]
    if crossings is not None and (not isinstance(crossings, int) or isinstance(crossings, bool) or crossings < 0):
        fail("teacher_validation_threshold_crossings must be a nonnegative integer")

    for key in ("nominal_input_change_percent", "applied_input_change_percent"):
        value = raw[key]
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 100:
            fail(f"{key} must be an integer percentage")
    root_changes = {}
    for field in ("nominal_root_changes", "applied_root_changes_per_toggle"):
        value = raw.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            fail(f"{field} must be a nonnegative integer count")
        root_changes[field] = value
    predicate_updates_applicable = raw.get("predicate_updates_applicable")
    if not isinstance(predicate_updates_applicable, bool):
        fail("predicate_updates_applicable must be Boolean")

    return {
        "round": record.get("round"),
        "backend": raw["backend"],
        "raw_record_path": record.get("path"),
        "raw_record_sha256": record.get("sha256"),
        "identity": raw_identity,
        "timing": timing,
        "cold_preparation_components_ms": cold_parts,
        "excluded_preparation_diagnostics_ms": excluded,
        "replay_payload_bytes": replay_bytes,
        "root_changes": root_changes,
        "ste_derivative_candidates": raw.get("ste_derivative_candidates"),
        "predicate_updates_applicable": predicate_updates_applicable,
        "phase_probe_scope": raw.get("phase_probe_scope"),
        "phase_probe": phases,
        "memory_bytes": memory,
        "teacher": teacher,
        "teacher_input_rule": raw.get("teacher_input_rule"),
        "teacher_initial_fit_in_recurrent_horizon": raw.get("teacher_initial_fit_in_recurrent_horizon"),
    }


def load_record_set(
    study: Path,
    packet: dict[str, Any],
    references: Any,
    *,
    update_mode: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    provenance = packet.get("provenance")
    if not isinstance(provenance, dict):
        fail(f"qualified packet {packet.get('job_id')} lacks source provenance")
    for key in ("archive_sha256", "binary_sha256", "contract_sha256", "matrix_sha256"):
        value = provenance.get(key)
        if not isinstance(value, str) or len(value) != 64:
            fail(f"qualified packet {packet.get('job_id')} lacks {key}")
    uuid = packet.get("device_uuid")
    if not isinstance(uuid, str) or not uuid.startswith("GPU-"):
        fail(f"qualified packet {packet.get('job_id')} lacks a GPU UUID")
    if not isinstance(references, list) or not references:
        fail(f"qualified packet {packet.get('job_id')} has no raw record references")

    seen: set[tuple[int, str]] = set()
    records = []
    root = study.resolve()
    for reference in references:
        if not isinstance(reference, dict):
            fail("raw record reference must be an object")
        if update_mode is not None and reference.get("update_mode") != update_mode:
            fail(f"update-ablation reference is outside its declared {update_mode} group")
        relative = Path(reference.get("path", ""))
        if relative.is_absolute() or ".." in relative.parts:
            fail(f"raw record path escapes study workspace: {relative}")
        path = study / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
            fail(f"raw record is missing, linked, or outside the study workspace: {relative}")
        digest = sha256(path)
        if digest != reference.get("sha256"):
            fail(f"raw record hash mismatch: {relative}")
        raw = read_json(path)
        record = round_value(raw, packet, reference)
        if update_mode is not None and record["identity"]["condition_update"] != update_mode:
            fail(f"update-ablation raw record does not match its {update_mode} label")
        key = (record["round"], record["backend"])
        if key in seen:
            fail(f"duplicate qualified round/backend: {key}")
        seen.add(key)
        records.append(record)

    first_identity = records[0]["identity"]
    if any(record["identity"] != first_identity for record in records):
        fail(f"qualified packet {packet.get('job_id')} mixes workload identities")
    rounds = {record["round"] for record in records}
    backends = {record["backend"] for record in records}
    expected = {(round_id, backend) for round_id in rounds for backend in backends}
    if seen != expected:
        fail(f"qualified packet {packet.get('job_id')} has missing or extra arm-round observations")
    return first_identity, records


def load_packet_records(study: Path, packet: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return load_record_set(study, packet, packet.get("raw_records"))


def phase_summary(records: list[dict[str, Any]], phase_key: str) -> dict[str, Any]:
    phase_names = sorted({phase for record in records for phase in record["phase_probe"][phase_key]})
    output = {}
    for phase in phase_names:
        per_round = []
        for record in records:
            values = record["phase_probe"][phase_key].get(phase, [])
            per_round.append(median(values, f"{phase_key}.{phase}") if values else None)
        present = [value for value in per_round if value is not None]
        output[phase] = {
            "independent_round_medians_ms": per_round,
            "median_across_rounds_ms": median(present, f"{phase_key}.{phase} round medians") if present else None,
        }
    return output


def summarize_arm(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_round = sorted(records, key=lambda item: item["round"])
    identity = by_round[0]["identity"]
    for record in by_round[1:]:
        if record["identity"] != identity:
            fail("arm observations differ in their frozen scenario identity")
    timing = [record["timing"] for record in by_round]

    def round_values(name: str) -> list[float]:
        return [float(item[name]) for item in timing]

    memory_summary = {}
    for field in MEMORY_FIELDS:
        values = [record["memory_bytes"][field] for record in by_round]
        memory_summary[field] = {
            "median_bytes": median(values, field),
            "minimum_bytes": min(values),
            "maximum_bytes": max(values),
        }

    phase_key_host = "phase_probe_host_ms"
    phase_key_device = "phase_probe_device_ms"
    teacher_values = [record["teacher"] for record in by_round]
    teacher_summary: dict[str, Any] = {}
    for key in TEACHER_FIELDS:
        values = [item[key] for item in teacher_values]
        if all(value is None for value in values):
            teacher_summary[key] = None
        elif all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            teacher_summary[key] = {
                "median_across_independent_rounds": median([float(value) for value in values], key),
                "round_values": values,
            }
        else:
            distinct = []
            for value in values:
                if value not in distinct:
                    distinct.append(value)
            teacher_summary[key] = {"common_value": distinct[0] if len(distinct) == 1 else None, "distinct_round_values": distinct}

    horizon = identity["reuse_horizon"]
    intercepts = [item["linear_model_intercept_round_ms"] for item in timing]
    recurrent_per_evaluation = [item["recurrent_per_evaluation_round_median_ms"] for item in timing]
    complete = [item["complete_operational_round_median_us"] for item in timing]
    backend = by_round[0]["backend"]
    if any(record["backend"] != backend for record in by_round):
        fail("arm group contains multiple backends")
    return {
        "backend": backend,
        "identity": identity,
        "independent_round_count": len(by_round),
        "rounds": [
            {
                "round": record["round"],
                "backend": record["backend"],
                "raw_record_path": record["raw_record_path"],
                "raw_record_sha256": record["raw_record_sha256"],
                "cold_execution_preparation_ms": record["timing"]["cold_execution_preparation_ms"],
                "cold_preparation_components_ms": record["cold_preparation_components_ms"],
                "excluded_preparation_diagnostics_ms": record["excluded_preparation_diagnostics_ms"],
                "replay_payload_bytes": record["replay_payload_bytes"],
                "root_changes": record["root_changes"],
                "recurrent_horizon_samples_ms": record["timing"]["recurrent_samples_ms"],
                "output_recovery_samples_ms": record["timing"]["output_recovery_samples_ms"],
                "complete_operational_samples_us": record["timing"]["complete_operational_samples_us"],
                "phase_probe_scope": record["phase_probe_scope"],
                "memory_bytes": record["memory_bytes"],
            }
            for record in by_round
        ],
        "primary_confirmation_cost": {
            "boundary": "cold_execution_preparation + recurrent_complete_horizon + output_recovery; per-round timing arrays are paired elementwise",
            "cold_execution_preparation_ms_median": median(round_values("cold_execution_preparation_ms"), "cold preparation rounds"),
            "cold_execution_preparation_ms_range": [min(round_values("cold_execution_preparation_ms")), max(round_values("cold_execution_preparation_ms"))],
            "cold_preparation_component_medians_ms": {
                field: median([record["cold_preparation_components_ms"][field] for record in by_round], field)
                for field in COLD_COMPONENT_FIELDS
            },
            "replay_payload_bytes": {
                field: median([record["replay_payload_bytes"][field] for record in by_round], field)
                for field in ("replay_payload_bytes_hashed", "replay_payload_bytes_read")
            },
            "excluded_host_and_oracle_diagnostic_medians_ms": {
                field: median([record["excluded_preparation_diagnostics_ms"][field] for record in by_round], field)
                for field in EXCLUDED_DIAGNOSTIC_FIELDS
            },
            "recurrent_horizon_ms_median": median(round_values("recurrent_horizon_round_median_ms"), "recurrent rounds"),
            "recurrent_horizon_ms_range": [min(round_values("recurrent_horizon_round_median_ms")), max(round_values("recurrent_horizon_round_median_ms"))],
            "output_recovery_ms_median": median(round_values("output_recovery_round_median_ms"), "recovery rounds"),
            "output_recovery_ms_range": [min(round_values("output_recovery_round_median_ms")), max(round_values("output_recovery_round_median_ms"))],
            "complete_operational_cost_us_median": median(complete, "complete operational round costs"),
            "complete_operational_cost_us_iqr": [quantile_linear(complete, 0.25), quantile_linear(complete, 0.75)],
            "complete_operational_round_costs_us": complete,
            "linear_amortization_inputs": {
                "offset_ms_per_reuse_cycle_median_cold_plus_recovery": median(intercepts, "amortization intercepts"),
                "recurrent_cost_ms_per_evaluation_median": median(recurrent_per_evaluation, "per evaluation recurrent costs"),
                "observed_reuse_horizon_evaluations": horizon,
                "model_assumption": "future evaluations repeat the same scenario's mean recurrent cost; cold execution preparation and one output recovery occur once per reuse cycle",
                "estimate_only": True,
                "confidence_interval": None,
            },
        },
        "first_step_phase_probe_diagnostic_not_primary_timing": {
            "scope": by_round[0]["phase_probe_scope"],
            "independent_round_count": len(by_round),
            "host_ms": phase_summary(by_round, phase_key_host),
            "device_ms": phase_summary(by_round, phase_key_device),
        },
        "memory_observations": {
            "scope_note": "These are runner-reported allocations/bytes. condition_device_bytes includes the common harness representations; device_total_bytes is cudaMemGetInfo total capacity, not candidate-attributable usage.",
            "fields": memory_summary,
        },
        "workload_and_learning_observations": {
            "nominal_input_change_percent": identity["nominal_input_change_percent"],
            "applied_input_change_percent": identity["applied_input_change_percent"],
            "condition_update": identity["condition_update"],
            "parameter_update": identity["parameter_update"],
            "nominal_and_applied_root_changes_per_toggle": {
                "nominal": by_round[0]["root_changes"]["nominal_root_changes"],
                "applied": by_round[0]["root_changes"]["applied_root_changes_per_toggle"],
            },
            "predicate_updates_applicable": by_round[0]["predicate_updates_applicable"],
            "ste_derivative_candidates": by_round[0]["ste_derivative_candidates"],
            "reuse_horizon_evaluations": horizon,
            "teacher_input_rule": by_round[0]["teacher_input_rule"],
            "teacher_initial_fit_in_recurrent_horizon": by_round[0]["teacher_initial_fit_in_recurrent_horizon"],
            "teacher_metrics": teacher_summary,
        },
    }


def linear_amortization(candidate: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    cand = candidate["primary_confirmation_cost"]["linear_amortization_inputs"]
    ctrl = control["primary_confirmation_cost"]["linear_amortization_inputs"]
    intercept_delta = cand["offset_ms_per_reuse_cycle_median_cold_plus_recovery"] - ctrl["offset_ms_per_reuse_cycle_median_cold_plus_recovery"]
    slope_delta = cand["recurrent_cost_ms_per_evaluation_median"] - ctrl["recurrent_cost_ms_per_evaluation_median"]
    answer: dict[str, Any] = {
        "estimate_only": True,
        "comparison": f"{candidate['backend']} versus {control['backend']}",
        "control_backend": control["backend"],
        "candidate_backend": candidate["backend"],
        "candidate_minus_control_offset_ms": intercept_delta,
        "candidate_minus_control_recurrent_ms_per_evaluation": slope_delta,
        "assumptions": [
            "same frozen scenario, gate/model update policy, and workload trajectory continue",
            "each additional evaluation has the observed horizon-average recurrent cost",
            "cold execution preparation and output recovery occur once per reuse cycle",
            "no uncertainty interval is inferred from the six arm medians",
        ],
        "confidence_interval": None,
    }

    candidate_rounds = {row["round"]: row for row in candidate["rounds"]}
    control_rounds = {row["round"]: row for row in control["rounds"]}
    paired_round_sensitivity = []
    for round_id in sorted(candidate_rounds.keys() & control_rounds.keys()):
        cand_round, ctrl_round = candidate_rounds[round_id], control_rounds[round_id]
        horizon = candidate["identity"]["reuse_horizon"]
        cand_offset = cand_round["cold_execution_preparation_ms"] + median(cand_round["output_recovery_samples_ms"], "candidate paired-round recovery")
        ctrl_offset = ctrl_round["cold_execution_preparation_ms"] + median(ctrl_round["output_recovery_samples_ms"], "control paired-round recovery")
        cand_slope = median(cand_round["recurrent_horizon_samples_ms"], "candidate paired-round recurrence") / horizon
        ctrl_slope = median(ctrl_round["recurrent_horizon_samples_ms"], "control paired-round recurrence") / horizon
        paired_intercept_delta = cand_offset - ctrl_offset
        paired_slope_delta = cand_slope - ctrl_slope
        if math.isclose(paired_slope_delta, 0.0, rel_tol=0.0, abs_tol=1e-12):
            paired_status = "parallel lines; no finite intersection"
            paired_crossing = None
        else:
            intersection = -paired_intercept_delta / paired_slope_delta
            if math.isfinite(intersection) and intersection > 0:
                paired_status = "positive intersection estimate"
                paired_crossing = intersection
            else:
                paired_status = "no positive intersection"
                paired_crossing = None
        paired_round_sensitivity.append({
            "round": round_id,
            "candidate_minus_control_offset_ms": paired_intercept_delta,
            "candidate_minus_control_recurrent_ms_per_evaluation": paired_slope_delta,
            "positive_intersection_evaluations_estimate": paired_crossing,
            "status": paired_status,
        })
    positive_round_crossings = [
        row["positive_intersection_evaluations_estimate"]
        for row in paired_round_sensitivity
        if row["positive_intersection_evaluations_estimate"] is not None
    ]
    answer["paired_round_line_sensitivity_descriptive_not_interval"] = {
        "rounds": paired_round_sensitivity,
        "positive_intersection_round_count": len(positive_round_crossings),
        "round_count": len(paired_round_sensitivity),
        "positive_intersection_range_evaluations": [min(positive_round_crossings), max(positive_round_crossings)] if positive_round_crossings else None,
        "confidence_interval": None,
        "interpretation": "These paired round-specific lines show sensitivity to measured cold preparation. They are descriptive observations, not a confidence interval or a tested reuse sweep.",
    }
    # Equality solves delta_offset + n * delta_slope = 0. A positive solution
    # exists only when intercept and slope differences have opposite signs.
    if math.isclose(slope_delta, 0.0, rel_tol=0.0, abs_tol=1e-12):
        answer.update({"status": "no finite crossover: modeled recurrent slopes are equal", "break_even_evaluations_estimate": None})
        return answer
    evaluations = -intercept_delta / slope_delta
    if not math.isfinite(evaluations) or evaluations <= 0:
        direction = "candidate lower for all positive evaluation counts under this line model" if intercept_delta + slope_delta < 0 else "control lower for all positive evaluation counts under this line model"
        answer.update({"status": "no positive crossover under this model", "direction_for_positive_counts": direction, "break_even_evaluations_estimate": None})
        return answer
    answer.update({
        "status": "linear amortization crossover estimate; not a measured threshold",
        "break_even_evaluations_estimate": evaluations,
        "first_integer_evaluation_after_estimated_crossover": math.floor(evaluations) + 1,
        "direction_below_crossover": "candidate is cheaper" if intercept_delta < 0 else "control is cheaper",
        "direction_above_crossover": "candidate is cheaper" if slope_delta < 0 else "control is cheaper",
        "crossing_role": (
            "upper reuse cutoff for the candidate" if intercept_delta < 0
            else "amortization threshold for the candidate"
        ),
    })
    return answer


def build_summary(study: Path, qualification_path: Path) -> dict[str, Any]:
    qualification = read_json(qualification_path)
    if qualification.get("schema") != "cgs.qualification.v1":
        fail("unsupported qualification schema")
    packets = qualification.get("packets")
    if not isinstance(packets, list):
        fail("qualification packets must be a list")

    result_groups = []
    update_groups = []
    update_group_index: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}

    def make_group(packet: dict[str, Any], identity: dict[str, Any], records: list[dict[str, Any]], *, measurement_set: str, update_mode: str | None = None) -> dict[str, Any]:
        arms: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            arms.setdefault(record["backend"], []).append(record)
        arm_summaries = {backend: summarize_arm(rows) for backend, rows in sorted(arms.items())}
        group = {
            "measurement_set": measurement_set,
            "update_mode": update_mode,
            "job_id": packet["job_id"],
            "device_uuid": packet["device_uuid"],
            "source_archive_sha256": packet["provenance"]["archive_sha256"],
            "binary_sha256": packet["provenance"]["binary_sha256"],
            "contract_sha256": packet["provenance"]["contract_sha256"],
            "matrix_sha256": packet["provenance"]["matrix_sha256"],
            "packet_metrics_sha256": packet["provenance"].get("packet_metrics_sha256"),
            "update_ablation_summary_sha256": packet.get("update_ablation_sha256") if update_mode is not None else None,
            "identity": identity,
            "arms": arm_summaries,
            "strongest_control_comparisons": [],
            "matched_staging_comparisons": [],
        }
        return group

    for packet in packets:
        identity, records = load_packet_records(study, packet)
        group = make_group(packet, identity, records, measurement_set="base-matrix")
        result_groups.append(group)
        update_references = packet.get("update_raw_records", [])
        if update_references:
            if not isinstance(update_references, list):
                fail("update_raw_records must be a list")
            summary_sha = packet.get("update_ablation_sha256")
            if not isinstance(summary_sha, str) or len(summary_sha) != 64:
                fail("update-ablation records lack their qualified summary hash")
            modes = {reference.get("update_mode") for reference in update_references if isinstance(reference, dict)}
            if any(not isinstance(mode, str) for mode in modes):
                fail("every update-ablation record must declare its mode")
            for mode in sorted(modes):
                if mode not in ("rebuild", "incremental"):
                    fail(f"unsupported update-ablation mode: {mode}")
                mode_refs = [reference for reference in update_references if isinstance(reference, dict) and reference.get("update_mode") == mode]
                update_identity, update_records = load_record_set(study, packet, mode_refs, update_mode=mode)
                update_group = make_group(packet, update_identity, update_records, measurement_set="update-ablation", update_mode=mode)
                update_groups.append(update_group)
                for backend in update_group["arms"]:
                    index = (packet["job_id"], packet["device_uuid"], update_identity["case"], backend, mode)
                    if index in update_group_index:
                        fail(f"duplicate update-ablation group: {index}")
                    update_group_index[index] = update_group

    def find_group(job_id: str, case: str) -> dict[str, Any] | None:
        matches = [group for group in result_groups if group["job_id"] == job_id and group["identity"]["case"] == case]
        return matches[0] if len(matches) == 1 else None

    comparison_index: dict[tuple[str, str], dict[str, Any]] = {}
    for comparison in qualification.get("comparisons", []):
        if not isinstance(comparison, dict):
            continue
        job_id, case, candidate_name = comparison.get("job_id"), comparison.get("case"), comparison.get("candidate")
        group = find_group(job_id, case)
        if group is None or candidate_name not in group["arms"]:
            continue
        control_name = comparison.get("strongest_control")
        control = group["arms"].get(control_name)
        candidate = group["arms"][candidate_name]
        pair_records = comparison.get("pairs", [])
        recorded_pairs = []
        candidate_rounds = {row["round"]: row for row in candidate["rounds"]}
        control_rounds = {row["round"]: row for row in control["rounds"]} if control else {}
        for pair in pair_records:
            round_id = pair.get("repeat")
            if round_id not in candidate_rounds or round_id not in control_rounds:
                fail(f"qualification pair lacks matching cost records for {job_id}/{candidate_name}/{round_id}")
            candidate_us = median(candidate_rounds[round_id]["complete_operational_samples_us"], "candidate pair samples")
            control_us = median(control_rounds[round_id]["complete_operational_samples_us"], "control pair samples")
            if not math.isclose(candidate_us, pair.get("candidate_us"), rel_tol=1e-9, abs_tol=1e-6) or not math.isclose(control_us, pair.get("control_us"), rel_tol=1e-9, abs_tol=1e-6):
                fail(f"qualified paired timing disagrees with raw cost components for {job_id}/{candidate_name}/{round_id}")
            recorded_pairs.append({"round": round_id, "candidate_us": candidate_us, "control_us": control_us, "raw_pair_verified": True})
        entry = {
            "candidate": candidate_name,
            "strongest_control": control_name,
            "qualification_decision": comparison.get("decision"),
            "qualification_calculation": comparison.get("calculation"),
            "qualified_paired_cost_rounds": recorded_pairs,
            "linear_amortization_estimate": linear_amortization(candidate, control) if control else None,
        }
        group["strongest_control_comparisons"].append(entry)
        comparison_index[(job_id, candidate_name)] = entry

    for staging in qualification.get("staging_ablations", []):
        if not isinstance(staging, dict):
            continue
        job_id = staging.get("job_id")
        group = find_group(job_id, staging.get("case"))
        if group is None:
            continue
        candidate_name, control_name = staging.get("candidate"), staging.get("control")
        candidate = group["arms"].get(candidate_name)
        control = group["arms"].get(control_name)
        if candidate is None or control is None:
            continue
        pair_records = staging.get("pairs", [])
        cand_rounds = {row["round"]: row for row in candidate["rounds"]}
        ctrl_rounds = {row["round"]: row for row in control["rounds"]}
        verified_pairs = []
        for pair in pair_records:
            round_id = pair.get("repeat")
            if round_id not in cand_rounds or round_id not in ctrl_rounds:
                fail(f"staging ablation lacks raw arm costs for round {round_id}")
            cand_us = median(cand_rounds[round_id]["complete_operational_samples_us"], "staging candidate samples")
            ctrl_us = median(ctrl_rounds[round_id]["complete_operational_samples_us"], "staging control samples")
            if not math.isclose(cand_us, pair.get("candidate_us"), rel_tol=1e-9, abs_tol=1e-6) or not math.isclose(ctrl_us, pair.get("control_us"), rel_tol=1e-9, abs_tol=1e-6):
                fail("qualified staging pair disagrees with raw cost components")
            verified_pairs.append({"round": round_id, "candidate_us": cand_us, "control_us": ctrl_us, "raw_pair_verified": True})
        group["matched_staging_comparisons"].append({
            "candidate": candidate_name,
            "control": control_name,
            "qualification_decision": staging.get("decision"),
            "qualified_paired_cost_rounds": verified_pairs,
            "linear_amortization_estimate": linear_amortization(candidate, control),
            "scope": "matched tile ownership/staging diagnostic; specialization decision remains against strongest applicable conventional control",
        })

    update_strategy_comparisons = []
    for comparison in qualification.get("update_ablations", []):
        if not isinstance(comparison, dict):
            continue
        job_id, device_uuid, case = comparison.get("job_id"), comparison.get("device_uuid"), comparison.get("case")
        backend = comparison.get("backend")
        rebuild_group = update_group_index.get((job_id, device_uuid, case, backend, "rebuild"))
        incremental_group = update_group_index.get((job_id, device_uuid, case, backend, "incremental"))
        if rebuild_group is None or incremental_group is None:
            fail(f"update-strategy comparison lacks separate rebuild/incremental groups: {job_id}/{backend}")
        rebuild_arm = rebuild_group["arms"].get(backend)
        incremental_arm = incremental_group["arms"].get(backend)
        if rebuild_arm is None or incremental_arm is None:
            fail(f"update-strategy comparison lacks arm costs: {job_id}/{backend}")
        rebuild_rounds = {row["round"]: row for row in rebuild_arm["rounds"]}
        incremental_rounds = {row["round"]: row for row in incremental_arm["rounds"]}
        qualified_pairs = []
        for pair in comparison.get("pairs", []):
            round_id = pair.get("repeat")
            if pair.get("device_uuid") != device_uuid or round_id not in rebuild_rounds or round_id not in incremental_rounds:
                fail(f"update-strategy pair lacks same-device raw records: {job_id}/{backend}/{round_id}")
            rebuild_us = median(rebuild_rounds[round_id]["complete_operational_samples_us"], "rebuild paired samples")
            incremental_us = median(incremental_rounds[round_id]["complete_operational_samples_us"], "incremental paired samples")
            if not math.isclose(rebuild_us, pair.get("control_us"), rel_tol=1e-9, abs_tol=1e-6) or not math.isclose(incremental_us, pair.get("candidate_us"), rel_tol=1e-9, abs_tol=1e-6):
                fail(f"update-strategy qualification pair differs from raw costs: {job_id}/{backend}/{round_id}")
            qualified_pairs.append({
                "round": round_id,
                "rebuild_us": rebuild_us,
                "incremental_us": incremental_us,
                "raw_pair_verified": True,
            })
        update_strategy_comparisons.append({
            "job_id": job_id,
            "device_uuid": device_uuid,
            "source_archive_sha256": incremental_group["source_archive_sha256"],
            "binary_sha256": incremental_group["binary_sha256"],
            "case": case,
            "backend": backend,
            "rebuild_update_mode": "rebuild",
            "incremental_update_mode": "incremental",
            "qualification_decision": comparison.get("decision"),
            "qualification_calculation": comparison.get("calculation"),
            "qualified_paired_cost_rounds": qualified_pairs,
            "linear_amortization_estimate_incremental_vs_rebuild": linear_amortization(incremental_arm, rebuild_arm),
            "grouping_note": "Each mode is summarized in its own update-ablation group. Raw samples from rebuild and incremental are never pooled.",
        })

    unsupported_crossovers = "Horizons 1, 16 and 256 occur in distinct frozen workloads. They are not a measured reuse sweep and are never pooled or treated as a crossover."
    limitations = [
        "This exporter consumes receipt-qualified packets. It rechecks referenced raw record hashes and paired component arithmetic, but does not repeat Scheduler, KDA, archive, binary, interference, or sanitizer qualification.",
        "The six paired rounds are independent units. Values in timing arrays within each raw record are summarized inside their round, not counted as independent rounds.",
        unsupported_crossovers,
        "Linear amortization is a scenario-local estimate without a confidence interval. It assumes the same workload, update policy, and mean recurrent work continue, with preparation and one output recovery paid per reuse cycle.",
        "phase_probe timings are first-step diagnostics collected separately from confirmation costs; they are not added to the primary cost.",
        "Memory fields follow the runner contract. Common condition representation allocations and device capacity are not attributed to an individual kernel; net device bytes is the before/after CUDA allocation delta reported by the runner.",
    ]
    return {
        "schema": "cgs.cost-summary.v1",
        "qualification_path": str(qualification_path.resolve()),
        "qualification_sha256": sha256(qualification_path),
        "qualification_schema": qualification["schema"],
        "qualification_contract_sha256": qualification.get("contract_sha256"),
        "qualification_matrix_sha256": qualification.get("matrix_sha256"),
        "measurement_boundary": qualification.get("measurement_boundary"),
        "accepted_packet_count": len(packets),
        "scenario_group_count": len(result_groups),
        "groups": result_groups,
        "update_ablation_group_count": len(update_groups),
        "update_ablation_groups": update_groups,
        "update_strategy_comparisons": update_strategy_comparisons,
        "limitations": limitations,
        "coverage": qualification.get("coverage"),
    }


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Qualified cost summary",
        "",
        "This report summarizes only packets listed as accepted by the selected qualification export. The source qualification and raw campaign records remain authoritative for admission, correctness, sanitizer and performance decisions.",
        "",
        f"Qualification SHA-256: `{summary['qualification_sha256']}`  ",
        f"Contract SHA-256: `{summary.get('qualification_contract_sha256')}`  ",
        f"Matrix SHA-256: `{summary.get('qualification_matrix_sha256')}`",
        "",
    ]
    if not summary["groups"]:
        lines.extend(["No accepted measurement packets are present.", ""])
    for group in summary["groups"]:
        identity = group["identity"]
        lines.extend([
            f"## {identity['case']}",
            "",
            f"Gate `{identity['gate']}`, body `{identity['body']}` / `{identity['precision']}`, input `{identity['input']}`, update `{identity['condition_update']}`, reuse horizon `{identity['reuse_horizon']}`. Device `{group['device_uuid']}`; job `{group['job_id']}`.",
            "",
            f"Source archive `{group['source_archive_sha256']}`; binary `{group['binary_sha256']}`.",
            "",
            "### Complete operational costs",
            "",
            "Each row summarizes six independent rounds where available. Recurrent and recovery samples inside a round are paired and summarized within that round.",
            "",
            "| Arm | Rounds | Cold prep ms (median, range) | Recurrent horizon ms (median, range) | Output recovery ms (median, range) | Complete cost µs (median, IQR) | Net allocated bytes (median) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for backend, arm in group["arms"].items():
            cost = arm["primary_confirmation_cost"]
            prep_range = cost["cold_execution_preparation_ms_range"]
            rec_range = cost["recurrent_horizon_ms_range"]
            out_range = cost["output_recovery_ms_range"]
            iqr = cost["complete_operational_cost_us_iqr"]
            allocated = arm["memory_observations"]["fields"]["net_device_allocated_bytes"]["median_bytes"]
            lines.append(
                f"| `{backend}` | {arm['independent_round_count']} | "
                f"{cost['cold_execution_preparation_ms_median']:.6g} ({prep_range[0]:.6g}–{prep_range[1]:.6g}) | "
                f"{cost['recurrent_horizon_ms_median']:.6g} ({rec_range[0]:.6g}–{rec_range[1]:.6g}) | "
                f"{cost['output_recovery_ms_median']:.6g} ({out_range[0]:.6g}–{out_range[1]:.6g}) | "
                f"{cost['complete_operational_cost_us_median']:.3f} ({iqr[0]:.3f}–{iqr[1]:.3f}) | {allocated:.0f} |"
            )
        lines.extend(["", "### Amortization estimates", ""])
        if group["strongest_control_comparisons"]:
            lines.extend(["These scenario-local line estimates have no confidence interval and do not replace the qualification's paired complete-cost decision.", ""])
            for comparison in group["strongest_control_comparisons"]:
                model = comparison["linear_amortization_estimate"]
                estimate = model.get("break_even_evaluations_estimate")
                amount = "not applicable" if estimate is None else f"{estimate:.4g} evaluations"
                direction = ""
                if estimate is not None:
                    direction = f" The fitted line favors the {model['direction_below_crossover'].split()[0]} below the crossing and the {model['direction_above_crossover'].split()[0]} above it; this is an estimated {model['crossing_role']}."
                sensitivity = model["paired_round_line_sensitivity_descriptive_not_interval"]
                positive = sensitivity["positive_intersection_round_count"]
                total = sensitivity["round_count"]
                spread = sensitivity["positive_intersection_range_evaluations"]
                if estimate is not None and total:
                    if positive and spread:
                        sensitivity_note = f" Paired-round lines have positive intersections in {positive}/{total} rounds (range {spread[0]:.3g}–{spread[1]:.3g}); other rounds do not cross at positive reuse. The cold intercept is noisy and this has no confidence interval."
                    else:
                        sensitivity_note = f" No paired-round line has a positive intersection; cold-intercept variation makes the median-line crossing unstable, with no confidence interval."
                else:
                    sensitivity_note = " No confidence interval is available for this model."
                measured = comparison.get("qualification_calculation") or {}
                reduction = measured.get("median_latency_reduction")
                speedup = measured.get("median_speedup")
                interval = measured.get("paired_bootstrap_95_percent_interval")
                if reduction is not None and speedup is not None and interval:
                    qualified = f"Measured paired median latency reduction {100*reduction:.3f}% (speedup {speedup:.5f}, 95% interval [{interval[0]:.5f}, {interval[1]:.5f}]); {comparison['qualification_decision']}"
                else:
                    qualified = f"Qualified paired decision: {comparison['qualification_decision']}"
                lines.append(f"- `{comparison['candidate']}` vs `{comparison['strongest_control']}`: {model['status']}; estimate {amount}.{direction}{sensitivity_note} {qualified}.")
        else:
            lines.append("No qualified candidate/control comparison is present for this group.")
        if group["matched_staging_comparisons"]:
            lines.extend(["", "Matched staging ablations:"])
            for comparison in group["matched_staging_comparisons"]:
                model = comparison["linear_amortization_estimate"]
                amount = model.get("break_even_evaluations_estimate")
                shown = "not applicable" if amount is None else f"{amount:.4g} evaluations"
                lines.append(f"- `{comparison['candidate']}` vs `{comparison['control']}`: {model['status']}; estimate {shown}; {comparison['scope']}.")
        teacher = next(iter(group["arms"].values()))["workload_and_learning_observations"]["teacher_metrics"] if group["arms"] else {}
        soft_uniform = teacher.get("teacher_uniform_soft_hard_mean")
        soft_weighted = teacher.get("teacher_workload_weighted_soft_hard_mean")
        if soft_uniform is not None and isinstance(soft_uniform, dict):
            crossings = teacher["teacher_validation_threshold_crossings"]
            crossings_value = crossings.get("median_across_independent_rounds") if isinstance(crossings, dict) else crossings
            lines.extend([
                "",
                "Teacher soft to hard change is recorded separately from circuit equivalence and timing:",
                f"- Uniform mean: {soft_uniform['median_across_independent_rounds']:.6g}.",
                f"- Workload weighted mean: {soft_weighted['median_across_independent_rounds']:.6g}.",
                f"- Teacher row hardening threshold crossings across validator checks, median per round: {crossings_value:g}.",
                f"- Observed teacher rows: {teacher['teacher_observed_rows'].get('common_value') or teacher['teacher_observed_rows'].get('distinct_round_values')}.",
            ])
        lines.extend(["", "First-step host and device phase probes are in the JSON under `first_step_phase_probe_diagnostic_not_primary_timing`; they are excluded from the primary complete-cost samples.", ""])
    lines.extend([
        "## Rebuild and incremental update ablations",
        "",
        "When present, v3 update-ablation records are grouped by mode and backend. Rebuild and incremental observations stay in separate groups; only their same-packet, same-round pairs are compared.",
        "",
        "| Scenario | Backend | Update mode | Rounds | Cold prep ms | Recurrent horizon ms | Output recovery ms | Complete cost µs |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    if summary["update_ablation_groups"]:
        for group in summary["update_ablation_groups"]:
            for backend, arm in group["arms"].items():
                cost = arm["primary_confirmation_cost"]
                lines.append(
                    f"| `{group['identity']['case']}` | `{backend}` | `{group['update_mode']}` | "
                    f"{arm['independent_round_count']} | {cost['cold_execution_preparation_ms_median']:.6g} | "
                    f"{cost['recurrent_horizon_ms_median']:.6g} | {cost['output_recovery_ms_median']:.6g} | "
                    f"{cost['complete_operational_cost_us_median']:.3f} |"
                )
    else:
        lines.append("| No qualified update-ablation records | — | — | 0 | — | — | — | — |")
    if summary["update_strategy_comparisons"]:
        lines.extend(["", "| Backend | Scenario | Paired decision | Incremental versus rebuild estimate |"])
        lines.append("|---|---|---|---|")
        for comparison in summary["update_strategy_comparisons"]:
            model = comparison["linear_amortization_estimate_incremental_vs_rebuild"]
            amount = model.get("break_even_evaluations_estimate")
            text = "not applicable" if amount is None else f"{amount:.4g} evaluations"
            directions = ""
            if amount is not None:
                directions = f"; favors {model['direction_below_crossover']} below, {model['direction_above_crossover']} above (estimated {model['crossing_role']})"
            lines.append(f"| `{comparison['backend']}` | `{comparison['case']}` | {comparison['qualification_decision']} | {model['status']}; {text}{directions}; no confidence interval |")
    lines.append("")
    lines.extend(["## Limits", ""])
    lines.extend([f"- {note}" for note in summary["limitations"]])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--qualification", type=Path, default=DEFAULT_QUALIFICATION)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    study = args.study.resolve()
    qualification_path = args.qualification.resolve()
    if not qualification_path.is_file():
        fail(f"qualification JSON does not exist: {qualification_path}")
    summary = build_summary(study, qualification_path)
    json_path = args.json_output or study / "analysis/cost-summary.json"
    markdown_path = args.markdown_output or study / "analysis/cost-summary.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    markdown_path.write_text(markdown_report(summary))
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path), "groups": len(summary["groups"])}, sort_keys=True))


if __name__ == "__main__":
    main()
