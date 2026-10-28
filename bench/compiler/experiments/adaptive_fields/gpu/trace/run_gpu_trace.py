#!/usr/bin/env python3
"""Scheduler-only small E7 low/high/low trace with measured-cost adaptation.

All cost-table, source-identity, fixture, package, and scheduler checks happen
before importing the frozen E7 runner or Torch. A missing measured table exits
without initializing CUDA.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time


HERE = Path(__file__).resolve().parent
GPU_ROOT = HERE.parent
REPO_ROOT = HERE.parents[5]
E7_RUNNER = GPU_ROOT / "e7_relation_runner.py"
DEFAULT_PACKAGE = Path(
    "/home/tumlinson/Cellerator/build-sparse-support/evidence/"
    "torch-v14-optional-policies/package")
DEFAULT_INDEXER = GPU_ROOT / "indexer/artifacts/libadaptive_fields_indexer.so"
TRACE_CANDIDATES = (
    "scan_simt_full",
    "ordered_postings_simt_full",
)
TRACE_STEPS = 48
WINDOW_STEPS = 16
LOW_DENSITY_CUTOFF = 0.20


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def measured_cost_preflight(path: Path | None) -> tuple[dict | None, dict]:
    if path is None:
        return None, {
            "status": "unresolved_missing_measured_cost_table",
            "gpu_execution_performed": False,
            "message": "E7 GPU trace is disabled until a source-hash-verified measured cost table is supplied",
        }
    table_path = path.resolve(strict=True)
    table = json.loads(table_path.read_text())
    if (table.get("schema") !=
            "cellerator.adaptive_fields.e7_trace_measured_cost_table.v1"):
        raise ValueError("unsupported measured cost table schema")
    if table.get("evidence_kind") != "measured_e7":
        raise ValueError("scenario or modeled costs cannot authorize the GPU adaptation trace")
    identity = table.get("source_identity")
    if not isinstance(identity, dict) or not identity.get("path") or not identity.get("sha256"):
        raise ValueError("measured table must identify its source report path and SHA-256")
    source_path = Path(identity["path"])
    if not source_path.is_absolute():
        source_path = table_path.parent / source_path
    source_path = source_path.resolve(strict=True)
    source_hash = sha256(source_path)
    if source_hash != identity["sha256"]:
        raise ValueError("measured cost table source report SHA-256 mismatch")
    if table.get("fixture") != "small":
        raise ValueError("the bounded GPU trace requires cost measurements for fixture 'small'")
    if table.get("complete_step_metric") != "synchronized_training_step_wall_ms":
        raise ValueError("cost table must use complete synchronized E7 training-step wall time")
    if int(table.get("evaluation_interval_steps", 0)) != WINDOW_STEPS:
        raise ValueError("measured table must use the declared 16-step evaluation horizon")
    policy_eval = float(table.get("policy_evaluation_ms", -1))
    if not math.isfinite(policy_eval) or policy_eval <= 0:
        raise ValueError("measured table must declare a finite positive policy_evaluation_ms")
    costs = table.get("candidate_costs_ms", {})
    if set(costs) != set(TRACE_CANDIDATES):
        raise ValueError(f"measured table must contain exactly {list(TRACE_CANDIDATES)}")
    if table.get("initial_candidate") not in TRACE_CANDIDATES:
        raise ValueError("measured table initial_candidate must be one of the trace candidates")
    normalized = {}
    for candidate in TRACE_CANDIDATES:
        row = costs[candidate]
        density = row.get("step_by_density", {})
        if not {"low", "high"}.issubset(density):
            raise ValueError(f"{candidate}: low/high measured step costs are required")
        values = [float(density["low"]), float(density["high"]),
                  float(row.get("candidate_prepare_ms", -1)),
                  float(row.get("candidate_switch_ms", -1))]
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError(f"{candidate}: measured cost values must be finite and positive")
        normalized[candidate] = {
            "step_by_density": {key: float(density[key]) for key in ("low", "high")},
            "candidate_prepare_ms": float(row["candidate_prepare_ms"]),
            "candidate_switch_ms": float(row["candidate_switch_ms"]),
        }
    identity = {**identity, "path": str(source_path), "verified_sha256": source_hash}
    for identity_key in ("e7_runner_sha256", "indexer_library_sha256",
                         "fixture_manifest_sha256", "trace_runner_sha256"):
        if not identity.get(identity_key):
            raise ValueError(f"measured source identity must include {identity_key}")
    normalized_table = {**table, "source_identity": identity,
                        "candidate_costs_ms": normalized,
                        "policy_evaluation_ms": policy_eval,
                        "table_path": str(table_path),
                        "table_sha256": sha256(table_path)}
    return normalized_table, {
        "status": "measured_cost_table_verified",
        "path": str(table_path),
        "sha256": normalized_table["table_sha256"],
        "source_identity": identity,
        "evidence_kind": "measured_e7",
        "candidate_ids": list(TRACE_CANDIDATES),
    }


def static_preflight(args: argparse.Namespace, table: dict) -> dict:
    fixture_root = args.fixture_root.resolve(strict=True)
    package_root = args.package_root.resolve(strict=True)
    indexer_library = args.indexer_library.resolve(strict=True)
    identity_file = package_root.parent / "identity.json"
    if not identity_file.is_file():
        raise FileNotFoundError(f"frozen package identity is missing: {identity_file}")
    package_identity_sha = sha256(identity_file)
    fixture_manifest = fixture_root / "small/manifest.json"
    if not fixture_manifest.is_file():
        raise FileNotFoundError(f"small E7 fixture manifest is missing: {fixture_manifest}")
    fixture_manifest_sha = sha256(fixture_manifest)
    identities = table["source_identity"]
    expected_identities = {
        "e7_runner_sha256": sha256(E7_RUNNER),
        "indexer_library_sha256": sha256(indexer_library),
        "fixture_manifest_sha256": fixture_manifest_sha,
        "trace_runner_sha256": sha256(Path(__file__).resolve()),
    }
    for key, actual in expected_identities.items():
        if identities[key] != actual:
            raise ValueError(f"measured source identity {key} does not match current input")
    return {
        "fixture_root": str(fixture_root),
        "fixture_manifest_sha256": fixture_manifest_sha,
        "frozen_e7_runner": str(E7_RUNNER),
        "frozen_e7_runner_sha256": sha256(E7_RUNNER),
        "trace_runner_sha256": sha256(Path(__file__).resolve()),
        "package_root": str(package_root),
        "package_identity_sha256": package_identity_sha,
        "indexer_library": str(indexer_library),
        "indexer_library_sha256": sha256(indexer_library),
        "cost_table_sha256": table["table_sha256"],
        "source_report_sha256": table["source_identity"]["verified_sha256"],
        "trace_steps": TRACE_STEPS,
        "window_steps": WINDOW_STEPS,
        "candidate_ids": list(TRACE_CANDIDATES),
        "gpu_execution_performed": False,
    }


def scheduler_preflight() -> dict:
    if os.environ.get("CELLERATOR_SPARSE_SUPPORT_BENCH_ADMITTED") != "1":
        raise RuntimeError("missing parent experiment marker; no GPU operation was started")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if (not visible.startswith(("GPU-", "MIG-")) or "," in visible):
        raise RuntimeError("one UUID-routed CUDA_VISIBLE_DEVICES value is required")
    run_key = os.environ.get("ADAPTIVE_FIELDS_RUN_KEY")
    if not run_key:
        raise RuntimeError("parent JobSpec correlation key is required")
    return {
        "run_key_correlation": run_key,
        "visible_device_uuid": visible,
        "marker_role": "accidental-run guard only; not authorization or isolation",
        "scheduler_authority": "external JobSpec, submission, admission, terminal, and cleanup receipts",
    }


def import_frozen_e7():
    sys.path.insert(0, str(GPU_ROOT))
    spec = importlib.util.spec_from_file_location("adaptive_fields_frozen_e7",
                                                  E7_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen E7 runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class GenerationIndexer:
    """Narrow adapter supplies trace state generations to the frozen E7 step."""
    def __init__(self, indexer):
        self.indexer = indexer
        self.state_generation = 1
        self.last_result = None

    def query_prepared(self, mode, *, state_generation, verify=False):
        self.last_result = self.indexer.query_prepared(
            mode, state_generation=self.state_generation, verify=verify)
        return self.last_result


def make_trace_fixture(e7, fixture, phase: str):
    import numpy as np
    if phase == "high":
        all_sources = np_unique(fixture.sources)
        rows = tuple(all_sources.copy() for _ in range(fixture.cells))
    else:
        rows = tuple(np.asarray([row[0]], dtype="int64") if len(row)
                     else np.asarray([], dtype="int64")
                     for row in fixture.query_rows)
    return replace(fixture, query_rows=rows, context_entities=rows)


def np_unique(values):
    # Keep NumPy import local so preflight never imports the Torch stack.
    import numpy as np
    return np.unique(np.asarray(values, dtype=np.int64))


def create_indexer(e7, fixture, support, epoch: int, args,
                   *, prepare_postings: bool) -> tuple[GenerationIndexer, dict]:
    started = time.perf_counter()
    indexer = e7.Indexer(
        args.indexer_library, support.edge_ids, support.sources,
        support.destinations, fixture.entities, epoch,
        relation_id=7301, structure_id=7302)
    create_ms = (time.perf_counter() - started) * 1000.0
    buffer_started = time.perf_counter()
    buffers = indexer.prepare_output_buffers(fixture.cells)
    buffer_ms = (time.perf_counter() - buffer_started) * 1000.0
    posting_ms = 0.0
    if prepare_postings:
        posting_started = time.perf_counter()
        indexer.prepare_ordered_postings()
        posting_ms = (time.perf_counter() - posting_started) * 1000.0
    proxy = GenerationIndexer(indexer)
    return proxy, {"handle_create_ms": create_ms,
                   "output_buffer_prepare_ms": buffer_ms,
                   "ordered_postings_prepare_ms": posting_ms,
                   "total_indexer_prepare_ms": create_ms + buffer_ms + posting_ms,
                   "host_buffers": buffers,
                   "structure_epoch": epoch}


def expected_workset_ids(e7, fixture, query_rows, support):
    import numpy as np
    from dataclasses import replace as dataclass_replace
    arrays = dict(fixture.arrays)
    arrays.update({
        "edge_ids": np.asarray(support.edge_ids, dtype=np.int64),
        "source_entity_id": np.asarray(support.sources, dtype=np.int64),
        "participant_entity_id": np.asarray(support.destinations, dtype=np.int64),
    })
    query_fixture = dataclass_replace(
        fixture, arrays=arrays, query_rows=query_rows,
        context_entities=query_rows)
    _, ids, ordinals = e7.host_scan_oracle(query_fixture)
    return ids, ordinals


def classify_density(densities) -> tuple[str, float]:
    if not densities:
        raise ValueError("density classification requires observed worksets")
    mean_density = statistics.mean(densities)
    return ("high" if mean_density >= LOW_DENSITY_CUTOFF else "low", mean_density)


def choose_candidate(table: dict, active: str, observed_class: str,
                     horizon: int, *, actual_observation_ms: float,
                     estimated_policy_evaluation_ms: float,
                     actual_candidate_prepare_ms: dict[str, float],
                     recent_actual_switch_ms: float | None,
                     common_rebuild_ms: float | None,
                     common_support_publish_ms: float | None) -> dict:
    alternatives = [candidate for candidate in TRACE_CANDIDATES if candidate != active]
    target = min(alternatives, key=lambda name:
                 table["candidate_costs_ms"][name]["step_by_density"][observed_class])
    costs = table["candidate_costs_ms"]
    gross_savings = (
        costs[active]["step_by_density"][observed_class]
        - costs[target]["step_by_density"][observed_class]) * horizon
    candidate_prepare = actual_candidate_prepare_ms.get(
        target, costs[target]["candidate_prepare_ms"])
    switch_estimate = (recent_actual_switch_ms if recent_actual_switch_ms is not None
                       else costs[target]["candidate_switch_ms"])
    # Support/indexer rebuild is a shared required cost for either candidate.
    # Keep it explicit in the full cost record, but cancel it in the incremental
    # switch inequality because fixed execution must pay it too.
    hurdle = (candidate_prepare + switch_estimate + actual_observation_ms
              + estimated_policy_evaluation_ms)
    net = gross_savings - hurdle
    return {
        "active_candidate_before_decision": active,
        "target_candidate_by_measured_step_cost": target,
        "observed_density_class": observed_class,
        "evaluation_horizon_steps": horizon,
        "gross_projected_savings_ms": gross_savings,
        "actual_candidate_prepare_cost_ms": candidate_prepare,
        "switch_cost_ms_used_in_hurdle": switch_estimate,
        "actual_observation_cost_ms": actual_observation_ms,
        "estimated_policy_evaluation_cost_ms": estimated_policy_evaluation_ms,
        "actual_common_support_indexer_rebuild_ms": common_rebuild_ms,
        "actual_common_support_publish_ms": common_support_publish_ms,
        "common_rebuild_is_in_total_cost_but_cancels_between_candidates": True,
        "net_projected_savings_ms": net,
        "selected_candidate": target if net > 0 else active,
        "switch_accepted": net > 0,
    }


def stage_and_publish(module, epoch: int, source: int, destination: int,
                      logical_edge_id: int) -> dict:
    if module._read_tickets:
        module.retire_tapes()
    if module._read_tickets:
        raise RuntimeError("native support still has live forward tapes at the window boundary")
    prior_ids = tuple(map(int, module.support.edge_ids))
    module.stage_promotion(logical_edge_id, source, destination,
                           discovery_round_id=17, scorer_generation=3,
                           proposal_score=0.875)
    started = time.perf_counter()
    module.publish_support(epoch + 1)
    elapsed = (time.perf_counter() - started) * 1000.0
    if module.structure_epoch != epoch + 1:
        raise RuntimeError("support publication did not advance the native structure epoch")
    published_ids = tuple(map(int, module.support.edge_ids))
    if published_ids != tuple(sorted((*prior_ids, logical_edge_id))):
        raise RuntimeError("support publication changed an old logical identity or omitted the promotion")
    return {"from_epoch": epoch, "to_epoch": module.structure_epoch,
            "logical_edge_id": logical_edge_id,
            "published_edge_count": len(module.support.edge_ids),
            "prior_logical_edge_ids": list(prior_ids),
            "published_logical_edge_ids": list(published_ids),
            "publish_wall_ms": elapsed,
            "live_read_tickets_after_publish": len(module._read_tickets)}


def run_one_trace(e7, torch, sparse_support, mechanism, fixture, target,
                  table, args, *, adaptive: bool, fixed_candidate: str,
                  original_id_count: int) -> dict:
    torch.cuda.reset_peak_memory_stats(torch.device("cuda"))
    seed = args.seed
    module_started = time.perf_counter()
    module = e7.prepare_module(
        torch, sparse_support, fixture, seed, cegeo=False,
        compiled_geometry=None, max_primal_items=original_id_count)
    module_prepare_ms = (time.perf_counter() - module_started) * 1000.0
    optimizer = torch.optim.Adam([module.parameters_flat], lr=1.0e-3,
                                 foreach=False, fused=False)
    guarded_step = mechanism.guarded_step
    initial_candidate = table["initial_candidate"] if adaptive else fixed_candidate
    current_candidate = initial_candidate
    current_mode, _, current_values = e7.parse_candidate(current_candidate)
    active_indexer = None
    indexer_preparation = []
    switch_rows = []
    policy_rows = []
    step_rows = []
    phase_preparation_rows = []
    run_overhead_ms = module_prepare_ms
    recent_actual_switch_ms = None
    prepared_postings = adaptive or current_mode == "ordered_postings"
    active_indexer, prep = create_indexer(
        e7, fixture, module.support, 1, args,
        prepare_postings=prepared_postings)
    indexer_preparation.append(prep)
    run_overhead_ms += prep["total_indexer_prepare_ms"]

    with torch.no_grad(), module._pinned_forward():
        entity = module.block("entity_embedding").detach().clone()
        rank = module.block("rank_embedding").detach().clone()
    contexts = {}
    inputs_by_phase = {}
    for phase in ("low", "high"):
        trace_fixture = make_trace_fixture(e7, fixture, phase)
        input_started = time.perf_counter()
        context = e7.build_context(torch, trace_fixture, entity, rank,
                                   module.parameters_flat.device)
        inputs_by_phase[phase] = context.to(torch.float16).contiguous()
        contexts[phase] = context
        phase_preparation_rows.append({
            "phase": phase,
            "input_context_prepare_ms": (time.perf_counter() - input_started) * 1000.0,
        })
        run_overhead_ms += phase_preparation_rows[-1]["input_context_prepare_ms"]

    target_device = target.to(module.parameters_flat.device)
    output_device = module.parameters_flat.device
    current_phase = None
    total_train_wall_ms = 0.0
    policy_candidate_prepare_ms = {
        "scan_simt_full": 0.0,
        "ordered_postings_simt_full": (
            indexer_preparation[0]["ordered_postings_prepare_ms"]
            if prepared_postings else 0.0),
    }
    promotion = None

    for step in range(TRACE_STEPS):
        phase = "low" if step < WINDOW_STEPS or step >= 2 * WINDOW_STEPS else "high"
        if step == WINDOW_STEPS:
            torch.cuda.synchronize()
            promotion = stage_and_publish(
                module, active_indexer.indexer.structure_epoch,
                int(fixture.sources[0]),
                (int(fixture.sources[0]) + 1) % fixture.entities,
                int(max(fixture.edge_ids)) + 1)
            run_overhead_ms += promotion["publish_wall_ms"]
            old_indexer = active_indexer
            rebuild_start = time.perf_counter()
            active_indexer, prep = create_indexer(
                e7, fixture, module.support, module.structure_epoch, args,
                prepare_postings=adaptive or current_mode == "ordered_postings")
            prep["support_publish_ms_preceded_rebuild"] = promotion["publish_wall_ms"]
            prep["epoch_rebuild_wall_ms_including_handle_and_buffers"] = (
                time.perf_counter() - rebuild_start) * 1000.0
            indexer_preparation.append(prep)
            run_overhead_ms += prep["epoch_rebuild_wall_ms_including_handle_and_buffers"]
            policy_candidate_prepare_ms["ordered_postings_simt_full"] = (
                prep["ordered_postings_prepare_ms"])
            old_indexer.indexer.close()
            if adaptive:
                density_start = time.perf_counter()
                observed_class, mean_density = classify_density(
                    [row["workset_density"] for row in step_rows[-WINDOW_STEPS:]])
                observation_ms = (time.perf_counter() - density_start) * 1000.0
                previous_switches = [row["actual_switch_ms"] for row in switch_rows]
                switch_estimate = (statistics.mean(previous_switches)
                                   if previous_switches else None)
                decision_start = time.perf_counter()
                decision = choose_candidate(
                    table, current_candidate, observed_class, WINDOW_STEPS,
                    actual_observation_ms=observation_ms,
                    estimated_policy_evaluation_ms=(
                        policy_rows[-1]["policy_evaluation_ms"] if policy_rows
                        else table["policy_evaluation_ms"]),
                    actual_candidate_prepare_ms=policy_candidate_prepare_ms,
                    recent_actual_switch_ms=switch_estimate,
                    common_rebuild_ms=(prep["handle_create_ms"]
                                       + prep["output_buffer_prepare_ms"]),
                    common_support_publish_ms=promotion["publish_wall_ms"])
                decision["policy_evaluation_ms"] = (
                    time.perf_counter() - decision_start) * 1000.0
                decision["observed_mean_density"] = mean_density
                decision["window_observation_ms"] = observation_ms
                policy_rows.append(decision)
                run_overhead_ms += observation_ms + decision["policy_evaluation_ms"]
                next_candidate = decision["selected_candidate"]
                if next_candidate != current_candidate:
                    switch_started = time.perf_counter()
                    previous_candidate = current_candidate
                    current_candidate = next_candidate
                    current_mode, _, current_values = e7.parse_candidate(current_candidate)
                    actual_switch_ms = (time.perf_counter() - switch_started) * 1000.0
                    switch_rows.append({"step": step,
                                        "from_candidate": previous_candidate,
                                        "to_candidate": current_candidate,
                                        "actual_switch_ms": actual_switch_ms})
                    run_overhead_ms += actual_switch_ms
                    recent_actual_switch_ms = actual_switch_ms
        elif step == 2 * WINDOW_STEPS and adaptive:
            density_start = time.perf_counter()
            observed_class, mean_density = classify_density(
                [row["workset_density"] for row in step_rows[-WINDOW_STEPS:]])
            observation_ms = (time.perf_counter() - density_start) * 1000.0
            decision_start = time.perf_counter()
            decision = choose_candidate(
                table, current_candidate, observed_class, WINDOW_STEPS,
                actual_observation_ms=observation_ms,
                estimated_policy_evaluation_ms=(
                    policy_rows[-1]["policy_evaluation_ms"] if policy_rows
                    else table["policy_evaluation_ms"]),
                actual_candidate_prepare_ms=policy_candidate_prepare_ms,
                recent_actual_switch_ms=recent_actual_switch_ms,
                common_rebuild_ms=0.0,
                common_support_publish_ms=0.0)
            decision["policy_evaluation_ms"] = (
                time.perf_counter() - decision_start) * 1000.0
            decision["observed_mean_density"] = mean_density
            decision["window_observation_ms"] = observation_ms
            policy_rows.append(decision)
            run_overhead_ms += observation_ms + decision["policy_evaluation_ms"]
            next_candidate = decision["selected_candidate"]
            if next_candidate != current_candidate:
                switch_started = time.perf_counter()
                previous = current_candidate
                current_candidate = next_candidate
                current_mode, _, current_values = e7.parse_candidate(current_candidate)
                actual_switch_ms = (time.perf_counter() - switch_started) * 1000.0
                switch_rows.append({"step": step, "from_candidate": previous,
                                    "to_candidate": current_candidate,
                                    "actual_switch_ms": actual_switch_ms})
                run_overhead_ms += actual_switch_ms
                recent_actual_switch_ms = actual_switch_ms

        if phase != current_phase:
            trace_fixture = make_trace_fixture(e7, fixture, phase)
            offsets, terms = e7.make_terms(trace_fixture)
            query_start = time.perf_counter()
            active_indexer.indexer.prepare_query(offsets, terms)
            query_prepare_ms = (time.perf_counter() - query_start) * 1000.0
            run_overhead_ms += query_prepare_ms
            phase_preparation_rows.append({"phase": phase,
                                           "query_prepare_ms": query_prepare_ms,
                                           "structure_epoch": module.structure_epoch})
            proxy = active_indexer
            expected_ids, expected_ordinals = expected_workset_ids(
                e7, fixture, trace_fixture.query_rows, module.support)
            if phase not in contexts:
                input_started = time.perf_counter()
                context = e7.build_context(torch, trace_fixture, entity, rank,
                                           output_device)
                inputs_by_phase[phase] = context.to(torch.float16).contiguous()
                contexts[phase] = context
                phase_preparation_rows.append({
                    "phase": phase,
                    "input_context_prepare_ms": (time.perf_counter() - input_started) * 1000.0,
                })
            current_phase = phase

        proxy.state_generation = step + 1
        mode_name, _, current_values = e7.parse_candidate(current_candidate)
        started = time.perf_counter()
        loss, _ = e7.train_step(
            torch, trace_fixture, module, optimizer, guarded_step,
            contexts[phase], inputs_by_phase[phase], target_device,
            proxy, mode_name, current_values, 0.5, seed + step,
            diagnostic=False)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        total_train_wall_ms += elapsed_ms
        workset = proxy.last_result
        if workset is None:
            raise RuntimeError("frozen E7 step did not produce a trace workset")
        if (workset.structure_epoch != module.structure_epoch
                or workset.state_generation != step + 1):
            raise RuntimeError("workset epoch or state generation is stale")
        expected_epoch = 1 if step < WINDOW_STEPS else 2
        if workset.structure_epoch != expected_epoch:
            raise RuntimeError("workset did not follow the expected structure publication")
        if not np_equal(workset.logical_edge_ids, expected_ids):
            raise RuntimeError("C ABI workset identities differ from frozen host oracle")
        if not np_equal(workset.edge_ordinals, expected_ordinals):
            raise RuntimeError("C ABI canonical ordinals differ from frozen host oracle")
        density = workset.item_count / (fixture.cells * len(module.support.edge_ids))
        step_rows.append({
            "step": step,
            "window": step // WINDOW_STEPS,
            "phase": phase,
            "candidate": current_candidate,
            "structure_epoch": workset.structure_epoch,
            "state_generation": workset.state_generation,
            "support_count": len(module.support.edge_ids),
            "workset_item_count": workset.item_count,
            "workset_density": density,
            "workset_logical_edge_ids": list(map(int, workset.logical_edge_ids)),
            "workset_edge_ordinals": list(map(int, workset.edge_ordinals)),
            "workset_sha256": digest_u64(workset.logical_edge_ids),
            "loss": loss,
            "training_step_wall_ms": elapsed_ms,
        })

    torch.cuda.synchronize()
    module.retire_tapes()
    if active_indexer:
        active_indexer.indexer.close()
    return {
        "trace_kind": "adaptive_policy" if adaptive else "fixed_candidate",
        "fixed_candidate": fixed_candidate,
        "initial_candidate": initial_candidate,
        "final_candidate": current_candidate,
        "module_prepare_ms": module_prepare_ms,
        "indexer_preparation_by_epoch": indexer_preparation,
        "support_promotion_publication": promotion,
        "phase_and_query_preparation": phase_preparation_rows,
        "candidate_switches": switch_rows,
        "policy_decisions": policy_rows,
        "total_policy_setup_observation_switch_overhead_ms": run_overhead_ms,
        "total_training_step_wall_ms": total_train_wall_ms,
        "complete_trace_wall_ms": run_overhead_ms + total_train_wall_ms,
        "steps": step_rows,
        "peak_device_allocated_bytes": int(torch.cuda.max_memory_allocated(output_device)),
        "peak_device_reserved_bytes": int(torch.cuda.max_memory_reserved(output_device)),
    }


def np_equal(left, right) -> bool:
    import numpy as np
    return np.array_equal(np.asarray(left), np.asarray(right))


def digest_u64(values) -> str:
    import numpy as np
    return hashlib.sha256(np.asarray(values, dtype="<u8").tobytes()).hexdigest()


def run_gpu(args, table: dict, preflight: dict, scheduler: dict) -> dict:
    e7 = import_frozen_e7()
    fixture = e7.load_fixture("small", args.fixture_root)
    torch, sparse_support, mechanism, package_identity = e7.load_package(args.package_root)
    if package_identity["experiment_driver_sha256"] != sha256(E7_RUNNER):
        raise RuntimeError("frozen E7 runner identity changed after preflight")
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("admitted E7 trace needs the optional CUDA Torch runtime")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    original_ids = list(map(int, fixture.edge_ids))
    target_started = time.perf_counter()
    target_cpu = torch.Generator(device="cpu").manual_seed(args.seed + 111)
    target = (torch.randn((fixture.cells, fixture.entities, e7.WIDTH),
                          generator=target_cpu, dtype=torch.float32) * 0.1).to(device)
    shared_target_prepare_ms = (time.perf_counter() - target_started) * 1000.0
    fixed = args.fixed_candidate or table.get("initial_candidate")
    if fixed not in TRACE_CANDIDATES:
        raise ValueError("fixed candidate must be a candidate in the measured table")

    fixed_trace = run_one_trace(
        e7, torch, sparse_support, mechanism, fixture, target, table, args,
        adaptive=False, fixed_candidate=fixed,
        original_id_count=len(original_ids))
    torch.cuda.synchronize()
    adaptive_trace = run_one_trace(
        e7, torch, sparse_support, mechanism, fixture, target, table, args,
        adaptive=True, fixed_candidate=fixed,
        original_id_count=len(original_ids))
    for trace in (fixed_trace, adaptive_trace):
        trace["shared_target_prepare_ms"] = shared_target_prepare_ms
        trace["complete_trace_wall_ms"] += shared_target_prepare_ms
    if len(fixed_trace["steps"]) != TRACE_STEPS or len(adaptive_trace["steps"]) != TRACE_STEPS:
        raise RuntimeError("trace did not complete all 48 steps")
    for baseline_step, policy_step in zip(fixed_trace["steps"], adaptive_trace["steps"]):
        for key in ("step", "structure_epoch", "state_generation", "support_count",
                    "workset_item_count", "workset_logical_edge_ids",
                    "workset_edge_ordinals", "phase"):
            if baseline_step[key] != policy_step[key]:
                raise RuntimeError(f"fixed/policy traces disagree for {key} at step {baseline_step['step']}")
    return {
        "schema": "cellerator.adaptive_fields.e7_gpu_adaptive_trace.v1",
        "status": "completed_gpu_trace",
        "gpu_execution_performed": True,
        "fixture": fixture.identity(),
        "trace_shape": {"steps": TRACE_STEPS, "window_steps": WINDOW_STEPS,
                        "phases": ["low", "high", "low"]},
        "low_density_cutoff": LOW_DENSITY_CUTOFF,
        "measured_cost_table": {
            "path": table["table_path"], "sha256": table["table_sha256"],
            "source_identity": table["source_identity"],
            "candidate_costs_ms": table["candidate_costs_ms"],
        },
        "fixed_trace": fixed_trace,
        "adaptive_trace": adaptive_trace,
        "complete_trace_cost_delta_policy_minus_fixed_ms": (
            adaptive_trace["complete_trace_wall_ms"]
            - fixed_trace["complete_trace_wall_ms"]),
        "source_identity": {
            "e7_runner_sha256": sha256(E7_RUNNER),
            "trace_runner_sha256": sha256(Path(__file__).resolve()),
            "package_identity_sha256": preflight["package_identity_sha256"],
            "indexer_library_sha256": preflight["indexer_library_sha256"],
            "fixture_manifest_sha256": preflight["fixture_manifest_sha256"],
            "cost_table_sha256": table["table_sha256"],
            "source_report_sha256": table["source_identity"]["verified_sha256"],
        },
        "scheduler": scheduler,
        "runtime": {"python": sys.version, "platform": platform.platform(),
                    "torch": str(torch.__version__),
                    "torch_cuda": str(torch.version.cuda)},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost-table", type=Path)
    parser.add_argument("--fixture-root", type=Path, default=Path(
        "/tmp/cellerator_scaled_sparse_fixtures_887437fc"))
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--indexer-library", type=Path, default=DEFAULT_INDEXER)
    parser.add_argument("--fixed-candidate", choices=TRACE_CANDIDATES)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    table, cost_preflight = measured_cost_preflight(args.cost_table)
    if table is None:
        result = {
            "schema": "cellerator.adaptive_fields.e7_gpu_adaptive_trace_preflight.v1",
            **cost_preflight,
            "scope": "no GPU import, CUDA initialization, or device operation",
        }
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"output": str(output), "status": result["status"],
                          "gpu_execution_performed": False}, sort_keys=True))
        return 3
    preflight = static_preflight(args, table)
    if args.preflight_only:
        result = {"schema": "cellerator.adaptive_fields.e7_gpu_adaptive_trace_preflight.v1",
                  "status": "ready_for_parent_scheduler_admission",
                  "cost_table": cost_preflight, "static_inputs": preflight,
                  "gpu_execution_performed": False}
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"output": str(output), "status": result["status"],
                          "gpu_execution_performed": False}, sort_keys=True))
        return 0
    scheduler = scheduler_preflight()
    result = run_gpu(args, table, preflight, scheduler)
    output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "status": result["status"],
                      "complete_trace_cost_delta_ms": result[
                          "complete_trace_cost_delta_policy_minus_fixed_ms"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
