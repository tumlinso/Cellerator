#!/usr/bin/env python3
"""Measure the exact two-choice low/high E7 costs used by run_gpu_trace.py.

This is a GPU experiment entrypoint. It performs static source/fixture checks
and the scheduler marker/UUID guard before importing the E7 runner or Torch.
The scheduler's external JobSpec and receipts remain the execution authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
GPU_ROOT = HERE.parent
E7_RUNNER = GPU_ROOT / "e7_relation_runner.py"
DEFAULT_PACKAGE = Path(
    "/home/tumlinson/Cellerator/build-sparse-support/evidence/"
    "torch-v14-optional-policies/package")
DEFAULT_INDEXER = GPU_ROOT / "indexer/artifacts/libadaptive_fields_indexer.so"
CANDIDATES = ("scan_simt_full", "ordered_postings_simt_full")
PHASES = ("low", "high")
WARMUPS = 5
PAIRED_SAMPLES_PER_ROUND = 21
ROUNDS = ("exploration_1", "exploration_2",
          "confirmation_1", "confirmation_2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_preflight(args: argparse.Namespace) -> dict:
    fixture_root = args.fixture_root.resolve(strict=True)
    package_root = args.package_root.resolve(strict=True)
    indexer = args.indexer_library.resolve(strict=True)
    manifest = fixture_root / "small/manifest.json"
    identity_file = package_root.parent / "identity.json"
    for required in (manifest, identity_file):
        if not required.is_file():
            raise FileNotFoundError(required)
    return {
        "fixture_root": str(fixture_root),
        "fixture_manifest": str(manifest),
        "fixture_manifest_sha256": sha256(manifest),
        "package_root": str(package_root),
        "package_identity": str(identity_file),
        "package_identity_sha256": sha256(identity_file),
        "indexer_library": str(indexer),
        "indexer_library_sha256": sha256(indexer),
        "e7_runner": str(E7_RUNNER),
        "e7_runner_sha256": sha256(E7_RUNNER),
        "calibration_runner": str(Path(__file__).resolve()),
        "calibration_runner_sha256": sha256(Path(__file__).resolve()),
        "trace_runner_sha256": sha256(HERE / "run_gpu_trace.py"),
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


def measured_step(torch, e7, fixture, module, optimizer, guarded_step,
                  context, inputs, target, indexer, candidate, generation,
                  expected_ids, expected_ordinals, seed: int) -> dict:
    import numpy as np

    mode, _, values = e7.parse_candidate(candidate)
    indexer.state_generation = generation
    torch.cuda.synchronize()
    started = time.perf_counter()
    loss, _ = e7.train_step(
        torch, fixture, module, optimizer, guarded_step, context, inputs,
        target, indexer, mode, values, 0.5, seed, diagnostic=False)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - started) * 1000.0
    result = indexer.last_result
    if result is None or result.state_generation != generation:
        raise RuntimeError("calibration step returned a stale/missing indexer workset")
    if (not np.array_equal(result.logical_edge_ids, expected_ids)
            or not np.array_equal(result.edge_ordinals, expected_ordinals)):
        raise RuntimeError("calibration indexer workset differs from host scan oracle")
    return {"wall_ms": elapsed, "loss": loss,
            "workset_density": (result.item_count /
                                (fixture.cells * len(module.support.edge_ids))),
            "workset_sha256": hashlib.sha256(
                result.logical_edge_ids.astype("<u8", copy=False).tobytes()
                + result.edge_ordinals.astype("<u8", copy=False).tobytes()
                + result.row_offsets.astype("<u8", copy=False).tobytes()).hexdigest(),
            "state_generation": generation,
            "structure_epoch": result.structure_epoch}


def summarize(values: list[float]) -> dict:
    if not values:
        raise ValueError("cannot summarize an empty timing set")
    standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"mean_ms": statistics.mean(values),
            "median_ms": statistics.median(values),
            "min_ms": min(values), "max_ms": max(values),
            "sample_count": len(values),
            "sample_standard_deviation_ms": standard_deviation,
            "mean_95pct_normal_half_width_ms": (
                1.96 * standard_deviation / math.sqrt(len(values))
                if len(values) > 1 else 0.0)}


def measure_policy_cost(trace, step_costs: dict,
                        observed_densities: dict[str, list[float]]) -> tuple[dict, dict]:
    # Exercise the exact policy function with realistic measured-cost records.
    # This is Python decision overhead, kept separate from CUDA step timings.
    observation_started = time.perf_counter()
    observation_count = 2000
    observed = None
    for index in range(observation_count):
        phase = PHASES[index & 1]
        observed = trace.classify_density(observed_densities[phase][:16])
    observation_total_ms = (time.perf_counter() - observation_started) * 1000.0
    observation = {
        "sample_count": observation_count,
        "mean_wall_ms": observation_total_ms / observation_count,
        "method": "measured repeated calls to the exact 16-step trace density classifier",
    }

    prepare_costs = {
        candidate: float(step_costs[candidate]["candidate_prepare_ms"])
        for candidate in CANDIDATES
    }
    estimated_evaluation_ms = observation["mean_wall_ms"]
    batches = []
    count = 2000
    for batch_index in range(100):
        started = time.perf_counter()
        for index in range(count // 100):
            active = CANDIDATES[(batch_index + index) & 1]
            target_class, _ = observed
            trace.choose_candidate(
                {"candidate_costs_ms": step_costs}, active, target_class, 16,
                actual_observation_ms=observation["mean_wall_ms"],
                estimated_policy_evaluation_ms=estimated_evaluation_ms,
                actual_candidate_prepare_ms=prepare_costs,
                recent_actual_switch_ms=None,
                common_rebuild_ms=None, common_support_publish_ms=None)
        batches.append((time.perf_counter() - started) * 1000.0 / (count // 100))
        estimated_evaluation_ms = statistics.mean(batches)
    policy_evaluation = {
        **summarize(batches),
        "method": "100 measured batches of 20 calls to the exact run_gpu_trace.choose_candidate function",
    }
    return observation, policy_evaluation


def run_calibration(args: argparse.Namespace, preflight: dict,
                    scheduler: dict) -> tuple[dict, dict]:
    # Delayed imports: no Torch/CUDA initialization before source and scheduler
    # preflight completes.
    import torch

    sys.path.insert(0, str(GPU_ROOT))
    sys.path.insert(0, str(HERE))
    import run_gpu_trace as trace
    e7 = trace.import_frozen_e7()
    fixture = e7.load_fixture("small", Path(preflight["fixture_root"]))
    loaded_torch, support, mechanism, package_identity = e7.load_package(
        Path(preflight["package_root"]))
    if loaded_torch is not torch:
        raise RuntimeError("calibration loaded inconsistent Torch module identities")
    if package_identity["experiment_driver_sha256"] != preflight["e7_runner_sha256"]:
        raise RuntimeError("frozen E7 package identity changed after preflight")
    if not torch.cuda.is_available():
        raise RuntimeError("admitted E7 calibration needs the optional CUDA Torch runtime")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    module = e7.prepare_module(
        torch, support, fixture, args.seed, cegeo=False,
        compiled_geometry=None, max_primal_items=len(fixture.edge_ids))
    promotion = trace.stage_and_publish(
        module, module.structure_epoch, int(fixture.sources[0]),
        (int(fixture.sources[0]) + 1) % fixture.entities,
        int(max(fixture.edge_ids)) + 1)
    optimizer = torch.optim.Adam([module.parameters_flat], lr=1.0e-3,
                                 foreach=False, fused=False)
    snapshot = e7.FixedStructureTrainingSnapshot(module, optimizer, torch)
    guarded_step = mechanism.guarded_step
    with torch.no_grad(), module._pinned_forward():
        entity = module.block("entity_embedding").detach().clone()
        rank = module.block("rank_embedding").detach().clone()

    target_cpu = torch.Generator(device="cpu").manual_seed(args.seed + 111)
    target_started = time.perf_counter()
    target = (torch.randn((fixture.cells, fixture.entities, e7.WIDTH),
                          generator=target_cpu, dtype=torch.float32) * 0.1).to(device)
    target_prepare_ms = (time.perf_counter() - target_started) * 1000.0
    indexers = {}
    indexer_setup = {}
    query_fixtures = {}
    contexts = {}
    inputs = {}
    query_setup = {}
    expected_worksets = {}
    for phase in PHASES:
        query_fixture = trace.make_trace_fixture(e7, fixture, phase)
        query_fixtures[phase] = query_fixture
        context = e7.build_context(torch, query_fixture, entity, rank, device)
        contexts[phase] = context
        inputs[phase] = context.to(torch.float16).contiguous()
        offsets, terms = e7.make_terms(query_fixture)
        expected_ids, expected_ordinals = trace.expected_workset_ids(
            e7, fixture, query_fixture.query_rows, module.support)
        expected_worksets[phase] = (expected_ids, expected_ordinals)
        query_setup[phase] = {}
        for candidate in CANDIDATES:
            indexer = indexers.get(candidate)
            if indexer is None:
                indexer, costs = trace.create_indexer(
                    e7, fixture, module.support, module.structure_epoch, args,
                    prepare_postings=(candidate == "ordered_postings_simt_full"))
                indexers[candidate] = indexer
                indexer_setup[candidate] = costs
            started = time.perf_counter()
            indexer.indexer.prepare_query(offsets, terms)
            query_setup[phase][candidate] = (time.perf_counter() - started) * 1000.0

    candidate_rows = {candidate: {phase: [] for phase in PHASES}
                      for candidate in CANDIDATES}
    pair_rows = {phase: [] for phase in PHASES}
    generation = 0
    for phase in PHASES:
        trace_fixture = query_fixtures[phase]
        for warmup in range(WARMUPS):
            for candidate in CANDIDATES:
                snapshot.restore()
                generation += 1
                measured_step(
                    torch, e7, trace_fixture, module, optimizer, guarded_step,
                    contexts[phase], inputs[phase], target, indexers[candidate],
                    candidate, generation,
                    expected_worksets[phase][0],
                    expected_worksets[phase][1],
                    args.seed + warmup)
        for round_name in ROUNDS:
            for pair in range(PAIRED_SAMPLES_PER_ROUND):
                pair_index = len(pair_rows[phase])
                order = (CANDIDATES if pair_index % 2 == 0 else CANDIDATES[::-1])
                pair_result = {"round": round_name, "pair": pair_index,
                               "order": list(order), "candidates": {}}
                for candidate in order:
                    snapshot.restore()
                    generation += 1
                    sample = measured_step(
                        torch, e7, trace_fixture, module, optimizer,
                        guarded_step, contexts[phase], inputs[phase], target,
                        indexers[candidate], candidate, generation,
                        expected_worksets[phase][0],
                        expected_worksets[phase][1],
                        args.seed + pair_index)
                    candidate_rows[candidate][phase].append(sample["wall_ms"])
                    pair_result["candidates"][candidate] = sample
                if (pair_result["candidates"][CANDIDATES[0]]["workset_sha256"]
                        != pair_result["candidates"][CANDIDATES[1]]["workset_sha256"]):
                    raise RuntimeError("scan and ordered calibration produced different worksets")
                pair_rows[phase].append(pair_result)

    torch.cuda.synchronize()
    module.retire_tapes()
    for proxy in indexers.values():
        proxy.indexer.close()

    # The measured preparation increment is the ordered-postings build cost.
    # Shared index creation/query buffers are recorded separately and are paid
    # by both candidates in the trace.
    preparation = {
        candidate: indexer_setup[candidate]["total_indexer_prepare_ms"]
        for candidate in CANDIDATES
    }
    # Measure the same tuple parse/selection operation used at trace switches.
    switch_samples = {candidate: [] for candidate in CANDIDATES}
    active = CANDIDATES[0]
    for index in range(2000):
        target_candidate = CANDIDATES[1] if active == CANDIDATES[0] else CANDIDATES[0]
        started = time.perf_counter()
        mode, _, values = e7.parse_candidate(target_candidate)
        active, current_mode, current_values = target_candidate, mode, values
        switch_samples[target_candidate].append(
            (time.perf_counter() - started) * 1000.0)
    switch_costs = {candidate: statistics.mean(samples)
                    for candidate, samples in switch_samples.items()}
    switch_summaries = {candidate: summarize(samples)
                        for candidate, samples in switch_samples.items()}
    step_costs = {
        candidate: {"step_by_density": {
            phase: statistics.mean(candidate_rows[candidate][phase])
            for phase in PHASES},
            "candidate_prepare_ms": preparation[candidate],
            "candidate_switch_ms": switch_costs[candidate]}
        for candidate in CANDIDATES
    }
    candidate_summaries = {
        candidate: {phase: summarize(candidate_rows[candidate][phase])
                    for phase in PHASES}
        for candidate in CANDIDATES
    }
    paired_delta_summaries = {}
    for phase in PHASES:
        paired_delta_summaries[phase] = summarize([
            row["candidates"][CANDIDATES[1]]["wall_ms"]
            - row["candidates"][CANDIDATES[0]]["wall_ms"]
            for row in pair_rows[phase]
        ])
    observed_densities = {
        phase: [row["candidates"][CANDIDATES[0]]["workset_density"]
                for row in pair_rows[phase]]
        for phase in PHASES
    }
    observation_cost, policy_evaluation = measure_policy_cost(
        trace, step_costs, observed_densities)
    report = {
        "schema": "cellerator.adaptive_fields.e7_trace_cost_calibration.v1",
        "evidence_kind": "measured_e7",
        "fixture": "small",
        "fixture_identity": fixture.identity(),
        "fixture_manifest_sha256": preflight["fixture_manifest_sha256"],
        "calibration_support_publication": promotion,
        "workload_definition": {
            "low": "first query entity from each actual E7 small fixture row",
            "high": "all unique source entities in each actual E7 small fixture row",
            "phases": list(PHASES),
            "values": "full-support",
            "numerical_backend": "SIMT",
            "trace_candidate_scope": list(CANDIDATES),
        },
        "measurement_design": {
            "warmup_paired_steps_per_phase": WARMUPS,
            "paired_samples_per_round": PAIRED_SAMPLES_PER_ROUND,
            "rounds": list(ROUNDS),
            "paired_order": "alternates scan-first and ordered-first",
            "reset": "FixedStructureTrainingSnapshot.restore before every measured candidate step",
            "timing": "host wall around complete train_step including query, host workset, transfer, forward, VJP, guarded Adam; torch.cuda.synchronize before and after",
        },
        "candidate_costs_ms": step_costs,
        "candidate_timing_summaries_ms": candidate_summaries,
        "paired_ordered_minus_scan_delta_summaries_ms": paired_delta_summaries,
        "per_phase_paired_samples": pair_rows,
        "indexer_setup_ms": indexer_setup,
        "query_preparation_ms": query_setup,
        "policy_evaluation": policy_evaluation,
        "window_observation_cost": observation_cost,
        "candidate_switch_timing_ms": switch_summaries,
        "target_prepare_ms": target_prepare_ms,
        "source_identity": preflight,
        "scheduler": scheduler,
        "runtime": {"python": sys.version,
                    "torch": str(torch.__version__),
                    "torch_cuda": str(torch.version.cuda)},
    }
    table = {
        "schema": "cellerator.adaptive_fields.e7_trace_measured_cost_table.v1",
        "evidence_kind": "measured_e7",
        "fixture": "small",
        "complete_step_metric": "synchronized_training_step_wall_ms",
        "evaluation_interval_steps": 16,
        "policy_evaluation_ms": policy_evaluation["mean_ms"],
        "initial_candidate": CANDIDATES[0],
        "candidate_costs_ms": step_costs,
        "source_identity": {
            "path": str(args.report_output.resolve()),
            "sha256": "filled_after_calibration_report_write",
            "e7_runner_sha256": preflight["e7_runner_sha256"],
            "indexer_library_sha256": preflight["indexer_library_sha256"],
            "fixture_manifest_sha256": preflight["fixture_manifest_sha256"],
            "calibration_runner_sha256": preflight["calibration_runner_sha256"],
            "trace_runner_sha256": preflight["trace_runner_sha256"],
        },
    }
    return report, table


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-root", type=Path, default=Path(
        "/tmp/cellerator_scaled_sparse_fixtures_887437fc"))
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--indexer-library", type=Path, default=DEFAULT_INDEXER)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--cost-table-output", type=Path, required=True)
    args = parser.parse_args()
    preflight = source_preflight(args)
    scheduler = scheduler_preflight()
    report, table = run_calibration(args, preflight, scheduler)
    report_path = args.report_output.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True,
                                      allow_nan=False) + "\n")
    table["source_identity"]["path"] = str(report_path)
    table["source_identity"]["sha256"] = sha256(report_path)
    table_path = args.cost_table_output.resolve()
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(json.dumps(table, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n")
    print(json.dumps({"status": "measured_cost_table_written",
                      "report": str(report_path),
                      "cost_table": str(table_path),
                      "gpu_execution_performed": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
