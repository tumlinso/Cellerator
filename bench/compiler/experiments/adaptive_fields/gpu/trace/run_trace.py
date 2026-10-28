#!/usr/bin/env python3
"""CPU-only E7 support trace and optional measured-cost policy replay.

The support lifecycle is exercised by the production adaptive_support_manager_v1.
Worksets are checked with the frozen E7 independent host-scan helper. This does
not import Torch or initialize CUDA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent
GPU_ROOT = HERE.parent
REPO_ROOT = HERE.parents[5]
sys.path.insert(0, str(GPU_ROOT))
from e7_fixtures import E7Fixture, host_scan_oracle  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_host_support_trace(cxx: str, artifacts: Path) -> tuple[dict, dict]:
    source = HERE / "support_trace.cc"
    header = REPO_ROOT / "include/Cellerator/compute/support/adaptive_support.hh"
    logical_header = REPO_ROOT / "include/Cellerator/compute/support/logical_support.hh"
    adaptive_source = REPO_ROOT / "src/compute/support/adaptive_support.cc"
    logical_source = REPO_ROOT / "src/compute/support/logical_support.cc"
    executable = artifacts / "support_trace"
    command = [cxx, "-std=c++17", "-O2", "-I", str(REPO_ROOT / "include"),
               str(source), str(adaptive_source), str(logical_source),
               "-o", str(executable)]
    started = time.perf_counter()
    build = subprocess.run(command, text=True, capture_output=True, check=False)
    build_ms = (time.perf_counter() - started) * 1000.0
    if build.returncode:
        raise RuntimeError(f"CPU support trace build failed:\n{build.stdout}\n{build.stderr}")
    version = subprocess.run([cxx, "--version"], text=True, capture_output=True,
                             check=True)
    run = subprocess.run([str(executable)], text=True, capture_output=True,
                         check=False)
    if run.returncode:
        raise RuntimeError(f"CPU support trace failed:\n{run.stdout}\n{run.stderr}")
    raw = json.loads(run.stdout)
    return raw, {
        "compiler": cxx,
        "compiler_version": version.stdout.splitlines()[0],
        "command": command,
        "compile_wall_ms_cpu_only": build_ms,
        "source_sha256": sha256(source),
        "adaptive_support_source_sha256": sha256(adaptive_source),
        "logical_support_source_sha256": sha256(logical_source),
        "adaptive_support_header_sha256": sha256(header),
        "logical_support_header_sha256": sha256(logical_header),
        "executable": str(executable.resolve()),
        "executable_sha256": sha256(executable),
        "gpu_execution_performed": False,
    }


def validate_worksets(raw: dict) -> list[dict]:
    """Compare every C++ manager snapshot query with E7's independent oracle."""
    if raw.get("schema") != "cellerator.adaptive_fields.e7_support_trace.v1":
        raise ValueError("unexpected support trace schema")
    if len(raw.get("steps", ())) != 48:
        raise ValueError("trace must contain three pinned 16-step windows")
    base_ids = np.arange(1, 65, dtype=np.int64)
    base_sources = np.arange(64, dtype=np.int64) % 16
    base_destinations = (
        base_sources + 1 + np.arange(64, dtype=np.int64) // 16) % 16
    validated = []
    previous_generation = 0
    for index, row in enumerate(raw["steps"]):
        expected_phase = ("low_before_publication" if index < 16 else
                          "high_after_publication" if index < 32 else
                          "low_after_publication")
        expected_epoch = 1 if index < 16 else 2
        expected_support_count = 64 if expected_epoch == 1 else 65
        expected_workset_count = 4 if index < 16 else 65 if index < 32 else 5
        if row["phase"] != expected_phase or row["structure_epoch"] != expected_epoch:
            raise ValueError(f"step {index}: wrong phase or structure epoch")
        if row["state_generation"] != previous_generation + 1:
            raise ValueError(f"step {index}: state generations are not consecutive")
        previous_generation = int(row["state_generation"])
        if int(row["support_count"]) != expected_support_count:
            raise ValueError(f"step {index}: wrong support count")
        ids = base_ids.copy()
        sources = base_sources.copy()
        destinations = base_destinations.copy()
        if expected_epoch == 2:
            ids = np.concatenate((ids, np.asarray([100], dtype=np.int64)))
            sources = np.concatenate((sources, np.asarray([0], dtype=np.int64)))
            destinations = np.concatenate((destinations, np.asarray([4], dtype=np.int64)))
        query_entities = np.asarray(row["query_source_entities"], dtype=np.int64) - 1
        queries = (query_entities.copy(),)
        fixture = E7Fixture(
            name="trace_cpu",
            root=HERE,
            manifest={"metadata": {"entities": 16}},
            manifest_sha256="experiment-local-trace",
            arrays={"edge_ids": ids, "source_entity_id": sources,
                    "participant_entity_id": destinations},
            query_rows=queries,
            context_entities=queries,
        )
        _, expected_ids, _ = host_scan_oracle(fixture)
        actual_ids = np.asarray(row["logical_edge_ids"], dtype=np.uint64)
        if actual_ids.size != expected_workset_count:
            raise ValueError(f"step {index}: unexpected low/high/low workset density")
        if not np.array_equal(actual_ids, expected_ids):
            raise ValueError(f"step {index}: stable logical workset IDs disagree with host oracle")
        density = int(actual_ids.size) / expected_support_count
        validated.append({**row, "workset_density": density,
                          "density_class": "high" if density >= 0.2 else "low",
                          "workset_sha256": hashlib.sha256(
                              actual_ids.astype("<u8", copy=False).tobytes()).hexdigest()})
    publication = raw.get("publication", {})
    if (publication.get("pinned_attempt") != "live_forward_tapes"
            or publication.get("status") != "success"
            or publication.get("structure_epoch") != 2
            or publication.get("promoted_logical_edge_id") != 100
            or publication.get("state_generation") != 48):
        raise ValueError("support publication did not respect pinned-window and identity contract")
    return validated


def replay_cost_policy(steps: list[dict], cost_table: dict,
                       evaluation_interval: int = 16) -> dict:
    """Replay the actual choices using explicit per-step/setup/switch costs."""
    if cost_table.get("schema") != "cellerator.adaptive_fields.e7_cost_table.v1":
        raise ValueError("unsupported cost table schema")
    candidates = cost_table.get("candidates", {})
    if len(candidates) < 2:
        raise ValueError("cost table must contain at least two candidates")
    for candidate, row in candidates.items():
        if set(row.get("step_ms", {})) < {"low", "high"}:
            raise ValueError(f"{candidate}: step_ms must define low and high")
        for value in [row.get("prepare_ms"), row.get("switch_ms"),
                      *row["step_ms"].values()]:
            if value is None or float(value) < 0:
                raise ValueError(f"{candidate}: costs must be nonnegative")
    policy_overhead = float(cost_table.get("policy_evaluation_ms", -1))
    if policy_overhead < 0:
        raise ValueError("cost table must declare policy_evaluation_ms")
    names = sorted(candidates)
    fixed_totals = {}
    epochs = sorted({int(step["structure_epoch"]) for step in steps})
    for name in names:
        row = candidates[name]
        fixed_totals[name] = (
            sum(float(row["prepare_ms"]) for _epoch in epochs)
            + sum(float(row["step_ms"][step["density_class"]]) for step in steps))
    fixed_winner = min(fixed_totals, key=fixed_totals.get)

    active = cost_table.get("initial_candidate", names[0])
    if active not in candidates:
        raise ValueError("initial_candidate is missing from candidates")
    prepared_epoch: dict[str, int] = {}
    total = 0.0
    decisions = []
    for block_start in range(0, len(steps), evaluation_interval):
        block_end = min(block_start + evaluation_interval, len(steps))
        epoch = int(steps[block_start]["structure_epoch"])
        observed_class = (steps[block_start - 1]["density_class"]
                          if block_start else steps[block_start]["density_class"])
        policy_ms = 0.0 if block_start == 0 else policy_overhead
        prior = active
        selected = active
        gross_step_savings = 0.0
        incremental_setup = 0.0
        net_savings = 0.0
        horizon = block_end - block_start
        if block_start:
            alternatives = [name for name in names if name != active]
            alt = min(alternatives,
                      key=lambda name: candidates[name]["step_ms"][observed_class])
            staying_setup = (float(candidates[active]["prepare_ms"])
                             if prepared_epoch.get(active) != epoch else 0.0)
            switching_setup = (
                (float(candidates[alt]["prepare_ms"])
                 if prepared_epoch.get(alt) != epoch else 0.0)
                + float(candidates[alt]["switch_ms"]))
            gross_step_savings = (
                float(candidates[active]["step_ms"][observed_class])
                - float(candidates[alt]["step_ms"][observed_class])) * horizon
            incremental_setup = switching_setup - staying_setup
            net_savings = gross_step_savings - incremental_setup
            if net_savings > 0:
                selected = alt
        prepare_cost = (float(candidates[selected]["prepare_ms"])
                        if prepared_epoch.get(selected) != epoch else 0.0)
        switch_cost = (float(candidates[selected]["switch_ms"])
                       if selected != prior else 0.0)
        block_step_cost = sum(
            float(candidates[selected]["step_ms"][step["density_class"]])
            for step in steps[block_start:block_end])
        total += policy_ms + prepare_cost + switch_cost + block_step_cost
        prepared_epoch[selected] = epoch
        active = selected
        decisions.append({
            "block_start_step": block_start,
            "block_end_step_exclusive": block_end,
            "policy_evaluated": bool(block_start),
            "structure_epoch": epoch,
            "observed_density_class_from_previous_step": observed_class,
            "active_candidate_before_decision": prior,
            "selected_candidate": selected,
            "gross_step_savings_over_next_block_ms": gross_step_savings,
            "incremental_prepare_and_switch_cost_ms": incremental_setup,
            "net_estimated_savings_ms": net_savings,
            "policy_evaluation_cost_ms": policy_ms,
            "preparation_cost_ms": prepare_cost,
            "switch_cost_ms": switch_cost,
            "actual_block_step_cost_ms": block_step_cost,
            "selection_reason": "switch only when modeled savings exceed incremental preparation/switch cost",
        })
    return {
        "status": "modeled_replay_only",
        "cost_table_evidence_kind": cost_table.get("evidence_kind"),
        "cost_table_source_identity": cost_table.get("source_identity"),
        "evaluation_interval_steps": evaluation_interval,
        "fixed_candidate_costs_ms": fixed_totals,
        "best_fixed_candidate_from_supplied_table": fixed_winner,
        "adaptive_candidate_cost_ms": total,
        "adaptive_minus_fixed_winner_ms": total - fixed_totals[fixed_winner],
        "decisions": decisions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=HERE / "artifacts")
    parser.add_argument("--cost-table", type=Path,
                        help="optional measured E7 candidate cost table for policy replay")
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw, build = run_host_support_trace(args.cxx, args.output_dir)
    steps = validate_worksets(raw)
    fixture_helper = GPU_ROOT / "e7_fixtures.py"
    runner = HERE / "run_trace.py"
    result = {
        "schema": "cellerator.adaptive_fields.e7_adaptive_trace_report.v1",
        "status": "cpu_support_trace_validated_cost_policy_pending",
        "evidence_boundary": "CPU support lifecycle and host workset correctness only; no GPU execution, kernels, or timing measurements",
        "support_trace": raw,
        "validated_steps": steps,
        "support_lifecycle": {
            "production_component": "adaptive_support_manager_v1",
            "pinned_windows": ["steps 0-15 epoch 1", "steps 16-31 epoch 2", "steps 32-47 epoch 2"],
            "publication": raw["publication"],
            "stable_logical_identity_check": "all 64 original IDs retained and new ID 100 appears only after epoch 2 publication",
        },
        "indexer_integration": {
            "status": "unresolved_gpu_integration",
            "source_counterexample": "gpu/indexer/indexer_bridge.h exposes af_index_create_v1 for immutable support plus af_index_destroy_v1; it has no publish/update/epoch-transition API. GPU follow-up must create a new handle at epoch 2 after retiring epoch-1 readers, then replace the caller's handle at the window boundary.",
        },
        "cost_policy": {"status": "unresolved_missing_measured_cost_table"},
        "build": build,
        "sources": {
            "run_trace_sha256": sha256(runner),
            "support_trace_sha256": sha256(HERE / "support_trace.cc"),
            "fixture_helper_sha256": sha256(fixture_helper),
            "indexer_header_sha256": sha256(GPU_ROOT / "indexer/indexer_bridge.h"),
            "indexer_binding_sha256": sha256(GPU_ROOT / "indexer/ctypes_binding.py"),
        },
        "platform": platform.platform(),
        "gpu_execution_performed": False,
    }
    if args.cost_table:
        table = json.loads(args.cost_table.read_text())
        if table.get("evidence_kind") == "measured_e7":
            identity = table.get("source_identity")
            if not isinstance(identity, dict) or not identity.get("path") \
                    or not identity.get("sha256"):
                raise ValueError("measured cost table requires source_identity path and sha256")
            source_path = Path(identity["path"])
            if not source_path.is_absolute():
                source_path = args.cost_table.resolve().parent / source_path
            actual_source_hash = sha256(source_path.resolve(strict=True))
            if actual_source_hash != identity["sha256"]:
                raise ValueError("measured cost table source identity hash mismatch")
            identity["path"] = str(source_path.resolve())
        replay = replay_cost_policy(steps, table)
        result["cost_policy"] = replay
        if table.get("evidence_kind") != "measured_e7":
            result["cost_policy"]["status"] = "scenario_only_not_measurement"
        else:
            result["cost_policy"]["status"] = "measured_cost_table_replay"
    raw_path = args.output_dir / "support_trace.json"
    report_path = args.output_dir / "trace_report.json"
    raw_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema": "cellerator.adaptive_fields.e7_adaptive_trace_manifest.v1",
        "gpu_execution_performed": False,
        "runner_sha256": sha256(runner),
        "support_trace_source_sha256": sha256(HERE / "support_trace.cc"),
        "host_trace_executable_sha256": build["executable_sha256"],
        "adaptive_support_source_sha256": build["adaptive_support_source_sha256"],
        "logical_support_source_sha256": build["logical_support_source_sha256"],
        "fixture_helper_sha256": sha256(fixture_helper),
        "indexer_header_sha256": sha256(GPU_ROOT / "indexer/indexer_bridge.h"),
        "indexer_binding_sha256": sha256(GPU_ROOT / "indexer/ctypes_binding.py"),
        "cost_table": (str(args.cost_table.resolve()) if args.cost_table else None),
        "cost_table_sha256": sha256(args.cost_table) if args.cost_table else None,
        "raw_trace_output_sha256": sha256(raw_path),
        "report_sha256": sha256(report_path),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(report_path), "manifest": str(manifest_path),
                      "steps": len(steps), "gpu_execution_performed": False,
                      "cost_policy_status": result["cost_policy"]["status"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
