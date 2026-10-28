#!/usr/bin/env python3
"""Isolated E5 executor for an inspectable recurrent event field.

GPU execution is admitted only through the root-owned Device Scheduler. This
adapter uses the existing Torch sparse owner and native primitive entrypoints;
it does not modify Cellerator or claim that these Python stages are production
compiler IR.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
DEFAULT_PACKAGE = Path(
    "/home/tumlinson/Cellerator/build-sparse-support/evidence/"
    "torch-v14-optional-policies/package")
SEEDS = (20261027, 20261028, 20261029)
WIDTH = 64


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Op:
    name: str
    consumes: tuple[str, ...]
    produces: tuple[str, ...]
    effects: tuple[str, ...]
    invoke: Callable[[dict[str, Any]], dict[str, Any]]


class FieldPlan:
    """Small dependency-checked field invoker; callbacks are native APIs."""

    def __init__(self, operations: tuple[Op, ...]):
        self.operations = operations
        self.validate()

    def validate(self) -> None:
        available = {"module", "step", "prefix", "timestamps", "windows",
                     "provider", "state", "policy", "seed", "threshold",
                     "temperature", "prune", "value_policy", "genes", "ranks",
                     "workset", "window_counter"}
        produced: set[str] = set()
        for op in self.operations:
            missing = set(op.consumes) - available - produced
            if missing:
                raise ValueError(f"{op.name} has unresolved inputs: {sorted(missing)}")
            overlap = produced.intersection(op.produces)
            if overlap:
                raise ValueError(f"field outputs are not unique: {sorted(overlap)}")
            produced.update(op.produces)
        if "relation_output" not in produced:
            raise ValueError("field plan does not produce relation_output")

    def run(self, env: dict[str, Any]) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
        values = dict(env)
        trace = []
        for op in self.operations:
            started = time.perf_counter()
            outputs = op.invoke(values)
            missing = set(op.produces) - set(outputs)
            if missing:
                raise RuntimeError(f"{op.name} omitted outputs: {sorted(missing)}")
            values.update(outputs)
            trace.append({
                "operation": op.name,
                "consumes": list(op.consumes),
                "produces": list(op.produces),
                "effects": list(op.effects),
                "host_enqueue_ms": (time.perf_counter() - started) * 1000.0,
            })
        return values["relation_output"], trace, values


def make_field_plan(torch, history_sum, relation_input_fn):
    """Build the event recurrence from shared exposed operation callbacks."""

    def provider(env):
        work = env["provider"](
            env["step"], env["prefix"], env["timestamps"], env["windows"])
        return {"event_work": work}

    def transition(env):
        module = env["module"]
        work = env["event_work"]
        event = env["prefix"][:, -1].to(device=module.parameters_flat.device,
                                          dtype=torch.int64)
        embeddings = module.block("entity_embedding")
        event_input = embeddings[event]
        next_state = module._torch_native.sparse_latent_transition(
            env["state"].to(torch.float16), event_input.to(torch.float16),
            work[2], work[3].to(torch.int32).contiguous(),
            module.block("transition_A"), module.block("transition_B"),
            module.block("transition_bias"), module._native)
        return {"event": event, "event_input": event_input,
                "next_state": next_state}

    def context(env):
        module = env["module"]
        embeddings = module.block("entity_embedding")
        cells = env["event"].shape[0]
        current = embeddings.new_zeros((cells, embeddings.shape[0], embeddings.shape[1]))
        current.scatter_add_(
            1, env["event"][:, None, None].expand(-1, 1, embeddings.shape[1]),
            env["event_input"].unsqueeze(1))
        history = None
        if len(env["event_work"]) == 5:
            history = history_sum(embeddings, env["event_work"][4], cells=cells)
            gate_context = history + env["next_state"].float().mean(dim=1, keepdim=True)
        else:
            gate_context = current + env["next_state"].float().mean(dim=1, keepdim=True)
        return {"current_input": current, "history_input": history,
                "gate_context": gate_context.expand(
                    -1, embeddings.shape[0], -1).contiguous()}

    def score(env):
        gates = env["module"].score_candidates(
            env["gate_context"], env["event_work"][1],
            temperature=env["temperature"], threshold=env["threshold"],
            prune=env["prune"])
        return {"gates": gates}

    def select(env):
        primal, derivative = env["module"].select_gate_worksets(
            env["event_work"][1], env["gates"], seed=env["seed"],
            window_counter=env["event_work"][0].state_generation)
        return {"primal": primal, "derivative": derivative}

    def values(env):
        edge_values = env["module"].logical_edge_values(
            policy=env["value_policy"], primal=env["primal"],
            derivative=env["derivative"], cells=env["event"].shape[0])
        return {"edge_values": edge_values}

    def relation(env):
        source = relation_input_fn(
            env["current_input"], env["history_input"], env["policy"])
        output = env["module"].relation(
            source.to(torch.float16), env["edge_values"], env["gates"],
            env["primal"], env["derivative"])
        return {"relation_output": output}

    return FieldPlan((
        Op("event_workset_provider", ("provider", "step", "prefix", "timestamps", "windows"),
           ("event_work",), ("reads:temporal-state", "writes:workset-state"), provider),
        Op("native_latent_transition", ("module", "event_work", "prefix", "state"),
           ("event", "event_input", "next_state"), ("reads:owner", "writes:latent-state"),
           transition),
        Op("event_and_context", ("module", "event", "event_input", "event_work", "next_state"),
           ("current_input", "history_input", "gate_context"),
           ("reads:owner", "writes:intermediate"), context),
        Op("candidate_gate_scoring", ("module", "gate_context", "event_work", "temperature", "threshold", "prune"),
           ("gates",), ("reads:owner", "writes:gate-values"), score),
        Op("primal_derivative_selection", ("module", "event_work", "gates", "seed"),
           ("primal", "derivative"), ("reads:structure", "writes:worksets"), select),
        Op("shared_edge_values", ("module", "primal", "derivative", "event", "value_policy"),
           ("edge_values",), ("reads:owner", "reads:worksets"), values),
        Op("native_sparse_relation", ("module", "current_input", "history_input", "policy", "edge_values", "gates", "primal", "derivative"),
           ("relation_output",), ("reads:structure", "reads:owner", "writes:output"), relation),
    ))


def field_event_forward(torch, module, index, fixture, plan, seed, relation_policy,
                        value_policy):
    """Run one short recurrence through the local stage plan."""
    from cellerator.torch.sparse_support import _sum_history_sources
    import train_synthetic as base

    device = module.parameters_flat.device
    provider = base.event_provider(module, index, fixture, torch.device(device))
    module._ensure_mechanism_route_index()
    events = fixture["events"].to(device)
    timestamps = fixture["timestamps"].to(device)
    windows = fixture["count_windows"].to(device)
    position_offset = int(module.application_state.get("event_position", 0))
    state = module.block("entity_embedding").new_zeros((events.shape[0], 4, WIDTH))
    previous = module.application_state.get("latent_state")
    if previous is not None:
        state = torch.as_tensor(previous, dtype=torch.float32, device=device)
    encoded = None
    trace = []
    with module._pinned_forward():
        for position in range(events.shape[1]):
            env = {
                "module": module, "step": position_offset + position,
                "prefix": events[:, :position + 1],
                "timestamps": timestamps[:, :position + 1],
                "windows": windows, "provider": provider,
                "state": state, "policy": relation_policy,
                "seed": seed, "temperature": 1.0,
                "threshold": 0.5, "prune": True,
                "value_policy": value_policy,
            }
            encoded, operations, values = plan.run(env)
            state = values["next_state"]
            trace.append(operations)
        module.application_state["latent_state"] = state.detach()
        module.application_state["event_position"] = position_offset + int(events.shape[1])
        module.application_state["relation_input_policy"] = relation_policy
        module.application_state["last_event_ids"] = events[:, -1].detach()
        module.application_state["last_timestamps"] = timestamps[:, -1].detach()
        module.application_state["windows"] = windows.detach()
    return encoded, trace


def event_forward(torch, module, index, fixture, plan, seed, relation_policy,
                  value_policy, *, field_plan: bool):
    if field_plan:
        return field_event_forward(torch, module, index, fixture, plan, seed,
                                   relation_policy, value_policy)
    import train_synthetic as base
    provider = base.event_provider(module, index, fixture,
                                   torch.device(module.parameters_flat.device))
    encoded = module(
        fixture["events"].to(module.parameters_flat.device), provider,
        timestamps=fixture["timestamps"].to(module.parameters_flat.device),
        windows=fixture["count_windows"].to(module.parameters_flat.device),
        prune=True, threshold=0.5, derivative_seed=seed,
        relation_input=relation_policy, edge_value_policy=value_policy)
    return encoded, []


def make_ranked_field_plan(torch, relation_input_fn):
    """Ranked composition with the same shared gate/value/relation tail."""
    def ranked_inputs(env):
        module = env["module"]
        embeddings = module.block("entity_embedding")
        genes = env["genes"].to(device=embeddings.device, dtype=torch.int64)
        ranks = env["ranks"].to(device=embeddings.device, dtype=torch.int64)
        visible = genes >= 0
        gene = embeddings[genes.clamp_min(0)] * visible.unsqueeze(-1)
        rank = module.block("rank_embedding")[ranks] * visible.unsqueeze(-1)
        affected_offsets, affected = module.affected_mechanism_workset(genes)
        visible_count = visible.sum(dim=1).clamp_min(1).to(torch.float32)
        prior = gene.float().sum(dim=1) / visible_count[:, None]
        state = prior[:, None, :].expand(-1, module.block("transition_A").shape[0], -1).contiguous()
        transition_input = (gene + rank).float().sum(dim=1)
        source = embeddings.new_zeros((genes.shape[0], embeddings.shape[0], embeddings.shape[1]))
        source.scatter_add_(
            1, genes.clamp_min(0).unsqueeze(-1).expand(-1, -1, embeddings.shape[-1]),
            (gene + rank).float())
        return {"initial_state": state, "transition_input": transition_input,
                "affected_offsets": affected_offsets, "affected_mechanisms": affected,
                "source_input": source, "candidate_derivative": env["workset"]}

    def transition(env):
        module = env["module"]
        result = module._torch_native.sparse_latent_transition(
            env["initial_state"].to(torch.float16),
            env["transition_input"].to(torch.float16),
            env["affected_offsets"], env["affected_mechanisms"].to(torch.int32).contiguous(),
            module.block("transition_A"), module.block("transition_B"),
            module.block("transition_bias"), module._native)
        return {"next_state": result}

    def context(env):
        gate_context = env["source_input"] + env["next_state"].float().mean(dim=1)[:, None, :]
        return {"current_input": gate_context, "history_input": None,
                "gate_context": gate_context}

    def score(env):
        gates = env["module"].score_candidates(
            env["gate_context"], env["candidate_derivative"], temperature=env["temperature"],
            threshold=env["threshold"], prune=env["prune"])
        return {"gates": gates}

    def select(env):
        primal, derivative = env["module"].select_gate_worksets(
            env["candidate_derivative"], env["gates"], seed=env["seed"],
            window_counter=env["window_counter"])
        return {"primal": primal, "derivative": derivative}

    def values(env):
        return {"edge_values": env["module"].logical_edge_values(
            policy=env["value_policy"], primal=env["primal"],
            derivative=env["derivative"], cells=env["genes"].shape[0])}

    def relation(env):
        output = env["module"].relation(
            env["current_input"].to(torch.float16), env["edge_values"],
            env["gates"], env["primal"], env["derivative"])
        return {"relation_output": output}

    return FieldPlan((
        Op("ranked_input_and_candidate_binding", ("module", "genes", "ranks", "workset"),
           ("initial_state", "transition_input", "affected_offsets", "affected_mechanisms",
            "source_input", "candidate_derivative"), ("reads:owner", "reads:ranked-input"), ranked_inputs),
        Op("native_latent_transition", ("module", "initial_state", "transition_input",
           "affected_offsets", "affected_mechanisms"), ("next_state",),
           ("reads:owner", "writes:latent-state"), transition),
        Op("ranked_context", ("source_input", "next_state"),
           ("current_input", "history_input", "gate_context"),
           ("writes:intermediate",), context),
        Op("candidate_gate_scoring", ("module", "gate_context", "candidate_derivative", "temperature",
           "threshold", "prune"), ("gates",), ("reads:owner", "writes:gate-values"), score),
        Op("primal_derivative_selection", ("module", "candidate_derivative", "gates", "seed",
           "window_counter"), ("primal", "derivative"),
           ("reads:structure", "writes:worksets"), select),
        Op("shared_edge_values", ("module", "primal", "derivative", "genes", "value_policy"),
           ("edge_values",), ("reads:owner", "reads:worksets"), values),
        Op("native_sparse_relation", ("module", "current_input", "history_input", "policy",
           "edge_values", "gates", "primal", "derivative"), ("relation_output",),
           ("reads:structure", "reads:owner", "writes:output"), relation),
    ))


def digest_tensor(tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _copy_tensor(tensor):
    return tensor.detach().contiguous().cpu().numpy().copy()


def _capture_workset(target: dict[str, Any], prefix: str, workset) -> None:
    for name in ("cell_offsets", "edge_ordinals", "logical_edge_ids",
                 "gate_value_indices"):
        target[f"{prefix}_{name}"] = _copy_tensor(getattr(workset, name))


def instrument_native_calls(module, captures: dict[str, Any], prefix: str) -> None:
    """Record primitive boundaries without changing their return values."""
    originals = {
        "score_candidates": module.score_candidates,
        "select_gate_worksets": module.select_gate_worksets,
        "logical_edge_values": module.logical_edge_values,
        "relation": module.relation,
    }
    counters = {name: 0 for name in originals}

    def score(*args, **kwargs):
        result = originals["score_candidates"](*args, **kwargs)
        index = counters["score_candidates"]
        counters["score_candidates"] += 1
        base = f"{prefix}_step{index}_score"
        captures[f"{base}_context"] = _copy_tensor(args[0])
        _capture_workset(captures, f"{base}_candidate", args[1])
        for name in ("probabilities", "admitted", "train_values",
                     "logical_edge_ids", "edge_ordinals"):
            captures[f"{base}_{name}"] = _copy_tensor(getattr(result, name))
        return result

    def select(*args, **kwargs):
        result = originals["select_gate_worksets"](*args, **kwargs)
        index = counters["select_gate_worksets"]
        counters["select_gate_worksets"] += 1
        base = f"{prefix}_step{index}_selected"
        _capture_workset(captures, f"{base}_primal", result[0])
        _capture_workset(captures, f"{base}_derivative", result[1])
        return result

    def values(*args, **kwargs):
        result = originals["logical_edge_values"](*args, **kwargs)
        index = counters["logical_edge_values"]
        counters["logical_edge_values"] += 1
        captures[f"{prefix}_step{index}_edge_values"] = _copy_tensor(result)
        return result

    def relation(*args, **kwargs):
        result = originals["relation"](*args, **kwargs)
        index = counters["relation"]
        counters["relation"] += 1
        captures[f"{prefix}_step{index}_relation_input"] = _copy_tensor(args[0])
        captures[f"{prefix}_step{index}_relation_output"] = _copy_tensor(result)
        return result

    module.score_candidates = score
    module.select_gate_worksets = select
    module.logical_edge_values = values
    module.relation = relation


def recording_provider(provider, captures: dict[str, Any], prefix: str):
    counter = {"step": 0}

    def wrapped(position, history, timestamps, windows):
        work = provider(position, history, timestamps, windows)
        index = counter["step"]
        counter["step"] += 1
        base = f"{prefix}_step{index}_provided"
        _capture_workset(captures, f"{base}_primal", work[0])
        _capture_workset(captures, f"{base}_derivative", work[1])
        captures[f"{base}_affected_offsets"] = _copy_tensor(work[2])
        captures[f"{base}_affected_mechanisms"] = _copy_tensor(work[3])
        if len(work) == 5:
            captures[f"{base}_history_sources_json"] = np.asarray(
                json.dumps(work[4], separators=(",", ":")))
        return work
    return wrapped


def instrument_latent_transition(native_module, captures: dict[str, Any], prefix: str):
    original = native_module.sparse_latent_transition
    counter = {"step": 0}

    def wrapped(state, event, offsets, affected, a, b, bias, handle):
        index = counter["step"]
        base = f"{prefix}_step{index}_latent"
        counter["step"] += 1
        captures[f"{base}_state_input"] = _copy_tensor(state)
        captures[f"{base}_event_input"] = _copy_tensor(event)
        captures[f"{base}_affected_offsets"] = _copy_tensor(offsets)
        captures[f"{base}_affected_mechanisms"] = _copy_tensor(affected)
        result = original(state, event, offsets, affected, a, b, bias, handle)
        captures[f"{base}_state_output"] = _copy_tensor(result)
        return result

    native_module.sparse_latent_transition = wrapped

    def restore():
        native_module.sparse_latent_transition = original
    return restore


def run_ranked_case(torch, native, base, module_type, support, fixture, rows,
                    seed, value_policy, device, axis_words, captures):
    from cellerator.torch.sparse_support import _relation_source_inputs, prepare_sparse_slice
    from cellerator.torch.mechanism import guarded_step, load_checkpoint, save_checkpoint

    modules = []
    optimizers = []
    indexes = []
    for _ in range(2):
        module = prepare_sparse_slice(
            module_type, support, entities=24, mechanisms=4, rank_count=8,
            device=device, parameter_axis_words=axis_words, seed=seed,
            reader_capacity=16)
        modules.append(module)
        optimizers.append(torch.optim.Adam([module.parameters_flat], lr=0.02))
        indexes.append(native.NativeReverseIndex(module._support_index))
    if not torch.equal(modules[0].parameters_flat, modules[1].parameters_flat):
        raise RuntimeError("ranked direct and field owners do not start identically")
    captures[f"ranked_initial_parameters_{seed}_{value_policy}"] = _copy_tensor(
        modules[0].parameters_flat)
    primal_derivative = [base.workset_rows(
        indexes[i], support, rows, device, seed) for i in range(2)]
    plan = make_ranked_field_plan(torch, _relation_source_inputs)
    results, losses, gradients = [], [], []
    for index, module in enumerate(modules):
        module.train()
        prefix = f"ranked_{'direct' if index == 0 else 'field'}_{seed}_{value_policy}"
        instrument_native_calls(module, captures, prefix)
        restore = instrument_latent_transition(module._torch_native, captures, prefix)
        primal, derivative = primal_derivative[index]
        if index == 0:
            try:
                encoded = module(
                    fixture["genes"].to(device), fixture["ranks"].to(device),
                    primal, derivative, prune=True, threshold=0.5,
                    derivative_seed=seed, window_counter=seed,
                    edge_value_policy=value_policy)
            finally:
                restore()
        else:
            try:
                with module._pinned_forward():
                    module._ensure_mechanism_route_index()
                    env = {
                        "module": module, "genes": fixture["genes"].to(device),
                        "ranks": fixture["ranks"].to(device), "workset": derivative,
                        "temperature": 1.0, "threshold": 0.5, "prune": True,
                        "seed": seed, "window_counter": seed,
                        "value_policy": value_policy, "policy": "current_event",
                    }
                    encoded, trace, _values = plan.run(env)
                captures[f"{prefix}_plan_trace_json"] = np.asarray(
                    json.dumps(trace, sort_keys=True, separators=(",", ":")))
            finally:
                restore()
        loss = module.training_loss(encoded, fixture["targets"].to(device))
        loss.backward()
        results.append(encoded.detach().clone())
        losses.append(float(loss.detach().cpu()))
        gradients.append(module.parameters_flat.grad.detach().clone())
        captures[f"{prefix}_final_output"] = _copy_tensor(encoded)
        captures[f"{prefix}_loss"] = np.asarray(losses[-1], dtype=np.float64)
        captures[f"{prefix}_owner_gradient"] = _copy_tensor(module.parameters_flat.grad)
        if not guarded_step(module, optimizers[index]):
            raise FloatingPointError("ranked guarded owner update rejected gradient")
        captures[f"{prefix}_owner_after_step"] = _copy_tensor(module.parameters_flat)
        module.zero_grad(set_to_none=True)
    if not torch.allclose(results[0], results[1], atol=2e-3, rtol=2e-3):
        raise AssertionError("ranked direct and field-plan output mismatch")
    if abs(losses[0] - losses[1]) > 5e-4:
        raise AssertionError("ranked direct and field-plan loss mismatch")
    if not torch.allclose(gradients[0], gradients[1], atol=5e-3, rtol=5e-3):
        raise AssertionError("ranked direct and field-plan owner VJP mismatch")
    if not torch.allclose(modules[0].parameters_flat, modules[1].parameters_flat,
                          atol=5e-3, rtol=5e-3):
        raise AssertionError("ranked direct and field-plan update mismatch")
    captures[f"ranked_genes_{seed}"] = _copy_tensor(fixture["genes"])
    captures[f"ranked_ranks_{seed}"] = _copy_tensor(fixture["ranks"])
    captures[f"ranked_masked_positions_{seed}"] = _copy_tensor(fixture["masked_position"])
    captures[f"ranked_targets_{seed}"] = _copy_tensor(fixture["targets"])
    for block_name, block in modules[0]._blocks.items():
        captures[f"ranked_block_{block_name}_offset_shape"] = np.asarray(
            [block.offset, block.numel, *block.shape], dtype=np.int64)
    for name, array in (("ids", support.edge_ids), ("sources", support.sources),
                        ("destinations", support.destinations)):
        captures[f"ranked_support_{name}_{seed}"] = np.asarray(array, dtype=np.int64)
    for index, (primal, derivative) in enumerate(primal_derivative):
        _capture_workset(captures, f"ranked_workset{index}_primal", primal)
        _capture_workset(captures, f"ranked_workset{index}_derivative", derivative)
    return {
        "task": "ranked_masked_gene_one_transition",
        "fixture": {"seed": seed, "cells": 4, "selected": 8,
                    "entities": 24, "width": WIDTH},
        "policy": "current_event",
        "value_policy": value_policy,
        "output_sha256": [digest_tensor(value) for value in results],
        "loss": losses,
        "owner_gradient_max_abs": float((gradients[0] - gradients[1]).abs().max().cpu()),
        "owner_after_update_max_abs": float((modules[0].parameters_flat -
                                              modules[1].parameters_flat).abs().max().cpu()),
        "field_stage_graph": [
            {"name": op.name, "consumes": list(op.consumes),
             "produces": list(op.produces), "effects": list(op.effects)}
            for op in plan.operations],
        "boundary": "ranked operation stages are experiment-local compositions of existing native calls",
    }


def run_seed(args, seed: int) -> dict[str, Any]:
    if os.environ.get("CELLERATOR_SPARSE_SUPPORT_BENCH_ADMITTED") != "1":
        raise RuntimeError("E5 requires the root-admitted Device Scheduler job")
    package_root = Path(os.environ.get("CELLERATOR_SPARSE_PACKAGE_ROOT", DEFAULT_PACKAGE))
    if not package_root.is_dir():
        raise RuntimeError("CELLERATOR_SPARSE_PACKAGE_ROOT is not a frozen package")
    identity_path = package_root.parent / "identity.json"
    identity = json.loads(identity_path.read_text())
    identity_files = identity.get("files", identity)
    for relative in (
            "package/cellerator/torch/sparse_support.py",
            "package/cellerator/torch/mechanism.py",
            "package/cellerator/torch/_torch.cpython-313-x86_64-linux-gnu.so"):
        expected = identity_files.get(relative)
        actual = sha256(package_root.parent / relative)
        if not isinstance(expected, str) or actual != expected:
            raise RuntimeError(f"frozen package identity mismatch: {relative}")
    checkout_source = ROOT / "python/cellerator/torch/sparse_support.py"
    if sha256(checkout_source) != sha256(package_root / "cellerator/torch/sparse_support.py"):
        raise RuntimeError("frozen sparse-support source differs from the current checkout")
    sys.path.insert(0, str(package_root))
    example_dir = package_root.parent / "examples/sparse_support"
    sys.path.insert(0, str(example_dir))

    import torch
    from cellerator.torch.sparse_support import (
        KaryoplastEventSlice, RankedGeneSlice, _relation_source_inputs,
        _sum_history_sources, make_masked_gene_fixture, make_next_event_fixture,
        prepare_sparse_slice,
    )
    from cellerator.torch.mechanism import guarded_step
    from cellerator.torch import _torch as native
    import train_synthetic as base

    if not torch.cuda.is_available():
        raise RuntimeError("admitted job has no CUDA device")
    device = torch.cuda.current_device()
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    support = base.pair_support(seed)
    fixture = make_next_event_fixture(seed=seed, cells=4, steps=6, entities=24)
    axis_words = (101, 102, 201, 202, 301, 302, 401, 402)
    plan = make_field_plan(torch, _sum_history_sources, _relation_source_inputs)
    records = []
    raw_arrays: dict[str, Any] = {
        "event_ids": _copy_tensor(fixture["events"]),
        "timestamps": _copy_tensor(fixture["timestamps"]),
        "count_windows": _copy_tensor(fixture["count_windows"]),
        "time_windows": _copy_tensor(fixture["time_windows"]),
        "targets": _copy_tensor(fixture["targets"]),
    }
    policy = args.relation_policy
    for policy in (policy,):
        initial_state = {}
        modules = []
        optimizers = []
        indexes = []
        for _ in range(2):
            module = prepare_sparse_slice(
                KaryoplastEventSlice, support, entities=24, mechanisms=4,
                rank_count=8, device=device, parameter_axis_words=axis_words,
                seed=seed, reader_capacity=16)
            module.application_state.update(initial_state)
            modules.append(module)
            optimizers.append(torch.optim.Adam([module.parameters_flat], lr=0.02))
            indexes.append(native.NativeReverseIndex(module._support_index))
        if not torch.equal(modules[0].parameters_flat, modules[1].parameters_flat):
            raise RuntimeError("direct and field-plan owners do not start identically")
        raw_arrays[f"initial_parameters_{policy}_{args.value_policy}"] = _copy_tensor(
            modules[0].parameters_flat)
        for module in modules:
            module.train()
            base.reset_event_stream_state(module)
        for name, block in modules[0]._blocks.items():
            raw_arrays[f"block_{name}_offset_shape"] = np.asarray(
                [block.offset, block.numel, *block.shape], dtype=np.int64)
        raw_arrays[f"support_edge_ids_{policy}_{args.value_policy}"] = np.asarray(
            support.edge_ids, dtype=np.int64)
        raw_arrays[f"support_sources_{policy}_{args.value_policy}"] = np.asarray(
            support.sources, dtype=np.int64)
        raw_arrays[f"support_destinations_{policy}_{args.value_policy}"] = np.asarray(
            support.destinations, dtype=np.int64)
        outputs = []
        losses = []
        gradients = []
        traces = []
        snapshots = []
        for candidate_index, module in enumerate(modules):
            provider = base.event_provider(module, indexes[candidate_index], fixture,
                                           torch.device(device))
            capture_prefix = ("direct" if candidate_index == 0 else "field") + f"_{policy}_{args.value_policy}"
            provider = recording_provider(provider, raw_arrays, capture_prefix)
            instrument_native_calls(module, raw_arrays, capture_prefix)
            restore_transition = instrument_latent_transition(
                module._torch_native, raw_arrays, capture_prefix)
            if candidate_index == 0:
                try:
                    encoded = module(
                        fixture["events"].to(device), provider,
                        timestamps=fixture["timestamps"].to(device),
                        windows=fixture["count_windows"].to(device),
                        prune=True, threshold=0.5, derivative_seed=seed,
                        relation_input=policy,
                        edge_value_policy=args.value_policy)
                finally:
                    restore_transition()
                traces.append([{"native_callable": "KaryoplastEventSlice.forward",
                                "compiler_visible_internal_stages": False}])
            else:
                # Experiment-local event loop adapted from the current
                # `_forward_pinned`; every operation below calls the existing
                # native owner, latent transition, gate selection, value, or
                # relation API directly rather than invoking forward twice.
                try:
                    with module._pinned_forward():
                        module._ensure_mechanism_route_index()
                        state = module.block("entity_embedding").new_zeros((
                            fixture["events"].shape[0], 4, WIDTH))
                        encoded = None
                        step_traces = []
                        events = fixture["events"].to(device)
                        timestamps = fixture["timestamps"].to(device)
                        windows = fixture["count_windows"].to(device)
                        for position in range(events.shape[1]):
                            env = {
                                "module": module, "step": position,
                                "prefix": events[:, :position + 1],
                                "timestamps": timestamps[:, :position + 1],
                                "windows": windows, "provider": provider,
                                "state": state, "policy": policy,
                                "seed": seed, "temperature": 1.0,
                                "threshold": 0.5, "prune": True,
                                "value_policy": args.value_policy,
                            }
                            encoded, trace, values = plan.run(env)
                            state = values["next_state"]
                            step_traces.append(trace)
                        module.application_state["latent_state"] = state.detach()
                        module.application_state["event_position"] = int(events.shape[1])
                        module.application_state["relation_input_policy"] = policy
                        module.application_state["last_event_ids"] = events[:, -1].detach()
                        module.application_state["last_timestamps"] = timestamps[:, -1].detach()
                        module.application_state["windows"] = windows.detach()
                        traces.append(step_traces)
                finally:
                    restore_transition()
            loss = module.training_loss(encoded, fixture["targets"].to(device))
            loss.backward()
            if module.parameters_flat.grad is None:
                raise RuntimeError("field/native path did not produce owner gradient")
            outputs.append(encoded.detach().clone())
            losses.append(float(loss.detach().cpu()))
            gradients.append(module.parameters_flat.grad.detach().clone())
            raw_arrays[f"{capture_prefix}_final_output"] = _copy_tensor(encoded)
            raw_arrays[f"{capture_prefix}_loss"] = np.asarray(losses[-1], dtype=np.float64)
            raw_arrays[f"{capture_prefix}_owner_gradient"] = _copy_tensor(
                module.parameters_flat.grad)
            snapshots.append({
                "structure_epoch": module.structure_epoch,
                "value_generation": int(module.application_state.get("parameter_generation", 0)),
                "state_generation": int(module.application_state.get("event_position", 0)),
                "primal_derivative_trace_count": len(traces[-1]),
                "support_ids_sha256": hashlib.sha256(
                    json.dumps(module.support.edge_ids).encode()).hexdigest(),
            })
            if not guarded_step(module, optimizers[candidate_index]):
                raise FloatingPointError("guarded owner update rejected gradient")
            raw_arrays[f"{capture_prefix}_owner_after_step"] = _copy_tensor(
                module.parameters_flat.detach())
            module.zero_grad(set_to_none=True)
        if not torch.allclose(outputs[0], outputs[1], atol=2e-3, rtol=2e-3):
            raise AssertionError("direct and field-plan recurrent output mismatch")
        if abs(losses[0] - losses[1]) > 5e-4:
            raise AssertionError("direct and field-plan recurrent loss mismatch")
        if not torch.allclose(gradients[0], gradients[1], atol=5e-3, rtol=5e-3):
            raise AssertionError("direct and field-plan owner VJP mismatch")
        if not torch.allclose(modules[0].parameters_flat, modules[1].parameters_flat,
                              atol=5e-3, rtol=5e-3):
            raise AssertionError("direct and field-plan optimizer update mismatch")
        event_record = {
            "policy": policy,
            "fixture": {"seed": seed, "cells": 4, "steps": 6,
                        "entities": 24, "width": WIDTH},
            "output_sha256": [digest_tensor(value) for value in outputs],
            "loss": losses,
            "owner_gradient_sha256": [digest_tensor(value) for value in gradients],
            "owner_gradient_max_abs": float((gradients[0] - gradients[1]).abs().max().cpu()),
            "owner_after_update_max_abs": float((modules[0].parameters_flat -
                                                  modules[1].parameters_flat).abs().max().cpu()),
            "lifecycle": snapshots,
            "gate_policy": {"estimator": "straight_through", "prune": True,
                            "temperature": 1.0, "threshold": 0.5,
                            "dormant_budget": 4, "derivative_seed": seed,
                            "value_policy": args.value_policy,
                            "loss": "cross_entropy(next_event_logits, targets)"},
            "optimizer": {"type": "torch.optim.Adam", "lr": 0.02,
                          "betas": [0.9, 0.999], "eps": 1e-8,
                          "foreach": None, "fused": None, "steps": 1,
                          "state_initially_empty": True},
            "field_stage_graph": [
                {"name": op.name, "consumes": list(op.consumes),
                 "produces": list(op.produces), "effects": list(op.effects)}
                for op in plan.operations],
            "trace": traces,
            "boundary": "experiment-local event loop uses existing native operations; no production field contract",
        }
        if any(module._read_tickets for module in modules):
            raise RuntimeError("pinned parameter tape did not retire before support publication")
        revoked_id = int(support.edge_ids[-1])
        promoted_id = int(max(support.edge_ids) + 1000)
        publication_rows = []
        for module in modules:
            old_epoch = int(module.structure_epoch)
            module.stage_revocation(revoked_id)
            module.stage_promotion(
                promoted_id, 23, 0, discovery_round_id=1,
                scorer_generation=old_epoch, proposal_score=0.75)
            module.publish_support(old_epoch + 1)
            publication_rows.append({
                "before_epoch": old_epoch,
                "after_epoch": int(module.structure_epoch),
                "revoked_logical_id": revoked_id,
                "promoted_logical_id": promoted_id,
                "promoted_endpoints": [23, 0],
                "stable_survivor_ids_sha256": hashlib.sha256(json.dumps(
                    module.support.edge_ids).encode()).hexdigest(),
                "read_tickets_after_publish": len(module._read_tickets),
            })
        if modules[0].support != modules[1].support:
            raise RuntimeError("direct and field owners disagree after structural publication")
        checkpoint_path = args.output.parent / "raw" / (
            f"e5_checkpoint_seed{seed}_{policy}_{args.value_policy}.pt")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(checkpoint_path, modules[1], optimizers[1])
        restored = prepare_sparse_slice(
            KaryoplastEventSlice, support, entities=24, mechanisms=4,
            rank_count=8, device=device, parameter_axis_words=axis_words,
            seed=seed, reader_capacity=16)
        restored_optimizer = torch.optim.Adam([restored.parameters_flat], lr=0.02)
        load_checkpoint(checkpoint_path, restored, restored_optimizer)
        if restored.support != modules[1].support:
            raise RuntimeError("checkpoint did not restore published logical support")
        if not torch.equal(restored.parameters_flat, modules[1].parameters_flat):
            raise RuntimeError("checkpoint did not restore exact canonical owner values")
        for parameter, restored_parameter in zip(
                optimizers[1].state.values(), restored_optimizer.state.values()):
            for key in ("step", "exp_avg", "exp_avg_sq"):
                left, right = parameter[key], restored_parameter[key]
                if isinstance(left, torch.Tensor):
                    if not torch.equal(left.cpu(), right.cpu()):
                        raise RuntimeError(f"checkpoint optimizer state differs at {key}")
                elif left != right:
                    raise RuntimeError(f"checkpoint optimizer scalar differs at {key}")
        continuation_fixture = make_next_event_fixture(
            seed=seed + 5000, cells=4, steps=6, entities=24)
        direct_index = native.NativeReverseIndex(modules[0]._support_index)
        restored_index = native.NativeReverseIndex(restored._support_index)
        direct_output, _ = event_forward(
            torch, modules[0], direct_index, continuation_fixture, plan, seed,
            policy, args.value_policy, field_plan=False)
        restored_output, restore_trace = event_forward(
            torch, restored, restored_index, continuation_fixture, plan, seed,
            policy, args.value_policy, field_plan=True)
        direct_loss = modules[0].training_loss(
            direct_output, continuation_fixture["targets"].to(device))
        restored_loss = restored.training_loss(
            restored_output, continuation_fixture["targets"].to(device))
        direct_loss.backward()
        restored_loss.backward()
        if not torch.allclose(direct_output, restored_output, atol=2e-3, rtol=2e-3):
            raise AssertionError("post-publication checkpoint continuation output mismatch")
        if not torch.allclose(modules[0].parameters_flat.grad,
                              restored.parameters_flat.grad, atol=5e-3, rtol=5e-3):
            raise AssertionError("post-publication checkpoint continuation VJP mismatch")
        if not guarded_step(modules[0], optimizers[0]):
            raise FloatingPointError("direct continuation owner update rejected")
        if not guarded_step(restored, restored_optimizer):
            raise FloatingPointError("restored continuation owner update rejected")
        if not torch.allclose(modules[0].parameters_flat, restored.parameters_flat,
                              atol=5e-3, rtol=5e-3):
            raise AssertionError("post-publication checkpoint optimizer continuation mismatch")
        raw_arrays[f"continuation_direct_{policy}_{args.value_policy}_output"] = _copy_tensor(direct_output)
        raw_arrays[f"continuation_restored_{policy}_{args.value_policy}_output"] = _copy_tensor(restored_output)
        raw_arrays[f"continuation_direct_{policy}_{args.value_policy}_gradient"] = _copy_tensor(modules[0].parameters_flat.grad)
        raw_arrays[f"continuation_restored_{policy}_{args.value_policy}_gradient"] = _copy_tensor(restored.parameters_flat.grad)
        event_record["support_publication"] = publication_rows
        event_record["checkpoint"] = {
            "path": str(checkpoint_path), "sha256": sha256(checkpoint_path),
            "restored_epoch": int(restored.structure_epoch),
            "restored_support_id_sha256": hashlib.sha256(json.dumps(
                restored.support.edge_ids).encode()).hexdigest(),
            "optimizer_state_restored": True,
            "exact_owner_values_restored": True,
            "post_publication_continuation": {
                "fixture_seed": seed + 5000,
                "direct_loss": float(direct_loss.detach().cpu()),
                "restored_loss": float(restored_loss.detach().cpu()),
                "output_max_abs": float((direct_output - restored_output).abs().max().cpu()),
                "gradient_max_abs": float((modules[0].parameters_flat.grad -
                                            restored.parameters_flat.grad).abs().max().cpu()),
                "field_trace": restore_trace,
            },
        }
        records.append(event_record)
    ranked_fixture = make_masked_gene_fixture(seed=seed, cells=4, selected=8, entities=24)
    ranked_rows = [[int(gene) for gene in row.tolist() if gene >= 0]
                   for row in ranked_fixture["genes"]]
    ranked_record = run_ranked_case(
        torch, native, base, RankedGeneSlice, support, ranked_fixture, ranked_rows,
        seed, args.value_policy, device, axis_words, raw_arrays)
    records.append(ranked_record)
    return {
        "schema": "cellerator.adaptive_fields.e5_gpu.v1",
        "seed": seed,
        "relation_policy": args.relation_policy,
        "value_policy": args.value_policy,
        "decision": "supported" if all(
            row["owner_gradient_max_abs"] <= 5e-3 and
            row["owner_after_update_max_abs"] <= 5e-3 for row in records)
            else "rejected",
        "experiment": "E5", "gpu_execution_performed": True,
        "package_root": str(package_root.resolve()),
        "package_source_sha256": sha256(package_root / "cellerator/torch/sparse_support.py"),
        "native_extension_sha256": sha256(package_root / "cellerator/torch/_torch.cpython-313-x86_64-linux-gnu.so"),
        "scheduler": {"job_id": os.environ.get("DEVICE_SCHEDULER_JOB_ID"),
                      "attempt_id": os.environ.get("DEVICE_SCHEDULER_ATTEMPT_ID")},
        "records": records,
        "parameter_blocks": {name: {"offset": block.offset, "numel": block.numel,
                                     "shape": list(block.shape)}
                             for name, block in modules[0]._blocks.items()},
        "raw_arrays": raw_arrays,
        "uncovered": ["full Torch recurrent oracle", "support publication/checkpoint in field run", "explicit repeated gate-index trace"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, default=SEEDS)
    parser.add_argument("--relation-policies", choices=("current_event", "eligible_history"),
                        nargs="+", default=("current_event", "eligible_history"))
    parser.add_argument("--value-policies", choices=("all", "workset_union"),
                        nargs="+", default=("all", "workset_union"))
    args = parser.parse_args()
    rows = []
    from copy import copy
    for seed in args.seeds:
        for relation_policy in args.relation_policies:
            for value_policy in args.value_policies:
                case_args = copy(args)
                case_args.relation_policy = relation_policy
                case_args.value_policy = value_policy
                row = run_seed(case_args, seed)
                raw_values = row.pop("raw_arrays")
                raw_path = args.output.parent / "raw" / (
                    f"e5_seed{seed}_{relation_policy}_{value_policy}.npz")
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(raw_path, **raw_values)
                row["raw_arrays_artifact"] = {
                    "path": str(raw_path), "sha256": sha256(raw_path),
                    "array_keys": sorted(raw_values),
                }
                rows.append(row)
    result = {
        "schema": "cellerator.adaptive_fields.e5_gpu_campaign.v1",
        "decision": "supported" if all(row["decision"] == "supported" for row in rows)
            else "rejected",
        "experiment": "E5",
        "gpu_execution_performed": True,
        "small_seed_values": list(args.seeds),
        "relation_policies": list(args.relation_policies),
        "value_policies": list(args.value_policies),
        "records_by_seed": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": result["decision"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
