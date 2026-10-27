#!/usr/bin/env python3
"""Paired SIMT versus CE-GEO N64 sparse relation training benchmark.

The caller owns Device Scheduler admission. This driver refuses to run unless
the job explicitly sets CELLERATOR_SPARSE_SUPPORT_BENCH_ADMITTED=1. It uses the
frozen Torch package supplied by CELLERATOR_SPARSE_PACKAGE_ROOT, and writes
raw source-bound timing and preparation evidence to --output-dir.

Measured step scope:
  inference: gate scoring + shared edge-value calculation + sparse relation
  training: zero_grad + gate scoring/STE + edge values + relation + loss +
            backward (including derivative-only gate candidates) + guarded Adam

Support and per-cell worksets are frozen before timing. Packed training uses
the production CE-GEO N64 forward dispatch; its backward remains the same native
SIMT VJP as the control. This measures the current mixed packed-forward/SIMT-
backward implementation, not a packed backward kernel.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np


WIDTH = 64
ENTITIES = 32
MECHANISMS = 4
RANK_COUNT = 8
SEED = 20261027
IDENTITY_VALIDATED_FILES = (
    "package/cellerator/torch/sparse_support.py",
    "tests/test_cuda_training.py",
    "package/cellerator/torch/mechanism.py",
)
TORCH_EXTENSION_PATH = "package/cellerator/torch/_torch.cpython-313-x86_64-linux-gnu.so"
REPO = Path(__file__).resolve().parents[2]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def tensor_digest(tensors: list[Any]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        host = tensor.detach().contiguous().cpu()
        digest.update(str(host.dtype).encode())
        digest.update(json_bytes(list(host.shape)))
        digest.update(host.numpy().tobytes())
    return digest.hexdigest()


def make_support(SupportSpec):
    # Four locally dense 4x4 modules inside a 16x16 source/destination tile.
    # The graph is shared across cells; only current active source sets differ.
    edge_ids, sources, destinations = [], [], []
    for source in range(16):
        module = source // 4
        for local_target in range(4):
            edge_ids.append(source * 4 + local_target + 1)
            sources.append(source)
            destinations.append(16 + module * 4 + local_target)
    return SupportSpec(tuple(edge_ids), tuple(sources), tuple(destinations), 1)


def load_raw_bundle(bundle: Path, names: tuple[str, ...]):
    manifest_path = bundle / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != "cellerator.scaled_sparse_fixture_bundle.v1":
        raise RuntimeError(f"unrecognized scaled fixture bundle: {manifest_path}")
    arrays = {}
    for name in names:
        descriptor = manifest["arrays"][name]
        raw = (bundle / descriptor["file"]).read_bytes()
        digest = sha256_bytes(raw)
        if digest != descriptor["sha256"]:
            raise RuntimeError(f"fixture array hash mismatch: {name}")
        value = np.frombuffer(raw, dtype=np.dtype(descriptor["dtype"]))
        value = value.reshape(tuple(descriptor["shape"]))
        arrays[name] = value.copy()
    return arrays, manifest, sha256_bytes(manifest_bytes)


def make_candidate_rows_from_postings(source_indptr, source_ordinals,
                                      active_rows, edge_count: int):
    """Union source postings into exact edge-ordinal-sorted candidate rows."""
    rows = []
    mark = np.zeros(edge_count, dtype=np.bool_)
    for active in active_rows:
        for source in np.unique(np.asarray(active, dtype=np.int64)):
            begin = int(source_indptr[source])
            end = int(source_indptr[source + 1])
            mark[source_ordinals[begin:end]] = True
        rows.append(np.flatnonzero(mark).astype(np.int64, copy=False))
        for ordinals in rows[-1]:
            mark[ordinals] = False
    return rows


def make_candidate_rows(support, active_rows):
    """Small-control candidate construction with canonical stable ordering."""
    source_indptr = np.zeros(max(support.sources, default=-1) + 2, dtype=np.int64)
    for source in support.sources:
        source_indptr[int(source) + 1] += 1
    np.cumsum(source_indptr, out=source_indptr)
    source_ordinals = np.empty(len(support.sources), dtype=np.int64)
    cursor = source_indptr[:-1].copy()
    for ordinal, source in enumerate(support.sources):
        pos = cursor[int(source)]
        source_ordinals[pos] = ordinal
        cursor[int(source)] += 1
    return make_candidate_rows_from_postings(
        source_indptr, source_ordinals, active_rows, len(support.edge_ids))


def flatten_candidate_rows(rows):
    lengths = np.asarray([row.size for row in rows], dtype=np.int64)
    offsets = np.empty(lengths.size + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    items = np.concatenate(rows) if offsets[-1] else np.empty(0, dtype=np.int64)
    cell_ids = np.repeat(np.arange(lengths.size, dtype=np.int64), lengths)
    return offsets, items, cell_ids


def make_worksets(torch, PerCellWorkset, support, edge_ids, candidate_rows,
                  device, seed):
    derivative_offsets, ordinals_np, cell_ids_np = flatten_candidate_rows(candidate_rows)
    derivative_ids_np = edge_ids[ordinals_np]
    gate_indices_np = np.arange(ordinals_np.size, dtype=np.int64)
    admitted_np = ((derivative_ids_np * 13 + cell_ids_np * 7 + seed) % 2 == 0)
    primal_counts = np.bincount(cell_ids_np[admitted_np], minlength=len(candidate_rows))
    primal_offsets = np.empty(len(candidate_rows) + 1, dtype=np.int64)
    primal_offsets[0] = 0
    np.cumsum(primal_counts, out=primal_offsets[1:])
    primal_ordinals_np = ordinals_np[admitted_np]
    primal_ids_np = derivative_ids_np[admitted_np]
    primal_gate_indices_np = gate_indices_np[admitted_np]

    derivative = PerCellWorkset(
        torch.as_tensor(derivative_offsets, device=device, dtype=torch.int64),
        torch.as_tensor(ordinals_np, device=device, dtype=torch.int64),
        torch.as_tensor(derivative_ids_np, device=device, dtype=torch.int64),
        torch.as_tensor(gate_indices_np, device=device, dtype=torch.int64),
        support.structure_epoch, 1, ordinals_np.size)
    gate_mask = torch.as_tensor(admitted_np, device=device, dtype=torch.bool)
    # This provisional primal is rebuilt from the sigmoid threshold below.
    primal = PerCellWorkset(
        torch.as_tensor(primal_offsets, device=device, dtype=torch.int64),
        torch.as_tensor(primal_ordinals_np, device=device, dtype=torch.int64),
        torch.as_tensor(primal_ids_np, device=device, dtype=torch.int64),
        torch.as_tensor(primal_gate_indices_np, device=device, dtype=torch.int64),
        support.structure_epoch, 1, ordinals_np.size)
    return primal, derivative, gate_mask


def workset_with_mask(torch, PerCellWorkset, support, edge_ids,
                      candidate_rows, derivative, admitted_mask, device):
    _, ordinals_np, cell_ids_np = flatten_candidate_rows(candidate_rows)
    mask_np = admitted_mask.detach().to("cpu").numpy().astype(np.bool_, copy=False)
    if ordinals_np.size != mask_np.size:
        raise RuntimeError("frozen gate mask does not match the derivative workset")
    selected = np.flatnonzero(mask_np)
    primal_ordinals = ordinals_np[selected]
    primal_ids = edge_ids[primal_ordinals]
    primal_cell_ids = cell_ids_np[selected]
    counts = np.bincount(primal_cell_ids, minlength=len(candidate_rows))
    offsets = np.empty(len(candidate_rows) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    primal = PerCellWorkset(
        torch.as_tensor(offsets, device=device, dtype=torch.int64),
        torch.as_tensor(primal_ordinals, device=device, dtype=torch.int64),
        torch.as_tensor(primal_ids, device=device, dtype=torch.int64),
        torch.as_tensor(selected, device=device, dtype=torch.int64),
        support.structure_epoch, derivative.state_generation, derivative.capacity)
    return primal


def make_case(torch, fixture_module, support, device, case_name: str, seed: int,
              *, fixture_arrays=None, entities=ENTITIES, rank_count=RANK_COUNT):
    from cellerator.torch.sparse_support import make_masked_gene_fixture, make_next_event_fixture

    with torch.no_grad(), fixture_module._pinned_forward():
        entity = fixture_module.block("entity_embedding").detach()
        rank = fixture_module.block("rank_embedding").detach()

    if case_name in {"large_ranked", "large_events"}:
        if fixture_arrays is None:
            raise ValueError("large cases require a hash-verified scaled fixture bundle")
        edge_ids = fixture_arrays["edge_ids"]
        if case_name == "large_ranked":
            genes_np = fixture_arrays["ranked_entity_id"].astype(np.int64, copy=False)
            ranks_np = fixture_arrays["ranked_position"].astype(np.int64, copy=False)
            cells, selected = genes_np.shape
            presence_offsets = fixture_arrays["ranked_presence_offsets"]
            presence_genes = fixture_arrays["ranked_presence_entity_ids"]
            presence_values = fixture_arrays["ranked_presence_values"]
            presence_views = fixture_arrays["ranked_presence_view_id"]
            if (presence_views.shape != (cells,) or np.any(presence_views != 1)
                    or presence_offsets.shape != (cells + 1,)
                    or presence_offsets[0] != 0
                    or presence_offsets[-1] != presence_genes.size
                    or np.any(presence_values != 1)):
                raise RuntimeError("ranked-presence view metadata or CSR is malformed")
            for cell in range(cells):
                begin, end = map(int, presence_offsets[cell:cell + 2])
                if not np.array_equal(np.sort(presence_genes[begin:end]),
                                      np.sort(genes_np[cell])):
                    raise RuntimeError("ranked token IDs disagree with the presence support view")
            genes = torch.as_tensor(genes_np, device=device, dtype=torch.int64)
            ranks = torch.as_tensor(ranks_np, device=device, dtype=torch.int64)
            features = entity.index_select(0, genes.reshape(-1)).view(
                cells, selected, WIDTH)
            features = features + rank.index_select(0, ranks.reshape(-1)).view_as(features)
            context = torch.zeros((cells, entities, WIDTH), device=device,
                                  dtype=torch.float32)
            context.scatter_add_(1, genes.unsqueeze(-1).expand(-1, -1, WIDTH), features)
            active_rows = [row for row in genes_np]
            case_shape = {"cells": cells, "entities": entities, "selected_genes_per_cell": selected,
                          "relation_width": WIDTH}
            fixture_identity = {"fixture_profile": "large-ranked",
                                "token_ids_sha256": sha256_bytes(genes_np.tobytes()),
                                "rank_positions_sha256": sha256_bytes(ranks_np.tobytes()),
                                "rank_is_feature_position": True}
        else:
            event_offsets = fixture_arrays["event_offsets"]
            event_ids_np = fixture_arrays["event_entity_ids"].astype(np.int64, copy=False)
            event_times = fixture_arrays["event_timestamps"]
            event_ordinals = fixture_arrays["event_ordinals"]
            count_bounds = fixture_arrays["count_window_bounds"]
            time_bounds = fixture_arrays["time_window_bounds"]
            event_window_offsets = fixture_arrays["event_window_observed_indptr"]
            event_window_genes = fixture_arrays["event_window_observed_gene_ids"]
            event_window_cells = fixture_arrays["window_cell_id"]
            event_window_ids = fixture_arrays["window_id"]
            event_window_views = fixture_arrays["event_window_observed_view_id"]
            event_window_starts = fixture_arrays["window_start_ordinal"]
            event_window_stops = fixture_arrays["window_stop_ordinal_exclusive"]
            cells = int(event_offsets.size - 1)
            filtered_rows = []
            row_time_bounds = []
            row_count_bounds = []
            for cell in range(cells):
                begin, end = map(int, event_offsets[cell:cell + 2])
                count_lo, count_hi = map(int, count_bounds[cell])
                time_lo, time_hi = map(int, time_bounds[cell])
                keep = ((event_ordinals[begin:end] >= count_lo)
                        & (event_ordinals[begin:end] < count_hi)
                        & (event_times[begin:end] >= time_lo)
                        & (event_times[begin:end] < time_hi))
                filtered = event_ids_np[begin:end][keep]
                window_index = cell * 2 + 1  # the fixture's second 256-event window
                wbegin, wend = map(int, event_window_offsets[window_index:window_index + 2])
                if (event_window_cells[window_index] != cell
                        or event_window_ids[window_index] != 1
                        or event_window_views[window_index] != 2
                        or event_window_starts[window_index] != count_lo
                        or event_window_stops[window_index] != count_hi
                        or not np.array_equal(filtered, event_window_genes[wbegin:wend])):
                    raise RuntimeError("event count/time window disagrees with its labeled view")
                filtered_rows.append(filtered)
                row_time_bounds.append((time_lo, time_hi))
                row_count_bounds.append((count_lo, count_hi))
            max_events = max((row.size for row in filtered_rows), default=0)
            event_tokens = torch.full((cells, max_events), -1, device=device,
                                      dtype=torch.int64)
            context = torch.zeros((cells, entities, WIDTH), device=device,
                                  dtype=torch.float32)
            active_rows = []
            filtered_counts = []
            for cell, row in enumerate(filtered_rows):
                filtered_counts.append(int(row.size))
                if row.size:
                    tokens = torch.as_tensor(row, device=device, dtype=torch.int64)
                    event_tokens[cell, :row.size] = tokens
                    event_features = entity.index_select(0, tokens)
                    context[cell].index_add_(0, tokens, event_features)
                active_rows.append(np.unique(row))
            case_shape = {"cells": cells, "entities": entities,
                          "events_per_cell_in_full_declared_window": filtered_counts,
                          "event_occurrences_after_both_windows": filtered_counts,
                          "relation_width": WIDTH}
            fixture_identity = {
                "fixture_profile": "large-events",
                "event_occurrences_sha256": sha256_bytes(event_ids_np.tobytes()),
                "event_timestamps_sha256": sha256_bytes(event_times.tobytes()),
                "count_bounds_sha256": sha256_bytes(count_bounds.tobytes()),
                "time_bounds_sha256": sha256_bytes(time_bounds.tobytes()),
                "window_filter": "fixture count and timestamp half-open intersection before candidate construction",
                "filtered_event_counts": filtered_counts,
                "count_bounds": row_count_bounds,
                "time_bounds": row_time_bounds,
                "repeated_occurrences_preserved_in_context": True,
            }
        candidate_rows = make_candidate_rows_from_postings(
            fixture_arrays["source_edge_indptr"],
            fixture_arrays["source_edge_ordinals"], active_rows, len(edge_ids))
        candidate_offsets, candidate_ordinals, _candidate_cell_ids = flatten_candidate_rows(candidate_rows)
        candidate_ids = edge_ids[candidate_ordinals]
        rows_for_worksets = candidate_rows
        selected_per_cell = [int(row.size) for row in active_rows]
        estimated_candidate_workspace_bytes = int(candidate_ordinals.size * WIDTH * 4 * 8)
        estimated_dense_workspace_bytes = int(cells * entities * WIDTH * 4 * 6)
        estimated_global_edge_refresh_bytes = int(len(edge_ids) * WIDTH * 4 * 3)
        case_shape.update({
            "active_entities_per_cell": selected_per_cell,
            "candidate_items": int(candidate_ordinals.size),
            "max_candidate_items_per_cell": int(np.diff(candidate_offsets).max(initial=0)),
            "candidate_feature_workspace_estimate_bytes": estimated_candidate_workspace_bytes,
            "dense_context_and_output_estimate_bytes": estimated_dense_workspace_bytes,
            "global_edge_value_refresh_intermediate_estimate_bytes":
                estimated_global_edge_refresh_bytes,
            "estimated_numerical_workspace_bytes_before_ce_geo": (
                estimated_candidate_workspace_bytes + estimated_dense_workspace_bytes
                + estimated_global_edge_refresh_bytes),
        })
        fixture_identity["candidate_ordinals_sha256"] = sha256_bytes(candidate_ordinals.tobytes())
        fixture_identity["active_query_counts"] = selected_per_cell
        fixture_identity["candidate_ids_sha256"] = sha256_bytes(candidate_ids.tobytes())
        context_inputs = context
        if case_name == "large_ranked":
            context_inputs = context
        # The same cell context feeds the FP16 sparse relation. For event
        # streams it is the count-preserving aggregate of occurrences that
        # survived both temporal bounds; the benchmark does not model a full
        # per-position recurrent loop.
        inputs = context_inputs.to(dtype=torch.float16).contiguous()
        target_gen = torch.Generator(device="cpu").manual_seed(seed + 111)
        target_cpu = torch.randn((cells, entities, WIDTH), generator=target_gen,
                                 dtype=torch.float32) * 0.1
        target = target_cpu.to(device=device)
        metadata = {
            "case": case_name,
            "shape": case_shape,
            "fixture_identity": fixture_identity,
        }
        return context, inputs, target, rows_for_worksets, metadata

    if case_name == "event_local":
        fixture = make_next_event_fixture(seed=seed, cells=4, steps=6, entities=ENTITIES)
        tokens = fixture["events"][:, -1].to(device=device, dtype=torch.int64)
        cells = int(tokens.numel())
        context = torch.zeros((cells, ENTITIES, WIDTH), device=device, dtype=torch.float32)
        event_features = entity.index_select(0, tokens)
        context.scatter_add_(1, tokens[:, None, None].expand(-1, 1, WIDTH),
                             event_features[:, None, :])
        active_rows = [[int(token)] for token in tokens.detach().cpu().tolist()]
        case_shape = {"cells": cells, "event_occurrences_per_cell": 1,
                      "selected_entities_per_cell": 1, "relation_width": WIDTH}
        fixture_identity = {
            "fixture": "make_next_event_fixture", "seed": seed,
            "cells": 4, "steps": 6, "entities": ENTITIES,
            "selected_occurrences": "last observed event per cell",
        }
    elif case_name == "ranked_batch":
        fixture = make_masked_gene_fixture(seed=seed, cells=32, selected=8,
                                           entities=ENTITIES)
        genes = fixture["genes"].to(device=device, dtype=torch.int64)
        ranks = fixture["ranks"].to(device=device, dtype=torch.int64)
        valid = genes >= 0
        safe_genes = genes.clamp_min(0)
        features = entity.index_select(0, safe_genes.reshape(-1)).view(
            genes.shape[0], genes.shape[1], WIDTH)
        features = (features + rank.index_select(0, ranks.reshape(-1)).view_as(features))
        features = features * valid.unsqueeze(-1)
        context = torch.zeros((genes.shape[0], ENTITIES, WIDTH), device=device,
                              dtype=torch.float32)
        context.scatter_add_(1, safe_genes.unsqueeze(-1).expand(-1, -1, WIDTH), features)
        active_rows = [sorted({int(gene) for gene in row if 0 <= int(gene) < 16})
                       for row in fixture["genes"].tolist()]
        cells = int(genes.shape[0])
        case_shape = {"cells": cells, "selected_genes_per_cell": 8,
                      "relation_width": WIDTH}
        fixture_identity = {
            "fixture": "make_masked_gene_fixture", "seed": seed,
            "cells": 32, "selected": 8, "entities": ENTITIES,
            "observed_fields": ["genes", "ranks"],
        }
    else:
        raise ValueError(f"unknown benchmark case: {case_name}")

    inputs = context.to(dtype=torch.float16).contiguous()
    edge_ids_np = np.asarray(support.edge_ids, dtype=np.int64)
    rows = make_candidate_rows(support, active_rows)
    target_gen = torch.Generator(device="cpu").manual_seed(seed + 111)
    target_cpu = torch.randn((cells, entities, WIDTH), generator=target_gen,
                             dtype=torch.float32) * 0.1
    target = target_cpu.to(device=device)
    metadata = {
        "case": case_name,
        "shape": case_shape,
        "fixture_identity": fixture_identity,
        "active_source_counts": [len(row) for row in active_rows],
        "candidate_items": int(sum(row.size for row in rows)),
    }
    return context, inputs, target, rows, metadata


def gate_values_for_step(torch, GateValues, module, context, derivative, hard_mask,
                         *, threshold: float, capture: bool = False):
    scored = module.score_candidates(
        context, derivative, temperature=1.0, threshold=threshold, prune=False)
    probabilities = scored.probabilities
    if capture and probabilities.requires_grad:
        probabilities.retain_grad()
    mask = hard_mask.to(dtype=probabilities.dtype)
    train_values = probabilities + (probabilities * mask - probabilities).detach()
    gates = GateValues(
        probabilities, hard_mask, train_values, scored.logical_edge_ids,
        scored.edge_ordinals, scored.structure_epoch, scored.state_generation)
    return gates, probabilities


def make_optimizer(torch, module):
    return torch.optim.Adam([module.parameters_flat], lr=1.0e-3,
                            foreach=False, fused=False)


def inference_step(torch, GateValues, module, context, inputs, primal, derivative,
                   hard_mask, threshold):
    module.eval()
    with torch.no_grad(), module._pinned_forward():
        gates, _ = gate_values_for_step(torch, GateValues, module, context, derivative,
                                        hard_mask, threshold=threshold)
        edge_values = module.logical_edge_values()
        return module.relation(inputs, edge_values, gates, primal, derivative)


def training_step(torch, GateValues, guarded_step, module, optimizer, context,
                  inputs, target, primal, derivative, hard_mask, threshold,
                  *, capture: bool = False):
    module.train()
    optimizer.zero_grad(set_to_none=True)
    with module._pinned_forward():
        gates, probabilities = gate_values_for_step(
            torch, GateValues, module, context, derivative, hard_mask,
            threshold=threshold, capture=capture)
        edge_values = module.logical_edge_values()
        output = module.relation(inputs, edge_values, gates, primal, derivative)
        loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    if module.parameters_flat.grad is None:
        raise RuntimeError("native sparse owner received no gradient")
    result = None
    if capture:
        result = {
            "loss": float(loss.detach().cpu()),
            "output": output.detach().clone(),
            "parameter_grad": module.parameters_flat.grad.detach().clone(),
            "probability_grad": (None if probabilities.grad is None
                                 else probabilities.grad.detach().clone()),
        }
    if not guarded_step(module, optimizer):
        raise FloatingPointError("guarded Adam rejected a nonfinite sparse gradient")
    return result


def global_edge_value_refresh_step(module):
    """Isolate the all-edge refresh plus a simple full-width VJP.

    The complete training timing below remains authoritative. This diagnostic
    exposes the cost that the sparse active workset cannot currently avoid;
    its all-ones edge-value cotangent is deliberately simpler than the
    relation kernel's actual cotangent and is not a replacement for full-step
    training measurements.
    """
    module.parameters_flat.grad = None
    with module._pinned_forward():
        edge_values = module.logical_edge_values()
        edge_values.sum().backward()
    module.parameters_flat.grad = None
    if len(module._read_tickets):
        raise RuntimeError("global edge-value refresh left a parameter read lease outstanding")


def event_pair_ms(start_event, end_event, torch, fn, iterations: int):
    torch.cuda.synchronize()
    start_event.record()
    wall_start = time.perf_counter()
    for _ in range(iterations):
        fn()
    end_event.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0 / iterations
    device_ms = start_event.elapsed_time(end_event) / iterations
    return {"cuda_event_ms_per_step": device_ms, "synchronized_wall_ms_per_step": wall_ms}


def stats(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "count": len(samples),
        "median": statistics.median(samples),
        "mean": statistics.mean(samples),
        "stdev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "p10": ordered[max(0, math.floor(0.10 * (len(ordered) - 1)))],
        "p90": ordered[min(len(ordered) - 1, math.ceil(0.90 * (len(ordered) - 1)))],
    }


def paired_ratio_ci(ratios: list[float], seed: int, draws: int = 10000):
    rng = random.Random(seed)
    values = []
    for _ in range(draws):
        values.append(statistics.median(rng.choices(ratios, k=len(ratios))))
    values.sort()
    return [values[int(0.025 * (len(values) - 1))],
            values[int(0.975 * (len(values) - 1))]]


def prepare_module(torch, prepare_sparse_slice, module_type, support, *, device,
                   seed, packed, compiled, cells, entities=ENTITIES,
                   rank_count=RANK_COUNT, primal_capacity=1):
    kwargs = {}
    if packed:
        kwargs = {
            "ce_geo_image": compiled["image"],
            "source_group_members": compiled["source_group_members"],
            "destination_group_members": compiled["destination_group_members"],
            "geometry_edge_ids": compiled["geometry_edge_ids_in_support_order"],
            "projection_id": int(compiled["projection_id"]),
            "max_cells": cells,
            "max_primal_items": max(1, int(primal_capacity)),
        }
    torch.cuda.synchronize()
    start_alloc = torch.cuda.memory_allocated(device)
    wall_start = time.perf_counter()
    module = prepare_sparse_slice(
        module_type, support, entities=entities, mechanisms=MECHANISMS,
        rank_count=rank_count, device=device,
        parameter_axis_words=(31, 32, 41, 42, 51, 52, 61, 62),
        seed=seed, reader_capacity=16, **kwargs)
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    alloc_delta = torch.cuda.memory_allocated(device) - start_alloc
    return module, {"wall_ms": wall_ms, "allocated_delta_bytes": alloc_delta}


def release_module(torch, module):
    del module
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def measure_module_setup_pairs(torch, package, module_type, support, compiled,
                               *, device, seed, cells, entities, rank_count,
                               primal_capacity, pairs):
    samples = {"simt": [], "packed": []}
    for pair_index in range(pairs):
        order = ("simt", "packed") if pair_index % 2 == 0 else ("packed", "simt")
        for label in order:
            module, sample = prepare_module(
                torch, package.prepare_sparse_slice, module_type, support,
                device=device, seed=seed, packed=(label == "packed"),
                compiled=compiled, cells=cells, entities=entities,
                rank_count=rank_count, primal_capacity=primal_capacity)
            sample.update({"pair": pair_index, "candidate": label,
                           "order": list(order)})
            samples[label].append(sample)
            release_module(torch, module)
    summaries = {}
    for label, rows in samples.items():
        summaries[label] = {
            "wall_ms": stats([float(row["wall_ms"]) for row in rows]),
            "allocated_delta_bytes": stats(
                [float(row["allocated_delta_bytes"]) for row in rows]),
            "raw_samples": rows,
        }
    return summaries


def benchmark_case(torch, package, support, case_name, seed, args, compiled):
    from cellerator.torch.mechanism import guarded_step
    from cellerator.torch.sparse_support import (
        GateValues, KaryoplastEventSlice, PerCellWorkset, RankedGeneSlice,
    )

    module_type = (KaryoplastEventSlice if case_name in {"event_local", "large_events"}
                   else RankedGeneSlice)
    fixture_arrays = args.fixture_arrays if case_name in {"large_ranked", "large_events"} else None
    entities = (int(args.fixture_manifest["metadata"]["entities"])
                if fixture_arrays is not None else ENTITIES)
    rank_count = (int(fixture_arrays["ranked_entity_id"].shape[1])
                  if fixture_arrays is not None and case_name == "large_ranked"
                  else RANK_COUNT)
    cells = (int(args.fixture_manifest["metadata"]["cells"])
             if fixture_arrays is not None
             else (4 if case_name == "event_local" else 32))
    fixture_module, fixture_module_setup = prepare_module(
        torch, package.prepare_sparse_slice, module_type, support,
        device=torch.cuda.current_device(), seed=seed, packed=False,
        compiled=compiled, cells=cells, entities=entities,
        rank_count=rank_count, primal_capacity=1)
    case_prep_start = time.perf_counter()
    context, inputs, target, candidate_rows, shape_meta = make_case(
        torch, fixture_module, support, torch.cuda.current_device(), case_name, seed,
        fixture_arrays=fixture_arrays, entities=entities, rank_count=rank_count)
    torch.cuda.synchronize()
    shape_meta["shared_workset_and_input_prep_ms"] = (
        time.perf_counter() - case_prep_start) * 1000.0
    edge_ids_np = np.asarray(support.edge_ids, dtype=np.int64)
    primal, derivative, _provisional_mask = make_worksets(
        torch, PerCellWorkset, support, edge_ids_np, candidate_rows,
        torch.cuda.current_device(), seed)
    shape_meta["candidate_items"] = int(derivative.edge_ordinals.numel())
    shape_meta["candidate_workset_bytes"] = int(sum(
        tensor.numel() * tensor.element_size() for tensor in (
            primal.cell_offsets, primal.edge_ordinals, primal.logical_edge_ids,
            primal.gate_value_indices, derivative.cell_offsets,
            derivative.edge_ordinals, derivative.logical_edge_ids,
            derivative.gate_value_indices)))
    shape_meta["estimated_numerical_workspace_bytes_before_ce_geo"] = int(
        shape_meta.get("estimated_numerical_workspace_bytes_before_ce_geo", 0)
        + shape_meta["candidate_workset_bytes"])
    if fixture_arrays is not None:
        free_bytes, total_bytes = torch.cuda.mem_get_info(torch.cuda.current_device())
        tile_count = int(compiled["tile_count"])
        candidate_count = int(derivative.edge_ordinals.numel())
        packed_active_tile_workspace = min(candidate_count, cells * tile_count) * 256 * 2
        estimated_peak = (shape_meta["estimated_numerical_workspace_bytes_before_ce_geo"]
                          + packed_active_tile_workspace)
        shape_meta["gpu_memory_preflight"] = {
            "free_bytes_before_candidate_scoring": int(free_bytes),
            "total_device_bytes": int(total_bytes),
            "estimated_packed_active_tile_workspace_bytes": int(packed_active_tile_workspace),
            "estimated_peak_bytes_before_admission": int(estimated_peak),
        "estimate_formula": "8 x candidate_count x width x fp32_bytes + 6 x B x N x width x fp32_bytes + 3 x E x width x fp32_bytes for autograd-saved full logical_edge_values gathers/product + min(candidate_count,B*tile_count) x 256 x fp16_bytes + workset tensors",
        }
        if estimated_peak > free_bytes * 0.85:
            raise RuntimeError(
                "scaled workload exceeds the 85% free-device-memory preflight; "
                f"estimated={estimated_peak} free={free_bytes}, no candidate work was truncated")

    # Choose a genuine sigmoid threshold once from the frozen initial parameter
    # state. The resulting hard decisions and primal workset stay fixed across
    # both paths; probabilities and their STE gradients are recomputed per step.
    freeze_start = time.perf_counter()
    with torch.no_grad(), fixture_module._pinned_forward():
        initial_gates = fixture_module.score_candidates(
            context, derivative, temperature=1.0, threshold=0.5, prune=False)
        initial_p = initial_gates.probabilities.detach()
        threshold = float(torch.quantile(initial_p.float(), 0.5).cpu())
        hard_mask = initial_p >= threshold
        primal = workset_with_mask(torch, PerCellWorkset, support, edge_ids_np, candidate_rows,
                                   derivative, hard_mask, torch.cuda.current_device())
        initial_active = int(hard_mask.sum().item())
    torch.cuda.synchronize()
    shape_meta["hard_gate_freeze_and_primal_workset_ms"] = (
        time.perf_counter() - freeze_start) * 1000.0
    shape_meta["frozen_tensor_bytes"] = sum(
        tensor.numel() * tensor.element_size() for tensor in (
            context, inputs, target, primal.cell_offsets, primal.edge_ordinals,
            primal.logical_edge_ids, primal.gate_value_indices,
            derivative.cell_offsets, derivative.edge_ordinals,
            derivative.logical_edge_ids, derivative.gate_value_indices, hard_mask))
    shape_meta.update({
        "threshold": threshold,
        "hard_admitted_items": initial_active,
        "hard_derivative_only_items": int(hard_mask.numel() - initial_active),
        "primal_items": int(primal.edge_ordinals.numel()),
        "derivative_items": int(derivative.edge_ordinals.numel()),
        "derivative_only_work_fraction": (
            0.0 if derivative.edge_ordinals.numel() == 0 else
            float(derivative.edge_ordinals.numel() - primal.edge_ordinals.numel()) /
            float(derivative.edge_ordinals.numel())),
        "workset_digest": tensor_digest([
            primal.cell_offsets, primal.edge_ordinals, primal.logical_edge_ids,
            primal.gate_value_indices, derivative.cell_offsets,
            derivative.edge_ordinals, derivative.logical_edge_ids,
            derivative.gate_value_indices, hard_mask]),
        "frozen_input_digest": tensor_digest([context, inputs, target]),
    })
    if initial_active == 0 or initial_active == hard_mask.numel():
        raise RuntimeError("median sigmoid threshold did not produce both primal and derivative-only candidates")

    # Keep the fixture owner setup separate, then prime both constructors and
    # measure backend module setup with balanced AB/BA order after common
    # CUDA/Torch initialization. CE-GEO host geometry compilation is reported
    # separately as packed-only setup.
    release_module(torch, fixture_module)
    priming_orders = (("simt", "packed"), ("packed", "simt"))
    for order in priming_orders:
        for label in order:
            module, _ = prepare_module(
                torch, package.prepare_sparse_slice, module_type, support,
                device=torch.cuda.current_device(), seed=seed, packed=(label == "packed"),
                compiled=compiled, cells=cells, entities=entities,
                rank_count=rank_count, primal_capacity=derivative.capacity)
            release_module(torch, module)
    setup_samples = measure_module_setup_pairs(
        torch, package, module_type, support, compiled,
        device=torch.cuda.current_device(), seed=seed, cells=cells,
        entities=entities, rank_count=rank_count,
        primal_capacity=derivative.capacity, pairs=args.setup_pairs)
    simt_setup = setup_samples["simt"]
    packed_setup = setup_samples["packed"]

    baseline, _ = prepare_module(
        torch, package.prepare_sparse_slice, module_type, support,
        device=torch.cuda.current_device(), seed=seed, packed=False,
        compiled=compiled, cells=cells, entities=entities,
        rank_count=rank_count, primal_capacity=derivative.capacity)
    packed, _ = prepare_module(
        torch, package.prepare_sparse_slice, module_type, support,
        device=torch.cuda.current_device(), seed=seed, packed=True,
        compiled=compiled, cells=cells, entities=entities,
        rank_count=rank_count, primal_capacity=derivative.capacity)
    if not torch.equal(baseline.parameters_flat.detach(), packed.parameters_flat.detach()):
        raise RuntimeError("SIMT and packed owners do not start from identical parameters")

    simt_optimizer = make_optimizer(torch, baseline)
    packed_optimizer = make_optimizer(torch, packed)
    funcs = {
        "simt": {
            "infer": lambda: inference_step(
                torch, GateValues, baseline, context, inputs, primal, derivative,
                hard_mask, threshold),
            "train": lambda: training_step(
                torch, GateValues, guarded_step, baseline, simt_optimizer,
                context, inputs, target, primal, derivative, hard_mask, threshold),
            "refresh": lambda: global_edge_value_refresh_step(baseline),
        },
        "packed": {
            "infer": lambda: inference_step(
                torch, GateValues, packed, context, inputs, primal, derivative,
                hard_mask, threshold),
            "train": lambda: training_step(
                torch, GateValues, guarded_step, packed, packed_optimizer,
                context, inputs, target, primal, derivative, hard_mask, threshold),
            "refresh": lambda: global_edge_value_refresh_step(packed),
        },
    }

    # Independent numerical check before timed updates. It exercises the
    # derivative-only STE candidate gradients as well as the same forward.
    check_results = {}
    for label in ("simt", "packed"):
        optimizer = simt_optimizer if label == "simt" else packed_optimizer
        module = baseline if label == "simt" else packed
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            output = funcs[label]["infer"]()
        check_results[label] = {"inference_output": output.detach().clone()}
        captured = training_step(
            torch, GateValues, guarded_step, module, optimizer, context, inputs,
            target, primal, derivative, hard_mask, threshold, capture=True)
        check_results[label].update(captured)
    output_ok = torch.allclose(check_results["simt"]["inference_output"],
                               check_results["packed"]["inference_output"],
                               atol=5.0e-3, rtol=5.0e-3)
    gradient_ok = torch.allclose(check_results["simt"]["parameter_grad"],
                                  check_results["packed"]["parameter_grad"],
                                  atol=6.0e-3, rtol=6.0e-3)
    loss_ok = math.isclose(check_results["simt"]["loss"],
                           check_results["packed"]["loss"],
                           abs_tol=5.0e-4, rel_tol=5.0e-3)
    if not (output_ok and gradient_ok and loss_ok):
        raise AssertionError("SIMT/packed numerical check failed: " + json.dumps({
            "output_max_abs": float((check_results["simt"]["inference_output"] -
                                      check_results["packed"]["inference_output"]).abs().max().cpu()),
            "grad_max_abs": float((check_results["simt"]["parameter_grad"] -
                                    check_results["packed"]["parameter_grad"]).abs().max().cpu()),
            "loss_simt": check_results["simt"]["loss"],
            "loss_packed": check_results["packed"]["loss"],
        }))
    derivative_grad_nonzero = {}
    for label in ("simt", "packed"):
        pgrad = check_results[label]["probability_grad"]
        if pgrad is None:
            derivative_grad_nonzero[label] = 0
        else:
            derivative_grad_nonzero[label] = int(
                ((pgrad != 0) & (~hard_mask)).sum().item())
    if any(value == 0 for value in derivative_grad_nonzero.values()):
        raise AssertionError("derivative-only gate candidates did not contribute STE gradients")

    # Warm both variants in alternating order so Adam state and allocator state
    # are initialized before collecting paired samples.
    for i in range(args.warmup):
        order = ("simt", "packed") if i % 2 == 0 else ("packed", "simt")
        for label in order:
            funcs[label]["infer"]()
    torch.cuda.synchronize()
    for i in range(args.warmup):
        order = ("simt", "packed") if i % 2 == 0 else ("packed", "simt")
        for label in order:
            funcs[label]["refresh"]()
    torch.cuda.synchronize()
    for i in range(args.warmup):
        order = ("simt", "packed") if i % 2 == 0 else ("packed", "simt")
        for label in order:
            funcs[label]["train"]()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    phase_samples: dict[str, dict[str, list[dict[str, Any]]]] = {
        phase: {"simt": [], "packed": []}
        for phase in ("inference", "global_edge_value_refresh_fwd_plus_ones_vjp", "training")}
    peak_extra: dict[str, dict[str, int]] = {
        phase: {"simt": 0, "packed": 0} for phase in phase_samples}
    for phase, fn_key in (("inference", "infer"),
                          ("global_edge_value_refresh_fwd_plus_ones_vjp", "refresh"),
                          ("training", "train")):
        for round_index in range(args.rounds):
            order = (("simt", "packed") if round_index % 2 == 0
                     else ("packed", "simt"))
            pair: dict[str, Any] = {"round": round_index, "order": list(order)}
            for label in order:
                torch.cuda.synchronize()
                start_alloc = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                sample = event_pair_ms(start_event, end_event, torch,
                                       funcs[label][fn_key], args.iterations)
                torch.cuda.synchronize()
                sample["peak_allocated_delta_bytes"] = max(
                    0, int(torch.cuda.max_memory_allocated() - start_alloc))
                peak_extra[phase][label] = max(peak_extra[phase][label],
                                               sample["peak_allocated_delta_bytes"])
                sample["round"] = round_index
                sample["candidate"] = label
                phase_samples[phase][label].append(sample)
                pair[label] = sample
            pair["paired_cuda_ratio_simt_over_packed"] = (
                pair["simt"]["cuda_event_ms_per_step"] /
                pair["packed"]["cuda_event_ms_per_step"])
            pair["paired_wall_ratio_simt_over_packed"] = (
                pair["simt"]["synchronized_wall_ms_per_step"] /
                pair["packed"]["synchronized_wall_ms_per_step"])
            phase_samples[phase].setdefault("pairs", []).append(pair)

    parameter_max_abs = float((baseline.parameters_flat.detach() -
                               packed.parameters_flat.detach()).abs().max().cpu())
    if len(baseline._read_tickets) or len(packed._read_tickets):
        raise RuntimeError("forward read lease remains outstanding after training steps")

    phase_results = {}
    for phase in phase_samples:
        phase_results[phase] = {}
        pairs = phase_samples[phase]["pairs"]
        cuda_ratios = [p["paired_cuda_ratio_simt_over_packed"] for p in pairs]
        wall_ratios = [p["paired_wall_ratio_simt_over_packed"] for p in pairs]
        for label in ("simt", "packed"):
            samples = phase_samples[phase][label]
            phase_results[phase][label] = {
                "cuda_event_ms_per_step": stats(
                    [s["cuda_event_ms_per_step"] for s in samples]),
                "synchronized_wall_ms_per_step": stats(
                    [s["synchronized_wall_ms_per_step"] for s in samples]),
                "max_peak_allocated_delta_bytes": peak_extra[phase][label],
                "raw_samples": samples,
            }
        phase_results[phase]["paired"] = {
            "simt_over_packed_cuda_ratio_median": statistics.median(cuda_ratios),
            "simt_over_packed_cuda_ratio_bootstrap_95pct": paired_ratio_ci(cuda_ratios, seed + 37),
            "simt_over_packed_wall_ratio_median": statistics.median(wall_ratios),
            "simt_over_packed_wall_ratio_bootstrap_95pct": paired_ratio_ci(wall_ratios, seed + 41),
            "raw_pairs": pairs,
        }

    packed_extra_module_setup_ms = (
        packed_setup["wall_ms"]["median"] - simt_setup["wall_ms"]["median"])
    packed_extra_total_setup_ms = (
        packed_extra_module_setup_ms + compiled["host_compile_ms"])
    break_even = {}
    for phase in ("inference", "training"):
        for unit, simt_key, packed_key, ci_key in (
                ("cuda_event", "cuda_event_ms_per_step", "cuda_event_ms_per_step",
                 "simt_over_packed_cuda_ratio_bootstrap_95pct"),
                ("synchronized_wall", "synchronized_wall_ms_per_step",
                 "synchronized_wall_ms_per_step",
                 "simt_over_packed_wall_ratio_bootstrap_95pct")):
            simt_step = phase_results[phase]["simt"][simt_key]["median"]
            packed_step = phase_results[phase]["packed"][packed_key]["median"]
            saving = simt_step - packed_step
            ci_low = phase_results[phase]["paired"][ci_key][0]
            saving_supported = ci_low > 1.0 and saving > 0.0
            break_even_supported = saving_supported and packed_extra_total_setup_ms > 0.0
            break_even[f"{phase}_{unit}"] = {
                "packed_extra_cold_setup_ms": packed_extra_total_setup_ms,
                "median_step_saving_ms": saving,
                "break_even_steps": (
                    packed_extra_total_setup_ms / saving if break_even_supported else None),
                "positive_step_saving_supported_by_paired_ci": saving_supported,
                "positive_extra_setup_cost": packed_extra_total_setup_ms > 0.0,
            }

    params = int(baseline.parameters_flat.numel())
    return {
        "case": shape_meta,
        "setup": {
            "common_cuda_torch_initialization_ms": args.common_cuda_init_ms,
            "fixture_owner_prepare_ms": fixture_module_setup["wall_ms"],
            "shared_geometry_compile_host_ms": compiled["host_compile_ms"],
            "module_setup_priming_order": [list(order) for order in priming_orders],
            "balanced_module_setup_pair_count": args.setup_pairs,
            "balanced_module_setup_order": [
                ("simt", "packed") if i % 2 == 0 else ("packed", "simt")
                for i in range(args.setup_pairs)],
            "simt": simt_setup,
            "packed": packed_setup,
            "packed_extra_module_setup_ms_median": packed_extra_module_setup_ms,
            "packed_extra_cold_setup_ms_including_geometry_compile": packed_extra_total_setup_ms,
            "shared_workset_and_input_prep_ms": shape_meta["shared_workset_and_input_prep_ms"],
            "hard_gate_freeze_and_primal_workset_ms":
                shape_meta["hard_gate_freeze_and_primal_workset_ms"],
            "break_even": break_even,
        },
        "checks": {
            "inference_output_allclose_atol_5e-3_rtol_5e-3": bool(output_ok),
            "training_parameter_gradient_allclose_atol_6e-3_rtol_6e-3": bool(gradient_ok),
            "loss_close_abs_5e-4_rel_5e-3": bool(loss_ok),
            "derivative_only_probability_grad_nonzero": derivative_grad_nonzero,
            "max_parameter_difference_after_measured_training": parameter_max_abs,
            "remaining_read_tickets": 0,
        },
        "parameters": {"shared_fp32_parameter_count": params,
                       "adam_state_initialized": True},
        "phases": phase_results,
        "backend_scope": {
            "simt": "SIMT native sparse relation forward and VJP",
            "packed": "CE-GEO N64 packed forward plus the same SIMT native VJP",
            "hard_support_and_admission_mask": "frozen and byte-identical across candidates",
            "gate_probabilities": "recomputed from trainable shared state per step",
            "derivative_only_candidates": "included in VJP and reported in case metadata",
        },
    }


def summarize_case(name: str, case: dict) -> str:
    lines = [f"### {name}"]
    lines.append(f"- Shape: `{json.dumps(case['case']['shape'], sort_keys=True)}`")
    lines.append(
        f"- Work: {case['case']['primal_items']} primal / "
        f"{case['case']['derivative_items']} derivative candidates; "
        f"{case['case']['derivative_only_work_fraction']:.1%} derivative-only.")
    for phase in ("inference", "training"):
        simt = case["phases"][phase]["simt"]
        packed = case["phases"][phase]["packed"]
        pair = case["phases"][phase]["paired"]
        lines.append(
            f"- {phase}: CUDA event SIMT {simt['cuda_event_ms_per_step']['median']:.6f} ms, "
            f"packed {packed['cuda_event_ms_per_step']['median']:.6f} ms "
            f"(paired SIMT/packed {pair['simt_over_packed_cuda_ratio_median']:.3f}, "
            f"95% bootstrap {pair['simt_over_packed_cuda_ratio_bootstrap_95pct']}); "
            f"synchronized wall SIMT "
            f"{simt['synchronized_wall_ms_per_step']['median']:.6f} ms, packed "
            f"{packed['synchronized_wall_ms_per_step']['median']:.6f} ms "
            f"(ratio {pair['simt_over_packed_wall_ratio_median']:.3f}).")
    refresh = case["phases"]["global_edge_value_refresh_fwd_plus_ones_vjp"]
    lines.append(
        "- Global all-edge value refresh plus diagnostic all-ones VJP: "
        f"SIMT-owner median {refresh['simt']['cuda_event_ms_per_step']['median']:.6f} ms, "
        f"packed-owner median {refresh['packed']['cuda_event_ms_per_step']['median']:.6f} ms; "
        "the VJP cotangent is simpler than the relation-derived gradient, so this is "
        "phase attribution, not a substitute for complete training timing.")
    simt_setup = case["setup"]["simt"]["wall_ms"]
    packed_setup = case["setup"]["packed"]["wall_ms"]
    lines.append(
        f"- Preparation: common CUDA/Torch initialization "
        f"{case['setup']['common_cuda_torch_initialization_ms']:.3f} ms; "
        f"fixture bundle load {case['setup'].get('fixture_bundle_load_ms', 0.0):.3f} ms; "
        f"support-spec construction {case['setup'].get('support_spec_prepare_ms', 0.0):.3f} ms; "
        f"fixture owner setup {case['setup']['fixture_owner_prepare_ms']:.3f} ms; "
        f"CE-GEO host compilation {case['setup']['shared_geometry_compile_host_ms']:.3f} ms.")
    lines.append(
        f"- Balanced module setup ({case['setup']['balanced_module_setup_pair_count']} AB/BA pairs): "
        f"SIMT median {simt_setup['median']:.3f} ms "
        f"(p10-p90 {simt_setup['p10']:.3f}-{simt_setup['p90']:.3f}); packed median "
        f"{packed_setup['median']:.3f} ms "
        f"(p10-p90 {packed_setup['p10']:.3f}-{packed_setup['p90']:.3f}); packed extra "
        f"including CE-GEO host compile "
        f"{case['setup']['packed_extra_cold_setup_ms_including_geometry_compile']:.3f} ms.")
    lines.append(f"- Setup amortization estimates: `{json.dumps(case['setup']['break_even'], sort_keys=True)}`")
    lines.append(
        f"- Checks: output={case['checks']['inference_output_allclose_atol_5e-3_rtol_5e-3']}, "
        f"gradients={case['checks']['training_parameter_gradient_allclose_atol_6e-3_rtol_6e-3']}, "
        f"derivative-only gate gradient counts="
        f"{case['checks']['derivative_only_probability_grad_nonzero']}.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", choices=("all", "event_local", "ranked_batch",
                                           "large_ranked", "large_events"), default="all")
    parser.add_argument("--fixture-root", type=Path,
                        default=Path("/tmp/cellerator_scaled_sparse_fixtures_887437fc"))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--setup-pairs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if os.environ.get("CELLERATOR_SPARSE_SUPPORT_BENCH_ADMITTED") != "1":
        parser.error("run only inside the admitted Device Scheduler job")
    package_root = Path(os.environ.get("CELLERATOR_SPARSE_PACKAGE_ROOT", ""))
    if not package_root.is_dir():
        parser.error("CELLERATOR_SPARSE_PACKAGE_ROOT must name a frozen Torch package directory")
    if min(args.warmup, args.rounds, args.iterations, args.setup_pairs) <= 0:
        parser.error("warmup, rounds, iterations, and setup pairs must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from cellerator.torch import _torch
    from cellerator.torch.sparse_support import (
        KaryoplastEventSlice, RankedGeneSlice, SupportSpec,
        compile_sparse_support_ce_geo, prepare_sparse_slice,
    )
    package = importlib.import_module("cellerator.torch.sparse_support")
    mechanism = importlib.import_module("cellerator.torch.mechanism")
    if not torch.cuda.is_available():
        raise RuntimeError("scheduler-admitted job has no CUDA device")
    common_init_start = time.perf_counter()
    torch.cuda.init()
    device = torch.cuda.current_device()
    common_init_probe = torch.empty((1,), device=device, dtype=torch.float32)
    common_init_probe.zero_()
    torch.cuda.synchronize(device)
    args.common_cuda_init_ms = (time.perf_counter() - common_init_start) * 1000.0
    del common_init_probe
    torch.cuda.empty_cache()
    if Path(package.__file__).resolve().is_relative_to(package_root.resolve()) is False:
        raise RuntimeError(f"imported package {package.__file__} is outside frozen root {package_root}")
    identity_path = package_root.parent / "identity.json"
    source_identities = {}
    frozen_identity = json.loads(identity_path.read_text())
    identity_files = frozen_identity.get("files", frozen_identity)
    if not isinstance(identity_files, dict):
        raise RuntimeError("frozen Torch identity must contain a file hash map")
    for relative in IDENTITY_VALIDATED_FILES:
        source_path = package_root.parent / relative
        manifest_expected = identity_files.get(relative)
        if not isinstance(manifest_expected, str) or not manifest_expected:
            raise RuntimeError(f"frozen Torch identity lacks a hash for {relative}")
        actual = sha256_file(source_path)
        if actual != manifest_expected:
            raise RuntimeError(f"frozen Torch source identity changed for {relative}: {actual}")
        source_identities[relative] = actual
    extension_path = Path(_torch.__file__).resolve()
    source_identities["frozen_torch_identity.json"] = sha256_file(identity_path)
    extension_hash = sha256_file(extension_path)
    expected_extension = identity_files.get(TORCH_EXTENSION_PATH)
    if not isinstance(expected_extension, str) or not expected_extension:
        raise RuntimeError(f"frozen Torch identity lacks a hash for {TORCH_EXTENSION_PATH}")
    if extension_hash != expected_extension:
        raise RuntimeError("frozen Torch extension hash mismatch")
    source_identities[TORCH_EXTENSION_PATH] = extension_hash
    source_identities["benchmark_driver"] = sha256_file(Path(__file__).resolve())
    fixture_generator_path = REPO / "bench/sparse_support/scaled_fixtures.py"
    if fixture_generator_path.is_file():
        source_identities["scaled_fixture_generator"] = sha256_file(fixture_generator_path)

    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    cases = ("event_local", "ranked_batch", "large_ranked", "large_events") \
        if args.case == "all" else (args.case,)
    results = {}
    for case_index, case_name in enumerate(cases):
        if case_name in {"large_ranked", "large_events"}:
            bundle = args.fixture_root / ("large-ranked" if case_name == "large_ranked"
                                          else "large-events")
            arrays_to_load = ["edge_ids", "source_entity_id", "participant_entity_id",
                              "source_edge_indptr", "source_edge_ordinals"]
            if case_name == "large_ranked":
                arrays_to_load.extend(["ranked_entity_id", "ranked_position",
                                       "ranked_presence_offsets", "ranked_presence_entity_ids",
                                       "ranked_presence_values", "ranked_presence_view_id"])
            else:
                arrays_to_load.extend(["event_offsets", "event_entity_ids",
                                       "event_timestamps", "event_ordinals",
                                       "count_window_bounds", "time_window_bounds",
                                       "event_window_observed_indptr",
                                       "event_window_observed_gene_ids",
                                       "event_window_observed_view_id", "window_cell_id",
                                       "window_id", "window_start_ordinal",
                                       "window_stop_ordinal_exclusive"])
            bundle_load_start = time.perf_counter()
            arrays, manifest, manifest_sha = load_raw_bundle(bundle, tuple(arrays_to_load))
            bundle_load_ms = (time.perf_counter() - bundle_load_start) * 1000.0
            args.fixture_arrays = arrays
            args.fixture_manifest = manifest
            args.fixture_bundle = str(bundle.resolve())
            args.fixture_manifest_sha256 = manifest_sha
            meta = manifest["metadata"]
            entities = int(meta["entities"])
            support_prepare_start = time.perf_counter()
            edge_ids_host = arrays["edge_ids"].astype(np.int64, copy=False)
            source_host = arrays["source_entity_id"].astype(np.uint32, copy=False)
            destination_host = arrays["participant_entity_id"].astype(np.uint32, copy=False)
            support = SupportSpec(tuple(map(int, edge_ids_host)),
                                  tuple(map(int, source_host)),
                                  tuple(map(int, destination_host)), 1)
            support_prepare_ms = (time.perf_counter() - support_prepare_start) * 1000.0
            topology_sha = str(meta["topology_sha256"])
            layout = str(meta["layout"])
            fixture_manifest_sha = manifest_sha
            fixture_generator_sha = sha256_file(REPO / "bench/sparse_support/scaled_fixtures.py")
            if fixture_generator_sha != "887437fc3418e06c25e515487891a3224f8dd56493b689bf41bc70e66ae5a76b":
                raise RuntimeError("scaled fixture generator differs from the frozen campaign source")
        else:
            bundle_load_ms = 0.0
            support_prepare_ms = 0.0
            support = make_support(SupportSpec)
            entities = ENTITIES
            args.fixture_arrays = None
            args.fixture_manifest = None
            args.fixture_bundle = None
            args.fixture_manifest_sha256 = None
            topology_sha = sha256_bytes(json_bytes({
                "edge_ids": support.edge_ids, "sources": support.sources,
                "destinations": support.destinations}))
            layout = "identity"
            fixture_manifest_sha = None
            fixture_generator_sha = None
        compile_start = time.perf_counter()
        compiled = compile_sparse_support_ce_geo(
            support, source_count=entities, destination_count=entities,
            projection_id=91 + case_index)
        compile_ms = (time.perf_counter() - compile_start) * 1000.0
        compiled["host_compile_ms"] = compile_ms
        compiled_meta = {key: compiled[key] for key in (
            "projection_id", "structure_epoch", "structure_identity", "tile_count",
            "residual_region_count", "value_map_count")}
        compiled_meta["image_sha256"] = sha256_bytes(compiled["image"])
        compiled_meta["host_compile_ms"] = compile_ms
        args.current_fixture_identity = {
            "bundle": args.fixture_bundle,
            "manifest_sha256": fixture_manifest_sha,
            "generator_sha256": fixture_generator_sha,
            "topology_sha256": topology_sha,
            "layout": layout,
        }
        results[case_name] = benchmark_case(
            torch, package, support, case_name, args.seed + case_index * 997,
            args, compiled)
        results[case_name]["setup"]["fixture_bundle_load_ms"] = bundle_load_ms
        results[case_name]["setup"]["support_spec_prepare_ms"] = support_prepare_ms
        results[case_name]["case"].update({
            "support_edges": len(support.edge_ids),
            "logical_entities": entities,
            "support_topology_sha256": topology_sha,
            "fixture_bundle_identity": args.current_fixture_identity,
            "compiled_geometry": compiled_meta,
        })

    props = torch.cuda.get_device_properties(device)
    config = {
        "benchmark_id": "cellerator-packed-vs-simt-training",
        "workload_family": "sparse_support_relation_training",
        "dataset_tier": "small_and_scaled",
        "backend_control": "same frozen Torch binary; handle projection toggles SIMT vs CE-GEO N64",
        "torch_package_root": str(package_root.resolve()),
        "torch_package_source": str(Path(package.__file__).resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device": {"index": device, "name": props.name,
                   "compute_capability": [props.major, props.minor],
                   "total_memory_bytes": props.total_memory},
        "source_identities": source_identities,
        "support": {"edge_count_by_case": {
                        name: case["case"]["support_edges"] for name, case in results.items()},
                    "topology_sha256_by_case": {
                        name: case["case"]["support_topology_sha256"]
                        for name, case in results.items()},
                    "compiled_geometry_by_case": {
                        name: case["case"]["compiled_geometry"]
                        for name, case in results.items()}},
        "timing": {"warmup_steps_per_candidate": args.warmup,
                   "paired_rounds": args.rounds,
                   "iterations_per_sample": args.iterations,
                   "balanced_setup_pairs": args.setup_pairs,
                   "cuda_event": "events enclose each candidate sample; synchronized before/after",
                   "wall": "perf_counter around submitted sample through final CUDA sync",
                   "paired_order": "alternates SIMT-first and packed-first",
                   "setup_order": "primes both backends, then alternates setup order across paired samples",
                   "common_cuda_torch_initialization_ms": args.common_cuda_init_ms,
                   "optimizer": "torch.optim.Adam through guarded_step, lr=1e-3, foreach/fused=false"},
        "scheduler": {"job_id": os.environ.get("DEVICE_SCHEDULER_JOB_ID"),
                      "attempt_id": os.environ.get("DEVICE_SCHEDULER_ATTEMPT_ID"),
                      "admitted_marker": os.environ.get("CELLERATOR_SPARSE_SUPPORT_BENCH_ADMITTED")},
        "scaled_fixture_generator_sha256": (
            sha256_file(REPO / "bench/sparse_support/scaled_fixtures.py")),
        "cases": list(cases),
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (output_dir / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    summary = ["# Packed versus SIMT sparse-support training benchmark", "",
               f"Source identity: `{source_identities['benchmark_driver']}`",
               f"Frozen Torch extension: `{extension_hash}`", "",
               "SIMT control and packed candidate use identical support, source inputs, "
               "workset rows, hard admission bits, and initial shared parameters. "
               "Packed backward is the existing SIMT VJP.", ""]
    summary.extend(summarize_case(name, value) for name, value in results.items())
    (output_dir / "summary.md").write_text("\n\n".join(summary) + "\n")
    print(json.dumps({"output_dir": str(output_dir), "cases": list(cases),
                      "results": str(output_dir / "results.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
