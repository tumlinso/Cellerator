#!/usr/bin/env python3
"""Train both native sparse slices on controlled held-out synthetic fixtures.

The CE-GEO image and compact group membership arrays come from the normal
Cellerator geometry/packing pipeline. Candidate construction is timed/setup
work through the native reverse and temporal indexes; it is not hidden inside
the numerical relation launch.
"""
from __future__ import annotations

import argparse
import gc
import random
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch

from cellerator.torch.mechanism import (
    guarded_step, load_checkpoint, save_checkpoint,
)
from cellerator.torch.sparse_support import (
    KaryoplastEventSlice, PerCellWorkset, RankedGeneSlice, SupportSpec,
    compile_sparse_support_ce_geo, make_masked_gene_fixture,
    make_next_event_fixture, prepare_sparse_slice,
)
from cellerator.torch import _torch


ENTITIES = 24
MECHANISMS = 4
WIDTH = 64
GATE_THRESHOLD = 0.5


def assert_transition_gradients(module, *, require_state_gradient=True):
    """Fail the fixture if its learned latent path is numerically disconnected."""
    flat_gradient = module.parameters_flat.grad
    if flat_gradient is None:
        raise RuntimeError("sparse slice produced no shared-parameter gradient")
    required = ("transition_B", "transition_bias")
    if require_state_gradient:
        required += ("transition_A",)
    for name in required:
        block = module._blocks[name]
        gradient = flat_gradient.narrow(0, block.offset, block.numel)
        if not torch.isfinite(gradient).all() or float(gradient.norm()) == 0.0:
            raise RuntimeError(f"synthetic task did not train latent parameter block {name}")


def pair_support(seed: int = 0) -> SupportSpec:
    # Seeded structural proposals cover most possible routes without reading
    # the fixture's validator-only hidden support. Each source has six of the
    # eight possible targets; eligibility still depends on the observed cell.
    rng = random.Random(seed)
    edges = []
    for source in range(16):
        targets = sorted(rng.sample(range(8), 6))
        edges.extend((1 + source * 8 + target, source, 16 + target)
                     for target in targets)
    edges.sort()
    return SupportSpec(tuple(item[0] for item in edges),
                       tuple(item[1] for item in edges),
                       tuple(item[2] for item in edges))


def workset_rows(index, support: SupportSpec, participant_rows, device, generation):
    ids_rows, ordinal_rows = [], []
    capacity_per_cell = len(support.edge_ids)
    for participant_ids in participant_rows:
        unique = sorted(set(participant_ids))
        if unique:
            result = index.query_any(
                np.zeros(len(unique), dtype=np.int32),
                np.asarray([entity + 1 for entity in unique], dtype=np.int64),
                capacity_per_cell)
            ids_rows.append(result["logical_edge_ids"].tolist())
            ordinal_rows.append(result["edge_ordinals"].tolist())
        else:
            ids_rows.append([])
            ordinal_rows.append([])
    offsets = [0]
    flat_ids, flat_ordinals = [], []
    for edge_ids, ordinals in zip(ids_rows, ordinal_rows):
        if len(edge_ids) != len(ordinals):
            raise RuntimeError("native posting query returned inconsistent arrays")
        begin = len(flat_ids)
        flat_ids.extend(edge_ids)
        flat_ordinals.extend(ordinals)
        offsets.append(len(flat_ids))
    gates = np.arange(len(flat_ids), dtype=np.int64)
    workset = PerCellWorkset(
        torch.tensor(offsets, device=device, dtype=torch.int64),
        torch.tensor(flat_ordinals, device=device, dtype=torch.int64),
        torch.tensor(flat_ids, device=device, dtype=torch.int64),
        torch.tensor(gates, device=device, dtype=torch.int64),
        support.structure_epoch, generation,
        max(1, len(participant_rows) * capacity_per_cell))
    # This is the bounded current eligibility pool. The module later separates
    # admitted forward items from the native rotating derivative sample.
    return workset, workset


def clipped_event_windows(position, count_bounds, time_bounds, current_time):
    """Return inclusive count/time bounds clipped to the observed prefix."""
    minimum_ordinal = int(count_bounds[0]) if count_bounds is not None else 0
    maximum_ordinal = min(int(count_bounds[1]), position) if count_bounds is not None else position
    if time_bounds is None:
        minimum_time = maximum_time = None
    else:
        minimum_time = int(time_bounds[0])
        maximum_time = min(int(time_bounds[1]), int(current_time))
    if minimum_ordinal > maximum_ordinal or (
            minimum_time is not None and minimum_time > maximum_time):
        return None
    return minimum_ordinal, maximum_ordinal, minimum_time, maximum_time


def train_ranked(module, index, fixture, device, optimizer, *, prune=False):
    module.train()
    genes = fixture["genes"]
    rows = [[int(gene) for gene in row.tolist() if gene >= 0] for row in genes]
    primal, derivative = workset_rows(
        index, module.support, rows, device, int(fixture["seed"]))
    ranks = fixture["ranks"].to(device)
    targets = fixture["targets"].to(device)
    encoded = module(genes.to(device), ranks, primal, derivative,
                     prune=prune, threshold=GATE_THRESHOLD,
                     derivative_seed=int(fixture["seed"]),
                     window_counter=int(fixture["seed"]))
    loss = module.training_loss(encoded, targets)
    loss.backward()
    assert_transition_gradients(module)
    guarded_step(module, optimizer)
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach()), module.gate_admission_rate()


def event_provider(module, index, fixture, device):
    events = fixture["events"]
    time_values = fixture["timestamps"]
    time_windows = fixture["time_windows"]
    cell_count, steps = events.shape
    rings = {}
    ring_capacity = max(steps, int(module.application_state.get(
        "temporal_ring_capacity", fixture.get("ring_capacity", steps))))
    for key, records in module.application_state.get(
            "temporal_occurrence_rings", {}).items():
        cell, entity = map(int, key.split(":"))
        ring = _torch.NativeTemporalOccurrenceRing(cell + 1, entity + 1, ring_capacity)
        for occurrence_id, ordinal, timestamp, flags in zip(
                records["occurrence_ids"], records["ordinals"],
                records["timestamps"], records["flags"]):
            ring.append(int(occurrence_id), int(ordinal),
                        int(timestamp) if int(flags) & 1 else None)
        rings[(cell, entity)] = ring
    generation = int(module.application_state.get("temporal_ring_counter", 0))

    def provider(position, history, timestamps, windows):
        nonlocal generation
        participant_rows, source_rows = [], []
        mechanism_entities = history[:, -1:]
        for cell in range(cell_count):
            current = int(history[cell, -1])
            current_time = int(timestamps[cell, -1]) if timestamps is not None else None
            key = (cell, current)
            if key not in rings:
                rings[key] = _torch.NativeTemporalOccurrenceRing(
                    cell + 1, current + 1, steps)
            occurrence_id = ((cell + 1) << 48) + position + 1
            rings[key].append(occurrence_id, position, current_time)
            all_entities = sorted(
                {int(value) for value in history[cell].tolist()}
                | {entity for (ring_cell, entity) in rings if ring_cell == cell})
            eligible = []
            count_bounds = windows[cell] if windows is not None else None
            time_bounds = time_windows[cell] if timestamps is not None else None
            clipped = clipped_event_windows(
                position, count_bounds, time_bounds, current_time)
            for entity in all_entities:
                ring = rings.get((cell, entity))
                if ring is None or clipped is None:
                    continue
                occurrences = ring.query_window(
                    *clipped)
                eligible.extend([entity] * len(occurrences["occurrence_ids"]))
            participant_rows.append(eligible)
            source_rows.append(eligible)
        snapshots = {}
        for (cell, entity), ring in rings.items():
            records = ring.query_window(None, None, None, None)
            snapshots[f"{cell}:{entity}"] = {
                key: np.asarray(records[key], dtype=np.int64).tolist()
                for key in ("occurrence_ids", "ordinals", "timestamps", "flags")
            }
        module.application_state["temporal_occurrence_rings"] = snapshots
        module.application_state["temporal_ring_capacity"] = ring_capacity
        module.application_state["temporal_ring_counter"] = int(position) + 1
        affected_offsets, affected_mechanisms = module.affected_mechanism_workset(
            mechanism_entities)
        generation += 1
        primal, derivative = workset_rows(
            index, module.support, participant_rows, device, generation)
        return (primal, derivative,
                affected_offsets, affected_mechanisms, source_rows)

    return provider


def train_events(module, index, fixture, device, optimizer, *, prune=False):
    module.train()
    # Each fixture batch is a new cohort; recurrence is within its event stream,
    # not across independent examples or optimizer windows.
    for key in ("latent_state", "event_position", "last_event_ids", "last_timestamps",
                "windows", "temporal_occurrence_rings", "temporal_ring_capacity",
                "temporal_ring_counter"):
        module.application_state.pop(key, None)
    events = fixture["events"]
    time_values = fixture["timestamps"]

    output = module(events, event_provider(module, index, fixture, device),
                    timestamps=time_values, windows=fixture["count_windows"],
                    prune=prune, threshold=GATE_THRESHOLD,
                    derivative_seed=int(fixture["seed"]))
    loss = module.training_loss(output, fixture["targets"].to(device))
    loss.backward()
    assert_transition_gradients(module)
    guarded_step(module, optimizer)
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach()), module.gate_admission_rate()


@torch.no_grad()
def evaluate_ranked(module, index, fixture, device, *, prune):
    module.eval()
    rows = [[int(gene) for gene in row.tolist() if gene >= 0]
            for row in fixture["genes"]]
    primal, derivative = workset_rows(index, module.support, rows, device, 1)
    encoded = module(fixture["genes"].to(device), fixture["ranks"].to(device),
                     primal, derivative, prune=prune, threshold=GATE_THRESHOLD,
                     derivative_seed=int(fixture["seed"]),
                     window_counter=int(fixture["seed"]))
    logits = module.masked_gene_logits(encoded)
    targets = fixture["targets"].to(device)
    return (float(torch.nn.functional.cross_entropy(logits, targets)),
            float((logits.argmax(dim=-1) == targets).float().mean()),
            module.gate_admission_rate())


@torch.no_grad()
def evaluate_events(module, index, fixture, device, *, prune):
    module.eval()
    for key in ("latent_state", "event_position", "last_event_ids", "last_timestamps",
                "windows", "temporal_occurrence_rings", "temporal_ring_capacity",
                "temporal_ring_counter"):
        module.application_state.pop(key, None)
    encoded = module(
        fixture["events"], event_provider(module, index, fixture, device),
        timestamps=fixture["timestamps"], windows=fixture["count_windows"],
        prune=prune, threshold=GATE_THRESHOLD,
        derivative_seed=int(fixture["seed"]))
    logits = module.next_event_logits(encoded)
    targets = fixture["targets"].to(device)
    return (float(torch.nn.functional.cross_entropy(logits, targets)),
            float((logits.argmax(dim=-1) == targets).float().mean()),
            module.gate_admission_rate())


def verify_checkpoint_continuation(prepare, support, device, seed,
                                   ranked, ranked_optimizer, events, event_optimizer,
                                   ranked_fixture, event_fixture):
    """Round-trip both adapted owners and compare the next optimizer window."""
    with tempfile.TemporaryDirectory(prefix="cellerator-sparse-checkpoint-") as directory:
        for kind, source, source_optimizer, fixture in (
                ("ranked", ranked, ranked_optimizer, ranked_fixture),
                ("events", events, event_optimizer, event_fixture)):
            path = Path(directory) / f"{kind}.pt"
            save_checkpoint(path, source, source_optimizer)
            module_type = RankedGeneSlice if kind == "ranked" else KaryoplastEventSlice
            # Recreate the same logical parameter-axis identity before loading.
            restored = prepare(module_type, support, 1 if kind == "ranked" else 2,
                               seed if kind == "ranked" else seed + 1)
            optimizer = torch.optim.Adam([restored.parameters_flat], lr=0.02)
            load_checkpoint(path, restored, optimizer)
            if restored.support.edge_ids != source.support.edge_ids:
                raise RuntimeError(f"{kind} checkpoint did not restore adaptive support")
            source_index = _torch.NativeReverseIndex(source._support_index)
            restored_index = _torch.NativeReverseIndex(restored._support_index)
            if kind == "ranked":
                reference = evaluate_ranked(source, source_index, fixture, device, prune=True)
                actual = evaluate_ranked(restored, restored_index, fixture, device, prune=True)
            else:
                reference = evaluate_events(source, source_index, fixture, device, prune=True)
                actual = evaluate_events(restored, restored_index, fixture, device, prune=True)
            if any(abs(left - right) > 1e-5 for left, right in zip(reference, actual)):
                raise RuntimeError(f"{kind} checkpoint changed held-out outputs")
            if kind == "ranked":
                train_ranked(source, source_index, fixture, device, source_optimizer, prune=True)
                train_ranked(restored, restored_index, fixture, device, optimizer, prune=True)
            else:
                train_events(source, source_index, fixture, device, source_optimizer, prune=True)
                train_events(restored, restored_index, fixture, device, optimizer, prune=True)
            if not torch.allclose(source.parameters_flat, restored.parameters_flat,
                                  atol=2e-6, rtol=2e-6):
                raise RuntimeError(f"{kind} checkpoint continuation diverged after one optimizer step")
            print(f"checkpoint_continuation={kind} support_edges={len(restored.support.edge_ids)} ok=1")


def reset_event_stream_state(module):
    for key in ("latent_state", "event_position", "last_event_ids", "last_timestamps",
                "windows", "temporal_occurrence_rings", "temporal_ring_capacity",
                "temporal_ring_counter"):
        module.application_state.pop(key, None)


def event_fixture_slice(fixture, start, stop):
    """Slice observations while keeping absolute timestamp/count windows."""
    result = dict(fixture)
    result["events"] = fixture["events"][:, start:stop].contiguous()
    result["timestamps"] = fixture["timestamps"][:, start:stop].contiguous()
    result["ring_capacity"] = int(fixture["events"].shape[1])
    return result


@torch.no_grad()
def verify_event_stream_checkpoint(prepare, support, device, seed,
                                   source, source_optimizer, index):
    """Restore latent and temporal posting state halfway through one stream."""
    fixture = make_next_event_fixture(seed=seed + 500_000, cells=32,
                                      steps=12, entities=ENTITIES)
    reset_event_stream_state(source)
    full = source(fixture["events"].to(device),
                  event_provider(source, index, fixture, device),
                  timestamps=fixture["timestamps"],
                  windows=fixture["count_windows"],
                  prune=True, threshold=GATE_THRESHOLD,
                  derivative_seed=int(fixture["seed"]))
    full_reference = full.detach().cpu()
    # The native forward context pins its support epoch until the output/context
    # retires. Checkpoint publication therefore happens only after dropping
    # this reference result.
    del full
    gc.collect()
    reset_event_stream_state(source)
    prefix = event_fixture_slice(fixture, 0, 6)
    source(prefix["events"].to(device), event_provider(source, index, prefix, device),
           timestamps=prefix["timestamps"],
           windows=prefix["count_windows"], prune=True,
           threshold=GATE_THRESHOLD, derivative_seed=int(fixture["seed"]))
    with tempfile.TemporaryDirectory(prefix="cellerator-event-stream-") as directory:
        path = Path(directory) / "mid-stream.pt"
        save_checkpoint(path, source, source_optimizer)
        restored = prepare(KaryoplastEventSlice, support, 2, seed + 1)
        restored_optimizer = torch.optim.Adam([restored.parameters_flat], lr=0.02)
        load_checkpoint(path, restored, restored_optimizer)
        restored_index = _torch.NativeReverseIndex(restored._support_index)
        suffix = event_fixture_slice(fixture, 6, 12)
        resumed = restored(suffix["events"].to(device),
                           event_provider(restored, restored_index, suffix, device),
                           timestamps=suffix["timestamps"],
                           windows=suffix["count_windows"], prune=True,
                           threshold=GATE_THRESHOLD,
                           derivative_seed=int(fixture["seed"]))
        if not torch.allclose(full_reference, resumed.detach().cpu(),
                              atol=3e-4, rtol=3e-4):
            raise RuntimeError("mid-stream checkpoint changed resumed event output")
        print("checkpoint_event_stream=events prefix=6 suffix=6 ok=1")


def shuffled_support(support: SupportSpec, seed: int) -> SupportSpec:
    """Keep edge count and endpoint degrees while breaking route identity."""
    destinations = list(support.destinations)
    random.Random(seed).shuffle(destinations)
    if destinations == list(support.destinations):
        destinations = destinations[1:] + destinations[:1]
    return SupportSpec(support.edge_ids, support.sources, tuple(destinations),
                       support.structure_epoch)


def dormant_sample_summary(module):
    evidence = module.application_state.get("derivative_sampler", {})
    selected = sum(evidence.get("selected_dormant_counts", ()))
    eligible = sum(evidence.get("eligible_dormant_counts", ()))
    return selected, eligible


def discover_route(module, observations, seed: int):
    """Run bounded native proposal, then rank only novel directed pairs."""
    rows = observations.reshape(-1, observations.shape[-1])
    visible = rows.reshape(-1)
    visible = visible[visible >= 0].to(torch.int64).cpu().numpy()
    entity_ids = np.arange(1, ENTITIES + 1, dtype=np.int64)
    destination_seeds = np.asarray(module.support.destinations, dtype=np.int64) + 1
    posting_seeds = np.unique(np.concatenate((visible + 1, destination_seeds))).astype(
        np.int64, copy=False)
    observed_sources = sorted(set(int(value) for value in visible if 0 <= value < 16))
    possible_destinations = sorted(set(map(int, module.support.destinations)))
    route_seeds = [(source + 1, destination + 1)
                   for source in observed_sources
                   for destination in possible_destinations]
    random.Random(seed).shuffle(route_seeds)
    route_seeds = route_seeds[:64]
    if visible.size >= 2:
        pairs = torch.stack((rows[:, :-1], rows[:, 1:]), dim=-1)
        valid_pairs = ((pairs[..., 0] >= 0) & (pairs[..., 1] >= 0)
                       & (pairs[..., 0] != pairs[..., 1]))
        adjacent = pairs[valid_pairs].to(torch.int64).cpu().numpy() + 1
        recent_pairs = adjacent[-64:].tolist()
    else:
        recent_pairs = []
    coactivations = np.ascontiguousarray(
        recent_pairs + route_seeds, dtype=np.int64).reshape(-1, 2)
    known_participants = np.zeros((len(module.support.edge_ids), 3), dtype=np.int64)
    known_participants[:, 0] = np.asarray(module.support.sources, dtype=np.int64) + 1
    known_participants[:, 1] = np.asarray(module.support.destinations, dtype=np.int64) + 1
    proposals = _torch.propose_native_support(
        entity_ids, np.arange(1, MECHANISMS + 1, dtype=np.int64),
        posting_seeds, coactivations, 2, seed, 64, 2048,
        np.ones(len(module.support.edge_ids), dtype=np.int64),
        known_participants)
    existing = set(zip(module.support.sources, module.support.destinations))
    valid = []
    for row in proposals["participant_ids"].tolist():
        source, destination = int(row[0]) - 1, int(row[1]) - 1
        if (0 <= source < 16 and 16 <= destination < 24
                and (source, destination) not in existing):
            valid.append((source, destination))
    valid = sorted(set(valid))
    if not valid:
        return None, {key: proposals[key] for key in (
            "attempts", "candidate_space_upper_bound",
            "missed_candidate_lower_bound", "sampled_coverage_lower_bound",
            "candidate_space_upper_bound_saturated", "missed_candidate_count_exact",
            "next_seed")}, 0
    scores = module.score_pair_proposals(
        [pair[0] for pair in valid], [pair[1] for pair in valid])
    selected = int(scores.argmax())
    score = float(scores[selected].cpu())
    evidence = {key: proposals[key] for key in (
        "attempts", "candidate_space_upper_bound",
        "missed_candidate_lower_bound", "sampled_coverage_lower_bound",
        "candidate_space_upper_bound_saturated", "missed_candidate_count_exact",
        "next_seed")}
    return (valid[selected][0], valid[selected][1], score), evidence, len(valid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ce-geo-image", type=Path,
                        help="optional prebuilt image; otherwise compile it from the support")
    parser.add_argument("--source-group-members", type=int, nargs="+")
    parser.add_argument("--destination-group-members", type=int, nargs="+")
    parser.add_argument("--geometry-edge-ids", type=int, nargs="+",
                        help="image edge IDs in logical support order for a supplied image")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--soft-epochs", type=int, default=10)
    parser.add_argument("--adapt-epoch", type=int, default=20,
                        help="promote one explored route between windows; use -1 to disable")
    parser.add_argument("--support-growth-budget", type=int, default=16,
                        help="maximum support additions reserved in each prepared projection")
    parser.add_argument("--seed", type=int, default=421)
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    torch.manual_seed(args.seed)
    device = torch.device("cuda", args.device)
    support = pair_support(args.seed)
    axis_words = (101, 102, 201, 202, 301, 302, 401, 402)
    preparation_costs = {"geometry_compile_seconds": 0.0,
                         "slice_prepare_seconds": 0.0}
    if args.soft_epochs < 0 or args.soft_epochs > args.epochs:
        parser.error("--soft-epochs must be between zero and --epochs")
    if args.support_growth_budget < 0:
        parser.error("--support-growth-budget must be nonnegative")

    def prepare(kind, current_support, discriminator, seed):
        if args.ce_geo_image is None:
            compile_started = time.perf_counter()
            compiled = compile_sparse_support_ce_geo(
                current_support, source_count=ENTITIES, destination_count=ENTITIES,
                projection_id=discriminator)
            preparation_costs["geometry_compile_seconds"] += time.perf_counter() - compile_started
            image = compiled["image"]
            source_members = compiled["source_group_members"]
            destination_members = compiled["destination_group_members"]
            geometry_edge_ids = compiled["geometry_edge_ids_in_support_order"]
        else:
            if args.source_group_members is None or args.destination_group_members is None:
                parser.error("prebuilt images require both group member arrays")
            image = args.ce_geo_image.read_bytes()
            source_members = args.source_group_members
            destination_members = args.destination_group_members
            geometry_edge_ids = args.geometry_edge_ids
        local_axis = axis_words[:-1] + (axis_words[-1] + discriminator,)
        max_cells = 64
        max_primal_items = max_cells * (
            len(current_support.edge_ids) + args.support_growth_budget)
        prepare_started = time.perf_counter()
        module = prepare_sparse_slice(
            kind, current_support, entities=ENTITIES, mechanisms=MECHANISMS,
            rank_count=8, device=args.device, parameter_axis_words=local_axis,
            seed=seed, ce_geo_image=image,
            source_group_members=source_members,
            destination_group_members=destination_members,
            geometry_edge_ids=geometry_edge_ids,
            projection_id=discriminator, max_cells=max_cells,
            max_primal_items=max_primal_items)
        preparation_costs["slice_prepare_seconds"] += time.perf_counter() - prepare_started
        module.application_state["prepared_capacity"] = {
            "max_cells": max_cells,
            "max_primal_items": max_primal_items,
            "support_growth_budget": args.support_growth_budget,
            "initial_support_edges": len(current_support.edge_ids),
        }
        return module

    ranked = prepare(RankedGeneSlice, support, 1, args.seed)
    events = prepare(KaryoplastEventSlice, support, 2, args.seed + 1)
    ranked_soft = prepare(RankedGeneSlice, support, 5, args.seed)
    events_soft = prepare(KaryoplastEventSlice, support, 6, args.seed + 1)
    # Shuffled-support controls preserve the number of edges and degree profile.
    shuffled = shuffled_support(support, args.seed + 17)
    ranked_shuffled = prepare(RankedGeneSlice, shuffled, 3, args.seed)
    events_shuffled = prepare(KaryoplastEventSlice, shuffled, 4, args.seed + 1)
    ranked_index = _torch.NativeReverseIndex(ranked._support_index)
    event_index = _torch.NativeReverseIndex(events._support_index)
    ranked_shuffled_index = _torch.NativeReverseIndex(ranked_shuffled._support_index)
    events_shuffled_index = _torch.NativeReverseIndex(events_shuffled._support_index)
    ranked_soft_index = _torch.NativeReverseIndex(ranked_soft._support_index)
    events_soft_index = _torch.NativeReverseIndex(events_soft._support_index)
    ranked_optimizer = torch.optim.Adam([ranked.parameters_flat], lr=0.02)
    event_optimizer = torch.optim.Adam([events.parameters_flat], lr=0.02)
    ranked_shuffled_optimizer = torch.optim.Adam(
        [ranked_shuffled.parameters_flat], lr=0.02)
    events_shuffled_optimizer = torch.optim.Adam(
        [events_shuffled.parameters_flat], lr=0.02)
    ranked_soft_optimizer = torch.optim.Adam([ranked_soft.parameters_flat], lr=0.02)
    events_soft_optimizer = torch.optim.Adam([events_soft.parameters_flat], lr=0.02)

    torch.cuda.reset_peak_memory_stats(device)
    print(f"preparation geometry_compile_seconds={preparation_costs['geometry_compile_seconds']:.6f} "
          f"slice_prepare_seconds={preparation_costs['slice_prepare_seconds']:.6f}")

    for epoch in range(args.epochs):
        epoch_started = time.perf_counter()
        ranked_data = make_masked_gene_fixture(seed=args.seed + epoch, cells=32,
                                              selected=8, entities=ENTITIES)
        event_data = make_next_event_fixture(seed=args.seed + epoch, cells=32,
                                             steps=12, entities=ENTITIES)
        # Hidden validator fields are intentionally never handed to train_*.
        prune = epoch >= args.soft_epochs
        ranked_loss, ranked_admission = train_ranked(
            ranked, ranked_index, ranked_data, device, ranked_optimizer, prune=prune)
        event_loss, event_admission = train_events(
            events, event_index, event_data, device, event_optimizer, prune=prune)
        train_ranked(ranked_shuffled, ranked_shuffled_index, ranked_data, device,
                     ranked_shuffled_optimizer, prune=prune)
        train_events(events_shuffled, events_shuffled_index, event_data, device,
                     events_shuffled_optimizer, prune=prune)
        train_ranked(ranked_soft, ranked_soft_index, ranked_data, device,
                     ranked_soft_optimizer, prune=False)
        train_events(events_soft, events_soft_index, event_data, device,
                     events_soft_optimizer, prune=False)
        if epoch == args.adapt_epoch:
            # Proposal effort is bounded; discovered routes are scored by
            # shared learned embeddings and published between tape windows.
            ranked_route, ranked_evidence, ranked_count = discover_route(
                ranked, ranked_data["genes"], args.seed + epoch)
            event_route, event_evidence, event_count = discover_route(
                events, event_data["events"], args.seed + epoch + 1)
            for module, route, evidence, count in (
                    (ranked, ranked_route, ranked_evidence, ranked_count),
                    (events, event_route, event_evidence, event_count)):
                module.application_state["exploration"] = {
                    **evidence, "proposal_count": count,
                    "seed": args.seed + epoch,
                }
                label = "ranked" if module is ranked else "events"
                print(f"proposal={label} proposal_count={count} attempts={evidence['attempts']} "
                      f"candidate_upper_bound={evidence['candidate_space_upper_bound']} "
                      f"upper_bound_saturated={int(evidence['candidate_space_upper_bound_saturated'])} "
                      f"missed_lower_bound={evidence['missed_candidate_lower_bound']} "
                      f"missed_exact={int(evidence['missed_candidate_count_exact'])} "
                      f"sampled_coverage_lower_bound={evidence['sampled_coverage_lower_bound']:.6f}")
                if route is None:
                    print(f"promotion={label} route=none")
                    continue
                source, destination, proposal_score = route
                edge_id = max(module.support.edge_ids) + 1
                module.stage_promotion(
                    edge_id, source, destination, discovery_round_id=epoch,
                    scorer_generation=_torch.sparse_generation(module._native),
                    proposal_score=proposal_score)
                module.publish_support(module.structure_epoch + 1)
                print(f"promotion={label} source={source} destination={destination} "
                      f"score={proposal_score:.6f} edges={len(module.support.edge_ids)} "
                      f"epoch={module.structure_epoch}")
            ranked_index = _torch.NativeReverseIndex(ranked._support_index)
            event_index = _torch.NativeReverseIndex(events._support_index)
        print(f"epoch={epoch:03d} prune={int(prune)} gate_threshold={GATE_THRESHOLD:.2f} "
              f"ranked_loss={ranked_loss:.6f} ranked_admission={ranked_admission:.4f} "
              f"event_loss={event_loss:.6f} event_admission={event_admission:.4f} "
              f"epoch_seconds={time.perf_counter() - epoch_started:.3f}")

    ranked_heldout = make_masked_gene_fixture(seed=args.seed + 100_000,
                                               cells=64, selected=8,
                                               entities=ENTITIES)
    event_heldout = make_next_event_fixture(seed=args.seed + 200_000,
                                             cells=64, steps=12,
                                             entities=ENTITIES)
    for label, module, index in (
            ("ranked", ranked, ranked_index),
            ("ranked_soft_only", ranked_soft, ranked_soft_index),
            ("ranked_shuffled", ranked_shuffled, ranked_shuffled_index)):
        soft = evaluate_ranked(module, index, ranked_heldout, device, prune=False)
        pruned = evaluate_ranked(module, index, ranked_heldout, device, prune=True)
        print(f"heldout={label} soft_loss={soft[0]:.6f} soft_acc={soft[1]:.4f} "
              f"soft_admission={soft[2]:.4f} pruned_loss={pruned[0]:.6f} "
              f"pruned_acc={pruned[1]:.4f} pruned_admission={pruned[2]:.4f} "
              f"dormant_sample={dormant_sample_summary(module)}")
    continuation_ranked = make_masked_gene_fixture(
        seed=args.seed + 300_000, cells=32, selected=8, entities=ENTITIES)
    continuation_events = make_next_event_fixture(
        seed=args.seed + 400_000, cells=32, steps=12, entities=ENTITIES)
    verify_checkpoint_continuation(
        prepare, support, device, args.seed, ranked, ranked_optimizer,
        events, event_optimizer, continuation_ranked, continuation_events)
    for label, module, index in (
            ("events", events, event_index),
            ("events_soft_only", events_soft, events_soft_index),
            ("events_shuffled", events_shuffled, events_shuffled_index)):
        soft = evaluate_events(module, index, event_heldout, device, prune=False)
        pruned = evaluate_events(module, index, event_heldout, device, prune=True)
        print(f"heldout={label} soft_loss={soft[0]:.6f} soft_acc={soft[1]:.4f} "
              f"soft_admission={soft[2]:.4f} pruned_loss={pruned[0]:.6f} "
              f"pruned_acc={pruned[1]:.4f} pruned_admission={pruned[2]:.4f} "
              f"dormant_sample={dormant_sample_summary(module)}")
    verify_event_stream_checkpoint(
        prepare, support, device, args.seed, events, event_optimizer, event_index)
    modules = (ranked, events, ranked_soft, events_soft,
               ranked_shuffled, events_shuffled)
    print(f"costs route_cache_refresh_seconds={sum(float(m.application_state.get('route_cache_refresh_seconds', 0.0)) for m in modules):.6f} "
          f"support_publish_seconds={sum(float(m.application_state.get('support_publish_seconds', 0.0)) for m in modules):.6f} "
          f"peak_allocated_bytes={torch.cuda.max_memory_allocated(device)}")


if __name__ == "__main__":
    main()
