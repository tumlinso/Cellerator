"""Deterministic, scalable sparse-support fixtures for host-side experiments.

The generated arrays describe synthetic directed edges.  They are engineering
fixtures, not biological data.  All edge results are expressed in stable
logical IDs; a separate permutation supplies a physical storage-layout control.
Generation and storage are linear in edges plus emitted observations, and no
edge-by-edge or entity-by-entity dense matrix is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ScaledSparseFixture:
    """Arrays and metadata for one reproducible sparse-support workload."""

    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]

    def __getitem__(self, name: str) -> np.ndarray:
        return self.arrays[name]


def _csr_from_keys(keys: np.ndarray, key_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Build stable key-to-row CSR in O(number of entries + key_count)."""
    counts = np.bincount(keys, minlength=key_count).astype(np.int64, copy=False)
    indptr = np.empty(key_count + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    cursors = indptr[:-1].copy()
    values = np.empty(keys.size, dtype=np.int64)
    for ordinal, key in enumerate(keys):
        pos = cursors[key]
        values[pos] = ordinal
        cursors[key] += 1
    return indptr, values


def _unique_directed_relation(edges: int, entities: int, modules: int,
                              max_source_degree: int, rng: np.random.Generator,
                              local_fraction: float = 0.65
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build E unique, non-self directed pairs with bounded skewed outdegree."""
    if entities < 2:
        raise ValueError("a directed non-self relation needs at least two entities")
    cap = min(max_source_degree, entities - 1)
    if edges > entities * cap:
        raise ValueError(f"{edges} unique pairs exceed entity/capacity bound {entities * cap}")
    active_count = min(edges, entities)
    active = rng.choice(entities, size=active_count, replace=False)
    degree = np.zeros(entities, dtype=np.int64)
    degree[active] = 1
    remaining = edges - active_count
    weights = rng.pareto(1.35, size=entities) + 1.0
    hot_count = max(1, min(16, entities // 128 or 1))
    hot = rng.choice(entities, size=hot_count, replace=False)
    weights[hot] *= 16.0
    while remaining:
        available = degree < cap
        available_ids = np.flatnonzero(available)
        probabilities = weights[available]
        probabilities /= probabilities.sum()
        proposed = rng.multinomial(remaining, probabilities)
        accepted = np.minimum(proposed, cap - degree[available])
        degree[available_ids] += accepted
        used = int(accepted.sum())
        if used == 0:
            raise RuntimeError("degree-cap redistribution made no progress")
        remaining -= used

    source = np.repeat(np.arange(entities, dtype=np.int64), degree)
    destination = np.empty(edges, dtype=np.int64)
    entity_modules = np.arange(entities, dtype=np.int64) % modules
    module_members = [np.flatnonzero(entity_modules == module)
                      for module in range(modules)]
    cursor = 0
    for entity, outdegree in enumerate(degree):
        count = int(outdegree)
        if not count:
            continue
        local_pool = module_members[int(entity_modules[entity])]
        local_pool = local_pool[local_pool != entity]
        local_count = (count if modules == 1 else
                       min(int(count * local_fraction), local_pool.size))
        if local_count:
            local_targets = rng.choice(local_pool, size=local_count, replace=False)
            destination[cursor:cursor + local_count] = local_targets
        cross_count = count - local_count
        if cross_count:
            modulus = entities - 1
            stride = int(rng.integers(1, modulus + 1))
            while math.gcd(stride, modulus) != 1:
                stride = int(rng.integers(1, modulus + 1))
            start = int(rng.integers(modulus))
            scan = 0
            found: list[np.ndarray] = []
            found_count = 0
            while found_count < cross_count:
                width = min(modulus - scan, max(32, 2 * (cross_count - found_count)))
                ranks = (start + np.arange(scan, scan + width, dtype=np.int64) * stride) % modulus
                candidates = ranks + (ranks >= entity)
                candidates = candidates[entity_modules[candidates] != entity_modules[entity]]
                take = min(cross_count - found_count, candidates.size)
                if take:
                    found.append(candidates[:take])
                    found_count += take
                scan += width
                if scan >= modulus and found_count < cross_count:
                    raise RuntimeError("cross-module neighborhood is too small for unique pairs")
            destination[cursor + local_count:cursor + count] = np.concatenate(found)
        cursor += count
    return source, destination, degree


def _csr_union(posting_indptr: np.ndarray, posting_ids: np.ndarray,
               query_indptr: np.ndarray, terms: np.ndarray, row_count: int,
               edge_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact sorted stable-ID unions for query rows without dense matrices."""
    q_indptr = np.empty(row_count + 1, dtype=np.int64)
    q_indptr[0] = 0
    rows: list[np.ndarray] = []
    mark = np.zeros(edge_count, dtype=np.bool_)
    for row in range(row_count):
        begin, end = int(query_indptr[row]), int(query_indptr[row + 1])
        touched: list[np.ndarray] = []
        for entity in terms[begin:end]:
            lo, hi = int(posting_indptr[entity]), int(posting_indptr[entity + 1])
            touched.append(posting_ids[lo:hi])
        if touched:
            merged = np.concatenate(touched)
            if merged.size:
                mark[merged] = True
                union = np.flatnonzero(mark).astype(np.int64, copy=False)
                mark[union] = False
            else:
                union = np.empty(0, dtype=np.int64)
        else:
            union = np.empty(0, dtype=np.int64)
        rows.append(union)
        q_indptr[row + 1] = q_indptr[row] + union.size
    result = np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)
    return q_indptr, result


def _digest_arrays(arrays: dict[str, np.ndarray], names: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name in names:
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def make_scaled_fixture(
    *,
    edges: int = 1_000_000,
    entities: int = 20_000,
    cells: int = 512,
    seed: int = 20261027,
    modules: int = 256,
    max_source_degree: int = 512,
    local_edge_fraction: float = 0.65,
    events_per_cell: int = 64,
    rank_count: int = 8,
    cohort_size: int = 16,
    layout: str = "identity",
    query_sizes: tuple[int, ...] | None = None,
) -> ScaledSparseFixture:
    """Generate support, query, event, gate, and ranked-set workloads.

    Source entities use a hub-plus-Zipf distribution.  Module anchors create
    shared topology while participant entities add an independent heavy-tailed
    incidence distribution.  Cells in a cohort share selected query support;
    their values, event streams, and gate states remain independently seeded.
    """
    if min(edges, entities, cells, modules, max_source_degree, rank_count,
           events_per_cell, cohort_size) < 1:
        raise ValueError("all shape parameters must be positive")
    if rank_count > entities:
        raise ValueError("rank_count must not exceed entity count")
    if not 0.0 <= local_edge_fraction <= 1.0:
        raise ValueError("local_edge_fraction must be between zero and one")
    if modules > entities:
        raise ValueError("modules must not exceed entities")
    if layout not in {"identity", "shuffled"}:
        raise ValueError("layout must be 'identity' or 'shuffled'")
    rng = np.random.default_rng(seed)
    edge_ord = np.arange(edges, dtype=np.int64)
    source, participant, source_degree = _unique_directed_relation(
        edges, entities, modules, max_source_degree, rng, local_edge_fraction)
    module_id = (source % modules).astype(np.int32)

    source_indptr, source_ids = _csr_from_keys(source, entities)
    all_keys = np.concatenate((source, participant))
    all_indptr = np.empty(entities + 1, dtype=np.int64)
    all_counts = np.bincount(all_keys, minlength=entities)
    all_indptr[0] = 0
    np.cumsum(all_counts, out=all_indptr[1:])
    all_cursor = all_indptr[:-1].copy()
    all_ids = np.empty(all_keys.size, dtype=np.int64)
    for ordinal, (src, dst) in enumerate(zip(source, participant)):
        pos = all_cursor[src]
        all_ids[pos] = ordinal
        all_cursor[src] += 1
        pos = all_cursor[dst]
        all_ids[pos] = ordinal
        all_cursor[dst] += 1
    # A duplicated source/participant pair is prevented above, so each edge
    # appears at most once in an all-role entity row.

    if query_sizes is None:
        query_sizes = tuple(sorted({1, min(8, entities), min(32, entities), min(128, entities),
                                    min(1024, entities), min(2048, entities)}))
    if any(q < 0 or q > entities for q in query_sizes):
        raise ValueError("query sizes must be between zero and entity count")
    query_entities: list[int] = []
    query_indptr = [0]
    query_case_sizes: list[int] = []
    query_case_names: list[str] = []
    # Explicit empty/overlap coverage is useful for the small oracle fixture.
    if 0 in query_sizes:
        query_case_sizes.append(0)
        query_case_names.append("empty")
        query_indptr.append(0)
    for q in query_sizes:
        if q == 0:
            continue
        chosen = rng.choice(entities, size=q, replace=False)
        query_entities.extend(chosen.tolist())
        query_indptr.append(len(query_entities))
        query_case_sizes.append(q)
        query_case_names.append(f"unique_{q}")
        # Duplicate-term control keeps the same unique support while exercising
        # deduplication in consumers.
        if q >= 2:
            duplicate_count = min(4, q - 1)
            query_entities.extend(chosen[:q - duplicate_count].tolist())
            query_entities.extend(chosen[:duplicate_count].tolist())
            query_indptr.append(len(query_entities))
            query_case_sizes.append(q)
            query_case_names.append(f"duplicate_{q}")
    query_entities_array = np.asarray(query_entities, dtype=np.int64)
    query_indptr_array = np.asarray(query_indptr, dtype=np.int64)
    query_edge_indptr, query_edge_ids = _csr_union(
        source_indptr, source_ids, query_indptr_array, query_entities_array,
        len(query_case_sizes), edges)

    cohort_ids = np.arange(cells, dtype=np.int32) // cohort_size
    cohort_query: dict[int, np.ndarray] = {}
    cell_query_indptr = [0]
    cell_query_entities: list[int] = []
    cell_value_seed = np.empty(cells, dtype=np.uint64)
    cell_selected_values: list[np.ndarray] = []
    selected_count = min(24, entities)
    for cell in range(cells):
        cohort = int(cohort_ids[cell])
        if cohort not in cohort_query:
            cohort_query[cohort] = rng.choice(entities, size=selected_count, replace=False)
        selected = np.sort(cohort_query[cohort])
        cell_query_entities.extend(selected.tolist())
        cell_query_indptr.append(len(cell_query_entities))
        cell_value_seed[cell] = rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64)
        # Independent values with common selected support make cohort reuse
        # explicit without sharing mutable per-cell state.
        cell_selected_values.append(rng.standard_normal(selected_count).astype(np.float32))

    # Per-cell burst streams contain repeated edge occurrences and explicit
    # monotonic timestamps.  Rows are cell-major; timestamps are local seconds.
    event_cells = np.repeat(np.arange(cells, dtype=np.int32), events_per_cell)
    event_edge = np.empty(event_cells.size, dtype=np.int64)
    event_time = np.empty(event_cells.size, dtype=np.int64)
    for cell in range(cells):
        start = cell * events_per_cell
        stop = start + events_per_cell
        burst_count = min(4, max(1, events_per_cell // 8))
        query_lo, query_hi = cell_query_indptr[cell:cell + 2]
        active_postings = [source_ids[source_indptr[gene]:source_indptr[gene + 1]]
                           for gene in cell_query_entities[query_lo:query_hi]]
        active_edges = np.concatenate(active_postings)
        if active_edges.size == 0:
            active_edges = edge_ord
        burst_edges = rng.choice(active_edges, size=burst_count, replace=True)
        picks = rng.integers(0, burst_count, size=events_per_cell)
        fresh_active = rng.choice(active_edges, size=events_per_cell, replace=True)
        fresh_background = rng.integers(0, edges, size=events_per_cell, dtype=np.int64)
        fresh = np.where(rng.random(events_per_cell) < 0.5,
                         fresh_active, fresh_background)
        burst_mask = rng.random(events_per_cell) < 0.55
        event_edge[start:stop] = np.where(burst_mask, burst_edges[picks], fresh)
        event_time[start:stop] = np.arange(events_per_cell, dtype=np.int64) * 10
    window_start = np.asarray([0, events_per_cell // 2], dtype=np.int32)
    window_stop = np.asarray([events_per_cell // 2, events_per_cell], dtype=np.int32)
    window_indptr = [0]
    window_event_indices: list[int] = []
    event_window_query_indices: list[int] = []
    event_window_query_indptr = [0]
    window_cells: list[int] = []
    window_ids: list[int] = []
    for cell in range(cells):
        base = cell * events_per_cell
        query_lo, query_hi = cell_query_indptr[cell:cell + 2]
        selected_query_entities = np.asarray(
            cell_query_entities[query_lo:query_hi], dtype=np.int64)
        for w, (lo, hi) in enumerate(zip(window_start, window_stop)):
            window_cells.append(cell)
            window_ids.append(w)
            event_indices = np.arange(base + int(lo), base + int(hi), dtype=np.int64)
            window_event_indices.extend(event_indices.tolist())
            matches = event_indices[np.isin(source[event_edge[event_indices]],
                                            selected_query_entities)]
            event_window_query_indices.extend(matches.tolist())
            event_window_query_indptr.append(len(event_window_query_indices))
            window_indptr.append(len(window_event_indices))

    # Local observed predicates are sparse per cell and keyed by gene ID.
    observed_predicate_values = np.concatenate([
        rng.integers(0, 2, size=selected_count, dtype=np.uint8)
        for _ in range(cells)])
    local_change_cell: list[int] = []
    local_change_gene: list[int] = []
    local_change_time: list[int] = []
    local_change_value: list[int] = []
    local_change_group: list[int] = []
    local_changes_per_cell = min(3, selected_count)
    for cell in range(cells):
        qlo, qhi = cell_query_indptr[cell:cell + 2]
        selected = np.asarray(cell_query_entities[qlo:qhi], dtype=np.int64)
        changed = rng.choice(selected, size=local_changes_per_cell, replace=False)
        state_lo = cell * selected_count
        state_by_gene = dict(zip(selected.tolist(),
                                 observed_predicate_values[state_lo:state_lo + selected_count].tolist()))
        times = np.sort(rng.choice(max(1, events_per_cell * 10),
                                   size=local_changes_per_cell, replace=False))
        for local_index, (gene_id, timestamp) in enumerate(zip(changed, times)):
            local_change_cell.append(cell)
            local_change_gene.append(int(gene_id))
            local_change_time.append(int(timestamp))
            local_change_value.append(1 - state_by_gene[int(gene_id)])
            local_change_group.append(cell * local_changes_per_cell + local_index)
    local_change_cell_a = np.asarray(local_change_cell, dtype=np.int32)
    local_change_gene_a = np.asarray(local_change_gene, dtype=np.int64)
    local_change_time_a = np.asarray(local_change_time, dtype=np.int64)
    local_change_value_a = np.asarray(local_change_value, dtype=np.uint8)
    local_change_group_a = np.asarray(local_change_group, dtype=np.int32)

    # One latent predicate is shared by each source module. Cohort-synchronized
    # flips change many module-local edges while retaining explicit dependency IDs.
    latent_values = rng.integers(0, 2, size=(cells, modules), dtype=np.uint8)
    latent_change_cell: list[int] = []
    latent_change_module: list[int] = []
    latent_change_time: list[int] = []
    latent_change_value: list[int] = []
    latent_change_group: list[int] = []
    latent_changes_per_cohort = min(modules, max(1, modules // 16))
    for cohort in np.unique(cohort_ids):
        cohort_cells = np.flatnonzero(cohort_ids == cohort)
        selected_modules = rng.choice(modules, size=latent_changes_per_cohort,
                                      replace=False)
        for change_index, module_id_value in enumerate(selected_modules):
            timestamp = int((change_index + 1) * events_per_cell * 10
                            // (latent_changes_per_cohort + 1))
            change_group = int(cohort * latent_changes_per_cohort + change_index)
            module_id_int = int(module_id_value)
            for cell in cohort_cells:
                latent_change_cell.append(int(cell))
                latent_change_module.append(module_id_int)
                latent_change_time.append(timestamp)
                latent_change_value.append(1 - int(latent_values[cell, module_id_int]))
                latent_change_group.append(change_group)
    latent_change_cell_a = np.asarray(latent_change_cell, dtype=np.int32)
    latent_change_module_a = np.asarray(latent_change_module, dtype=np.int32)
    latent_change_time_a = np.asarray(latent_change_time, dtype=np.int64)
    latent_change_value_a = np.asarray(latent_change_value, dtype=np.uint8)
    latent_change_group_a = np.asarray(latent_change_group, dtype=np.int32)

    ranked_entities = np.empty((cells, rank_count), dtype=np.int64)
    ranked_cohort_support: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    shared_rank_count = int(rank_count * 0.75)
    for cell in range(cells):
        cohort = int(cohort_ids[cell])
        if cohort not in ranked_cohort_support:
            shared = rng.choice(entities, size=shared_rank_count, replace=False)
            available = np.setdiff1d(np.arange(entities, dtype=np.int64), shared,
                                     assume_unique=False)
            ranked_cohort_support[cohort] = (shared, available)
        shared, available = ranked_cohort_support[cohort]
        residual_count = rank_count - shared_rank_count
        residual = rng.choice(available, size=residual_count, replace=False)
        ranked_entities[cell] = rng.permutation(np.concatenate((shared, residual)))
    ranked_position = np.broadcast_to(np.arange(rank_count, dtype=np.int32),
                                      (cells, rank_count)).copy()

    physical_to_logical = (rng.permutation(edges).astype(np.int64)
                           if layout == "shuffled" else edge_ord.copy())
    logical_to_physical = np.empty(edges, dtype=np.int64)
    logical_to_physical[physical_to_logical] = edge_ord
    event_offsets = np.arange(cells + 1, dtype=np.int64) * events_per_cell
    count_window_bounds = np.tile(
        np.asarray([events_per_cell // 2, events_per_cell], dtype=np.int32),
        (cells, 1))
    time_window_bounds = np.column_stack((
        np.full(cells, (events_per_cell // 2) * 10, dtype=np.int64),
        np.full(cells, events_per_cell * 10, dtype=np.int64)))
    rank_offsets = np.arange(cells + 1, dtype=np.int64) * rank_count
    edge_condition_indptr = np.arange(edges + 1, dtype=np.int64) * 2
    edge_observed_predicate_id = source.copy()
    edge_latent_predicate_id = (entities + module_id.astype(np.int64))
    edge_condition_predicate_id = np.column_stack((edge_observed_predicate_id,
                                                    edge_latent_predicate_id)).reshape(-1)
    window_event_indices_array = np.asarray(window_event_indices, dtype=np.int64)
    event_window_observed_genes = source[event_edge[window_event_indices_array]]
    event_window_observed_edges = event_edge[window_event_indices_array] + 1
    arrays: dict[str, np.ndarray] = {
        "logical_edge_id": edge_ord + 1,
        "edge_ids": edge_ord + 1,
        "edge_ordinal": edge_ord,
        "source_entity_id": source,
        "source_ids": source,
        "participant_entity_id": participant,
        "destination_ids": participant,
        "edge_module_id": module_id.astype(np.int32),
        "edge_observed_predicate_id": edge_observed_predicate_id,
        "edge_latent_predicate_id": edge_latent_predicate_id,
        "source_edge_indptr": source_indptr,
        "source_edge_ids": source_ids + 1,
        "source_edge_ordinals": source_ids,
        "entity_edge_indptr": all_indptr,
        "entity_edge_ids": all_ids + 1,
        "query_role": np.zeros(len(query_case_sizes), dtype=np.int8),
        "query_indptr": query_indptr_array,
        "query_entities": query_entities_array,
        "query_case_size": np.asarray(query_case_sizes, dtype=np.int32),
        "query_edge_indptr": query_edge_indptr,
        "query_edge_ids": query_edge_ids + 1,
        "query_edge_ordinals": query_edge_ids,
        "cell_cohort_id": cohort_ids,
        "cell_query_indptr": np.asarray(cell_query_indptr, dtype=np.int64),
        "cell_query_entities": np.asarray(cell_query_entities, dtype=np.int64),
        "cell_query_terms": np.asarray(cell_query_entities, dtype=np.int64),
        "query_offsets": np.asarray(cell_query_indptr, dtype=np.int64),
        "query_terms": np.asarray(cell_query_entities, dtype=np.int64),
        "cell_value_seed": cell_value_seed,
        "cell_selected_values": np.concatenate(cell_selected_values),
        "cell_observed_predicate_offsets": np.asarray(cell_query_indptr, dtype=np.int64),
        "cell_observed_predicate_ids": np.asarray(cell_query_entities, dtype=np.int64),
        "cell_observed_predicate_values": observed_predicate_values,
        "cell_observed_predicate_view_id": np.zeros(cells, dtype=np.uint8),
        "cell_query_candidate_counts": np.asarray([
            int(source_degree[cell_query_entities[cell_query_indptr[cell]:
                                                 cell_query_indptr[cell + 1]]].sum())
            for cell in range(cells)], dtype=np.int64),
        "event_cell_id": event_cells,
        "event_edge_id": event_edge + 1,
        "event_offsets": event_offsets,
        "event_entity_ids": source[event_edge],
        "event_timestamps": event_time,
        "event_ordinals": np.tile(np.arange(events_per_cell, dtype=np.int32), cells),
        "event_timestamp": event_time,
        "count_window_bounds": count_window_bounds,
        "time_window_bounds": time_window_bounds,
        "window_cell_id": np.asarray(window_cells, dtype=np.int32),
        "window_id": np.asarray(window_ids, dtype=np.int8),
        "window_start_ordinal": np.tile(window_start, cells),
        "window_stop_ordinal_exclusive": np.tile(window_stop, cells),
        "window_event_indptr": np.asarray(window_indptr, dtype=np.int64),
        "window_event_indices": np.asarray(window_event_indices, dtype=np.int64),
        "event_window_query_indptr": np.asarray(event_window_query_indptr, dtype=np.int64),
        "event_window_query_indices": np.asarray(event_window_query_indices, dtype=np.int64),
        "event_window_query_edge_ids": event_edge[np.asarray(
            event_window_query_indices, dtype=np.int64)] + 1,
        "event_window_observed_indptr": np.asarray(window_indptr, dtype=np.int64),
        "event_window_observed_gene_ids": event_window_observed_genes,
        "event_window_observed_edge_ids": event_window_observed_edges,
        "event_window_observed_values": np.ones(window_event_indices_array.size,
                                                 dtype=np.uint8),
        "event_window_observed_view_id": np.full(2 * cells, 2, dtype=np.uint8),
        "latent_predicate_values": latent_values,
        "predicate_values": latent_values,
        "edge_condition_indptr": edge_condition_indptr,
        "edge_condition_predicate_id": edge_condition_predicate_id,
        "gate_initial_state": latent_values,
        "local_gene_change_cell_id": local_change_cell_a,
        "local_gene_change_gene_id": local_change_gene_a,
        "local_gene_change_timestamp": local_change_time_a,
        "local_gene_change_value": local_change_value_a,
        "local_gene_change_group_id": local_change_group_a,
        "latent_change_cell_id": latent_change_cell_a,
        "latent_change_module_id": latent_change_module_a,
        "latent_change_timestamp": latent_change_time_a,
        "latent_change_value": latent_change_value_a,
        "latent_change_group_id": latent_change_group_a,
        # Compatibility aliases for the earlier gate-only loader.
        "gate_change_cell_id": latent_change_cell_a,
        "gate_change_predicate_id": latent_change_module_a,
        "gate_change_timestamp": latent_change_time_a,
        "gate_change_value": latent_change_value_a,
        "gate_change_group_id": latent_change_group_a,
        "ranked_entity_id": ranked_entities,
        "ranked_position": ranked_position,
        "rank_offsets": rank_offsets,
        "rank_entity_ids": ranked_entities.reshape(-1),
        "rank_positions": ranked_position.reshape(-1),
        "ranked_presence_offsets": rank_offsets,
        "ranked_presence_entity_ids": ranked_entities.reshape(-1),
        "ranked_presence_values": np.ones(cells * rank_count, dtype=np.uint8),
        "ranked_presence_view_id": np.ones(cells, dtype=np.uint8),
        "logical_to_physical": logical_to_physical,
        "physical_to_logical": physical_to_logical,
        "layout_tag": np.asarray([0 if layout == "identity" else 1], dtype=np.uint8),
        "physical_to_logical_ordinal": physical_to_logical,
        "physical_source_entity_id": source[physical_to_logical],
        "physical_participant_entity_id": participant[physical_to_logical],
    }
    degree = source_degree
    local_change_fanout = np.zeros(cells, dtype=np.int64)
    latent_change_fanout = np.zeros(cells, dtype=np.int64)
    for cell in range(cells):
        local_genes = local_change_gene_a[local_change_cell_a == cell]
        local_change_fanout[cell] = source_degree[local_genes].sum()
        changed_modules = np.unique(
            latent_change_module_a[latent_change_cell_a == cell])
        latent_change_fanout[cell] = np.count_nonzero(np.isin(module_id, changed_modules))
    arrays["local_gene_change_edge_fanout"] = local_change_fanout
    arrays["latent_change_edge_fanout"] = latent_change_fanout
    arrays["query_case_candidate_counts"] = np.diff(query_edge_indptr)
    arrays["event_window_unique_edge_count"] = np.asarray([
        np.unique(event_edge[cell * events_per_cell + events_per_cell // 2:
                             (cell + 1) * events_per_cell]).size
        for cell in range(cells)], dtype=np.int64)
    hist_edges = np.unique(np.concatenate((
        np.asarray([0, 1, 2], dtype=np.int64),
        np.ceil(np.geomspace(2, max(3, int(degree.max(initial=0)) + 1), 33))
        .astype(np.int64))))
    if hist_edges[-1] <= degree.max(initial=0):
        hist_edges = np.append(hist_edges, degree.max() + 1)
    degree_hist_counts, degree_hist_edges = np.histogram(degree, bins=hist_edges)
    arrays["source_degree_histogram_edges"] = degree_hist_edges.astype(np.int64)
    arrays["source_degree_histogram_counts"] = degree_hist_counts.astype(np.int64)
    latent_changed_fraction = latent_change_fanout / max(1, edges)
    local_changed_fraction = local_change_fanout / max(1, edges)
    total_bytes = sum(value.nbytes for value in {id(value): value
                                                  for value in arrays.values()}.values())
    source_digest = _digest_arrays(arrays, (
        "logical_edge_id", "source_entity_id", "participant_entity_id", "edge_module_id"))
    local_module_edge_count = int(np.count_nonzero((source % modules) == (participant % modules)))
    metadata: dict[str, Any] = {
        "schema": "cellerator.scaled_sparse_fixture.v1",
        "seed": int(seed), "edges": int(edges), "entities": int(entities),
        "cells": int(cells), "modules": int(modules),
        "events_per_cell": int(events_per_cell), "layout": layout,
        "max_source_degree_cap": int(min(max_source_degree, entities - 1)),
        "source_role": 0,
        "query_case_names": query_case_names,
        "topology_sha256": source_digest,
        "physical_layout_sha256": _digest_arrays(arrays, ("physical_to_logical_ordinal",)),
        "array_bytes": int(total_bytes),
        "dense_entity_affinity_bytes_if_materialized": int(entities * entities * 4),
        "dense_cell_edge_state_bytes_if_materialized": int(cells * edges),
        "source_degree": {
            "zero": int(np.count_nonzero(degree == 0)),
            "one": int(np.count_nonzero(degree == 1)),
            "p50": int(np.quantile(degree, 0.50)),
            "p90": int(np.quantile(degree, 0.90)),
            "p99": int(np.quantile(degree, 0.99)),
            "max": int(degree.max(initial=0)),
            "histogram_edges": degree_hist_edges.tolist(),
            "histogram_counts": degree_hist_counts.tolist(),
        },
        "unique_directed_endpoint_pairs": int(edges),
        "duplicate_directed_endpoint_pairs": 0,
        "module_local_edge_count": local_module_edge_count,
        "cross_module_edge_count": int(edges - local_module_edge_count),
        "module_local_edge_fraction": float(local_module_edge_count / edges),
        "query_case_candidate_counts": np.diff(query_edge_indptr).tolist(),
        "cell_query_candidate_counts": arrays["cell_query_candidate_counts"].tolist(),
        "cohort_support_reuse_fraction": 1.0,
        "cell_state_values_independent": True,
        "edge_predicate_participants_per_edge": 2,
        "observed_gene_predicate_count": int(entities),
        "shared_latent_predicate_count": int(modules),
        "local_gene_change_count": int(local_change_gene_a.size),
        "local_gene_changed_edge_fraction_per_cell_mean": float(local_changed_fraction.mean()),
        "local_gene_change_edge_fanout_per_cell": local_change_fanout.tolist(),
        "latent_change_count": int(latent_change_module_a.size),
        "latent_change_correlated_cohort_groups": int(np.unique(latent_change_group_a).size),
        "latent_changed_edge_fraction_per_cell_mean": float(latent_changed_fraction.mean()),
        "latent_change_edge_fanout_per_cell": latent_change_fanout.tolist(),
        "event_window_unique_edge_count_per_cell": arrays["event_window_unique_edge_count"].tolist(),
        "event_window_query_occurrences": int(len(event_window_query_indices)),
        "event_window_query_occurrences_per_window": np.diff(
            event_window_query_indptr).tolist(),
        "event_occurrences": int(event_edge.size),
        "event_unique_edge_fraction": float(np.unique(event_edge).size / max(1, event_edge.size)),
        "ranked_identity_and_position_separate": True,
        "rank_count_per_cell": int(rank_count),
        "ranked_cohort_support_overlap_fraction": float(
            shared_rank_count / rank_count),
        "observed_view_labels": {
            "0": "generic_query_state",
            "1": "ranked_presence",
            "2": "event_window_history",
        },
    }
    return ScaledSparseFixture(arrays, metadata)


def profile(name: str, *, seed: int = 20261027, layout: str = "identity") -> ScaledSparseFixture:
    """Construct a documented small oracle or indexing/numerical-scale fixture."""
    if name == "small":
        return make_scaled_fixture(edges=96, entities=64, cells=8, modules=8,
                                   events_per_cell=16, cohort_size=4,
                                   seed=seed, layout=layout,
                                   query_sizes=(0, 1, 8, 32))
    if name == "large-indexing":
        return make_scaled_fixture(edges=1_000_000, entities=20_000, cells=512,
                                   modules=256, events_per_cell=64,
                                   cohort_size=16, seed=seed, layout=layout)
    if name == "large-indexing-2m":
        return make_scaled_fixture(edges=2_000_000, entities=20_000, cells=1024,
                                   modules=256, events_per_cell=64,
                                   cohort_size=16, seed=seed, layout=layout)
    if name == "large-retrieval":
        return make_scaled_fixture(edges=1_000_000, entities=20_000, cells=32,
                                   modules=256, events_per_cell=64,
                                   cohort_size=8, seed=seed, layout=layout)
    if name == "large-numerical":
        # Starts within the bounded dense B*N*D envelope used by the campaign.
        return make_scaled_fixture(edges=262_144, entities=8_192, cells=128,
                                   modules=128, events_per_cell=64,
                                   cohort_size=8, seed=seed, layout=layout)
    if name == "large-ranked":
        return make_scaled_fixture(edges=262_144, entities=4_096, cells=32,
                                   modules=128, events_per_cell=64, rank_count=512,
                                   cohort_size=8, seed=seed, layout=layout)
    if name == "large-events":
        return make_scaled_fixture(edges=1_280_000, entities=20_000, cells=8,
                                   modules=256, events_per_cell=512,
                                   cohort_size=4, seed=seed, layout=layout)
    raise ValueError(f"unknown fixture profile: {name}")


def write_raw_bundle(fixture: ScaledSparseFixture, directory: str | Path) -> Path:
    """Write canonical little-endian raw arrays and a compact integrity manifest."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    descriptors: dict[str, Any] = {}
    storage_by_signature: dict[tuple[str, tuple[int, ...], str], tuple[str, str]] = {}
    for name, array in sorted(fixture.arrays.items()):
        value = np.ascontiguousarray(array)
        dtype = value.dtype.newbyteorder("<")
        little = value.astype(dtype, copy=False)
        raw = little.tobytes(order="C")
        filename = f"{name}.bin"
        checksum = hashlib.sha256(raw).hexdigest()
        signature = (dtype.str, tuple(value.shape), checksum)
        if signature in storage_by_signature:
            filename, alias_target = storage_by_signature[signature]
        else:
            (root / filename).write_bytes(raw)
            storage_by_signature[signature] = (filename, name)
            alias_target = ""
        descriptor = {
            "file": filename,
            "dtype": dtype.str,
            "shape": list(value.shape),
            "nbytes": len(raw),
            "sha256": checksum,
        }
        if alias_target:
            descriptor["alias_of"] = alias_target
        descriptors[name] = descriptor
    manifest = {
        "schema": "cellerator.scaled_sparse_fixture_bundle.v1",
        "generator": "bench/sparse_support/scaled_fixtures.py",
        "metadata": fixture.metadata,
        "arrays": descriptors,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("small", "large-indexing",
                                               "large-indexing-2m", "large-numerical",
                                               "large-retrieval",
                                               "large-ranked", "large-events"),
                        default="small")
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--layout", choices=("identity", "shuffled"), default="identity")
    parser.add_argument("--metadata-only", action="store_true",
                        help="print metadata without writing fixture arrays")
    parser.add_argument("--output", help="optional NPZ output path")
    parser.add_argument("--bundle-dir", help="write raw little-endian arrays and manifest")
    args = parser.parse_args()
    fixture = profile(args.profile, seed=args.seed, layout=args.layout)
    print(json.dumps(fixture.metadata, sort_keys=True, indent=2))
    if args.output:
        np.savez_compressed(args.output, **fixture.arrays)
    if args.bundle_dir:
        write_raw_bundle(fixture, args.bundle_dir)
    if not args.output and not args.bundle_dir and not args.metadata_only:
        parser.error("provide --output or --metadata-only")


if __name__ == "__main__":
    main()
