"""Validate supplied sparse token records and emit exact, sparse raw bundles."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np


RAW_BUNDLE_SCHEMA = "cellerator.scaled_sparse_fixture_bundle.v1"
EVALUATION_SCHEMA = "cellerator.sparse_support_evaluation.v1"


class CapacityExceeded(ValueError):
    """Raised before output when a caller-specified bound cannot hold input."""

    def __init__(self, resource: str, capacity: int, required_at_least: int):
        self.resource = resource
        self.capacity = int(capacity)
        self.required_at_least = int(required_at_least)
        super().__init__(f"{resource} capacity {capacity} exceeded; "
                         f"required at least {required_at_least}")


@dataclass(frozen=True)
class CapacityLimits:
    """Optional explicit caller limits. Omitted limits do not truncate data."""

    edges: int | None = None
    query_records: int | None = None
    candidate_items_per_view: int | None = None
    condition_participants: int | None = None

    def __post_init__(self):
        for name, value in vars(self).items():
            if value is not None:
                if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                    raise ValueError(f"{name} capacity must be an integer")
                if int(value) < 0:
                    raise ValueError(f"{name} capacity must be nonnegative")


@dataclass(frozen=True)
class EvaluationBundle:
    """Exact packed arrays plus provenance and a sparse coverage summary."""

    arrays: Mapping[str, np.ndarray]
    metadata: Mapping[str, object]
    summary: Mapping[str, object]


def _vector(name: str, values, *, kind: str = "integer") -> np.ndarray:
    try:
        out = np.asarray(values)
    except Exception as exc:
        raise ValueError(f"{name} must be an array-like value") from exc
    if out.ndim != 1:
        raise ValueError(f"{name} must be rank one")
    if out.size == 0:
        return out.astype(np.float64 if kind == "numeric" else np.int64, copy=False)
    if out.dtype.hasobject or out.dtype.kind not in (
            "iuf" if kind == "numeric" else "iu"):
        raise ValueError(f"{name} must contain {kind} values without object dtype")
    if out.dtype.kind == "f" and not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain finite values")
    return out


def _int64(name: str, values) -> np.ndarray:
    out = _vector(name, values)
    if out.size:
        if out.dtype.kind == "u" and int(out.max()) > np.iinfo(np.int64).max:
            raise ValueError(f"{name} values must fit signed int64")
        if out.dtype.kind == "i" and (int(out.min()) < np.iinfo(np.int64).min
                                       or int(out.max()) > np.iinfo(np.int64).max):
            raise ValueError(f"{name} values must fit signed int64")
    return out.astype(np.int64, copy=False)


def _capacity(value: int | None, resource: str, required: int) -> None:
    if value is not None and required > int(value):
        raise CapacityExceeded(resource, int(value), required)


def _unique_ids(name: str, values: np.ndarray) -> None:
    if values.size > 1 and np.unique(values).size != values.size:
        raise ValueError(f"{name} must be unique")


def _mapping_id_key(key, *, name: str) -> int:
    if isinstance(key, (bool, np.bool_)):
        raise ValueError(f"{name} keys must be integer IDs, not booleans")
    if isinstance(key, (int, np.integer)):
        return int(key)
    if isinstance(key, str) and re.fullmatch(r"[+-]?[0-9]+", key):
        return int(key)
    raise ValueError(f"{name} keys must be integer IDs or decimal JSON object keys")


def _normalise_mapping(mapping: Mapping | None, *, name: str) -> dict[int, tuple[int, ...]]:
    if mapping is None:
        return {}
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{name} must be a mapping")
    result = {}
    for key, values in mapping.items():
        try:
            ident = _mapping_id_key(key, name=name)
            row_values = tuple(values)
            if any(isinstance(v, (bool, np.bool_))
                   or not isinstance(v, (int, np.integer)) for v in row_values):
                raise ValueError
            row = tuple(int(v) for v in row_values)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} keys and values must be integer IDs") from exc
        if ident in result:
            raise ValueError(f"{name} contains duplicate normalized key {ident}")
        result[ident] = row
    return result


def _window_mapping(mapping: Mapping | None, *, name: str,
                    integer_bounds: bool = False) -> dict[int, tuple[object, object]]:
    if mapping is None:
        return {}
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{name} must map cell IDs to half-open [start, stop) bounds")
    result = {}
    for key, pair in mapping.items():
        try:
            cell = _mapping_id_key(key, name=name)
            if len(pair) != 2:
                raise ValueError
            if cell in result:
                raise ValueError(f"{name} contains duplicate normalized cell ID {cell}")
            lo, hi = pair
            if integer_bounds:
                if any(isinstance(v, (bool, np.bool_))
                       or not isinstance(v, (int, np.integer)) for v in (lo, hi)):
                    raise ValueError
            else:
                if any(isinstance(v, (bool, np.bool_))
                       or not isinstance(v, (int, float, np.integer, np.floating))
                       or not np.isfinite(v) for v in (lo, hi)):
                    raise ValueError
        except (TypeError, ValueError) as exc:
            kind = "integer " if integer_bounds else "finite numeric "
            raise ValueError(f"{name} values must be two-item {kind}half-open bounds") from exc
        if hi < lo:
            raise ValueError(f"{name} stop must be greater than or equal to start")
        result[cell] = (lo, hi)
    return result


def _csr_postings(keys: np.ndarray, edge_ordinals: np.ndarray):
    if not keys.size:
        return (np.empty(0, dtype=np.int64), np.zeros(1, dtype=np.int64),
                np.empty(0, dtype=np.int64))
    permutation = np.argsort(keys, kind="stable")
    sorted_keys = keys[permutation]
    unique, starts, counts = np.unique(sorted_keys, return_index=True, return_counts=True)
    offsets = np.empty(unique.size + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return unique.astype(np.int64, copy=False), offsets, edge_ordinals[permutation]


def _posting_slice(keys: np.ndarray, offsets: np.ndarray,
                   ordinals: np.ndarray, key: int) -> np.ndarray:
    index = int(np.searchsorted(keys, key))
    if index == keys.size or int(keys[index]) != key:
        return np.empty(0, dtype=np.int64)
    return ordinals[int(offsets[index]):int(offsets[index + 1])]


def _bundle(*, mode: str, edge_ids, sources, destinations, token_cells,
            token_entities, record_ids, input_indices, sequence_positions,
            rank_positions, query_eligible, query_role: str,
            token_arrays: dict[str, np.ndarray], cell_universe,
            event_position_windows, event_timestamp_windows, event_timestamps,
            conditions, edge_conditions, active_conditions, targets,
            target_kind, provenance, capacities) -> EvaluationBundle:
    ids = _int64("edge_ids", edge_ids)
    src = _int64("source_entity_ids", sources)
    dst = _int64("destination_entity_ids", destinations)
    if not (ids.size == src.size == dst.size):
        raise ValueError("edge_ids, source_entity_ids, and destination_entity_ids must align")
    _unique_ids("edge_ids", ids)
    _capacity(capacities.edges, "edges", int(ids.size))
    if query_role not in ("source", "destination", "either"):
        raise ValueError("query_role must be source, destination, or either")

    edge_order = np.argsort(ids, kind="stable")
    ids = ids[edge_order]
    src = src[edge_order]
    dst = dst[edge_order]
    edge_input_index = edge_order.astype(np.int64, copy=False)
    edge_ordinals = np.arange(ids.size, dtype=np.int64)
    src_keys, src_offsets, src_ordinals = _csr_postings(src, edge_ordinals)
    dst_keys, dst_offsets, dst_ordinals = _csr_postings(dst, edge_ordinals)

    cells = _int64("cell_id_universe", cell_universe)
    if cells.size > 1:
        cells = np.unique(cells)
    token_cells = _int64("record cell IDs", token_cells)
    token_entities = _int64("record entity IDs", token_entities)
    if not (token_cells.size == token_entities.size == record_ids.size
            == input_indices.size == sequence_positions.size == query_eligible.size):
        raise ValueError("token identity and query arrays must have the same length")
    _capacity(capacities.query_records, "query_records", int(token_cells.size))
    if not np.isin(token_cells, cells, assume_unique=False).all():
        raise ValueError("record cell IDs must be listed in cell_id_universe")

    # Stable grouping preserves each cell's caller order. Rank/ordinal values
    # are fields and never determine sequence order.
    token_order = np.argsort(token_cells, kind="stable")
    token_cells = token_cells[token_order]
    token_entities = token_entities[token_order]
    record_ids = record_ids[token_order]
    input_indices = input_indices[token_order]
    sequence_positions = sequence_positions[token_order]
    query_eligible = query_eligible[token_order].astype(np.uint8, copy=False)
    if rank_positions is not None:
        rank_positions = rank_positions[token_order]

    query_offsets = np.searchsorted(token_cells, cells, side="left").astype(np.int64)
    query_offsets = np.concatenate((query_offsets, np.asarray([token_cells.size], dtype=np.int64)))
    if cells.size:
        for index, cell in enumerate(cells):
            lo, hi = int(query_offsets[index]), int(query_offsets[index + 1])
            sequence_positions[lo:hi] = np.arange(hi - lo, dtype=np.int64)

    condition_map = _normalise_mapping(conditions, name="conditions")
    edge_condition_map = _normalise_mapping(edge_conditions, name="edge_conditions")
    active_map = _normalise_mapping(active_conditions, name="active_conditions")
    if condition_map:
        participant_count = sum(len(x) for x in condition_map.values())
        _capacity(capacities.condition_participants, "condition_participants", participant_count)
    unknown_edge_keys = set(edge_condition_map).difference(int(x) for x in ids)
    if unknown_edge_keys:
        raise ValueError(f"edge_conditions references unknown logical edge ID {min(unknown_edge_keys)}")
    unknown_predicates = {p for row in edge_condition_map.values() for p in row}.difference(condition_map)
    if unknown_predicates:
        raise ValueError(f"edge_conditions references unknown condition ID {min(unknown_predicates)}")
    unknown_active = {p for row in active_map.values() for p in row}.difference(condition_map)
    if unknown_active:
        raise ValueError(f"active_conditions references unknown condition ID {min(unknown_active)}")
    unknown_cells = set(active_map).difference(int(x) for x in cells)
    if unknown_cells:
        raise ValueError(f"active_conditions references unknown cell ID {min(unknown_cells)}")

    condition_ids = np.asarray(sorted(condition_map), dtype=np.int64)
    condition_offsets = [0]
    condition_participants = []
    for condition_id in condition_ids:
        condition_participants.extend(condition_map[int(condition_id)])
        condition_offsets.append(len(condition_participants))
    edge_condition_offsets = [0]
    edge_condition_values = []
    for edge_id in ids:
        edge_condition_values.extend(edge_condition_map.get(int(edge_id), ()))
        edge_condition_offsets.append(len(edge_condition_values))

    active_offsets = [0]
    active_values = []
    active_sets = {}
    for cell in cells:
        values = tuple(sorted(set(active_map.get(int(cell), ()))))
        active_values.extend(values)
        active_offsets.append(len(active_values))
        active_sets[int(cell)] = set(values)

    # The CPU intake keeps caller identity untouched. These parallel arrays
    # provide explicit, reversible positive IDs for current native contracts.
    participant_entity_values = np.asarray(condition_participants, dtype=np.int64)
    entity_universe = np.unique(np.concatenate((src, dst, token_entities,
                                                participant_entity_values)))
    native_entity_ids = np.arange(1, entity_universe.size + 1, dtype=np.uint64)
    entity_to_native = {int(external): int(native)
                        for external, native in zip(entity_universe, native_entity_ids)}
    native_cell_ids = np.arange(1, cells.size + 1, dtype=np.uint64)
    cell_to_native = {int(external): int(native)
                      for external, native in zip(cells, native_cell_ids)}
    record_id_values = np.asarray(record_ids, dtype=np.int64)
    native_record_external_ids = np.unique(record_id_values)
    native_record_ids = np.arange(1, native_record_external_ids.size + 1,
                                  dtype=np.uint64)
    record_to_native = {int(external): int(native)
                        for external, native in zip(native_record_external_ids,
                                                    native_record_ids)}
    native_edge_ids = np.arange(1, ids.size + 1, dtype=np.uint64)

    position_windows = _window_mapping(event_position_windows, name="event_position_windows",
                                       integer_bounds=True)
    timestamp_windows = _window_mapping(event_timestamp_windows, name="event_timestamp_windows")
    if mode == "ranked" and (position_windows or timestamp_windows):
        raise ValueError("ranked token order is not temporal; event windows cannot be applied")
    if timestamp_windows and event_timestamps is None:
        raise ValueError("event_timestamp_windows require supplied event_timestamps")
    for name, mapping in (("event_position_windows", position_windows),
                          ("event_timestamp_windows", timestamp_windows)):
        unknown = set(mapping).difference(int(x) for x in cells)
        if unknown:
            raise ValueError(f"{name} references unknown cell ID {min(unknown)}")

    edge_condition_rows = []
    for edge_id in ids:
        edge_condition_rows.append(edge_condition_map.get(int(edge_id), ()))
    structural_rows = []
    eligible_rows = []
    for query_index, entity in enumerate(token_entities):
        cell = int(token_cells[query_index])
        seq_pos = int(sequence_positions[query_index])
        in_window = bool(query_eligible[query_index])
        if mode == "events":
            if cell in position_windows:
                lo, hi = position_windows[cell]
                in_window &= int(lo) <= seq_pos < int(hi)
            if cell in timestamp_windows:
                lo, hi = timestamp_windows[cell]
                stamp = event_timestamps[input_indices[query_index]]
                in_window &= lo <= stamp < hi
        query_eligible[query_index] = np.uint8(in_window)
        if not in_window:
            candidates = np.empty(0, dtype=np.int64)
        else:
            matches = []
            if query_role in ("source", "either"):
                matches.append(_posting_slice(src_keys, src_offsets, src_ordinals, int(entity)))
            if query_role in ("destination", "either"):
                matches.append(_posting_slice(dst_keys, dst_offsets, dst_ordinals, int(entity)))
            candidates = (np.unique(np.concatenate(matches)) if matches
                          else np.empty(0, dtype=np.int64))
        structural_rows.append(candidates)
        if active_conditions is None or not edge_condition_map:
            eligible_rows.append(candidates)
        else:
            active = active_sets[cell]
            filtered = [int(edge_ordinal) for edge_ordinal in candidates
                        if set(edge_condition_rows[int(edge_ordinal)]).issubset(active)]
            eligible_rows.append(np.asarray(filtered, dtype=np.int64))
    structural_required = sum(int(row.size) for row in structural_rows)
    eligible_required = sum(int(row.size) for row in eligible_rows)
    _capacity(capacities.candidate_items_per_view, "candidate_items_per_view",
              max(structural_required, eligible_required))

    def rows_to_csr(rows):
        offsets = np.zeros(len(rows) + 1, dtype=np.int64)
        if rows:
            np.cumsum([row.size for row in rows], out=offsets[1:])
            flat = np.concatenate(rows) if offsets[-1] else np.empty(0, dtype=np.int64)
        else:
            flat = np.empty(0, dtype=np.int64)
        return offsets, flat

    candidate_offsets, candidate_ordinals = rows_to_csr(structural_rows)
    eligible_offsets, eligible_ordinals = rows_to_csr(eligible_rows)
    candidate_ids = ids[candidate_ordinals]
    eligible_ids = ids[eligible_ordinals]
    source_counts = np.diff(src_offsets)
    destination_counts = np.diff(dst_offsets)
    unique_candidate_ids = np.unique(candidate_ids)
    unique_eligible_ids = np.unique(eligible_ids)
    input_digest = hashlib.sha256()
    for name, value in sorted(token_arrays.items()):
        contiguous = np.ascontiguousarray(value)
        input_digest.update(name.encode("utf-8"))
        input_digest.update(contiguous.dtype.str.encode("ascii"))
        input_digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
        input_digest.update(contiguous.tobytes())
    input_digest.update(np.ascontiguousarray(edge_order).tobytes())

    arrays = {
        "edge_ids": ids.astype("<i8", copy=False),
        "native_edge_ids": native_edge_ids,
        "native_edge_ordinals": np.arange(ids.size, dtype=np.uint64),
        "source_entity_id": src.astype("<i8", copy=False),
        "participant_entity_id": dst.astype("<i8", copy=False),
        "native_edge_source_entity_ids": np.asarray(
            [entity_to_native[int(x)] for x in src], dtype=np.uint64),
        "native_edge_destination_entity_ids": np.asarray(
            [entity_to_native[int(x)] for x in dst], dtype=np.uint64),
        "edge_input_index": edge_input_index,
        "source_posting_entity_ids": src_keys,
        "source_posting_offsets": src_offsets,
        "source_posting_edge_ordinals": src_ordinals,
        "destination_posting_entity_ids": dst_keys,
        "destination_posting_offsets": dst_offsets,
        "destination_posting_edge_ordinals": dst_ordinals,
        "cell_ids": cells,
        "native_cell_ids": native_cell_ids,
        "native_cell_external_ids": cells.copy(),
        "query_cell_offsets": query_offsets,
        "query_cell_ids": token_cells,
        "native_query_cell_ids": np.asarray(
            [cell_to_native[int(x)] for x in token_cells], dtype=np.uint64),
        "query_record_ids": record_ids,
        "query_entity_ids": token_entities,
        "native_query_record_ids": np.asarray(
            [record_to_native[int(x)] for x in record_ids], dtype=np.uint64),
        "native_query_entity_ids": np.asarray(
            [entity_to_native[int(x)] for x in token_entities], dtype=np.uint64),
        "query_input_indices": input_indices.astype(np.int64, copy=False),
        "query_sequence_positions": sequence_positions,
        "query_window_eligible": query_eligible,
        "candidate_offsets": candidate_offsets,
        "candidate_edge_ordinals": candidate_ordinals,
        "candidate_edge_ids": candidate_ids,
        "native_candidate_edge_ids": native_edge_ids[candidate_ordinals],
        "condition_eligible_offsets": eligible_offsets,
        "condition_eligible_edge_ordinals": eligible_ordinals,
        "condition_eligible_edge_ids": eligible_ids,
        "native_condition_eligible_edge_ids": native_edge_ids[eligible_ordinals],
        "condition_ids": condition_ids,
        "condition_participant_offsets": np.asarray(condition_offsets, dtype=np.int64),
        "condition_participant_entity_ids": participant_entity_values,
        "native_entity_ids": native_entity_ids,
        "native_entity_external_ids": entity_universe.copy(),
        "native_condition_participant_entity_ids": np.asarray(
            [entity_to_native[int(x)] for x in participant_entity_values], dtype=np.uint64),
        "edge_condition_offsets": np.asarray(edge_condition_offsets, dtype=np.int64),
        "edge_condition_ids": np.asarray(edge_condition_values, dtype=np.int64),
        "cell_active_condition_offsets": np.asarray(active_offsets, dtype=np.int64),
        "cell_active_condition_ids": np.asarray(active_values, dtype=np.int64),
    }
    arrays.update(token_arrays)
    if mode == "events":
        arrays["native_event_ordinals"] = sequence_positions.astype(np.uint64, copy=False)
        arrays["native_event_ids"] = arrays["native_query_record_ids"].copy()
    if rank_positions is not None:
        arrays["rank_positions"] = rank_positions.astype(np.int64, copy=False)
    if targets is not None:
        target_array = np.asarray(targets)
        if target_array.dtype.hasobject or target_array.dtype.kind not in "biufUS":
            raise ValueError("targets must be numeric or fixed-width string arrays")
        arrays["targets"] = target_array

    metadata = {
        "evaluation_schema": EVALUATION_SCHEMA,
        "mode": mode,
        "query_role": query_role,
        "provenance": dict(provenance or {}),
        "query_records_sha256": input_digest.hexdigest(),
        "target_kind": target_kind if targets is not None else None,
        "condition_evaluation": ("active predicates applied as a conjunction"
                                 if active_conditions is not None else "not requested"),
        "window_semantics": {
            "event_position_windows": "half-open per-cell input-sequence positions",
            "event_timestamp_windows": "half-open timestamp values, independent of occurrence order",
            "event_ordinals": "preserved data field; not used to reorder or define count windows",
            "rank_positions": "preserved feature positions; never interpreted as time",
        },
        "materialized_dense_cell_edge_matrix": False,
        "native_abi_mapping": {
            "provided": True,
            "scheme": "positive dense IDs in canonical sorted external-ID order",
            "external_id_dtype": "signed int64",
            "external_ids_preserved": True,
            "edge_ids": "native_edge_ids and native_edge_ordinals map to edge_ids order",
            "entity_ids": "native_entity_ids maps one shared external entity domain",
            "cell_ids": "native_cell_ids maps cell_ids order",
            "record_ids": "native_query_record_ids maps query_record_ids; event mode also exports native_event_ids",
            "event_ordinals": "native_event_ordinals are dense per-cell input-order positions; external event_ordinals are preserved",
            "timestamps": "preserved as supplied; caller must choose an explicit integer time scale for native timestamp ABI",
        },
        "target_shape": list(np.asarray(targets).shape) if targets is not None else None,
    }
    try:
        json.dumps(metadata, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("provenance and target_kind must be finite JSON values") from exc
    summary = {
        "edges": int(ids.size),
        "cells": int(cells.size),
        "query_records": int(token_cells.size),
        "window_eligible_records": int(np.count_nonzero(query_eligible)),
        "structural_candidate_items": int(candidate_ordinals.size),
        "condition_eligible_items": int(eligible_ordinals.size),
        "unique_structural_candidate_edges": int(unique_candidate_ids.size),
        "unique_condition_eligible_edges": int(unique_eligible_ids.size),
        "structural_edge_coverage": float(unique_candidate_ids.size / ids.size) if ids.size else 0.0,
        "condition_eligible_edge_coverage": float(unique_eligible_ids.size / ids.size) if ids.size else 0.0,
        "empty_queries": int(np.count_nonzero(np.diff(candidate_offsets) == 0)),
        "max_source_degree": int(source_counts.max()) if source_counts.size else 0,
        "max_destination_degree": int(destination_counts.max()) if destination_counts.size else 0,
        "max_query_degree": int(np.diff(candidate_offsets).max()) if token_cells.size else 0,
        "duplicate_entity_records": int(token_entities.size - np.unique(token_entities).size),
        "dense_cell_edge_matrix_allocated": False,
    }
    return EvaluationBundle(arrays, metadata, summary)


def _cell_universe(token_cells: np.ndarray, explicit, *mappings) -> np.ndarray:
    pieces = [token_cells]
    if explicit is not None:
        pieces.append(_int64("cell_id_universe", explicit))
    for mapping in mappings:
        pieces.append(np.asarray(list(mapping), dtype=np.int64))
    nonempty = [piece for piece in pieces if piece.size]
    return np.unique(np.concatenate(nonempty)) if nonempty else np.empty(0, dtype=np.int64)


def prepare_ranked(*, edge_ids, source_entity_ids, destination_entity_ids,
                   cell_ids, gene_ids, rank_positions, token_ids=None,
                   cell_id_universe=None, query_role: str = "source",
                   conditions: Mapping | None = None,
                   edge_conditions: Mapping | None = None,
                   active_conditions: Mapping | None = None,
                   targets=None, target_kind: str | None = None,
                   provenance: Mapping | None = None,
                   capacities: CapacityLimits = CapacityLimits()) -> EvaluationBundle:
    """Prepare rank-aware tokens without assigning temporal meaning to rank.

    Flat arrays preserve caller order within each cell. If ``token_ids`` is
    omitted, stable input positions supply token identity. Repeated genes and
    rank values are retained.
    """
    raw_cells = _int64("ranked cell_ids", cell_ids)
    genes = _int64("gene_ids", gene_ids)
    ranks = _int64("rank_positions", rank_positions)
    if not (raw_cells.size == genes.size == ranks.size):
        raise ValueError("ranked cell_ids, gene_ids, and rank_positions must align")
    if np.any(ranks < 0):
        raise ValueError("rank_positions must be nonnegative integers")
    tokens = (np.arange(genes.size, dtype=np.int64) if token_ids is None
              else _int64("token_ids", token_ids))
    if tokens.size != genes.size:
        raise ValueError("token_ids must align with ranked genes")
    _unique_ids("token_ids", tokens)
    active = _normalise_mapping(active_conditions, name="active_conditions")
    cells = _cell_universe(raw_cells, cell_id_universe, active)
    order = np.argsort(raw_cells, kind="stable")
    token_arrays = {
        "ranked_cell_ids": raw_cells[order],
        "ranked_token_ids": tokens[order],
        "ranked_entity_ids": genes[order],
        "rank_positions": ranks[order],
        "ranked_input_indices": order.astype(np.int64, copy=False),
    }
    return _bundle(
        mode="ranked", edge_ids=edge_ids, sources=source_entity_ids,
        destinations=destination_entity_ids, token_cells=raw_cells,
        token_entities=genes, record_ids=tokens, input_indices=np.arange(genes.size),
        sequence_positions=np.empty(genes.size, dtype=np.int64), rank_positions=ranks,
        query_eligible=np.ones(genes.size, dtype=np.uint8), query_role=query_role,
        token_arrays=token_arrays, cell_universe=cells,
        event_position_windows=None, event_timestamp_windows=None,
        event_timestamps=None, conditions=conditions, edge_conditions=edge_conditions,
        active_conditions=active if active_conditions is not None else None,
        targets=targets, target_kind=target_kind, provenance=provenance,
        capacities=capacities)


def prepare_events(*, edge_ids, source_entity_ids, destination_entity_ids,
                   cell_ids, entity_ids, occurrence_ids, event_ordinals,
                   timestamps=None, event_values=None, cell_id_universe=None,
                   position_windows: Mapping | None = None,
                   timestamp_windows: Mapping | None = None,
                   query_role: str = "source", conditions: Mapping | None = None,
                   edge_conditions: Mapping | None = None,
                   active_conditions: Mapping | None = None,
                   targets=None, target_kind: str | None = None,
                   provenance: Mapping | None = None,
                   capacities: CapacityLimits = CapacityLimits()) -> EvaluationBundle:
    """Prepare positive event occurrences; order, ordinal, and time stay distinct.

    Input order within each cell is the event sequence. ``event_ordinals`` are
    opaque caller fields and may be repeated or nonmonotonic. Count windows use
    input-sequence positions; timestamp windows are an independent predicate.
    """
    raw_cells = _int64("event cell_ids", cell_ids)
    entities = _int64("event entity_ids", entity_ids)
    occurrences = _int64("occurrence_ids", occurrence_ids)
    ordinals = _int64("event_ordinals", event_ordinals)
    n = entities.size
    if not (raw_cells.size == n == occurrences.size == ordinals.size):
        raise ValueError("event cell/entity/occurrence/ordinal arrays must align")
    _unique_ids("occurrence_ids", occurrences)
    stamps = None if timestamps is None else _vector("timestamps", timestamps, kind="numeric")
    if stamps is not None and stamps.size != n:
        raise ValueError("timestamps must align with event occurrences")
    values = None if event_values is None else _vector("event_values", event_values, kind="numeric")
    if values is not None and (values.size != n or np.any(values <= 0)):
        raise ValueError("event_values must align and contain only positive activations")
    pos_windows = _window_mapping(position_windows, name="position_windows",
                                  integer_bounds=True)
    time_windows = _window_mapping(timestamp_windows, name="timestamp_windows")
    if time_windows and stamps is None:
        raise ValueError("timestamp_windows require supplied timestamps")
    active = _normalise_mapping(active_conditions, name="active_conditions")
    cells = _cell_universe(raw_cells, cell_id_universe, pos_windows, time_windows, active)
    order = np.argsort(raw_cells, kind="stable")
    sorted_cells = raw_cells[order]
    input_cell_offsets = np.searchsorted(sorted_cells, cells, side="left").astype(np.int64)
    input_cell_offsets = np.concatenate((input_cell_offsets, np.asarray([n], dtype=np.int64)))
    sequence_positions = np.empty(n, dtype=np.int64)
    for index, cell in enumerate(cells):
        lo, hi = int(input_cell_offsets[index]), int(input_cell_offsets[index + 1])
        count = hi - lo
        if int(cell) in pos_windows:
            start, stop = pos_windows[int(cell)]
            if int(start) < 0 or int(stop) > count:
                raise ValueError(f"position window for cell {int(cell)} is outside its {count} input occurrences")
        sequence_positions[lo:hi] = np.arange(count, dtype=np.int64)
    preliminary_eligible = np.ones(n, dtype=np.uint8)
    for sorted_index, original_index in enumerate(order):
        cell = int(sorted_cells[sorted_index])
        pos = int(sequence_positions[sorted_index])
        if cell in pos_windows:
            lo, hi = pos_windows[cell]
            preliminary_eligible[original_index] &= np.uint8(int(lo) <= pos < int(hi))
        if cell in time_windows:
            lo, hi = time_windows[cell]
            preliminary_eligible[original_index] &= np.uint8(lo <= stamps[original_index] < hi)
    token_arrays = {
        "event_cell_ids": raw_cells[order],
        "event_occurrence_ids": occurrences[order],
        "event_entity_ids": entities[order],
        "event_ordinals": ordinals[order],
        "event_input_indices": order.astype(np.int64, copy=False),
        "event_window_eligible": preliminary_eligible[order],
    }
    if stamps is not None:
        token_arrays["event_timestamps"] = stamps[order]
    if values is not None:
        token_arrays["event_values"] = values[order]
    return _bundle(
        mode="events", edge_ids=edge_ids, sources=source_entity_ids,
        destinations=destination_entity_ids, token_cells=raw_cells,
        token_entities=entities, record_ids=occurrences,
        input_indices=np.arange(n), sequence_positions=sequence_positions,
        rank_positions=None, query_eligible=preliminary_eligible,
        query_role=query_role, token_arrays=token_arrays, cell_universe=cells,
        event_position_windows=pos_windows, event_timestamp_windows=time_windows,
        event_timestamps=stamps, conditions=conditions, edge_conditions=edge_conditions,
        active_conditions=active if active_conditions is not None else None,
        targets=targets, target_kind=target_kind, provenance=provenance,
        capacities=capacities)


def write_raw_bundle(bundle: EvaluationBundle, directory: str | Path) -> Path:
    """Write a raw-v1-compatible bundle without overwriting existing evidence."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty evaluation bundle: {root}")
    descriptors = {}
    for name, array in sorted(bundle.arrays.items()):
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            raise ValueError(f"invalid array name: {name}")
        value = np.ascontiguousarray(array)
        if value.dtype.hasobject:
            raise ValueError(f"object dtype is not supported for {name}")
        little_dtype = value.dtype.newbyteorder("<")
        raw = value.astype(little_dtype, copy=False).tobytes(order="C")
        digest = hashlib.sha256(raw).hexdigest()
        filename = f"{name}.bin"
        (root / filename).write_bytes(raw)
        descriptors[name] = {"file": filename, "dtype": little_dtype.str,
                              "shape": list(value.shape), "nbytes": len(raw),
                              "sha256": digest}
    manifest = {"schema": RAW_BUNDLE_SCHEMA,
                "generator": "bench/sparse_support/evaluation/inputs.py",
                "metadata": {"evaluation": dict(bundle.metadata),
                             "summary": dict(bundle.summary)},
                "arrays": descriptors}
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")
    return path


def load_raw_bundle(directory: str | Path, names: Sequence[str] | None = None):
    """Read and hash-verify bundles using the existing scaled-fixture schema."""
    root = Path(directory)
    manifest_path = root / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != RAW_BUNDLE_SCHEMA:
        raise ValueError("bundle does not use the existing raw sparse-bundle schema")
    descriptors = manifest.get("arrays")
    if not isinstance(descriptors, dict):
        raise ValueError("manifest arrays must be a descriptor mapping")
    selected = tuple(descriptors) if names is None else tuple(names)
    arrays = {}
    for name in selected:
        if name not in descriptors:
            raise KeyError(f"array not present in bundle: {name}")
        descriptor = descriptors[name]
        filename = descriptor.get("file")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"unsafe or invalid bundle filename for {name}")
        raw = (root / filename).read_bytes()
        if len(raw) != int(descriptor.get("nbytes", -1)):
            raise ValueError(f"bundle byte length mismatch for {name}")
        if hashlib.sha256(raw).hexdigest() != descriptor.get("sha256"):
            raise ValueError(f"bundle hash mismatch for {name}")
        dtype = np.dtype(descriptor["dtype"])
        shape = tuple(int(dim) for dim in descriptor["shape"])
        if np.prod(shape, dtype=np.int64) * dtype.itemsize != len(raw):
            raise ValueError(f"bundle shape/dtype size mismatch for {name}")
        arrays[name] = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    return arrays, manifest, hashlib.sha256(manifest_bytes).hexdigest()
