"""ctypes facade for the experiment-local adaptive-fields index bridge.

Importing this module does not import Torch or initialize CUDA. An
``Indexer`` instance calls the CUDA-backed C ABI and therefore must only be
used inside the parent-owned Device Scheduler window.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCAN = 0
ORDERED_POSTINGS = 1


class PostingTermV1(ctypes.Structure):
    _fields_ = [("role", ctypes.c_uint32),
                ("reserved", ctypes.c_uint32),
                ("entity_id", ctypes.c_uint64)]


class IndexOutputV1(ctypes.Structure):
    _fields_ = [
        ("row_offsets", ctypes.POINTER(ctypes.c_uint64)),
        ("row_offsets_capacity", ctypes.c_uint64),
        ("logical_edge_ids", ctypes.POINTER(ctypes.c_uint64)),
        ("edge_ordinals", ctypes.POINTER(ctypes.c_uint64)),
        ("gate_value_indices", ctypes.POINTER(ctypes.c_uint64)),
        ("item_capacity", ctypes.c_uint64),
        ("item_count", ctypes.c_uint64),
        ("structure_epoch", ctypes.c_uint64),
        ("state_generation", ctypes.c_uint64),
        ("ordered_fast_path_rows", ctypes.c_uint64),
        ("exact_fallback_rows", ctypes.c_uint64),
        ("path_diagnostics_valid", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("required_capacity", ctypes.c_uint64),
    ]


@dataclass(frozen=True)
class WorksetCSR:
    row_offsets: np.ndarray
    logical_edge_ids: np.ndarray
    edge_ordinals: np.ndarray
    gate_value_indices: np.ndarray
    structure_epoch: int
    state_generation: int
    ordered_fast_path_rows: int | None
    exact_fallback_rows: int | None

    @property
    def item_count(self) -> int:
        return len(self.edge_ordinals)

    def to_torch_workset(self, torch, device):
        """Create a PerCellWorkset from the host result.

        Host materialization and this host-to-device conversion belong inside
        complete-step timing for E7. The caller supplies the correct support
        capacity; this helper uses the exact discovered item count as capacity.
        """
        from cellerator.torch.sparse_support import PerCellWorkset

        return PerCellWorkset(
            torch.as_tensor(self.row_offsets, dtype=torch.int64, device=device),
            torch.as_tensor(self.edge_ordinals, dtype=torch.int64, device=device),
            torch.as_tensor(self.logical_edge_ids, dtype=torch.int64, device=device),
            torch.as_tensor(self.gate_value_indices, dtype=torch.int64, device=device),
            self.structure_epoch, self.state_generation,
            max(1, self.item_count))


class Indexer:
    def __init__(self, library: str | Path, edge_ids, sources, destinations,
                 entity_count: int, structure_epoch: int,
                 relation_id: int = 1, structure_id: int = 1):
        self._lib = ctypes.CDLL(str(Path(library).resolve(strict=True)))
        u64p = ctypes.POINTER(ctypes.c_uint64)
        self._lib.af_index_create_v1.argtypes = [
            ctypes.c_uint64, u64p, u64p, u64p, ctypes.c_uint64,
            ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        self._lib.af_index_create_v1.restype = ctypes.c_void_p
        self._lib.af_index_capacity_v1.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, u64p]
        self._lib.af_index_capacity_v1.restype = ctypes.c_uint32
        self._lib.af_index_prepare_ordered_postings_v1.argtypes = [ctypes.c_void_p]
        self._lib.af_index_prepare_ordered_postings_v1.restype = ctypes.c_uint32
        self._lib.af_index_run_v1.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64, u64p,
            ctypes.POINTER(PostingTermV1), ctypes.c_uint64,
            ctypes.c_uint64, ctypes.c_int, ctypes.POINTER(IndexOutputV1)]
        self._lib.af_index_run_v1.restype = ctypes.c_uint32
        self._lib.af_index_destroy_v1.argtypes = [ctypes.c_void_p]
        self._lib.af_index_destroy_v1.restype = None
        self._lib.af_index_last_error_v1.argtypes = []
        self._lib.af_index_last_error_v1.restype = ctypes.c_char_p
        if not (len(edge_ids) == len(sources) == len(destinations)):
            raise ValueError("edge ID/source/destination arrays must have equal lengths")
        self.edge_count = len(edge_ids)
        self._handle = None
        self._cells = None
        self._capacity = None
        self._row_offsets_c = None
        self._ids_c = None
        self._ordinals_c = None
        self._gate_indices_c = None
        self._query_cache = None
        self._support_ids = tuple(map(int, edge_ids))
        self._sources = tuple(map(int, sources))
        self._destinations = tuple(map(int, destinations))
        self._edge_ids_c = (ctypes.c_uint64 * self.edge_count)(*self._support_ids)
        self._sources_c = (ctypes.c_uint64 * self.edge_count)(*self._sources)
        self._destinations_c = (ctypes.c_uint64 * self.edge_count)(*self._destinations)
        self._handle = self._lib.af_index_create_v1(
            self.edge_count, self._edge_ids_c, self._sources_c,
            self._destinations_c, int(entity_count), int(structure_epoch),
            int(relation_id), int(structure_id))
        if not self._handle:
            raise RuntimeError(self._error())
        self.structure_epoch = int(structure_epoch)

    def _error(self) -> str:
        raw = self._lib.af_index_last_error_v1()
        return raw.decode("utf-8", errors="replace") if raw else "unknown C ABI failure"

    def capacity(self, cells: int) -> int:
        capacity = ctypes.c_uint64()
        status = self._lib.af_index_capacity_v1(
            self._handle, int(cells), ctypes.byref(capacity))
        if status:
            raise RuntimeError(f"capacity query failed with status {status}: {self._error()}")
        return int(capacity.value)

    def prepare_ordered_postings(self) -> None:
        """Build the optional host/device posting projection once."""
        status = self._lib.af_index_prepare_ordered_postings_v1(self._handle)
        if status:
            raise RuntimeError(
                f"ordered-posting preparation failed with status {status}: {self._error()}")

    def prepare_output_buffers(self, cells: int) -> dict[str, int]:
        """Allocate worst-case host result storage once per cell count.

        The cold buffer allocation is setup cost. Every query still performs
        the native query/synchronization/D2H work and each result is copied to
        CUDA by ``to_torch_workset`` inside the measured training step.
        """
        if self._cells == int(cells):
            return {"cells": self._cells, "item_capacity": self._capacity,
                    "host_buffer_bytes": self._capacity * 3 * 8 + (cells + 1) * 8,
                    "allocated_now": False}
        capacity = self.capacity(cells)
        self._cells = int(cells)
        self._capacity = capacity
        self._row_offsets_c = (ctypes.c_uint64 * (cells + 1))()
        self._ids_c = (ctypes.c_uint64 * max(1, capacity))()
        self._ordinals_c = (ctypes.c_uint64 * max(1, capacity))()
        self._gate_indices_c = (ctypes.c_uint64 * max(1, capacity))()
        return {"cells": self._cells, "item_capacity": self._capacity,
                "host_buffer_bytes": self._capacity * 3 * 8 + (cells + 1) * 8,
                "allocated_now": True}

    def prepare_query(self, query_offsets, terms) -> dict[str, int]:
        """Cache immutable query terms and offsets once per fixture lifetime."""
        offsets_values = tuple(map(int, query_offsets))
        terms_values = tuple((int(role), int(entity_id)) for role, entity_id in terms)
        if (not offsets_values or offsets_values[0] != 0
                or offsets_values[-1] != len(terms_values)):
            raise ValueError("query_offsets must partition all terms and start at zero")
        if any(a > b for a, b in zip(offsets_values, offsets_values[1:])):
            raise ValueError("query offsets must be monotonically nondecreasing")
        cells = len(offsets_values) - 1
        self.prepare_output_buffers(cells)
        offsets_c = (ctypes.c_uint64 * len(offsets_values))(*offsets_values)
        terms_c = (PostingTermV1 * max(1, len(terms_values)))(
            *(PostingTermV1(role, 0, entity) for role, entity in terms_values))
        self._query_cache = (cells, len(terms_values), offsets_c, terms_c)
        return {"cells": cells, "query_terms": len(terms_values),
                "query_buffer_bytes": len(offsets_values) * 8
                + len(terms_values) * ctypes.sizeof(PostingTermV1)}

    def query(self, mode: int, query_offsets, terms, *, state_generation: int,
              verify: bool = False) -> WorksetCSR:
        offsets_values = tuple(map(int, query_offsets))
        terms_values = tuple((int(role), int(entity_id)) for role, entity_id in terms)
        if not offsets_values or offsets_values[0] != 0 or offsets_values[-1] != len(terms_values):
            raise ValueError("query_offsets must partition all terms and start at zero")
        if any(a > b for a, b in zip(offsets_values, offsets_values[1:])):
            raise ValueError("query offsets must be monotonically nondecreasing")
        cells = len(offsets_values) - 1
        self.prepare_output_buffers(cells)
        offsets_c = (ctypes.c_uint64 * len(offsets_values))(*offsets_values)
        terms_c = (PostingTermV1 * max(1, len(terms_values)))(
            *(PostingTermV1(role, 0, entity) for role, entity in terms_values))
        capacity = self._capacity
        row_offsets_c = self._row_offsets_c
        ids_c = self._ids_c
        ordinals_c = self._ordinals_c
        gate_indices_c = self._gate_indices_c
        output = IndexOutputV1(
            row_offsets_c, cells + 1, ids_c, ordinals_c, gate_indices_c,
            capacity, 0, 0, 0, 0, 0, 0, 0, 0)
        status = self._lib.af_index_run_v1(
            self._handle, int(mode), cells, offsets_c, terms_c,
            len(terms_values), int(state_generation), int(bool(verify)),
            ctypes.byref(output))
        if status:
            raise RuntimeError(f"index query failed with status {status}: {self._error()}")
        count = int(output.item_count)
        row_offsets = np.ctypeslib.as_array(row_offsets_c)
        ids = np.ctypeslib.as_array(ids_c)[:count]
        ordinals = np.ctypeslib.as_array(ordinals_c)[:count]
        gate_indices = np.ctypeslib.as_array(gate_indices_c)[:count]
        return WorksetCSR(
            row_offsets, ids, ordinals, gate_indices,
            int(output.structure_epoch), int(output.state_generation),
            int(output.ordered_fast_path_rows) if output.path_diagnostics_valid else None,
            int(output.exact_fallback_rows) if output.path_diagnostics_valid else None)

    def query_prepared(self, mode: int, *, state_generation: int,
                       verify: bool = False) -> WorksetCSR:
        """Execute using retained host query buffers; no per-step query copying."""
        if self._query_cache is None:
            raise RuntimeError("call prepare_query before query_prepared")
        cells, term_count, offsets_c, terms_c = self._query_cache
        if self._cells != cells:
            raise RuntimeError("query cache and result buffer cell counts disagree")
        if mode not in (SCAN, ORDERED_POSTINGS):
            raise ValueError(f"unknown native index mode: {mode}")
        output = IndexOutputV1(
            self._row_offsets_c, cells + 1, self._ids_c, self._ordinals_c,
            self._gate_indices_c, self._capacity, 0, 0, 0, 0, 0, 0, 0, 0)
        status = self._lib.af_index_run_v1(
            self._handle, int(mode), cells, offsets_c, terms_c, term_count,
            int(state_generation), int(bool(verify)), ctypes.byref(output))
        if status:
            raise RuntimeError(f"index query failed with status {status}: {self._error()}")
        count = int(output.item_count)
        return WorksetCSR(
            np.ctypeslib.as_array(self._row_offsets_c),
            np.ctypeslib.as_array(self._ids_c)[:count],
            np.ctypeslib.as_array(self._ordinals_c)[:count],
            np.ctypeslib.as_array(self._gate_indices_c)[:count],
            int(output.structure_epoch), int(output.state_generation),
            int(output.ordered_fast_path_rows) if output.path_diagnostics_valid else None,
            int(output.exact_fallback_rows) if output.path_diagnostics_valid else None)

    def close(self) -> None:
        if self._handle:
            self._lib.af_index_destroy_v1(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
