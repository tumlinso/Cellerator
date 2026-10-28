"""Frozen E7 relation-only fixtures and independent exact workset oracle.

This module is CPU-only. It builds the query terms and expected logical
worksets from the checked scaled-fixture arrays. GPU timing is performed by
``e7_relation_runner.py`` through the experiment-local C ABI indexer.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


FIXTURE_ROOT = Path("/tmp/cellerator_scaled_sparse_fixtures_887437fc")
FIXTURE_MANIFEST_SHA256 = {
    "small": "c33facc18f330739411c1f293609c237689685bd4003099ca79b331fded144be",
    "large_ranked": "674d5d4660befcb6ddbcd618b16b136203d3478ddc577d09a037e116bffd6e32",
    "large_events": "910a4c742b795a2a827cb21e62c510a3bc5039e6878ae75d4826fc0d33234be8",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class E7Fixture:
    name: str
    root: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    arrays: dict[str, np.ndarray]
    query_rows: tuple[np.ndarray, ...]
    context_entities: tuple[np.ndarray, ...]

    @property
    def cells(self) -> int:
        return len(self.query_rows)

    @property
    def entities(self) -> int:
        return int(self.manifest["metadata"]["entities"])

    @property
    def edge_ids(self) -> np.ndarray:
        return self.arrays["edge_ids"].astype(np.int64, copy=False)

    @property
    def sources(self) -> np.ndarray:
        return self.arrays["source_entity_id"].astype(np.int64, copy=False)

    @property
    def destinations(self) -> np.ndarray:
        return self.arrays["participant_entity_id"].astype(np.int64, copy=False)

    def identity(self) -> dict[str, Any]:
        return {
            "fixture": self.name,
            "manifest_sha256": self.manifest_sha256,
            "topology_sha256": self.manifest["metadata"]["topology_sha256"],
            "query_rows_sha256": sha256_bytes(
                np.concatenate(self.query_rows).astype(np.int64, copy=False).tobytes()),
            "query_offsets_sha256": sha256_bytes(
                offsets_for(self.query_rows).tobytes()),
            "context_entities_sha256": sha256_bytes(
                np.concatenate(self.context_entities).astype(np.int64, copy=False).tobytes()),
        }


def offsets_for(rows: tuple[np.ndarray, ...] | list[np.ndarray]) -> np.ndarray:
    lengths = np.asarray([len(row) for row in rows], dtype=np.uint64)
    offsets = np.empty(len(rows) + 1, dtype=np.uint64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    return offsets


def _read_bundle(root: Path, profile: str) -> tuple[dict[str, Any], str,
                                                    dict[str, np.ndarray]]:
    folder = root / {"large_ranked": "large-ranked",
                     "large_events": "large-events",
                     "small": "small"}[profile]
    raw_manifest = (folder / "manifest.json").read_bytes()
    manifest_sha = sha256_bytes(raw_manifest)
    manifest = json.loads(raw_manifest)
    expected_name = {"small": "small", "large_ranked": "large-ranked",
                     "large_events": "large-events"}[profile]
    expected = FIXTURE_MANIFEST_SHA256[profile]
    if manifest_sha != expected:
        raise ValueError(f"{profile} fixture manifest SHA mismatch: {manifest_sha}")
    if manifest.get("schema") != "cellerator.scaled_sparse_fixture_bundle.v1":
        raise ValueError(f"unsupported fixture schema in {folder}")
    required = {
        "edge_ids", "source_entity_id", "participant_entity_id",
        "cell_query_entities", "cell_query_indptr",
    }
    if profile == "large_ranked":
        required.update({"ranked_entity_id", "ranked_position"})
    if profile == "large_events":
        required.update({"event_offsets", "event_entity_ids", "event_timestamps",
                         "event_ordinals", "count_window_bounds", "time_window_bounds"})
    arrays: dict[str, np.ndarray] = {}
    for name in sorted(required):
        descriptor = manifest["arrays"][name]
        raw = (folder / descriptor["file"]).read_bytes()
        if sha256_bytes(raw) != descriptor["sha256"]:
            raise ValueError(f"fixture array checksum mismatch: {name}")
        arrays[name] = np.frombuffer(raw, dtype=np.dtype(descriptor["dtype"])) \
            .reshape(tuple(descriptor["shape"])).copy()
    if folder.name != expected_name:
        raise AssertionError("internal fixture profile mapping is inconsistent")
    return manifest, manifest_sha, arrays


def load_fixture(name: str, root: Path = FIXTURE_ROOT) -> E7Fixture:
    if name not in FIXTURE_MANIFEST_SHA256:
        raise ValueError(f"unsupported E7 fixture: {name}")
    manifest, manifest_sha, arrays = _read_bundle(root, name)
    cells = int(manifest["metadata"]["cells"])
    if name == "large_ranked":
        context_rows = tuple(
            np.asarray(row, dtype=np.int64).copy()
            for row in arrays["ranked_entity_id"]
        )
        # Use exact ranked identities as source queries. Sorting/deduplication
        # is a relation-query policy; the context keeps the original ordering.
        queries = tuple(np.unique(row) for row in context_rows)
    elif name == "large_events":
        offsets = arrays["event_offsets"].astype(np.int64, copy=False)
        event_ids = arrays["event_entity_ids"].astype(np.int64, copy=False)
        event_ordinals = arrays["event_ordinals"].astype(np.int64, copy=False)
        event_times = arrays["event_timestamps"]
        count_bounds = arrays["count_window_bounds"].astype(np.int64, copy=False)
        time_bounds = arrays["time_window_bounds"]
        context_list = []
        for cell in range(cells):
            begin, end = map(int, offsets[cell:cell + 2])
            count_lo, count_hi = map(int, count_bounds[cell])
            time_lo, time_hi = time_bounds[cell]
            keep = ((event_ordinals[begin:end] >= count_lo)
                    & (event_ordinals[begin:end] < count_hi)
                    & (event_times[begin:end] >= time_lo)
                    & (event_times[begin:end] < time_hi))
            context_list.append(event_ids[begin:end][keep].copy())
        context_rows = tuple(context_list)
        queries = tuple(np.unique(row) for row in context_rows)
    else:
        # The small control declares one fixed 24-term source-role query per
        # cell. These terms are also the entities contributing to its context.
        begin = arrays["cell_query_indptr"].astype(np.int64, copy=False)
        entities = arrays["cell_query_entities"].astype(np.int64, copy=False)
        context_rows = tuple(entities[int(begin[c]):int(begin[c + 1])].copy()
                             for c in range(cells))
        queries = tuple(np.unique(row) for row in context_rows)
    if len(context_rows) != cells or len(queries) != cells:
        raise ValueError("fixture query row count does not match declared cells")
    for row in queries:
        if row.size and (row.min() < 0 or row.max() >= int(manifest["metadata"]["entities"])):
            raise ValueError("query entity outside fixture domain")
    return E7Fixture(name, root / {"large_ranked": "large-ranked",
                                   "large_events": "large-events",
                                   "small": "small"}[name], manifest,
                     manifest_sha, arrays, queries, context_rows)


def host_scan_oracle(fixture: E7Fixture) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact CSR offsets, stable IDs, and ordinals for source queries."""
    edge_ids = fixture.edge_ids
    sources = fixture.sources
    if edge_ids.ndim != 1 or sources.shape != edge_ids.shape:
        raise ValueError("malformed support arrays")
    offsets = [0]
    ids: list[int] = []
    ordinals: list[int] = []
    for terms in fixture.query_rows:
        query = np.asarray(terms, dtype=np.int64)
        selected = np.flatnonzero(np.isin(sources, query, assume_unique=False))
        # Support arrays are canonical increasing logical IDs; ordinals are
        # preserved independently to catch accidental physical-order sorting.
        ids.extend(map(int, edge_ids[selected]))
        ordinals.extend(map(int, selected))
        offsets.append(len(ids))
    return (np.asarray(offsets, dtype=np.uint64),
            np.asarray(ids, dtype=np.uint64),
            np.asarray(ordinals, dtype=np.uint64))


def build_query_buffers(fixture: E7Fixture) -> tuple[np.ndarray, np.ndarray]:
    """Make the C ABI's one-based source-role query terms and cell offsets."""
    from ctypes import Structure, c_uint32, c_uint64

    class PostingTerm(Structure):
        _fields_ = [("role", c_uint32), ("reserved", c_uint32),
                    ("entity_id", c_uint64)]

    rows = tuple(np.asarray(row, dtype=np.uint64) + 1 for row in fixture.query_rows)
    offsets = offsets_for(rows)
    terms = np.zeros(int(offsets[-1]), dtype=np.dtype([
        ("role", np.uint32), ("reserved", np.uint32), ("entity_id", np.uint64)]))
    terms["entity_id"] = np.concatenate(rows) if rows else np.empty(0, dtype=np.uint64)
    if terms.dtype.itemsize != __import__("ctypes").sizeof(PostingTerm):
        raise RuntimeError("ctypes posting term layout differs from C ABI")
    return offsets, terms

