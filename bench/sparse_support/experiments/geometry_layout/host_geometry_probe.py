#!/usr/bin/env python3
"""Host-only CE-GEO grouping and tile-density probe.

The topology candidate sees only directed edge endpoints. Fixture module labels
are emitted as a diagnostic upper bound and never drive candidate grouping.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def load_bundle(bundle: Path, names: tuple[str, ...]):
    manifest_bytes = (bundle / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    arrays = {}
    for name in names:
        requested_name = name
        desc = manifest["arrays"][name]
        if "alias_of" in desc:
            desc = manifest["arrays"][desc["alias_of"]]
        raw = (bundle / desc["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != desc["sha256"]:
            raise ValueError(f"array hash mismatch: {requested_name}")
        arrays[requested_name] = np.frombuffer(raw, dtype=np.dtype(desc["dtype"])).reshape(
            tuple(desc["shape"])).copy()
    return arrays, manifest, hashlib.sha256(manifest_bytes).hexdigest()


def neighbor_greedy_order(source: np.ndarray, destination: np.ndarray,
                          entity_count: int, group_width: int = 16) -> np.ndarray:
    """Greedily pack high-degree vertices with their unplaced graph neighbors."""
    edges = np.column_stack((source, destination)).astype(np.int64, copy=False)
    undirected = np.concatenate((edges, edges[:, ::-1]), axis=0)
    undirected = np.unique(undirected, axis=0)
    row_counts = np.bincount(undirected[:, 0], minlength=entity_count)
    row_offsets = np.empty(entity_count + 1, dtype=np.int64)
    row_offsets[0] = 0
    np.cumsum(row_counts, out=row_offsets[1:])
    neighbors = undirected[:, 1].astype(np.int32, copy=False)
    remaining = np.ones(entity_count, dtype=np.bool_)
    degree = row_counts.astype(np.int64, copy=False)
    order = np.empty(entity_count, dtype=np.int32)
    cursor = 0

    while cursor < entity_count:
        remaining_ids = np.flatnonzero(remaining)
        seed = int(remaining_ids[np.argmax(degree[remaining_ids])])
        group = [seed]
        remaining[seed] = False
        affinity = np.zeros(entity_count, dtype=np.uint16)

        def add_neighbors(vertex: int) -> None:
            begin, end = int(row_offsets[vertex]), int(row_offsets[vertex + 1])
            ids = neighbors[begin:end]
            ids = ids[remaining[ids]]
            if ids.size:
                np.add.at(affinity, ids, 1)

        add_neighbors(seed)
        target = min(group_width, entity_count - cursor)
        while len(group) < target:
            candidates = np.flatnonzero(remaining)
            if not candidates.size:
                break
            candidate_scores = affinity[candidates].astype(np.int64)
            # Affinity first, then degree, then logical ID for stable ties.
            best = int(candidates[np.lexsort((candidates, -degree[candidates],
                                               -candidate_scores))[0]])
            if affinity[best] == 0:
                best = int(candidates[np.argmax(degree[candidates])])
            group.append(best)
            remaining[best] = False
            add_neighbors(best)
        order[cursor:cursor + len(group)] = group
        cursor += len(group)
    return order


def tile_statistics(source: np.ndarray, destination: np.ndarray,
                    logical_to_physical: np.ndarray, entity_count: int,
                    group_width: int = 16) -> dict:
    physical_source = logical_to_physical[source]
    physical_destination = logical_to_physical[destination]
    group_count = (entity_count + group_width - 1) // group_width
    keys = (physical_source // group_width) * group_count + \
        physical_destination // group_width
    counts = np.bincount(keys, minlength=group_count * group_count)
    nonzero = counts[counts > 0]
    output = {
        "group_count_per_axis": group_count,
        "tile_grid_capacity": int(group_count * group_count),
        "active_tiles": int(nonzero.size),
        "empty_tiles": int(counts.size - nonzero.size),
        "edge_count": int(source.size),
        "mean_edges_per_active_tile": float(nonzero.mean()) if nonzero.size else 0.0,
        "median_edges_per_active_tile": float(np.median(nonzero)) if nonzero.size else 0.0,
        "p90_edges_per_active_tile": float(np.quantile(nonzero, 0.9)) if nonzero.size else 0.0,
        "max_edges_per_active_tile": int(nonzero.max(initial=0)),
        "active_tile_histogram": {
            str(int(k)): int(np.count_nonzero(nonzero == k))
            for k in np.unique(nonzero)
        },
        "thresholds": {},
    }
    for threshold in (4, 8, 16, 32):
        eligible = nonzero >= threshold
        output["thresholds"][str(threshold)] = {
            "eligible_tiles": int(eligible.sum()),
            "mma_edges": int(nonzero[eligible].sum()),
            "residual_edges": int(source.size - nonzero[eligible].sum()),
            "mma_edge_fraction": float(nonzero[eligible].sum() / source.size),
        }
    return output


def packed_workset_tiles(source: np.ndarray, destination: np.ndarray,
                         logical_to_physical: np.ndarray, query_rows: np.ndarray,
                         source_offsets: np.ndarray, source_ordinals: np.ndarray,
                         entity_count: int, *, tile_threshold: int | None = None,
                         group_width: int = 16) -> dict:
    group_count = (entity_count + group_width - 1) // group_width
    physical_source = logical_to_physical[source]
    physical_destination = logical_to_physical[destination]
    keys = (physical_source // group_width) * group_count + \
        physical_destination // group_width
    counts = np.bincount(keys, minlength=group_count * group_count)
    eligible = np.ones(counts.size, dtype=np.bool_) if tile_threshold is None else \
        counts >= tile_threshold
    output = []
    for row in query_rows:
        ordinals = []
        for entity in np.unique(row):
            begin, end = map(int, source_offsets[int(entity):int(entity) + 2])
            ordinals.extend(source_ordinals[begin:end].tolist())
        if ordinals:
            ordinals = np.unique(np.asarray(ordinals, dtype=np.int64))
            tile_ids = np.unique(keys[ordinals])
            mma = tile_ids[eligible[tile_ids]]
            output.append({
                "candidate_edges": int(ordinals.size),
                "mma_tiles": int(mma.size),
                "residual_edges": int(ordinals.size - np.count_nonzero(eligible[keys[ordinals]])),
            })
        else:
            output.append({"candidate_edges": 0, "mma_tiles": 0,
                           "residual_edges": 0})
    return {
        "mean_candidate_edges_per_cell": float(np.mean([x["candidate_edges"] for x in output])),
        "p90_candidate_edges_per_cell": float(np.quantile([x["candidate_edges"] for x in output], 0.9)),
        "mean_mma_tiles_per_cell": float(np.mean([x["mma_tiles"] for x in output])),
        "mean_residual_edges_per_cell": float(np.mean([x["residual_edges"] for x in output])),
        "rows": output,
    }


def case_layouts(name: str, arrays: dict, manifest: dict) -> dict:
    meta = manifest["metadata"]
    n = int(meta["entities"])
    source = arrays["source_entity_id"].astype(np.int64, copy=False)
    destination = arrays["participant_entity_id"].astype(np.int64, copy=False)
    started = time.perf_counter()
    order = neighbor_greedy_order(source, destination, n)
    layout_ms = (time.perf_counter() - started) * 1000.0
    topo_inverse = np.empty(n, dtype=np.int64)
    topo_inverse[order] = np.arange(n, dtype=np.int64)
    identity = np.arange(n, dtype=np.int64)
    results = {
        "identity": tile_statistics(source, destination, identity, n),
        "graph_neighbor_greedy": tile_statistics(source, destination, topo_inverse, n),
    }
    if "edge_module_id" in arrays:
        labels = arrays["edge_module_id"].astype(np.int64, copy=False)
        # The edge attribute maps to source modules; recover per-entity labels
        # only to report a hidden-label upper bound. It never constructs the
        # graph-neighbor candidate above.
        entity_module = np.full(n, -1, dtype=np.int64)
        entity_module[source] = labels
        if np.all(entity_module >= 0):
            module_order = np.lexsort((np.arange(n), entity_module))
            module_inverse = np.empty(n, dtype=np.int64)
            module_inverse[module_order] = np.arange(n, dtype=np.int64)
            results["fixture_label_oracle_upper_bound"] = tile_statistics(
                source, destination, module_inverse, n)
    out = {
        "case": name,
        "topology_sha256": meta["topology_sha256"],
        "shape": {"entities": n, "edges": int(source.size),
                  "cells": int(meta["cells"]), "rank_count": meta.get("rank_count_per_cell")},
        "grouping_algorithm": "undirected edge-neighbor greedy; max-degree seed, then max unplaced neighbors with degree tie-break",
        "graph_neighbor_layout_prepare_ms_host": layout_ms,
        "logical_to_physical_order_sha256": hashlib.sha256(topo_inverse.astype("<i8").tobytes()).hexdigest(),
        "layouts": results,
    }
    if name == "large-ranked":
        ranked = arrays["ranked_entity_id"].astype(np.int64, copy=False)
        out["ranked_cell_query_tiles"] = {}
        for layout_name, inverse in (("identity", identity),
                                     ("graph_neighbor_greedy", topo_inverse)):
            out["ranked_cell_query_tiles"][layout_name] = packed_workset_tiles(
                source, destination, inverse, ranked, arrays["source_edge_indptr"],
                arrays["source_edge_ordinals"], n)
        out["ranked_cell_query_tiles"]["graph_neighbor_hybrid_threshold_8"] = \
            packed_workset_tiles(source, destination, topo_inverse, ranked,
                                 arrays["source_edge_indptr"], arrays["source_edge_ordinals"],
                                 n, tile_threshold=8)
        out["ranked_cell_query_tiles"]["graph_neighbor_hybrid_threshold_16"] = \
            packed_workset_tiles(source, destination, topo_inverse, ranked,
                                 arrays["source_edge_indptr"], arrays["source_edge_ordinals"],
                                 n, tile_threshold=16)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-root", type=Path,
                        default=Path("/tmp/cellerator_scaled_sparse_fixtures_887437fc"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"schema": "cellerator.geometry_layout_host_probe.v1", "cases": {}}
    for name, arrays_to_load in (
        ("small", ("source_entity_id", "participant_entity_id", "edge_module_id")),
        ("large-ranked", ("source_entity_id", "participant_entity_id", "edge_module_id",
                           "ranked_entity_id", "source_edge_indptr", "source_edge_ordinals")),
    ):
        arrays, manifest, manifest_sha = load_bundle(args.fixture_root / name,
                                                     arrays_to_load)
        row = case_layouts(name, arrays, manifest)
        row["manifest_sha256"] = manifest_sha
        result["cases"][name] = row
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: {
        "shape": row["shape"],
        "layout_metrics": {key: {
            "active_tiles": value["active_tiles"],
            "mean_occupancy": value["mean_edges_per_active_tile"],
            "thresholds": value["thresholds"],
        } for key, value in row["layouts"].items()},
        "ranked_queries": {
            key: {metric: value[metric] for metric in (
                "mean_candidate_edges_per_cell", "mean_mma_tiles_per_cell",
                "mean_residual_edges_per_cell")}
            for key, value in row.get("ranked_cell_query_tiles", {}).items()
        }
    } for name, row in result["cases"].items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
