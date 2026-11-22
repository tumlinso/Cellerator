#!/usr/bin/env python3
"""Prepare real PBMC3K examples and one signed TF-target support for CUDA timing."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import struct

import h5py
import numpy as np
from scipy.sparse import csr_matrix


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_counts(path):
    with h5py.File(path, "r") as h5:
        group = h5["X"]
        names = [value.decode() for value in h5["var"]["index"]]
        shape = tuple(group.attrs["h5sparse_shape"])
        counts = csr_matrix(
            (group["data"][:], group["indices"][:], group["indptr"][:]), shape=shape
        )
    if len(names) != shape[1] or not np.all(np.isfinite(counts.data) & (counts.data >= 0)):
        raise ValueError("Expected nonnegative raw counts and a matching gene axis")
    repeated = Counter(names)
    return counts, {name: i for i, name in enumerate(names) if repeated[name] == 1}


def read_support(path, measured):
    edges = {}
    stats = Counter()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        records = csv.DictReader(stream)
        if not {"source", "target", "weight"}.issubset(records.fieldnames or ()):
            raise ValueError("Expected the signed CollecTRI table")
        for record in records:
            stats["input_records"] += 1
            tf, target = record["source"], record["target"]
            weight = float(record["weight"])
            if not np.isfinite(weight) or weight == 0:
                raise ValueError("Expected a finite signed edge weight")
            if tf not in measured or target not in measured:
                stats["unmatched_or_ambiguous_endpoint"] += 1
            elif tf == target:
                stats["self_edge"] += 1
            else:
                key = (target, tf)
                sign = 1 if weight > 0 else -1
                if key in edges:
                    if edges[key] != sign:
                        raise ValueError(f"Conflicting signs for {tf} -> {target}")
                    stats["duplicate_edge"] += 1
                edges[key] = sign
    if not edges:
        raise ValueError("No unambiguous measured TF-target pairs")
    return edges, stats


def prepare(h5ad, regulatory_csv, output, manifest):
    counts, measured = read_counts(h5ad)
    edges, stats = read_support(regulatory_csv, measured)
    targets = sorted({target for target, _ in edges}, key=measured.__getitem__)
    regulators = sorted({tf for _, tf in edges}, key=measured.__getitem__)
    feature_axis = sorted(set(targets) | set(regulators), key=measured.__getitem__)
    columns = {symbol: i for i, symbol in enumerate(feature_axis)}
    target_ids = {symbol: i for i, symbol in enumerate(targets)}
    tf_ids = {symbol: i for i, symbol in enumerate(regulators)}
    ordered = sorted(edges, key=lambda pair: (target_ids[pair[0]], tf_ids[pair[1]]))
    degrees = Counter(target for target, _ in ordered)
    sizes = np.array([degrees[symbol] for symbol in targets], dtype="<u4")
    row_offsets = np.empty(len(targets) + 1, dtype="<u4")
    row_offsets[0] = 0
    np.cumsum(sizes, out=row_offsets[1:])
    minor = np.array([tf_ids[tf] for _, tf in ordered], dtype="<u4")
    initial = np.array([edges[pair] / degrees[pair[0]] for pair in ordered], dtype="<f4")
    source_map = np.array([columns[tf] for tf in regulators], dtype="<u4")
    target_map = np.array([columns[target] for target in targets], dtype="<u4")

    # The resident input table keeps assay order on the retained gene axis.
    # Neither cell order nor the expression value layout is packed for a route.
    library_size = np.asarray(counts.sum(axis=1, dtype=np.float64)).ravel()
    if np.any(library_size <= 0):
        raise ValueError("Expected positive library size for every retained cell")
    expression = counts[:, [measured[symbol] for symbol in feature_axis]].toarray()
    expression = np.log1p(expression * (1e4 / library_size[:, None])).astype("<f4")

    with output.open("wb") as stream:
        stream.write(struct.pack("<8I", 0x31445443, counts.shape[0], len(feature_axis),
                                 len(targets), len(regulators), len(edges), 0, 0))
        for plane in (row_offsets, minor, initial, target_map, source_map, expression):
            plane.tofile(stream)
    record = {
        "assay_sha256": file_hash(h5ad),
        "regulatory_csv_sha256": file_hash(regulatory_csv),
        "fixture_sha256": file_hash(output),
        "cells": counts.shape[0],
        "assayed_genes": counts.shape[1],
        "resident_genes": len(feature_axis),
        "targets": len(targets),
        "regulators": len(regulators),
        "edges": len(edges),
        "edge_filters": dict(stats),
        "normalization": "log1p(10000 * count / all-assayed-gene cell library size)",
        "initial_edge_coefficient": "regulatory sign / retained target in-degree",
        "ordered_assay_columns": [measured[symbol] for symbol in feature_axis],
    }
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({key: value for key, value in record.items() if key != "ordered_assay_columns"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("h5ad", "regulatory-csv", "output", "manifest"):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.h5ad, args.regulatory_csv, args.output, args.manifest)
