#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a deterministic day-stratified H5AD subset for fast benchmarking.",
    )
    parser.add_argument(
        "--source",
        default="data/test/reference/adata_JAX_dataset_1.h5ad",
        help="Input H5AD file.",
    )
    parser.add_argument(
        "--output",
        default="data/test/reference/adata_JAX_dataset_1_day_stratified_256_per_day.h5ad",
        help="Output H5AD file.",
    )
    parser.add_argument(
        "--day-column",
        default="day",
        help="obs column used for day-level stratification.",
    )
    parser.add_argument(
        "--cells-per-day",
        type=int,
        default=256,
        help="Maximum number of cells to sample for each day label.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Deterministic RNG seed.",
    )
    return parser.parse_args()


def day_sort_key(label: str):
    match = re.search(r"(\d+(?:\.\d+)?)", label)
    if match is None:
        return (1, label)
    return (0, float(match.group(1)), label)


def choose_rows(day_values, cells_per_day: int, seed: int):
    if cells_per_day <= 0:
        raise ValueError("cells_per_day must be positive")

    labels = np.asarray(day_values.astype(str))
    unique_labels = sorted(np.unique(labels), key=day_sort_key)
    rng = np.random.default_rng(seed)

    selected_parts = []
    manifest = []
    for label in unique_labels:
        group_rows = np.flatnonzero(labels == label)
        take = min(int(cells_per_day), int(group_rows.size))
        sampled = rng.choice(group_rows, size=take, replace=False)
        sampled.sort()
        selected_parts.append(sampled)
        manifest.append(
            {
                "day": label,
                "available_cells": int(group_rows.size),
                "sampled_cells": int(take),
            }
        )

    return np.concatenate(selected_parts), manifest


def make_subset(backed: ad.AnnData, row_ids: np.ndarray):
    subset = backed[row_ids, :].to_memory()
    x = subset.X
    if sp.issparse(x):
        subset.X = x.tocsr(copy=False)
        subset.X.sort_indices()
    return subset


def main() -> int:
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)

    backed = ad.read_h5ad(source, backed="r")
    if args.day_column not in backed.obs.columns:
        raise KeyError(f"missing obs column: {args.day_column}")

    row_ids, day_manifest = choose_rows(
        backed.obs[args.day_column],
        cells_per_day=args.cells_per_day,
        seed=args.seed,
    )

    subset = make_subset(backed, row_ids)
    backed.file.close()
    day_labels = [entry["day"] for entry in day_manifest]
    available_cells = [entry["available_cells"] for entry in day_manifest]
    sampled_cells = [entry["sampled_cells"] for entry in day_manifest]
    subset.uns["benchmark_subset"] = {
        "source_h5ad": str(source.resolve()),
        "strategy": "day_stratified_random_cells",
        "day_column": args.day_column,
        "cells_per_day": int(args.cells_per_day),
        "seed": int(args.seed),
        "n_obs": int(subset.n_obs),
        "n_vars": int(subset.n_vars),
        "day_labels": day_labels,
        "available_cells": np.asarray(available_cells, dtype=np.int64),
        "sampled_cells": np.asarray(sampled_cells, dtype=np.int64),
        "day_manifest_json": json.dumps(day_manifest),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    subset.write_h5ad(output, compression="lzf")

    summary = {
        "output_h5ad": str(output.resolve()),
        "n_obs": int(subset.n_obs),
        "n_vars": int(subset.n_vars),
        "days": day_manifest,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
