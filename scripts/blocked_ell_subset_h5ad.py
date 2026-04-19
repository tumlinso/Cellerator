#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.io
import scipy.sparse as sp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic MTX subsets for blocked-ELL optimization study.",
    )
    parser.add_argument(
        "--source",
        default="data/test/reference/GSE147520_all_cells.h5ad",
        help="Source H5AD file.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/test/generated/blocked_ell_optimization",
        help="Output directory for generated MTX files.",
    )
    parser.add_argument(
        "--manifest",
        default="bench/real_data/generated/blocked_ell_optimization/manifest.tsv",
        help="Output manifest TSV path.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Deterministic RNG seed.")
    parser.add_argument("--fast-rows", type=int, default=4096, help="Fast subset row count.")
    parser.add_argument(
        "--representative-rows",
        type=int,
        default=16384,
        help="Representative subset row count.",
    )
    parser.add_argument(
        "--rhs-cols",
        type=int,
        default=128,
        help="Default RHS column count stored in the manifest.",
    )
    return parser.parse_args()


def load_source_matrix(source: Path):
    backed = ad.read_h5ad(source, backed="r")
    if backed.raw is not None:
        matrix = backed.raw.X
        rows, cols = backed.raw.shape
    else:
        matrix = backed.X
        rows, cols = backed.shape
    if not sp.issparse(matrix) and not hasattr(matrix, "__getitem__"):
        raise RuntimeError("expected sparse or sliceable matrix in source h5ad")
    return backed, matrix, rows, cols


def choose_nested_rows(total_rows: int, representative_rows: int, fast_rows: int, seed: int):
    if representative_rows > total_rows:
        representative_rows = total_rows
    if fast_rows > representative_rows:
        fast_rows = representative_rows
    rng = np.random.default_rng(seed)
    representative = np.sort(rng.choice(total_rows, size=representative_rows, replace=False))
    fast = representative[:fast_rows].copy()
    return fast, representative


def subset_csr_rows(matrix, row_ids: np.ndarray) -> sp.csr_matrix:
    subset = matrix[row_ids]
    if hasattr(subset, "to_memory"):
        subset = subset.to_memory()
    if not sp.isspmatrix_csr(subset):
        subset = subset.tocsr()
    subset.sort_indices()
    return subset


def write_matrix_market(path: Path, matrix: sp.csr_matrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.mmwrite(str(path), matrix)


def write_manifest(path: Path, rowspecs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("dataset_id\tmatrix_path\trows\tcols\tnnz\tslice_rows\trhs_cols\n")
        for row in rowspecs:
            handle.write(
                f"{row['dataset_id']}\t{row['matrix_path']}\t{row['rows']}\t{row['cols']}\t"
                f"{row['nnz']}\t{row['rows']}\t{row['rhs_cols']}\n"
            )


def main() -> int:
    args = parse_args()
    source = Path(args.source)
    out_dir = Path(args.out_dir)
    manifest = Path(args.manifest)

    backed, matrix, total_rows, total_cols = load_source_matrix(source)
    fast_rows, representative_rows = choose_nested_rows(
        total_rows,
        args.representative_rows,
        args.fast_rows,
        args.seed,
    )

    specs = []
    metadata = {
        "source": str(source.resolve()),
        "seed": args.seed,
        "shape": [int(total_rows), int(total_cols)],
        "raw_used": backed.raw is not None,
        "subsets": [],
    }
    for name, row_ids in (
        ("gse147520_fast_r%d_allcols" % len(fast_rows), fast_rows),
        ("gse147520_representative_r%d_allcols" % len(representative_rows), representative_rows),
    ):
        subset = subset_csr_rows(matrix, row_ids)
        matrix_path = out_dir / f"{name}.mtx"
        write_matrix_market(matrix_path, subset)
        specs.append(
            {
                "dataset_id": name,
                "matrix_path": str(matrix_path.resolve()),
                "rows": int(subset.shape[0]),
                "cols": int(subset.shape[1]),
                "nnz": int(subset.nnz),
                "rhs_cols": int(args.rhs_cols),
            }
        )
        metadata["subsets"].append(
            {
                "dataset_id": name,
                "rows": int(subset.shape[0]),
                "cols": int(subset.shape[1]),
                "nnz": int(subset.nnz),
                "row_index_path": str((out_dir / f"{name}.rows.npy").resolve()),
            }
        )
        np.save(out_dir / f"{name}.rows.npy", row_ids)

    write_manifest(manifest, specs)
    info_path = out_dir / "subsets.json"
    with info_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(f"wrote {len(specs)} subsets to {out_dir}")
    print(f"manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
