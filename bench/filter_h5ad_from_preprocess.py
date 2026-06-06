#!/usr/bin/env python3

import argparse
import struct
from pathlib import Path

import anndata
import numpy as np
import scipy.sparse as sp


ANALYSIS_MAGIC = b"CPRA1\x00\x00\x00"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter an H5AD using a Cellerator preprocess analysis blob.")
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--matrix-source", default="raw_x")
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_analysis_blob(path: str) -> dict:
    with open(path, "rb") as handle:
        magic = handle.read(len(ANALYSIS_MAGIC))
        if magic != ANALYSIS_MAGIC:
            raise ValueError(f"{path}: invalid analysis blob magic")
        rows, cols = struct.unpack("<QQ", handle.read(16))

        def read_array(dtype: np.dtype, count: int) -> np.ndarray:
            byte_count = int(np.dtype(dtype).itemsize) * int(count)
            payload = handle.read(byte_count)
            if len(payload) != byte_count:
                raise ValueError(f"{path}: truncated analysis blob")
            return np.frombuffer(payload, dtype=dtype, count=count).copy()

        return {
            "rows": rows,
            "cols": cols,
            "cell_keep": read_array(np.float32, rows),  # cell_total_counts
            "cell_mito_counts": read_array(np.float32, rows),
            "cell_max_counts": read_array(np.float32, rows),
            "cell_detected_genes": read_array(np.uint32, rows),
            "cell_keep_mask": read_array(np.uint8, rows),
            "gene_sum": read_array(np.float32, cols),
            "gene_sq_sum": read_array(np.float32, cols),
            "gene_detected_cells": read_array(np.float32, cols),
            "gene_keep_mask": read_array(np.uint8, cols),
            "gene_flags": read_array(np.uint8, cols),
        }


def load_selected_matrix(adata: anndata.AnnData, matrix_source: str):
    if matrix_source == "raw_x":
        if adata.raw is None:
            raise ValueError("requested raw_x, but .raw is missing in the H5AD file")
        matrix = adata.raw.X
        var = adata.raw.var.copy()
    elif matrix_source == "x":
        matrix = adata.X
        var = adata.var.copy()
    else:
        raise ValueError(f"unsupported matrix source: {matrix_source}")
    if hasattr(matrix, "to_memory"):
        matrix = matrix.to_memory()
    if sp.issparse(matrix):
        matrix = matrix.tocsr(copy=True)
        matrix.sort_indices()
    else:
        matrix = sp.csr_matrix(np.asarray(matrix))
    return matrix, var


def main() -> int:
    args = parse_args()
    analysis = load_analysis_blob(args.analysis)
    adata = anndata.read_h5ad(args.h5ad)
    matrix, var = load_selected_matrix(adata, args.matrix_source)

    cell_keep = analysis["cell_keep_mask"] != 0
    gene_keep = analysis["gene_keep_mask"] != 0
    if matrix.shape[0] != cell_keep.shape[0] or matrix.shape[1] != gene_keep.shape[0]:
        raise ValueError("analysis dimensions do not match selected matrix")

    filtered_matrix = matrix[cell_keep][:, gene_keep].copy()
    filtered_obs = adata.obs.iloc[cell_keep].copy()
    filtered_var = var.iloc[gene_keep].copy()
    filtered = anndata.AnnData(X=filtered_matrix, obs=filtered_obs, var=filtered_var)

    for key, value in adata.obsm.items():
        if getattr(value, "shape", None) is not None and value.shape[0] == adata.n_obs:
            filtered.obsm[key] = value[cell_keep].copy()
    for key, value in adata.varm.items():
        if getattr(value, "shape", None) is not None and value.shape[0] == var.shape[0]:
            filtered.varm[key] = value[gene_keep].copy()
    filtered.uns = dict(adata.uns)
    filtered.raw = filtered.copy()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.write_h5ad(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
