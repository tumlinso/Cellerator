#!/usr/bin/env python3

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import anndata
import numpy as np
import scipy.sparse as sp


ANALYSIS_MAGIC = b"CPRA1\x00\x00\x00"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Cellerator preprocess analysis outputs against a Scanpy-comparable baseline.")
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--matrix-source", default="raw_x")
    parser.add_argument("--blocked-analysis", required=True)
    parser.add_argument("--sliced-analysis", required=True)
    parser.add_argument("--summary-tsv", required=True)
    parser.add_argument("--details-json", required=True)
    parser.add_argument("--target-sum", type=float, required=True)
    parser.add_argument("--min-counts", type=float, required=True)
    parser.add_argument("--min-genes", type=int, required=True)
    parser.add_argument("--max-mito-fraction", type=float, required=True)
    parser.add_argument("--min-gene-sum", type=float, required=True)
    parser.add_argument("--min-detected-cells", type=float, required=True)
    parser.add_argument("--min-variance", type=float, required=True)
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
            "cell_total_counts": read_array(np.float32, rows),
            "cell_mito_counts": read_array(np.float32, rows),
            "cell_max_counts": read_array(np.float32, rows),
            "cell_detected_genes": read_array(np.uint32, rows),
            "cell_keep": read_array(np.uint8, rows),
            "gene_sum": read_array(np.float32, cols),
            "gene_sq_sum": read_array(np.float32, cols),
            "gene_detected_cells": read_array(np.float32, cols),
            "gene_keep": read_array(np.uint8, cols),
            "gene_flags": read_array(np.uint8, cols),
        }


def load_matrix(h5ad_path: str, matrix_source: str):
    adata = anndata.read_h5ad(h5ad_path)
    if matrix_source == "raw_x":
        if adata.raw is None:
            raise ValueError("requested raw_x, but .raw is missing in the H5AD file")
        matrix = adata.raw.X
    elif matrix_source == "x":
        matrix = adata.X
    else:
        raise ValueError(f"unsupported matrix source: {matrix_source}")
    if hasattr(matrix, "to_memory"):
        matrix = matrix.to_memory()
    if sp.issparse(matrix):
        matrix = matrix.tocsr(copy=True)
        matrix.sort_indices()
        matrix = matrix.astype(np.float32, copy=False)
    else:
        matrix = np.asarray(matrix, dtype=np.float32)
    return matrix


def maybe_scanpy_normalize_log1p(matrix_kept, target_sum: float):
    try:
        import scanpy as sc
    except Exception:
        return None, "python_reference"

    adata = anndata.AnnData(matrix_kept.copy())
    sc.pp.normalize_total(adata, target_sum=target_sum, inplace=True)
    sc.pp.log1p(adata)
    x = adata.X
    if sp.issparse(x):
        return x.tocsr(copy=False), "scanpy"
    return sp.csr_matrix(np.asarray(x, dtype=np.float32)), "scanpy"


def build_reference(matrix, gene_flags: np.ndarray, args: argparse.Namespace) -> tuple[dict, str]:
    mito_mask = (gene_flags & 1) != 0
    if sp.issparse(matrix):
        csr = matrix.tocsr(copy=False)
        cell_total = np.asarray(csr.sum(axis=1)).ravel().astype(np.float32, copy=False)
        cell_max = np.asarray(csr.max(axis=1).toarray()).ravel().astype(np.float32, copy=False)
        cell_detected = np.diff(csr.indptr).astype(np.uint32, copy=False)
        if np.any(mito_mask):
            mito_counts = np.asarray(csr[:, mito_mask].sum(axis=1)).ravel().astype(np.float32, copy=False)
        else:
            mito_counts = np.zeros(csr.shape[0], dtype=np.float32)
    else:
        dense = np.asarray(matrix, dtype=np.float32)
        cell_total = dense.sum(axis=1, dtype=np.float64).astype(np.float32)
        cell_max = dense.max(axis=1).astype(np.float32)
        cell_detected = (dense != 0).sum(axis=1, dtype=np.uint32)
        mito_counts = dense[:, mito_mask].sum(axis=1, dtype=np.float64).astype(np.float32) if np.any(mito_mask) else np.zeros(dense.shape[0], dtype=np.float32)

    mito_fraction = np.divide(
        mito_counts.astype(np.float64),
        np.maximum(cell_total.astype(np.float64), 1e-12),
        out=np.zeros_like(cell_total, dtype=np.float64),
    )
    cell_keep = (
        (cell_total.astype(np.float64) >= float(args.min_counts))
        & (cell_detected.astype(np.uint64) >= int(args.min_genes))
        & (mito_fraction <= float(args.max_mito_fraction))
    ).astype(np.uint8)

    kept_mask = cell_keep != 0
    kept_cells = int(np.count_nonzero(kept_mask))
    gene_sum = np.zeros(gene_flags.shape[0], dtype=np.float32)
    gene_sq_sum = np.zeros(gene_flags.shape[0], dtype=np.float32)
    gene_detected_cells = np.zeros(gene_flags.shape[0], dtype=np.float32)
    baseline_mode = "python_reference"

    if kept_cells != 0:
        if sp.issparse(matrix):
            x_kept = matrix[kept_mask].tocsr(copy=True).astype(np.float32, copy=False)
            normalized, baseline_mode = maybe_scanpy_normalize_log1p(x_kept, float(args.target_sum))
            if normalized is None:
                row_nnz = np.diff(x_kept.indptr)
                scales = np.divide(
                    float(args.target_sum),
                    np.maximum(cell_total[kept_mask].astype(np.float64), 1e-12),
                    out=np.zeros(kept_cells, dtype=np.float64),
                ).astype(np.float32)
                x_kept.data *= np.repeat(scales, row_nnz)
                np.log1p(x_kept.data, out=x_kept.data)
                normalized = x_kept
            gene_sum = np.asarray(normalized.sum(axis=0)).ravel().astype(np.float32, copy=False)
            squared = normalized.copy()
            squared.data *= squared.data
            gene_sq_sum = np.asarray(squared.sum(axis=0)).ravel().astype(np.float32, copy=False)
            gene_detected_cells = np.diff(normalized.tocsc(copy=True).indptr).astype(np.float32, copy=False)
        else:
            dense_kept = np.asarray(matrix[kept_mask], dtype=np.float32).copy()
            scales = np.divide(
                float(args.target_sum),
                np.maximum(cell_total[kept_mask].astype(np.float64), 1e-12),
                out=np.zeros(kept_cells, dtype=np.float64),
            ).astype(np.float32)
            dense_kept *= scales[:, None]
            np.log1p(dense_kept, out=dense_kept)
            gene_sum = dense_kept.sum(axis=0, dtype=np.float64).astype(np.float32)
            gene_sq_sum = np.square(dense_kept, dtype=np.float32).sum(axis=0, dtype=np.float64).astype(np.float32)
            gene_detected_cells = (dense_kept != 0).sum(axis=0, dtype=np.float64).astype(np.float32)

    mean = np.divide(
        gene_sum.astype(np.float64),
        float(max(kept_cells, 1)),
        out=np.zeros_like(gene_sum, dtype=np.float64),
    )
    variance = np.maximum(
        np.divide(
            gene_sq_sum.astype(np.float64),
            float(max(kept_cells, 1)),
            out=np.zeros_like(gene_sq_sum, dtype=np.float64),
        ) - mean * mean,
        0.0,
    )
    gene_keep = (
        (gene_sum.astype(np.float64) >= float(args.min_gene_sum))
        & (gene_detected_cells.astype(np.float64) >= float(args.min_detected_cells))
        & (variance >= float(args.min_variance))
    ).astype(np.uint8)

    return {
        "cell_total_counts": cell_total,
        "cell_mito_counts": mito_counts,
        "cell_max_counts": cell_max,
        "cell_detected_genes": cell_detected,
        "cell_keep": cell_keep,
        "gene_sum": gene_sum,
        "gene_sq_sum": gene_sq_sum,
        "gene_detected_cells": gene_detected_cells,
        "gene_keep": gene_keep,
        "gene_flags": gene_flags.astype(np.uint8, copy=False),
        "kept_cells": kept_cells,
        "kept_genes": int(np.count_nonzero(gene_keep)),
    }, baseline_mode


def float_metrics(actual: np.ndarray, reference: np.ndarray) -> dict:
    actual64 = actual.astype(np.float64, copy=False)
    ref64 = reference.astype(np.float64, copy=False)
    abs_diff = np.abs(actual64 - ref64)
    denom = np.maximum(np.abs(ref64), 1e-8)
    rel_diff = abs_diff / denom
    return {
        "max_abs": float(abs_diff.max(initial=0.0)),
        "mean_abs": float(abs_diff.mean() if abs_diff.size else 0.0),
        "max_rel": float(rel_diff.max(initial=0.0)),
        "mean_rel": float(rel_diff.mean() if rel_diff.size else 0.0),
    }


def mask_metrics(actual: np.ndarray, reference: np.ndarray) -> dict:
    mismatches = np.flatnonzero(actual != reference)
    preview = mismatches[:8].astype(np.int64).tolist()
    return {
        "mismatch_count": int(mismatches.size),
        "first_mismatches": preview,
    }


def compare_branch(actual: dict, reference: dict) -> dict:
    return {
        "cell_total_counts": float_metrics(actual["cell_total_counts"], reference["cell_total_counts"]),
        "cell_mito_counts": float_metrics(actual["cell_mito_counts"], reference["cell_mito_counts"]),
        "cell_max_counts": float_metrics(actual["cell_max_counts"], reference["cell_max_counts"]),
        "cell_detected_genes": float_metrics(actual["cell_detected_genes"], reference["cell_detected_genes"]),
        "cell_keep": mask_metrics(actual["cell_keep"], reference["cell_keep"]),
        "gene_sum": float_metrics(actual["gene_sum"], reference["gene_sum"]),
        "gene_sq_sum": float_metrics(actual["gene_sq_sum"], reference["gene_sq_sum"]),
        "gene_detected_cells": float_metrics(actual["gene_detected_cells"], reference["gene_detected_cells"]),
        "gene_keep": mask_metrics(actual["gene_keep"], reference["gene_keep"]),
    }


def flatten(prefix: str, value, output: dict):
    if isinstance(value, dict):
        for key, child in value.items():
            flatten(f"{prefix}.{key}" if prefix else key, child, output)
        return
    if isinstance(value, list):
        output[prefix] = ",".join(str(item) for item in value)
        return
    output[prefix] = value


def main() -> int:
    args = parse_args()
    blocked = load_analysis_blob(args.blocked_analysis)
    sliced = load_analysis_blob(args.sliced_analysis)
    if blocked["rows"] != sliced["rows"] or blocked["cols"] != sliced["cols"]:
        raise ValueError("blocked and sliced analysis blobs disagree on matrix shape")
    if not np.array_equal(blocked["gene_flags"], sliced["gene_flags"]):
        raise ValueError("blocked and sliced analysis blobs disagree on gene_flags")

    matrix = load_matrix(args.h5ad, args.matrix_source)
    reference, baseline_mode = build_reference(matrix, blocked["gene_flags"], args)

    details = {
        "baseline_mode": baseline_mode,
        "reference": {
            "rows": int(blocked["rows"]),
            "cols": int(blocked["cols"]),
            "kept_cells": int(reference["kept_cells"]),
            "kept_genes": int(reference["kept_genes"]),
        },
        "blocked": compare_branch(blocked, reference),
        "sliced": compare_branch(sliced, reference),
    }

    summary = {}
    flatten("", details, summary)
    Path(args.details_json).write_text(json.dumps(details, indent=2), encoding="utf-8")
    with open(args.summary_tsv, "w", encoding="utf-8") as handle:
        for key in sorted(summary):
            handle.write(f"{key}\t{summary[key]}\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"preprocess_scanpy_reference.py: {exc}", file=sys.stderr)
        raise
