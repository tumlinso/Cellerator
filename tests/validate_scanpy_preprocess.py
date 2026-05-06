#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import pathlib
import sys
from dataclasses import dataclass

import numpy as np
import scanpy as sc
import scipy.sparse as sp


DEFAULT_GROUP_NAMES = ("mt", "ribo", "hb")


@dataclass
class CheckResult:
    name: str
    passed: bool
    max_abs: float = 0.0
    max_rel: float = 0.0
    mismatches: int = 0


def import_cellerator(build_dir: pathlib.Path | None, repo_root: pathlib.Path):
    python_dir = repo_root / "python"
    if str(python_dir) not in sys.path:
        sys.path.insert(0, str(python_dir))

    try:
        return importlib.import_module("cellerator")
    except ModuleNotFoundError as exc:
        if "cellerator._cellerator" not in str(exc) or build_dir is None:
            raise

    sys.modules.pop("cellerator", None)
    if str(build_dir) not in sys.path:
        sys.path.insert(0, str(build_dir))
    extension = importlib.import_module("_cellerator")
    sys.modules["cellerator._cellerator"] = extension
    return importlib.import_module("cellerator")


def as_csr_counts(adata) -> sp.csr_matrix:
    matrix = adata.X
    if not sp.issparse(matrix):
        matrix = sp.csr_matrix(matrix)
    matrix = matrix.tocsr().astype(np.float32)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix


def feature_ids(adata) -> list[str]:
    if "gene_ids" in adata.var:
        return [str(value) for value in adata.var["gene_ids"].to_numpy()]
    return [str(value) for value in adata.var_names]


def normalized_log1p_csr(raw: sp.csr_matrix, total_counts: np.ndarray, target_sum: float) -> sp.csr_matrix:
    normalized = raw.copy().astype(np.float32)
    rows = np.repeat(np.arange(normalized.shape[0], dtype=np.int64), np.diff(normalized.indptr))
    scales = np.zeros(normalized.shape[0], dtype=np.float32)
    np.divide(target_sum, total_counts, out=scales, where=total_counts > 0.0)
    normalized.data = np.log1p(normalized.data * scales[rows]).astype(np.float32)
    normalized.eliminate_zeros()
    return normalized


def scanpy_reference(cellerator, h5ad_path: pathlib.Path, target_sum: float) -> dict[str, np.ndarray]:
    adata = sc.read_h5ad(h5ad_path)
    raw = as_csr_counts(adata)

    ids = feature_ids(adata)
    names = [str(value) for value in adata.var_names]
    types = ["gene"] * raw.shape[1]
    modalities = ["rna"] * raw.shape[1]
    feature_masks = np.asarray(
        cellerator.compile_default_qc_feature_group_masks(
            feature_ids=ids,
            feature_names=names,
            feature_types=types,
            modalities=modalities,
        ),
        dtype=np.uint32,
    )

    total_counts = np.asarray(raw.sum(axis=1)).ravel().astype(np.float32)
    detected_genes = np.diff(raw.indptr).astype(np.uint32)
    max_counts = raw.max(axis=1).toarray().ravel().astype(np.float32)
    group_counts = np.zeros((raw.shape[0], len(DEFAULT_GROUP_NAMES)), dtype=np.float32)
    for group_index in range(len(DEFAULT_GROUP_NAMES)):
        cols = np.flatnonzero((feature_masks & np.uint32(1 << group_index)) != 0)
        if cols.size != 0:
            group_counts[:, group_index] = np.asarray(raw[:, cols].sum(axis=1)).ravel().astype(np.float32)
    group_fraction = np.divide(
        group_counts,
        total_counts[:, None],
        out=np.zeros_like(group_counts, dtype=np.float32),
        where=total_counts[:, None] > 0.0,
    )
    group_pct = group_fraction * np.float32(100.0)

    cell_keep = ((total_counts >= 1.0) & (detected_genes >= 1) & (group_fraction[:, 0] <= 1.0)).astype(np.uint8)
    normalized = normalized_log1p_csr(raw, total_counts, target_sum)
    normalized = normalized.multiply(cell_keep[:, None]).tocsr()
    normalized.eliminate_zeros()

    gene_sum = np.asarray(normalized.sum(axis=0)).ravel().astype(np.float32)
    gene_sq_sum = np.asarray(normalized.multiply(normalized).sum(axis=0)).ravel().astype(np.float32)
    gene_detected = np.diff(normalized.tocsc().indptr).astype(np.float32)
    kept_cells = float(cell_keep.sum())
    if kept_cells > 0.0:
        mean = gene_sum / kept_cells
        variance = np.maximum(gene_sq_sum / kept_cells - mean * mean, 0.0)
    else:
        variance = np.zeros(raw.shape[1], dtype=np.float32)
    gene_keep = ((gene_sum >= 0.0) & (gene_detected >= 0.0) & (variance >= 0.0)).astype(np.uint8)

    return {
        "shape": np.asarray(raw.shape, dtype=np.int64),
        "feature_group_masks": feature_masks,
        "cell_total_counts": total_counts,
        "cell_detected_genes": detected_genes,
        "cell_mito_counts": group_counts[:, 0],
        "cell_max_counts": max_counts,
        "cell_group_counts": group_counts.reshape(-1),
        "cell_group_pct": group_pct.reshape(-1),
        "cell_keep": cell_keep,
        "gene_sum": gene_sum,
        "gene_sq_sum": gene_sq_sum,
        "gene_detected_cells": gene_detected,
        "gene_keep": gene_keep,
    }


def summarize_numeric(name: str, observed: np.ndarray, expected: np.ndarray, atol: float, rtol: float) -> CheckResult:
    observed = np.asarray(observed)
    expected = np.asarray(expected)
    if observed.shape != expected.shape:
        return CheckResult(name, False, mismatches=-1)
    diff = np.abs(observed.astype(np.float64) - expected.astype(np.float64))
    denom = np.maximum(np.abs(expected.astype(np.float64)), 1.0)
    rel = diff / denom
    ok = (diff <= (atol + rtol * np.abs(expected.astype(np.float64)))) | (observed == expected)
    return CheckResult(
        name=name,
        passed=bool(np.all(ok)),
        max_abs=float(diff.max(initial=0.0)),
        max_rel=float(rel.max(initial=0.0)),
        mismatches=int(np.size(ok) - np.count_nonzero(ok)),
    )


def summarize_exact(name: str, observed: np.ndarray, expected: np.ndarray) -> CheckResult:
    observed = np.asarray(observed)
    expected = np.asarray(expected)
    if observed.shape != expected.shape:
        return CheckResult(name, False, mismatches=-1)
    ok = observed == expected
    return CheckResult(name=name, passed=bool(np.all(ok)), mismatches=int(np.size(ok) - np.count_nonzero(ok)))


def print_result(result: CheckResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    if result.max_abs or result.max_rel:
        print(
            f"{status} {result.name}: mismatches={result.mismatches} "
            f"max_abs={result.max_abs:.6g} max_rel={result.max_rel:.6g}"
        )
    else:
        print(f"{status} {result.name}: mismatches={result.mismatches}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Cellerator preprocessing metrics against a Scanpy reference.")
    parser.add_argument("--h5ad", required=True, type=pathlib.Path)
    parser.add_argument("--csh5", required=True, type=pathlib.Path)
    parser.add_argument("--build-dir", default=None, type=pathlib.Path)
    parser.add_argument("--target-sum", default=10000.0, type=float)
    parser.add_argument("--atol", default=1.0e-2, type=float)
    parser.add_argument("--rtol", default=2.0e-3, type=float)
    parser.add_argument("--gene-atol", default=8.0, type=float)
    parser.add_argument("--gene-rtol", default=8.0e-3, type=float)
    args = parser.parse_args()

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    cellerator = import_cellerator(args.build_dir, repo_root)
    reference = scanpy_reference(cellerator, args.h5ad, args.target_sum)

    options = cellerator.PreprocessOptions()
    options.target_sum = float(args.target_sum)
    session = cellerator.pp.preprocess(str(args.csh5), options=options)
    metrics = session.metrics()

    results: list[CheckResult] = [
        summarize_exact("shape", np.asarray(metrics["shape"], dtype=np.int64), reference["shape"]),
        summarize_exact("feature_group_masks", metrics["feature_group_masks"], reference["feature_group_masks"]),
        summarize_numeric("cell_total_counts", metrics["cell_total_counts"], reference["cell_total_counts"], args.atol, args.rtol),
        summarize_exact("cell_detected_genes", metrics["cell_detected_genes"], reference["cell_detected_genes"]),
        summarize_numeric("cell_mito_counts", metrics["cell_mito_counts"], reference["cell_mito_counts"], args.atol, args.rtol),
        summarize_numeric("cell_max_counts", metrics["cell_max_counts"], reference["cell_max_counts"], args.atol, args.rtol),
        summarize_numeric("cell_group_counts", metrics["cell_group_counts"], reference["cell_group_counts"], args.atol, args.rtol),
        summarize_numeric("cell_group_pct", metrics["cell_group_pct"], reference["cell_group_pct"], args.atol, args.rtol),
        summarize_exact("cell_keep", metrics["cell_keep"], reference["cell_keep"]),
        summarize_numeric("gene_sum", metrics["gene_sum"], reference["gene_sum"], args.gene_atol, args.gene_rtol),
        summarize_numeric("gene_sq_sum", metrics["gene_sq_sum"], reference["gene_sq_sum"], args.gene_atol, args.gene_rtol),
        summarize_numeric("gene_detected_cells", metrics["gene_detected_cells"], reference["gene_detected_cells"], args.atol, args.rtol),
        summarize_exact("gene_keep", metrics["gene_keep"], reference["gene_keep"]),
    ]

    print(f"Cellerator session: layout={session.layout} shape=({session.rows}, {session.cols}) nnz={session.nnz}")
    print(f"partitions_processed={session.partitions_processed} kept_cells={session.kept_cells} kept_genes={session.kept_genes}")
    for result in results:
        print_result(result)

    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
