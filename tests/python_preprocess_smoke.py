#!/usr/bin/env python3
import argparse
import importlib
import pathlib
import sys

import numpy as np


def import_cellerator(build_dir: str | None):
    try:
        return importlib.import_module("cellerator")
    except ModuleNotFoundError as exc:
        if "cellerator._cellerator" not in str(exc) or not build_dir:
            raise

    sys.modules.pop("cellerator", None)
    sys.path.insert(0, build_dir)
    extension = importlib.import_module("_cellerator")
    sys.modules["cellerator._cellerator"] = extension
    return importlib.import_module("cellerator")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default=None)
    parser.add_argument("--dataset", type=pathlib.Path, default=None)
    args = parser.parse_args()

    cellerator = import_cellerator(args.build_dir)
    assert cellerator.__version__ == "0.1.0"

    assert cellerator.validate_raw_count_state(
        assay="scrna",
        matrix_orientation="observations_by_features",
        matrix_state="raw_counts",
        raw_counts_available=True,
    )
    try:
        cellerator.validate_raw_count_state(matrix_state="normalized_log1p", raw_counts_available=True)
    except RuntimeError as exc:
        assert "raw_counts" in str(exc) or "processed" in str(exc)
    else:
        raise AssertionError("normalized input was accepted as raw")

    masks = cellerator.compile_default_qc_feature_group_masks(
        feature_ids=["ENSG_MT-ND1", "RPS3", "HBA1", "USER_GENE"],
        feature_names=["MT-ND1", "RPS3", "HBA1", "USER_GENE"],
        feature_types=["gene", "gene", "gene", "gene"],
        modalities=["rna", "rna", "rna", "rna"],
    )
    assert np.asarray(masks).dtype == np.uint32
    assert int(masks[0]) & 1
    assert int(masks[1]) & 2
    assert int(masks[2]) & 4

    plan = cellerator.pp.plan(path="input.h5ad", format="h5ad", matrix_source="counts")
    assert plan.layout == "blocked_ell"
    assert plan.adapt_to_cellshard_first

    if args.dataset is not None:
        session = cellerator.pp.preprocess(str(args.dataset))
        assert session.rows > 0
        assert session.cols > 0
        metrics = session.metrics()
        assert metrics["cell_keep"].shape[0] == session.rows
        assert metrics["gene_keep"].shape[0] == session.cols
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
