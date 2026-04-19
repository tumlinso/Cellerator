#!/usr/bin/env python3

import pathlib
import shutil

import cellerator


def write(path: pathlib.Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def main() -> int:
    base = pathlib.Path("/tmp/cellerator_python_smoke")
    if base.exists():
        if base.is_dir():
            shutil.rmtree(base)
        else:
            base.unlink()
    base.mkdir(parents=True, exist_ok=True)

    matrix_path = base / "matrix.mtx"
    feature_path = base / "features.tsv"
    barcode_path = base / "barcodes.tsv"
    metadata_path = base / "metadata.tsv"
    manifest_path = base / "manifest.tsv"
    dataset_path = base / "dataset.csh5"
    cache_root = base / "cache"
    preprocess_cache_root = base / "preprocess-cache"

    write(
        matrix_path,
        "%%MatrixMarket matrix coordinate integer general\n"
        "2 3 3\n"
        "1 1 5\n"
        "1 3 1\n"
        "2 2 7\n",
    )
    write(feature_path, "g0\tMT-CO1\tgene\ng1\tGeneB\tgene\ng2\tGeneC\tgene\n")
    write(barcode_path, "bc0\nbc1\n")
    write(metadata_path, "day\tembryo_id\tcell_id\nE8.5\tembryo_1\tbc0\nP0\tembryo_1\tbc1\n")
    write(
        manifest_path,
        "dataset\tpath\tformat\tfeatures\tbarcodes\tmetadata\trows\tcols\tnnz\n"
        f"sample_a\t{matrix_path}\tmtx\t{feature_path}\t{barcode_path}\t{metadata_path}\t2\t3\t3\n",
    )

    inspection = cellerator.inspect_manifest(str(manifest_path))
    if not inspection.ok:
        raise RuntimeError("manifest inspection failed")
    if len(inspection.sources) != 1:
        raise RuntimeError("expected exactly one source")

    policy = cellerator.ingest_policy()
    policy.max_part_nnz = 2
    policy.convert_window_bytes = 256 << 20
    policy.target_shard_bytes = 256 << 20
    policy.output_path = str(dataset_path)
    policy.cache_dir = str(cache_root)

    plan = cellerator.plan_ingest(inspection.sources, policy)
    if not plan.ok:
        raise RuntimeError("ingest plan failed")
    if not plan.parts or not plan.shards:
        raise RuntimeError("plan did not produce partitions and shards")

    report = cellerator.convert(plan)
    if not report.ok:
        raise RuntimeError("conversion failed")

    summary = cellerator.summarize_dataset(str(dataset_path))
    if not summary.ok:
        raise RuntimeError("dataset summary failed")
    if summary.rows != 2 or summary.cols != 3 or summary.nnz != 3:
        raise RuntimeError("unexpected summary shape")

    builder = cellerator.DatasetBuilder.from_manifest(str(manifest_path))
    builder.plan(policy)
    reopened_summary = builder.inspect_output()
    if reopened_summary.rows != 2:
        raise RuntimeError("builder inspect_output did not reopen dataset")

    preprocess = cellerator.preprocess_config()
    preprocess.cache_dir = str(preprocess_cache_root)
    preprocess.min_counts = 0.0
    preprocess.min_genes = 0
    preprocess.max_mito_fraction = 1.0
    preprocess.min_gene_sum = 0.0
    preprocess.min_detected_cells = 0.0
    preprocess.min_variance = 0.0
    preprocess_result = cellerator.preprocess(str(dataset_path), preprocess)
    if not preprocess_result.ok:
        raise RuntimeError("preprocess failed")

    dataset = cellerator.open(str(dataset_path))
    if dataset.shape != (2, 3):
        raise RuntimeError("cellshard open returned the wrong shape")
    head = dataset.head(1, format="csr")
    if head.rows != 1 or head.cols != 3:
        raise RuntimeError("dataset head returned the wrong shape")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
