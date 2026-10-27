#!/usr/bin/env python3
"""Freeze selected large-retrieval fixture arrays and nested query/oracle CSR."""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from bench.sparse_support.scaled_fixtures import profile  # noqa: E402


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_array(directory: pathlib.Path, name: str, array: np.ndarray) -> dict[str, object]:
    value = np.ascontiguousarray(array)
    dtype = value.dtype.newbyteorder("<")
    little = value.astype(dtype, copy=False)
    payload = little.tobytes(order="C")
    path = directory / f"{name}.bin"
    path.write_bytes(payload)
    return {
        "file": path.name,
        "dtype": dtype.str,
        "shape": list(value.shape),
        "nbytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def build_queries(fixture, query_sizes: tuple[int, ...], seed: int):
    cells = int(fixture.metadata["cells"])
    entities = int(fixture.metadata["entities"])
    source_indptr = fixture["source_edge_indptr"]
    source_ordinals = fixture["source_edge_ordinals"]
    edge_ids = fixture["logical_edge_id"]
    base_offsets = fixture["cell_query_indptr"]
    base_terms = fixture["cell_query_entities"]
    all_cases = {}

    extensions = []
    for cell in range(cells):
        begin, end = int(base_offsets[cell]), int(base_offsets[cell + 1])
        base = np.asarray(base_terms[begin:end], dtype=np.int64)
        if np.unique(base).size != base.size:
            raise RuntimeError("fixture cell query unexpectedly contains duplicates")
        available = np.setdiff1d(np.arange(entities, dtype=np.int64), base,
                                 assume_unique=False)
        rng = np.random.default_rng(seed ^ (0x9E3779B97F4A7C15 + cell * 0xD1B54A32D192ED03))
        extensions.append((base, rng.permutation(available)))

    for q in query_sizes:
        unique_rows = []
        duplicate_rows = []
        for base, extension in extensions:
            extra = max(0, q - base.size)
            row = np.concatenate((base[:q], extension[:extra])) if q < base.size else np.concatenate((base, extension[:extra]))
            row.sort()
            if row.size != q or np.unique(row).size != q:
                raise RuntimeError(f"failed to make {q} distinct query terms")
            unique_rows.append(row.astype(np.int64, copy=False))
            if q >= 32:
                duplicate = np.concatenate((row[:-4], row[:4])).astype(np.int64, copy=False)
                duplicate.sort()
                duplicate_rows.append(duplicate)

        variants = {"unique": unique_rows}
        if q >= 32:
            variants["duplicate4"] = duplicate_rows
        for variant, rows in variants.items():
            offsets = np.arange(cells + 1, dtype=np.int64) * q
            terms = np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)
            oracle_offsets = np.empty(cells + 1, dtype=np.int64)
            oracle_offsets[0] = 0
            oracle_rows = []
            raw_postings_per_row = np.empty(cells, dtype=np.uint64)
            degree = np.diff(source_indptr)
            for cell, row in enumerate(rows):
                mark = np.zeros(int(fixture.metadata["edges"]), dtype=np.bool_)
                raw_count = 0
                for entity in row:
                    lo, hi = int(source_indptr[entity]), int(source_indptr[entity + 1])
                    posting = source_ordinals[lo:hi]
                    raw_count += posting.size
                    mark[posting] = True
                oracle = np.flatnonzero(mark).astype(np.uint64, copy=False)
                oracle = oracle[np.argsort(edge_ids[oracle], kind="stable")]
                oracle_rows.append(oracle)
                oracle_offsets[cell + 1] = oracle_offsets[cell] + oracle.size
                raw_postings_per_row[cell] = raw_count
            oracle_ordinals = np.concatenate(oracle_rows) if oracle_rows else np.empty(0, dtype=np.uint64)
            case_name = f"q{q}_{variant}"
            all_cases[case_name] = {
                "query_offsets": offsets,
                "query_terms": terms,
                "oracle_offsets": oracle_offsets,
                "oracle_edge_ordinals": oracle_ordinals,
                "raw_posting_count_per_row": raw_postings_per_row,
                "required_per_row": np.diff(oracle_offsets).astype(np.uint64),
            }
    return all_cases


def main() -> None:
    out = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "bench/sparse_support/experiments/csr_union_bitmap/input/large-retrieval"
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    fixture = profile("large-retrieval", seed=20261027, layout="identity")
    fixture_build_ms = (time.perf_counter() - started) * 1000.0
    if (fixture.metadata["edges"], fixture.metadata["entities"], fixture.metadata["cells"]) != (1_000_000, 20_000, 32):
        raise RuntimeError("scaled fixture profile dimensions changed")

    support_dir = out / "support"
    support_dir.mkdir(exist_ok=True)
    source_names = (
        "logical_edge_id", "source_entity_id", "participant_entity_id",
        "source_edge_indptr", "source_edge_ordinals", "cell_query_indptr",
        "cell_query_entities", "physical_to_logical_ordinal",
    )
    support_arrays = {name: fixture[name] for name in source_names}
    support_descriptors = {
        name: write_array(support_dir, name, array)
        for name, array in support_arrays.items()
    }

    query_begin = time.perf_counter()
    cases = build_queries(fixture, (1, 32, 128, 1024, 2048), 20261027)
    query_build_ms = (time.perf_counter() - query_begin) * 1000.0
    case_descriptors = {}
    for case_name, arrays in cases.items():
        case_dir = out / "queries" / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        case_descriptors[case_name] = {
            name: write_array(case_dir, name, array)
            for name, array in arrays.items()
        }

    generator = ROOT / "bench/sparse_support/scaled_fixtures.py"
    manifest = {
        "schema": "cellerator.csr_union_bitmap_fixture.v1",
        "generator": "bench/sparse_support/scaled_fixtures.py",
        "generator_sha256": hashlib.sha256(generator.read_bytes()).hexdigest(),
        "fixture_metadata": fixture.metadata,
        "fixture_generation_ms": fixture_build_ms,
        "query_preparation_ms": query_build_ms,
        "support_arrays": support_descriptors,
        "query_cases": case_descriptors,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        "fixture_dir": str(out),
        "generator_sha256": manifest["generator_sha256"],
        "topology_sha256": fixture.metadata["topology_sha256"],
        "query_cases": list(cases),
        "fixture_generation_ms": fixture_build_ms,
        "query_preparation_ms": query_build_ms,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
