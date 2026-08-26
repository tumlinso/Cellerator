#!/usr/bin/env python3
"""Build and verify the CE-LIVE PBMC3K quantitative test fixture.

This is fixture tooling, not a Cellerator input adapter.  It understands only
the checksum-pinned legacy CSR H5AD shape documented by CE-LIVE-12.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA = "cellerator.ce-live.quantitative-fixture.v1"
SOURCE_SHA256 = "89a96f1beaa2dd83a687666d3f19a4513ac27a2a2d12581fcd77afed7ea653a1"
MASK64 = (1 << 64) - 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return value ^ (value >> 31)


def choose_rows(total_rows: int, requested_rows: int, seed: int) -> list[int]:
    if requested_rows <= 0 or requested_rows > total_rows:
        raise ValueError("selected row count is outside source bounds")
    ranked = ((splitmix64(seed ^ row), row) for row in range(total_rows))
    return sorted(row for _, row in heapq.nsmallest(requested_rows, ranked))


def string_sequence_digest(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
    return digest.hexdigest()


def array_digest(values: Any, dtype: str) -> str:
    import numpy as np  # type: ignore

    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes()).hexdigest()


def identity(label: str, *parts: str) -> str:
    return string_sequence_digest((SCHEMA, label, *parts))


def decode_field(records: Any, field: str) -> list[str]:
    return [bytes(value[field]).rstrip(b"\0").decode("utf-8") for value in records]


def stress_values(rows: Sequence[int], indptr: Any, indices: Any) -> Any:
    """Return deterministic, non-biological fp32 values on unchanged support."""
    import numpy as np  # type: ignore

    result = np.empty(int(indptr[-1]), dtype="<f4")
    for local_row, source_row in enumerate(rows):
        begin, end = int(indptr[local_row]), int(indptr[local_row + 1])
        for edge in range(begin, end):
            word = splitmix64(
                0xCE1A1E02
                ^ (source_row << 32)
                ^ int(indices[edge])
                ^ (edge - begin)
            )
            numerator = int(word % 2001) - 1000
            if numerator == 0:
                numerator = 1
            result[edge] = np.float32(numerator / 257.0)
    return result


def load_source(path: Path, selected_count: int = 512, seed: int = 7) -> dict[str, Any]:
    try:
        import h5py  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as error:
        raise RuntimeError("CE-LIVE fixture tooling requires h5py and numpy") from error

    source_digest = sha256_file(path)
    if source_digest != SOURCE_SHA256:
        raise ValueError(f"PBMC3K source checksum mismatch: {source_digest}")

    with h5py.File(path, "r") as handle:
        if "X" not in handle or not isinstance(handle["X"], h5py.Group):
            raise ValueError("expected the checksum-pinned CSR group at /X")
        matrix = handle["X"]
        if not {"data", "indices", "indptr"} <= set(matrix):
            raise ValueError("/X is not the expected CSR group")
        shape = matrix.attrs.get("h5sparse_shape")
        if shape is None or tuple(int(value) for value in shape) != (2700, 32738):
            raise ValueError("unexpected /X h5sparse_shape")
        if matrix.attrs.get("h5sparse_format") not in ("csr", b"csr"):
            raise ValueError("unexpected /X h5sparse_format")

        source_data = matrix["data"][:]
        source_indices = matrix["indices"][:]
        source_indptr = matrix["indptr"][:]
        if source_data.dtype != np.dtype("float32"):
            raise ValueError("expected stored /X/data dtype float32")
        if len(source_data) != len(source_indices) or len(source_indptr) != 2701:
            raise ValueError("invalid source CSR extents")
        if int(source_indptr[0]) != 0 or int(source_indptr[-1]) != len(source_data):
            raise ValueError("invalid source CSR offsets")
        if np.any(source_indptr[1:] < source_indptr[:-1]):
            raise ValueError("source CSR offsets are not monotonic")
        if np.any(source_indices < 0) or np.any(source_indices >= 32738):
            raise ValueError("source CSR column is outside the feature axis")

        observations = decode_field(handle["obs"][:], "index")
        variables = handle["var"][:]
        features = decode_field(variables, "index")
        gene_ids = decode_field(variables, "gene_ids")

    selected_rows = choose_rows(2700, selected_count, seed)
    counts = [int(source_indptr[row + 1] - source_indptr[row]) for row in selected_rows]
    indptr = np.zeros(selected_count + 1, dtype="<u8")
    indptr[1:] = np.cumsum(counts, dtype=np.uint64)
    indices = np.concatenate(
        [source_indices[source_indptr[row]:source_indptr[row + 1]] for row in selected_rows]
    ).astype("<u4", copy=False)
    values = np.concatenate(
        [source_data[source_indptr[row]:source_indptr[row + 1]] for row in selected_rows]
    ).astype("<f4", copy=False)
    selected_ids = [observations[row] for row in selected_rows]
    stress = stress_values(selected_rows, indptr, indices)

    feature_digest = string_sequence_digest(features)
    gene_digest = string_sequence_digest(gene_ids)
    row_digest = string_sequence_digest(selected_ids)
    support_indptr_digest = array_digest(indptr, "<u8")
    support_indices_digest = array_digest(indices, "<u4")
    stored_values_digest = array_digest(values, "<f4")
    stress_digest = array_digest(stress, "<f4")
    structure_id = identity(
        "structure", source_digest, support_indptr_digest, support_indices_digest
    )

    return {
        "schema": SCHEMA,
        "source": {
            "id": "pbmc3k-raw-local",
            "relative_path": "data/test/reference/pbmc3k_raw.h5ad",
            "sha256": source_digest,
            "size_bytes": path.stat().st_size,
            "matrix_path": "/X",
            "matrix_encoding": "legacy-h5ad-csr",
            "stored_dtype": "float32",
            "stored_orientation": "observations_by_features",
            "shape": [2700, 32738],
        },
        "selection": {
            "algorithm": "splitmix64-rank-v1",
            "seed": seed,
            "source_row_count": 2700,
            "selected_row_count": selected_count,
            "selected_source_rows": selected_rows,
            "selected_source_rows_csv_sha256": hashlib.sha256(
                ",".join(str(row) for row in selected_rows).encode("ascii")
            ).hexdigest(),
            "selected_source_rows_u32le_sha256": array_digest(selected_rows, "<u4"),
            "selected_observation_id_field": "/obs:index",
            "selected_observation_ids_sha256": row_digest,
        },
        "features": {
            "count": len(features),
            "index_field": "/var:index",
            "index_encoding": "uint64-length-prefixed-utf8-sequence",
            "index_sha256": feature_digest,
            "gene_id_field": "/var:gene_ids",
            "gene_ids_sha256": gene_digest,
        },
        "extracted_csr": {
            "shape": [selected_count, len(features)],
            "nnz": len(values),
            "row_order": "ascending-source-row-after-selection",
            "edge_order": "source-csr-order-within-row",
            "indptr_encoding": "uint64-little-endian",
            "indptr_sha256": support_indptr_digest,
            "indices_encoding": "uint32-little-endian",
            "indices_sha256": support_indices_digest,
            "values_encoding": "ieee754-float32-little-endian",
            "values_sha256": stored_values_digest,
        },
        "stored_value_validation": {
            "scope": "all-/X/data-values",
            "count": len(source_data),
            "finite": bool(np.all(np.isfinite(source_data))),
            "non_negative": bool(np.all(source_data >= 0)),
            "integral_exact_in_float32": bool(np.all(source_data == np.floor(source_data))),
            "minimum": float(source_data.min()),
            "maximum": float(source_data.max()),
            "sum_float64": float(source_data.astype(np.float64).sum()),
            "exactly_representable_after_float32_cast": True,
        },
        "generations": [
            {
                "generation": 1,
                "meaning": "validated stored integral values; explicit float32 identity cast",
                "values_sha256": stored_values_digest,
                "value_generation_id": identity(
                    "value-generation-1", structure_id, stored_values_digest
                ),
            },
            {
                "generation": 2,
                "meaning": "deterministic numerical stress only; not a biological transformation",
                "rule": "float32(nonzero((splitmix64(0xCE1A1E02 xor (source_row << 32) xor feature xor row_edge_ordinal) mod 2001)-1000)/257)",
                "values_sha256": stress_digest,
                "value_generation_id": identity(
                    "value-generation-2", structure_id, stress_digest
                ),
            },
        ],
        "identities": {
            "observation_domain_id": identity("observation-domain", source_digest),
            "feature_domain_id": identity("feature-domain", source_digest, feature_digest),
            "observation_order_id": identity("observation-order", row_digest),
            "feature_order_id": identity("feature-order", feature_digest),
            "geometry_id": identity(
                "geometry", row_digest, feature_digest, support_indices_digest
            ),
            "partition_id": identity("single-local-partition", str(selected_count)),
            "structure_id": structure_id,
        },
        "scientific_limit": (
            "Computational fixture only. It carries no donor, sample, chemistry, "
            "species, normalization, comparison, or interpretation claim."
        ),
    }


def verify_manifest(manifest: dict[str, Any], actual: dict[str, Any]) -> None:
    if manifest != actual:
        raise ValueError("fixture manifest does not match the checksum-pinned source")


def write_npz(path: Path, source: Path, manifest: dict[str, Any]) -> None:
    import h5py  # type: ignore
    import numpy as np  # type: ignore

    rows = manifest["selection"]["selected_source_rows"]
    with h5py.File(source, "r") as handle:
        matrix = handle["X"]
        source_indptr = matrix["indptr"][:]
        counts = [int(source_indptr[row + 1] - source_indptr[row]) for row in rows]
        indptr = np.zeros(len(rows) + 1, dtype="<u8")
        indptr[1:] = np.cumsum(counts, dtype=np.uint64)
        indices = np.concatenate(
            [matrix["indices"][source_indptr[row]:source_indptr[row + 1]] for row in rows]
        ).astype("<u4", copy=False)
        values = np.concatenate(
            [matrix["data"][source_indptr[row]:source_indptr[row + 1]] for row in rows]
        ).astype("<f4", copy=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        source_rows=np.asarray(rows, dtype="<u4"),
        indptr=indptr,
        indices=indices,
        generation_1_values=values,
        generation_2_values=stress_values(rows, indptr, indices),
    )


def csr_forward(
    indptr: Sequence[int],
    indices: Sequence[int],
    values: Sequence[float],
    x: Sequence[float],
) -> list[float]:
    return [
        math.fsum(
            float(values[edge]) * float(x[int(indices[edge])])
            for edge in range(int(indptr[row]), int(indptr[row + 1]))
        )
        for row in range(len(indptr) - 1)
    ]


def coordinate_forward(
    indptr: Sequence[int],
    indices: Sequence[int],
    values: Sequence[float],
    x: Sequence[float],
) -> list[float]:
    result = [0.0] * (len(indptr) - 1)
    for row in range(len(result)):
        products = []
        for edge in range(int(indptr[row]), int(indptr[row + 1])):
            products.append(float(values[edge]) * float(x[int(indices[edge])]))
        result[row] = sum(products)
    return result


def csr_transpose(
    indptr: Sequence[int],
    indices: Sequence[int],
    values: Sequence[float],
    x: Sequence[float],
    columns: int,
) -> list[float]:
    result = [0.0] * columns
    for row in range(len(indptr) - 1):
        for edge in range(int(indptr[row]), int(indptr[row + 1])):
            result[int(indices[edge])] += float(values[edge]) * float(x[row])
    return result


def coordinate_transpose(
    indptr: Sequence[int],
    indices: Sequence[int],
    values: Sequence[float],
    x: Sequence[float],
    columns: int,
) -> list[float]:
    edges: list[tuple[int, int, float]] = []
    for row in range(len(indptr) - 1):
        edges.extend(
            (int(indices[edge]), row, float(values[edge]))
            for edge in range(int(indptr[row]), int(indptr[row + 1]))
        )
    return [
        math.fsum(
            value * float(x[row])
            for column, row, value in edges
            if column == feature
        )
        for feature in range(columns)
    ]


def verify_smoke(path: Path) -> None:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    if fixture["schema"] != "cellerator.ce-live.quantitative-smoke.v1":
        raise ValueError("unexpected smoke fixture schema")
    csr = fixture["csr"]
    first = csr_forward(
        csr["indptr"], csr["indices"], csr["values"], fixture["dense_input"]
    )
    second = coordinate_forward(
        csr["indptr"], csr["indices"], csr["values"], fixture["dense_input"]
    )
    if first != fixture["expected_forward"] or second != fixture["expected_forward"]:
        raise ValueError("independent smoke referees disagree")
    columns = int(csr["shape"][1])
    first_t = csr_transpose(
        csr["indptr"], csr["indices"], csr["values"], fixture["transpose_input"], columns
    )
    second_t = coordinate_transpose(
        csr["indptr"], csr["indices"], csr["values"], fixture["transpose_input"], columns
    )
    if first_t != fixture["expected_transpose"] or second_t != fixture["expected_transpose"]:
        raise ValueError("independent transpose smoke referees disagree")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("manifest", "verify", "extract"):
        child = subparsers.add_parser(command)
        child.add_argument("--source", type=Path, required=True)
        if command == "verify":
            child.add_argument("--manifest", type=Path, required=True)
        if command == "extract":
            child.add_argument("--manifest", type=Path, required=True)
            child.add_argument("--output", type=Path, required=True)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--fixture", type=Path, required=True)
    arguments = parser.parse_args()

    if arguments.command == "smoke":
        verify_smoke(arguments.fixture)
        print("CE_LIVE_QUANTITATIVE_SMOKE_OK")
        return 0

    actual = load_source(arguments.source)
    if arguments.command == "manifest":
        json.dump(actual, sys.stdout, indent=2)
        print()
        return 0
    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    verify_manifest(manifest, actual)
    if arguments.command == "extract":
        write_npz(arguments.output, arguments.source, manifest)
        print(f"wrote local representative fixture: {arguments.output}")
    else:
        print("CELLERATOR_QUANTITATIVE_FIXTURE_V1_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
