#!/usr/bin/env python3
"""Deterministic, structure-only fixtures for CE-ARCH-30 evidence.

The compact JSON representation is for reviewable smoke fixtures. Large local
outputs use Matrix Market pattern matrices and remain gitignored. Neither form
is a Cellerator runtime or persistence format.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator


SCHEMA_VERSION = 1
MASK64 = (1 << 64) - 1
MAX_COMPACT_EDGES = 1_000_000


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return value ^ (value >> 31)


def random_word(seed: int, row: int, slot: int, attempt: int = 0) -> int:
    value = seed & MASK64
    value ^= ((row + 1) * 0xD6E8FEB86659FD93) & MASK64
    value ^= ((slot + 1) * 0xA5A3564E27F8862B) & MASK64
    value ^= ((attempt + 1) * 0x9E3779B97F4A7C15) & MASK64
    return splitmix64(value)


def bounded(value: int, limit: int) -> int:
    if limit <= 0:
        raise ValueError("bounded selection requires a positive limit")
    return value % limit


def degree_for_row(spec: dict[str, Any], row: int) -> int:
    columns = int(spec["columns"])
    family = spec["family"]
    if family == "long_tail":
        minimum = int(spec.get("minimum_degree", 1))
        maximum = int(spec["maximum_degree"])
        word = random_word(int(spec["seed"]), row, 0)
        exponent = min((word & -word).bit_length() - 1, 20) if word else 20
        degree = minimum + ((maximum - minimum) >> min(exponent, 16))
    elif family == "dense_fragment":
        fragment_rows = int(spec["fragment_rows"])
        degree = int(spec["fragment_degree"] if row < fragment_rows
                     else spec.get("background_degree", 1))
    else:
        degree = int(spec["degree"])
    return max(0, min(columns, degree))


def module_interval(columns: int, module_count: int, module: int) -> tuple[int, int]:
    begin = (columns * module) // module_count
    end = (columns * (module + 1)) // module_count
    return begin, max(begin + 1, end)


def candidate_feature(spec: dict[str, Any], row: int, slot: int, attempt: int) -> int:
    columns, seed = int(spec["columns"]), int(spec["seed"])
    family = spec["family"]
    word = random_word(seed, row, slot, attempt)
    if family in {"uniform_random", "long_tail"}:
        return bounded(word, columns)
    if family in {"modular_coherent", "sequence_regulatory", "sparse_learned"}:
        modules = int(spec.get("module_count", 8))
        module = row % modules
        if family == "sparse_learned":
            module = bounded(splitmix64(row + seed), modules)
        coherence_ppm = int(spec.get("coherence_ppm", 850_000))
        if bounded(word >> 16, 1_000_000) < coherence_ppm:
            begin, end = module_interval(columns, modules, module)
            return begin + bounded(word, end - begin)
        return bounded(word, columns)
    if family == "trajectory":
        modules = int(spec.get("module_count", 8))
        module = min(modules - 1, (row * modules) // int(spec["rows"]))
        if slot & 1 and module + 1 < modules:
            module += 1
        begin, end = module_interval(columns, modules, module)
        return begin + bounded(word, end - begin)
    if family == "hub_skew":
        hubs = max(1, int(spec.get("hub_count", max(1, columns // 100))))
        hub_ppm = int(spec.get("hub_probability_ppm", 800_000))
        if bounded(word >> 16, 1_000_000) < hub_ppm:
            return bounded(word, min(hubs, columns))
        return bounded(word, columns)
    if family == "rare_features":
        common = max(1, int(spec.get("common_feature_count", columns * 3 // 4)))
        rare_period = max(1, int(spec.get("rare_period", 16)))
        if slot == 0 and row % rare_period == 0 and common < columns:
            return common + bounded(word, columns - common)
        return bounded(word, min(common, columns))
    if family == "cell_neighborhood":
        radius = max(1, int(spec.get("radius", 8)))
        cross_ppm = int(spec.get("cross_edge_ppm", 50_000))
        if bounded(word >> 16, 1_000_000) < cross_ppm:
            return bounded(word, columns)
        delta = bounded(word, 2 * radius + 1) - radius
        return (row + delta) % columns
    if family == "dense_fragment":
        fragment_columns = min(columns, int(spec["fragment_columns"]))
        if row < int(spec["fragment_rows"]):
            return bounded(word, fragment_columns)
        return bounded(word, columns)
    raise ValueError(f"unsupported synthetic family: {family}")


def row_columns(spec: dict[str, Any], row: int) -> list[int]:
    columns, degree = int(spec["columns"]), degree_for_row(spec, row)
    selected: set[int] = set()
    for slot in range(degree):
        for attempt in range(columns + 1):
            feature = candidate_feature(spec, row, slot, attempt)
            if feature not in selected:
                selected.add(feature)
                break
        else:
            raise RuntimeError("could not construct unique deterministic row")
    return sorted(selected)


def rows_from_synthetic(spec: dict[str, Any]) -> Iterable[list[int]]:
    for row in range(int(spec["rows"])):
        yield row_columns(spec, row)


def summarize_rows(rows: Iterable[list[int]], column_count: int) -> dict[str, Any]:
    row_degrees: list[int] = []
    feature_counts = [0] * column_count
    block_totals = {width: [0, 0] for width in (8, 16, 32)}
    for row in rows:
        row_degrees.append(len(row))
        for feature in row:
            feature_counts[feature] += 1
        for width, totals in block_totals.items():
            occupied = len({feature // width for feature in row})
            totals[0] += occupied
            totals[1] += occupied * width
    occupied_features = sum(count != 0 for count in feature_counts)
    nnz = sum(row_degrees)
    histogram: dict[str, int] = {}
    for degree in row_degrees:
        bucket = "0" if degree == 0 else str(1 << (degree.bit_length() - 1))
        histogram[bucket] = histogram.get(bucket, 0) + 1
    block_summaries: dict[str, dict[str, Any]] = {}
    for width, (occupied, slots) in block_totals.items():
        block_summaries[str(width)] = {
            "occupied_row_blocks": occupied,
            "scalar_occupancy": 0.0 if slots == 0 else nnz / slots,
        }
    return {
        "nnz": nnz,
        "row_degree_min": min(row_degrees, default=0),
        "row_degree_max": max(row_degrees, default=0),
        "row_degree_mean": 0.0 if not row_degrees else nnz / len(row_degrees),
        "row_degree_histogram_power2_floor": histogram,
        "occupied_feature_count": occupied_features,
        "maximum_feature_frequency": max(feature_counts, default=0),
        "block_width_summaries": block_summaries,
    }


def canonical_payload_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def compact_trace(
    trace_id: str,
    rows: list[list[int]],
    column_count: int,
    axes: dict[str, str],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    row_offsets, column_indices = [0], []
    for row in rows:
        column_indices.extend(row)
        row_offsets.append(len(column_indices))
    if len(column_indices) > MAX_COMPACT_EDGES:
        raise ValueError(
            f"compact JSON is capped at {MAX_COMPACT_EDGES} edges; "
            "use Matrix Market for local large traces"
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "trace_id": trace_id,
        "trace_kind": "bipartite_support",
        "value_semantics": "structure_only",
        "row_count": len(rows),
        "column_count": column_count,
        "row_axis": axes["row_axis"],
        "column_axis": axes["column_axis"],
        "row_offsets": row_offsets,
        "column_indices": column_indices,
        "statistics": summarize_rows(rows, column_count),
        "provenance": provenance,
    }
    result = dict(payload)
    result["payload_sha256"] = canonical_payload_digest(payload)
    return result


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_matrix_market_iter(
    path: Path,
    trace_id: str,
    rows: Iterable[list[int]],
    row_count: int,
    column_count: int,
    nnz: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("%%MatrixMarket matrix coordinate pattern general\n")
        stream.write(f"% CE-ARCH-30 trace_id={trace_id} structure_only=true\n")
        stream.write(f"{row_count} {column_count} {nnz}\n")
        for row_index, row in enumerate(rows, start=1):
            for column in row:
                stream.write(f"{row_index} {column + 1}\n")


def write_matrix_market(
    path: Path,
    trace_id: str,
    rows: list[list[int]],
    column_count: int,
) -> None:
    write_matrix_market_iter(
        path, trace_id, rows, len(rows), column_count,
        sum(len(row) for row in rows),
    )


def validate_compact_trace(trace: dict[str, Any]) -> None:
    required = {
        "schema_version", "trace_id", "trace_kind", "value_semantics",
        "row_count", "column_count", "row_offsets", "column_indices",
        "row_axis", "column_axis", "statistics", "provenance",
        "payload_sha256",
    }
    missing = required - trace.keys()
    if missing:
        raise ValueError(f"compact trace missing fields: {sorted(missing)}")
    if trace["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported compact trace schema")
    offsets, indices = trace["row_offsets"], trace["column_indices"]
    if len(offsets) != int(trace["row_count"]) + 1 or offsets[0] != 0:
        raise ValueError("row_offsets do not match row_count")
    if offsets[-1] != len(indices):
        raise ValueError("final row offset does not match column_indices")
    columns = int(trace["column_count"])
    for row in range(int(trace["row_count"])):
        begin, end = offsets[row], offsets[row + 1]
        if begin > end or indices[begin:end] != sorted(set(indices[begin:end])):
            raise ValueError("trace rows must be sorted and duplicate-free")
        if any(feature < 0 or feature >= columns for feature in indices[begin:end]):
            raise ValueError("trace column index is out of range")
    payload = dict(trace)
    expected = payload.pop("payload_sha256")
    if canonical_payload_digest(payload) != expected:
        raise ValueError("compact trace payload checksum mismatch")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def find_workload(manifest: dict[str, Any], workload_id: str) -> dict[str, Any]:
    for workload in manifest["workloads"]:
        if workload["id"] == workload_id:
            return workload
    raise ValueError(f"unknown workload id: {workload_id}")


def choose_rows(total_rows: int, requested_rows: int, seed: int) -> list[int]:
    if requested_rows <= 0 or requested_rows > total_rows:
        raise ValueError("requested row count is outside source bounds")
    ranked = ((splitmix64(seed ^ row), row) for row in range(total_rows))
    return sorted(row for _, row in heapq.nsmallest(requested_rows, ranked))


def h5ad_support_factory(
    path: Path, requested_rows: int, seed: int,
) -> tuple[Callable[[], Iterator[list[int]]], int, dict[str, Any]]:
    try:
        import h5py  # type: ignore
    except ImportError as error:
        raise RuntimeError("extract-h5ad requires h5py") from error
    with h5py.File(path, "r") as handle:
        matrix = handle["X"]
        if not isinstance(matrix, h5py.Group) or not {"indptr", "indices"} <= set(matrix):
            raise ValueError("H5AD X must be CSR with indptr and indices")
        shape = matrix.attrs.get("shape", matrix.attrs.get("h5sparse_shape"))
        if shape is None or len(shape) != 2:
            raise ValueError("H5AD CSR shape metadata is missing")
        total_rows, columns = int(shape[0]), int(shape[1])
        selected = choose_rows(total_rows, requested_rows, seed)

    def rows() -> Iterator[list[int]]:
        with h5py.File(path, "r") as handle:
            matrix = handle["X"]
            indptr, indices = matrix["indptr"], matrix["indices"]
            for source_row in selected:
                begin = int(indptr[source_row])
                end = int(indptr[source_row + 1])
                row = sorted(set(int(value) for value in indices[begin:end]))
                if any(value < 0 or value >= columns for value in row):
                    raise ValueError("H5AD CSR column index is out of range")
                yield row

    selection_digest = hashlib.sha256(
        ",".join(str(row) for row in selected).encode()
    ).hexdigest()
    return rows, columns, {
        "source_row_count": total_rows,
        "selected_row_count": requested_rows,
        "row_selection": "splitmix64-rank-v1",
        "row_selection_seed": seed,
        "selected_rows_sha256": selection_digest,
    }


def h5ad_support_rows(
    path: Path, requested_rows: int, seed: int,
) -> tuple[list[list[int]], int, dict[str, Any]]:
    factory, columns, provenance = h5ad_support_factory(
        path, requested_rows, seed,
    )
    return list(factory()), columns, provenance


def emit_streamed_matrix_market(
    output: Path,
    trace_id: str,
    row_count: int,
    columns: int,
    axes: dict[str, str],
    provenance: dict[str, Any],
    row_factory: Callable[[], Iterable[list[int]]],
) -> None:
    statistics = summarize_rows(row_factory(), columns)
    write_matrix_market_iter(
        output, trace_id, row_factory(), row_count, columns,
        int(statistics["nnz"]),
    )
    sidecar = {
        "schema_version": SCHEMA_VERSION,
        "trace_id": trace_id,
        "trace_kind": "bipartite_support",
        "value_semantics": "structure_only",
        "row_count": row_count,
        "column_count": columns,
        "row_axis": axes["row_axis"],
        "column_axis": axes["column_axis"],
        "statistics": statistics,
        "provenance": provenance,
        "matrix_market_sha256": file_sha256(output),
    }
    write_json(output.with_suffix(output.suffix + ".meta.json"), sidecar)


def emit_trace(
    output: Path,
    output_format: str,
    trace_id: str,
    rows: list[list[int]],
    columns: int,
    axes: dict[str, str],
    provenance: dict[str, Any],
) -> None:
    if output_format == "compact-json":
        write_json(output, compact_trace(trace_id, rows, columns, axes, provenance))
    elif output_format == "matrix-market":
        write_matrix_market(output, trace_id, rows, columns)
        sidecar = {
            "schema_version": SCHEMA_VERSION,
            "trace_id": trace_id,
            "trace_kind": "bipartite_support",
            "value_semantics": "structure_only",
            "row_count": len(rows),
            "column_count": columns,
            "row_axis": axes["row_axis"],
            "column_axis": axes["column_axis"],
            "statistics": summarize_rows(rows, columns),
            "provenance": provenance,
            "matrix_market_sha256": file_sha256(output),
        }
        write_json(output.with_suffix(output.suffix + ".meta.json"), sidecar)
    else:
        raise ValueError(f"unsupported output format: {output_format}")


def command_generate(args: argparse.Namespace) -> None:
    manifest = load_json(args.workloads)
    workload = find_workload(manifest, args.workload)
    if "synthetic" not in workload:
        raise ValueError("selected workload is not a synthetic generator workload")
    spec = dict(workload["synthetic"])
    provenance = {
        "kind": "deterministic_synthetic",
        "generator": "splitmix64-structural-v1",
        "family": spec["family"],
        "seed": int(spec["seed"]),
        "workload_manifest": args.workloads.name,
    }
    if args.format == "matrix-market":
        emit_streamed_matrix_market(
            args.output,
            workload["id"],
            int(spec["rows"]),
            int(spec["columns"]),
            workload["axes"],
            provenance,
            lambda: rows_from_synthetic(spec),
        )
        return
    emit_trace(
        args.output,
        args.format,
        workload["id"],
        list(rows_from_synthetic(spec)),
        int(spec["columns"]),
        workload["axes"],
        provenance,
    )


def command_extract(args: argparse.Namespace) -> None:
    sources = load_json(args.sources)
    source = next((item for item in sources["sources"] if item["id"] == args.source), None)
    if source is None:
        raise ValueError(f"unknown source id: {args.source}")
    if not source.get("structure_only", False):
        raise ValueError("CE-ARCH-30 H5AD extraction must be structure-only")
    repository = Path(__file__).resolve().parents[2]
    source_path = repository / source["local_path"]
    row_factory, columns, selection = h5ad_support_factory(
        source_path, args.rows, args.seed,
    )
    trace_id = f"{source['id']}-support-r{args.rows}-s{args.seed}"
    provenance = {
        "kind": "real_support_extraction",
        "source_id": source["id"],
        "source_sha256": source["sha256"],
        "values_ignored": True,
        **selection,
    }
    if args.format == "matrix-market":
        emit_streamed_matrix_market(
            args.output,
            trace_id,
            args.rows,
            columns,
            source["axes"],
            provenance,
            row_factory,
        )
        return
    emit_trace(
        args.output,
        args.format,
        trace_id,
        list(row_factory()),
        columns,
        source["axes"],
        provenance,
    )


def command_derive_block(args: argparse.Namespace) -> None:
    if args.block_width <= 0:
        raise ValueError("feature block width must be positive")
    source = load_json(args.source_trace)
    validate_compact_trace(source)
    offsets = source["row_offsets"]
    indices = source["column_indices"]
    block_counts: dict[int, int] = {}
    for feature in indices:
        block = int(feature) // args.block_width
        block_counts[block] = block_counts.get(block, 0) + 1
    if not block_counts:
        raise ValueError("source trace has no occupied feature block")
    selected = min(block_counts, key=lambda block: (-block_counts[block], block))
    rows = [
        [feature for feature in indices[offsets[row]:offsets[row + 1]]
         if feature // args.block_width == selected]
        for row in range(int(source["row_count"]))
    ]
    provenance = {
        "kind": "real_support_derived_feature_block",
        "source_trace_id": source["trace_id"],
        "source_trace_sha256": file_sha256(args.source_trace),
        "source_payload_sha256": source["payload_sha256"],
        "selection": "maximum_observed_edges_then_lowest_block",
        "feature_block_width": args.block_width,
        "selected_feature_block": selected,
        "values_ignored": True,
    }
    write_json(args.output, compact_trace(
        args.trace_id,
        rows,
        int(source["column_count"]),
        {"row_axis": source["row_axis"], "column_axis": source["column_axis"]},
        provenance,
    ))


def command_validate(args: argparse.Namespace) -> None:
    trace = load_json(args.trace)
    validate_compact_trace(trace)
    print(f"valid compact trace: {trace['trace_id']}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="generate a synthetic support trace")
    generate.add_argument("--workloads", type=Path, required=True)
    generate.add_argument("--workload", required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--format", choices=("compact-json", "matrix-market"), default="compact-json")
    generate.set_defaults(function=command_generate)
    extract = subparsers.add_parser("extract-h5ad", help="extract deterministic H5AD CSR support")
    extract.add_argument("--sources", type=Path, required=True)
    extract.add_argument("--source", required=True)
    extract.add_argument("--rows", type=int, required=True)
    extract.add_argument("--seed", type=int, default=7)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--format", choices=("compact-json", "matrix-market"), default="compact-json")
    extract.set_defaults(function=command_extract)
    derive = subparsers.add_parser(
        "derive-block", help="derive the most occupied native feature block"
    )
    derive.add_argument("--source-trace", type=Path, required=True)
    derive.add_argument("--trace-id", required=True)
    derive.add_argument("--block-width", type=int, default=16)
    derive.add_argument("--output", type=Path, required=True)
    derive.set_defaults(function=command_derive_block)
    validate = subparsers.add_parser("validate", help="validate a compact support trace")
    validate.add_argument("--trace", type=Path, required=True)
    validate.set_defaults(function=command_validate)
    return result


def main() -> int:
    try:
        arguments = parser().parse_args()
        arguments.function(arguments)
        return 0
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"trace_tool: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
