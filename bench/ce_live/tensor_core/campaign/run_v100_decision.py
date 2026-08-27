#!/usr/bin/env python3
"""Run the bounded CE-LIVE-32 V100 promotion/rejection campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EXTENT = 16
SLOTS = EXTENT * EXTENT
QUALIFICATION_NNZ = SLOTS // 2


def array_digest(values: list[int], width: int) -> str:
    return hashlib.sha256(b"".join(
        int(value).to_bytes(width, "little") for value in values
    )).hexdigest()


def density_bucket(nnz: int) -> str:
    if nnz == 0:
        return "empty"
    if nnz < 64:
        return "low"
    if nnz < 128:
        return "medium"
    if nnz < 192:
        return "high"
    return "near_dense"


def classify(trace: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    rows = int(trace["row_count"])
    features = int(trace["column_count"])
    offsets = [int(value) for value in trace["row_offsets"]]
    indices = [int(value) for value in trace["column_indices"]]
    expected = manifest["extracted_csr"]
    if [rows, features] != expected["shape"] or len(indices) != expected["nnz"]:
        raise ValueError("trace shape does not match CE-LIVE fixture manifest")
    if array_digest(offsets, 8) != expected["indptr_sha256"]:
        raise ValueError("trace row offsets do not match fixture manifest")
    if array_digest(indices, 4) != expected["indices_sha256"]:
        raise ValueError("trace source indices do not match fixture manifest")

    occupied: dict[tuple[int, int], int] = defaultdict(int)
    for row in range(rows):
        for edge in range(offsets[row], offsets[row + 1]):
            occupied[(row // EXTENT, indices[edge] // EXTENT)] += 1
    buckets = Counter(density_bucket(nnz) for nnz in occupied.values())
    qualifying = [nnz for nnz in occupied.values() if nnz >= QUALIFICATION_NNZ]
    full_row_tiles = rows // EXTENT
    full_feature_tiles = features // EXTENT
    full_slots = full_row_tiles * full_feature_tiles * SLOTS
    return {
        "rows": rows,
        "features": features,
        "logical_edges": len(indices),
        "occupied_fragment_tiles": len(occupied),
        "density_buckets": {
            name: buckets.get(name, 0)
            for name in ("empty", "low", "medium", "high", "near_dense")
        },
        "qualification_threshold_nnz": QUALIFICATION_NNZ,
        "qualified_fragment_tiles": len(qualifying),
        "qualified_logical_edges": sum(qualifying),
        "maximum_tile_nnz": max(occupied.values(), default=0),
        "maximum_tile_density": max(occupied.values(), default=0) / SLOTS,
        "whole_structure_full_fragment_slots": full_slots,
        "whole_structure_padding_ratio": full_slots / len(indices),
        "row_tail": rows % EXTENT,
        "feature_tail": features % EXTENT,
    }


def run_baselines(options: argparse.Namespace, raw_path: Path) -> list[dict[str, Any]]:
    command = [
        "python3", str(options.runner), "--binary", str(options.binary),
        "--trace", str(options.trace), "--output", str(raw_path),
        "--warmups", str(options.warmups), "--repeats", str(options.repeats),
    ]
    for width in options.width:
        command.extend(("--n", str(width)))
    subprocess.run(command, check=True)
    records = [json.loads(line) for line in raw_path.read_text().splitlines()
        if line.strip()]
    if not records or any(not record.get("correct", False) for record in records):
        raise RuntimeError("strong baseline campaign did not pass correctness")
    return records


def summarize_baselines(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_width: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_width[int(record["n"])].append(record)
    summary: dict[str, Any] = {}
    for width, candidates in sorted(by_width.items()):
        one_shot = min(candidates, key=lambda record:
            float(record["median_total_ms"])
            + float(record["query_ms"])
            + float(record["projection_build_ms"])
            + float(record["value_pack_ms"])
            + float(record["backend_prepare_ms"]))
        reuse_eight = min(candidates,
            key=lambda record: float(record["amortized_total_ms"]))
        summary[str(width)] = {
            "one_shot_winner": one_shot["candidate"],
            "one_shot_complete_ms": (
                float(one_shot["median_total_ms"])
                + float(one_shot["query_ms"])
                + float(one_shot["projection_build_ms"])
                + float(one_shot["value_pack_ms"])
                + float(one_shot["backend_prepare_ms"])
            ),
            "reuse_8_winner": reuse_eight["candidate"],
            "reuse_8_amortized_ms": float(reuse_eight["amortized_total_ms"]),
            "candidates": [record["candidate"] for record in candidates],
            "maximum_mad_percent": max(
                float(record["mad_percent"]) for record in candidates),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--width", type=int, action="append", required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=11)
    options = parser.parse_args()
    if options.width != [16, 32, 64]:
        raise SystemExit("CE-LIVE-32 requires ordered widths 16, 32, and 64")
    if options.warmups < 1 or options.repeats < 5:
        raise SystemExit("campaign timing sample is too small")

    trace = json.loads(options.trace.read_text())
    manifest = json.loads(options.manifest.read_text())
    classification = classify(trace, manifest)
    options.raw_output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ce-live-32-") as directory:
        temporary_raw = Path(directory) / "baselines.jsonl"
        records = run_baselines(options, temporary_raw)
        options.raw_output.write_bytes(temporary_raw.read_bytes())

    promoted = classification["qualified_fragment_tiles"] != 0
    # Qualification is necessary, not sufficient. A future campaign would
    # still need a complete-cost win before promotion.
    decision = "inconclusive" if promoted else "measured_rejection"
    output = {
        "schema": "CELLERATOR-TENSOR-CORE-DECISION/1",
        "task": "CE-LIVE-32",
        "candidate": "v100-wmma-dense-fragment-f16-f32",
        "architecture": "sm_70",
        "logical_relation": "feature-source-to-row-destination",
        "fixture": {
            "trace_id": trace["trace_id"],
            "structure_id": manifest["identities"]["structure_id"],
            "computational_only": True,
        },
        "classification": classification,
        "baseline_summary": summarize_baselines(records),
        "numeric_policy": "f16 relation and dense input, f32 accumulation/output",
        "generations_validated_by_candidate_test": [1, 2],
        "decision": decision,
        "registered_in_builtin_catalog": False,
        "reason": (
            "No PBMC3K 16x16 relation tile reaches the frozen 50% shortlist "
            "threshold; the legal candidate owns no work, while global dense "
            "materialization would pad the relation by the reported ratio."
        ),
        "dense_library_baseline_applicability": (
            "not timed: whole-structure densification is rejected by the "
            "frozen CE-LIVE-16 contract before candidate selection"
        ),
        "candidate_kernel_timing_applicability": (
            "not timed on PBMC3K: zero legal qualified fragments"
        ),
    }
    if decision != "measured_rejection":
        raise RuntimeError("fixture classification did not support rejection")
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
