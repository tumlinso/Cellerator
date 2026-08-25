#!/usr/bin/env python3
"""Validate and annotate the bounded CE-ARCH-92 V100 campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


TRACE_PATHS = {
    "gse147520-local-x-support-r256-s7": Path(
        "bench/architecture_evidence/real_traces/gse147520-support-256.json"
    ),
    "gse147520-high-sharing-block-r256": Path(
        "bench/architecture_evidence/real_traces/gse147520-high-sharing-block-256.json"
    ),
    "pbmc3k-raw-local-support-r512-s7": Path(
        "bench/architecture_evidence/real_traces/pbmc3k-support-512.json"
    ),
    "adversarial-tiny-partial-blocks": Path(
        "data/manifests/architecture_evidence/smoke_traces/"
        "adversarial-tiny-partial-blocks.json"
    ),
}
WIDTHS = (1, 16, 32)
EXPECTED_CANDIDATES = {
    1: {"row_masked", "csr", "feature_major"},
    16: {"row_masked", "csr", "feature_major"},
    32: {"row_masked", "csr", "feature_major_cta"},
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--controller-evidence-id", required=True)
    options = parser.parse_args()

    traces = {}
    for trace_id, path in TRACE_PATHS.items():
        trace = json.loads(path.read_text())
        require(trace["trace_id"] == trace_id, f"trace id mismatch: {path}")
        traces[trace_id] = {
            "path": str(path),
            "sha256": sha256(path),
            "payload_sha256": trace["payload_sha256"],
            "trace_kind": trace["trace_kind"],
        }

    records = [json.loads(line) for line in options.raw.read_text().splitlines()]
    require(len(records) == len(traces) * len(WIDTHS) * 3,
            "campaign must contain exactly one record per candidate cell")
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    identities = set()
    annotated = []
    for record in records:
        trace_id = record["regime"]
        width = record["n"]
        require(trace_id in traces and width in WIDTHS, "unexpected trace/N cell")
        require(record["correct"] is True, "candidate failed referee")
        require(record["warmups"] == 3 and record["repeats"] == 11,
                "unequal timing policy")
        require(record["mad_percent"] <= 5.0, "timing variance exceeds 5%")
        require(record["output_effect"] == "overwrite", "output effect mismatch")
        require(record["input_order"] == "packed-row-major", "input order mismatch")
        require(record["output_order"] == "execution-row-major", "output order mismatch")
        identities.add((record["device"], record["sm"], record["cuda_driver"],
                        record["cuda_runtime"], record["rows"], record["features"],
                        record["nnz"], trace_id, width))
        groups[(trace_id, width)].append(record)
        annotated.append({
            **record,
            "source_schema": record["schema"],
            "schema": "CE-ARCH-92-EVIDENCE/1",
            "trace_path": traces[trace_id]["path"],
            "trace_sha256": traces[trace_id]["sha256"],
            "trace_payload_sha256": traces[trace_id]["payload_sha256"],
            "trace_kind": traces[trace_id]["trace_kind"],
            "controller_evidence_id": options.controller_evidence_id,
        })

    require(len(identities) == len(traces) * len(WIDTHS),
            "candidate device/build/shape identities differ within a cell")
    winners = []
    for (trace_id, width), cell in sorted(groups.items()):
        require({item["candidate"] for item in cell} == EXPECTED_CANDIDATES[width],
                "candidate set mismatch")
        ranked = sorted(cell, key=lambda item: item["amortized_total_ms"])
        margin = ranked[1]["amortized_total_ms"] / ranked[0]["amortized_total_ms"] - 1.0
        winners.append({
            "trace_id": trace_id,
            "n": width,
            "winner": ranked[0]["candidate"],
            "amortized_total_ms": ranked[0]["amortized_total_ms"],
            "runner_up_margin_percent": margin * 100.0,
            "clear": margin >= 0.02,
        })
    require(all(item["clear"] for item in winners), "campaign contains an ambiguous cell")
    require(any(item["winner"] == "csr" for item in winners),
            "campaign lacks a measured fallback regime")
    require(any(item["winner"] == "row_masked" for item in winners),
            "campaign lacks a measured row-masked win")
    require(any(item["winner"].startswith("feature_major") for item in winners),
            "campaign lacks a measured Cellerator-native win")

    options.output.write_text("".join(
        json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
        for item in annotated
    ))
    summary = {
        "schema": "CE-ARCH-92-SUMMARY/1",
        "controller_evidence_id": options.controller_evidence_id,
        "raw_sha256": sha256(options.raw),
        "record_count": len(annotated),
        "trace_count": len(traces),
        "widths": list(WIDTHS),
        "timing_basis": "amortized_total_ms_at_expected_reuse_8",
        "maximum_mad_percent": max(item["mad_percent"] for item in records),
        "winners": winners,
        "migration_exit_evidence": {
            "native_cellerator_win": True,
            "row_masked_real_regime": True,
            "csr_fallback_regime": True,
            "all_candidates_correct": True,
            "all_cells_clear_at_two_percent": True,
        },
    }
    options.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
