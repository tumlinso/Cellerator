#!/usr/bin/env python3
"""Build, run, validate, and package the CE-GEO-119 ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
BINARY = ROOT / "build/ceGeoPreprintAblations"
EXPECTED = {
    "reorder_grouping": {"logical_order", "grouped_edge_order"},
    "constraints": {"pinned_order", "adaptive_grouping"},
    "cost_model": {"forced_dense", "measured_sparse"},
    "support_refinement": {"coarse_support", "refined_support"},
    "order": {"persistent_order", "canonical_remap"},
    "value_mutability": {"reuse_structure", "reupload_structure"},
    "cover_density": {"sparse_cover", "dense_padded_cover"},
    "partial_cover": {"single_cover", "main_plus_residual"},
    "residual": {"drop_residual_negative", "exact_residual"},
    "cover_sharing": {"shared_padded_cover", "operation_specific_cover"},
}


def run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=ROOT, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compile-only", action="store_true")
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    BINARY.parent.mkdir(parents=True, exist_ok=True)
    compiled = run([
        "nvcc", "-std=c++17", "-arch=sm_70", "-O3", "-lineinfo",
        "-Xcompiler=-Wall,-Wextra,-Werror", "-I.", "-Iinclude",
        "bench/biology/ce_geo/ablations.cu", "-lcudart", "-o", str(BINARY),
    ])
    if compiled.stderr:
        print(compiled.stderr, file=sys.stderr, end="")
    if arguments.compile_only:
        print(json.dumps({"compile_valid": 1}, sort_keys=True))
        return 0
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    ablations = output / "ablations.jsonl"
    executed = run([str(BINARY), "--output", str(ablations),
                    "--warmups", "3", "--repeats", "11"])
    if executed.stderr:
        print(executed.stderr, file=sys.stderr, end="")
    rows = load_jsonl(ablations)
    measurements = [row for row in rows if row.get("record_type") == "measurement"]
    observed: dict[str, set[str]] = {}
    for row in measurements:
        observed.setdefault(str(row["family"]), set()).add(str(row["variant"]))
    if observed != EXPECTED:
        raise ValueError(f"ablation matrix mismatch: {observed}")
    if not any(row.get("correctness_passed") is False for row in measurements):
        raise ValueError("required negative correctness control is missing")
    if not all(row.get("accepted_for_promotion") is False for row in measurements):
        raise ValueError("an ablation improperly claims promotion")

    decisions: list[dict[str, object]] = []
    for family in sorted(EXPECTED):
        family_rows = [row for row in measurements if row["family"] == family]
        correct = [row for row in family_rows if row["correctness_passed"] is True]
        if not correct:
            raise ValueError(f"{family} has no correct variant")
        winner = min(correct, key=lambda row: float(row["complete_ns"]))
        decisions.append({
            "family": family,
            "selected_variant": winner["variant"],
            "selected_complete_ns": winner["complete_ns"],
            "disposition": "calibration_only",
            "negative_results": [
                {"variant": row["variant"], "complete_ns": row["complete_ns"],
                 "correctness_passed": row["correctness_passed"]}
                for row in family_rows if row is not winner
            ],
        })

    source_paths = sorted(
        list((ROOT / "bench/ce_geo/evidence/micro").glob("*.json*"))
        + list((ROOT / "bench/ce_geo/evidence/biology").glob("*.json*"))
    )
    sources = [{
        "path": str(path.relative_to(ROOT)),
        "sha256": digest(path),
        "bytes": path.stat().st_size,
    } for path in source_paths]
    summary = {
        "schema": "CELLERATOR-CE-GEO-PREPRINT-ABLATIONS/1",
        "task_id": "CE-GEO-119",
        "campaign_id": "preprint-ablations",
        "controller_evidence_id": "CE-GEO-119-preprint-ablations-v1",
        "evidence_valid": 1,
        "accepted_for_promotion": False,
        "disposition": "evaluated_not_promoted",
        "ablation_family_count": len(decisions),
        "measurement_count": len(measurements),
        "decisions": decisions,
        "source_evidence": sources,
        "limitations": [
            "Ablations use one deterministic synthetic degree-16 relation shape on V100; they are mechanism controls, not biological performance claims.",
            "Analytical padding and launch variants do not replace end-to-end planner campaigns.",
            "The hierarchy calibration in CE-GEO-118 is benchmark-local and remains non-promoted.",
            "Nsight Compute cache and stall counters were unavailable in CE-GEO-114 and remain unavailable here.",
            "Per-family fastest correct variants are calibration observations, not global planner defaults.",
        ],
        "negative_result_policy": "Incorrect or slower variants are retained verbatim; no failed result is removed from the package.",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema": "CELLERATOR-CE-GEO-PREPRINT-MANIFEST/1",
        "files": {
            "ablations.jsonl": digest(ablations),
            "summary.json": digest(output / "summary.json"),
        },
        "record_count": len(rows),
        "measurement_count": len(measurements),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "evidence_valid": 1,
        "ablation_family_count": len(decisions),
        "measurement_count": len(measurements),
        "negative_result_count": sum(
            row.get("correctness_passed") is False for row in measurements),
        "output": str(output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
