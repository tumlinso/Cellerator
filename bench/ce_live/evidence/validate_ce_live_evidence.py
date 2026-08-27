#!/usr/bin/env python3
"""Validate the CE-LIVE evidence fan-in without manufacturing measurements."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / "bench/ce_live/evidence/ce_live_evidence_v1.json"
FORWARD = ROOT / "bench/ce_live/forward/pbmc3k_forward_v1.jsonl"
TRAINING = ROOT / "bench/ce_live/training/native_training_v1_evidence.json"
CONCURRENCY = ROOT / "bench/ce_live/concurrency/acceptance_evidence.json"
REPLAY = ROOT / "docs/CE_LIVE_REPLAY.md"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"CE-LIVE-36 evidence invalid: {message}")


def main() -> None:
    evidence = json.loads(EVIDENCE.read_text())
    forward = [json.loads(line) for line in FORWARD.read_text().splitlines()]
    training = json.loads(TRAINING.read_text())
    concurrency = json.loads(CONCURRENCY.read_text())
    replay = REPLAY.read_text()

    widths = [1, 16, 17, 31, 32, 48, 64]
    reuse = [1, 8, 1024]
    require(evidence["schema"] == "CELLERATOR-LIVE-EVIDENCE/1", "schema")
    require(len(forward) == len(widths) * len(reuse), "forward record count")
    require(sorted({item["width"] for item in forward}) == widths, "widths")
    require(sorted({item["reuse"] for item in forward}) == reuse, "reuse")
    require(all(item["legal_candidates"] == 1 for item in forward),
            "legal candidate count changed; regenerate competitive evidence")
    require(all(item["generations"] == 2 for item in forward),
            "two-generation coverage")
    require(all(item["selected_total_ns"] == item["best_legal_total_ns"]
                and item["planner_regret_percent"] == 0 for item in forward),
            "planner regret")
    require(evidence["forward"]["maximum_planner_regret_percent"] == 0,
            "summary regret")
    require(training["correctness"]["passed"], "training referee")
    require(training["native"]["median_microseconds"] ==
            evidence["training"]["native_median_microseconds"],
            "native training timing")
    require(training["conventional"]["median_microseconds"] ==
            evidence["training"]["persistent_conventional_median_microseconds"],
            "conventional training timing")
    require(concurrency["correctness"]["passed"], "concurrency correctness")
    require(concurrency["compute_sanitizer"]["passed"], "concurrency memcheck")
    for identifier in (
        evidence["persistence_replay"]["controller_evidence_id"],
        evidence["persistence_replay"]["compute_sanitizer_evidence_id"],
    ):
        require(identifier in replay, "replay evidence provenance")
    require(evidence["fixture"]["logical_edges"] == 433808,
            "fixture identity")
    require(evidence["fixture"]["fmp1_projection_bytes"] == 3073340,
            "FMP1 memory accounting")
    print("CE-LIVE-36 evidence validated records=21 widths=7 reuse=3 "
          "generations=2 training_baselines=2 replay=1 concurrency=1")


if __name__ == "__main__":
    main()
