#!/usr/bin/env python3
"""CPU-only support/label-prior coverage for the fixed 512-cell trial fixture."""
from __future__ import annotations

import json
import hashlib
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/home/tumlinson/Cellerator")
DEFAULT_SNAPSHOT = ROOT / "build-sparse-support/evidence/torch-v12-validator"
SNAPSHOT = Path(os.environ.get("CELLERATOR_SPARSE_SUPPORT_SNAPSHOT",
                               str(DEFAULT_SNAPSHOT))).resolve()
sys.path.insert(0, str(SNAPSHOT / "package"))
sys.path.insert(0, str(SNAPSHOT / "examples/sparse_support"))

import train_synthetic as base  # noqa: E402
from cellerator.torch.sparse_support import make_next_event_fixture  # noqa: E402


SEED = 41027
HELDOUT_SEED = SEED + 200_000
CELLS = 512
STEPS = 12


def summarize(support, fixture):
    edges = set(zip(support.sources, support.destinations))
    current_supported = 0
    history_supported = 0
    history_sources_total = 0
    for cell in range(CELLS):
        target = int(fixture["targets"][cell])
        last_source = int(fixture["events"][cell, -1])
        current_supported += (last_source, target) in edges
        count_min, count_max = map(int, fixture["count_windows"][cell].tolist())
        time_min, time_max = map(int, fixture["time_windows"][cell].tolist())
        sources = []
        for position in range(max(0, count_min), min(STEPS - 1, count_max) + 1):
            timestamp = int(fixture["timestamps"][cell, position])
            if time_min <= timestamp <= time_max:
                sources.append(int(fixture["events"][cell, position]))
        history_sources_total += len(sources)
        history_supported += any((source, target) in edges for source in sources)
    return {
        "cells_with_current_last_event_target_route": current_supported,
        "cells_with_any_window_history_target_route": history_supported,
        "window_occurrence_count_total": history_sources_total,
        "window_occurrence_count_per_cell": history_sources_total / CELLS,
    }


def main():
    fixture = make_next_event_fixture(seed=HELDOUT_SEED, cells=CELLS,
                                      steps=STEPS, entities=24)
    # Deliberately whitelist observed values used by the model/report.
    observed = {key: fixture[key] for key in (
        "events", "timestamps", "count_windows", "time_windows", "targets")}
    if any(key.startswith("validator") for key in observed):
        raise RuntimeError("validator-only fixture data entered the CPU check")
    support = base.pair_support(SEED)
    shuffled = base.shuffled_support(support, SEED + 17)
    target_counts = Counter(map(int, observed["targets"].tolist()))
    source_target_counts = {source: Counter() for source in range(8, 16)}
    for row in range(CELLS):
        source_target_counts[int(observed["events"][row, -1])].update(
            [int(observed["targets"][row])])
    conditional_majority = sum(max(counts.values())
                               for counts in source_target_counts.values()) / CELLS
    result = {
        "diagnostic": "event_readout_route_coverage_fixed_512_v1",
        "package_snapshot": str(SNAPSHOT),
        "package_identity_sha256": hashlib.sha256(
            (SNAPSHOT / "identity.json").read_bytes()).hexdigest(),
        "train_seed": SEED,
        "heldout_seed": HELDOUT_SEED,
        "heldout_cells": CELLS,
        "target_counts_16_to_19": [target_counts.get(i, 0) for i in range(16, 20)],
        "constant_majority_target_id": min(target_counts, key=lambda x: (-target_counts[x], x)),
        "constant_majority_accuracy": max(target_counts.values()) / CELLS,
        "heldout_only_current_source_conditional_majority_accuracy": conditional_majority,
        "current_source_target_counts_8_to_15_by_target_16_to_19": {
            str(source): [counts.get(target, 0) for target in range(16, 20)]
            for source, counts in source_target_counts.items()
        },
        "learned_support": summarize(support, observed),
        "shuffled_support": summarize(shuffled, observed),
        "validator_only_fields_read": [],
        "readout_domain": 24,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
