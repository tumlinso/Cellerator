#!/usr/bin/env python3
"""Measure true-target route coverage for the frozen v9 event fixture.

This is an analytical CPU diagnostic. It reads only event inputs and target
labels from the public fixture API; it never reads validator_support or
validator_mechanism and does not alter labels or fixture generation.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from cellerator.torch.sparse_support import make_next_event_fixture


TRAIN_SEED = 41027
HELDOUT_SEED = TRAIN_SEED + 200_000
CELLS = 64
STEPS = 12
TARGET_IDS = tuple(range(16, 20))


def seeded_support(seed: int) -> set[tuple[int, int]]:
    rng = random.Random(seed)
    return {
        (source, 16 + target)
        for source in range(16)
        for target in sorted(rng.sample(range(8), 6))
    }


def main() -> None:
    fixture = make_next_event_fixture(
        seed=HELDOUT_SEED, cells=CELLS, steps=STEPS, entities=24)
    # Deliberately select only fields consumed by the held-out model/evaluator.
    events = fixture["events"].cpu().tolist()
    targets = fixture["targets"].cpu().tolist()
    timestamps = fixture["timestamps"].cpu().tolist()
    count_windows = fixture["count_windows"].cpu().tolist()
    time_windows = fixture["time_windows"].cpu().tolist()
    support = seeded_support(TRAIN_SEED)
    # v9 report records this one published event promotion. It cannot connect
    # held-out last-event sources (8..15) to class targets (16..19).
    promoted = (3, 23)
    post_adaptation_support = support | {promoted}

    assert len(events) == CELLS and all(len(row) == STEPS for row in events)
    assert set(targets) == set(TARGET_IDS)
    last_sources = [row[-1] for row in events]
    assert set(last_sources) == set(range(8, 16))

    # Reconstruct the observed occurrence IDs accepted by the same inclusive
    # count/time window at the final event position. This is the existing
    # provider's history-source contract, not hidden support.
    history_sources = []
    for cell in range(CELLS):
        count_min = int(count_windows[cell][0])
        count_max = min(int(count_windows[cell][1]), STEPS - 1)
        time_min = int(time_windows[cell][0])
        time_max = min(int(time_windows[cell][1]), int(timestamps[cell][-1]))
        history_sources.append([
            int(events[cell][position])
            for position in range(count_min, count_max + 1)
            if time_min <= int(timestamps[cell][position]) <= time_max
        ])
    assert all(last_sources[i] in history_sources[i] for i in range(CELLS))

    def covered(route_set: set[tuple[int, int]]) -> list[bool]:
        return [(source, target) in route_set
                for source, target in zip(last_sources, targets)]

    initial_mask = covered(support)
    adapted_mask = covered(post_adaptation_support)
    history_initial_mask = [
        any((source, targets[cell]) in support for source in history_sources[cell])
        for cell in range(CELLS)
    ]
    history_adapted_mask = [
        any((source, targets[cell]) in post_adaptation_support
            for source in history_sources[cell])
        for cell in range(CELLS)
    ]
    unconnected_output_ids = sorted(
        set(range(24)) - {destination for _, destination in post_adaptation_support})
    per_source = []
    for source in sorted(set(last_sources)):
        selected = [i for i, value in enumerate(last_sources) if value == source]
        target_route_classes = sorted(
            target for target in TARGET_IDS if (source, target) in support)
        per_source.append({
            "last_event_source": source,
            "heldout_examples": len(selected),
            "heldout_targets_supported": sum(initial_mask[i] for i in selected),
            "heldout_targets_unsupported": sum(not initial_mask[i] for i in selected),
            "supported_class_destinations": target_route_classes,
            "supported_class_destination_count": len(target_route_classes),
        })

    payload = {
        "diagnostic": "event_true_target_route_coverage_v1",
        "qualification_source": "torch-v9 corrected synthetic learning run",
        "train_seed": TRAIN_SEED,
        "heldout_seed": HELDOUT_SEED,
        "cells": CELLS,
        "steps": STEPS,
        "class_destinations": list(TARGET_IDS),
        "initial_support_edges": len(support),
        "reported_promotion_zero_based": list(promoted),
        "post_adaptation_support_edges": len(post_adaptation_support),
        "last_event_source_domain": [8, 15],
        "heldout_true_target_route_present_initial": sum(initial_mask),
        "heldout_true_target_route_missing_initial": len(initial_mask) - sum(initial_mask),
        "heldout_true_target_route_present_after_reported_promotion": sum(adapted_mask),
        "post_promotion_coverage_changed": initial_mask != adapted_mask,
        "current_readout_entity_count": 24,
        "valid_target_count": len(TARGET_IDS),
        "unconnected_output_ids_after_reported_promotion": unconnected_output_ids,
        "current_input_route_support_top1_ceiling_count": sum(adapted_mask),
        "current_input_route_support_top1_ceiling_fraction": (
            sum(adapted_mask) / CELLS),
        "heldout_true_target_route_present_from_any_window_source": sum(
            history_initial_mask),
        "heldout_true_target_route_missing_from_all_window_sources": (
            len(history_initial_mask) - sum(history_initial_mask)),
        "heldout_true_target_route_present_from_any_window_source_after_promotion": sum(
            history_adapted_mask),
        "history_window_source_count_min": min(map(len, history_sources)),
        "history_window_source_count_max": max(map(len, history_sources)),
        "history_window_source_count_mean": sum(map(len, history_sources)) / CELLS,
        "support_route_count_for_event_sources_to_class_destinations": sum(
            (source, target) in support
            for source in range(8, 16) for target in TARGET_IDS),
        "possible_routes_for_event_sources_to_class_destinations": 8 * len(TARGET_IDS),
        "per_source": per_source,
        "readout_invariant": (
            "at each event step inputs contain only the current event entity; "
            "history/state affect gates, while sparse relation output for a "
            "destination is zero if the current event source has no support edge "
            "to it. The same permitted count/time-window history has multiple "
            "source entities and can supply a route when the current source cannot; "
            "unconnected outputs have zero logits because the readout has no bias"
        ),
        "validator_support_read": False,
        "validator_mechanism_read": False,
    }
    output = Path(__file__).with_name("route_coverage.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
