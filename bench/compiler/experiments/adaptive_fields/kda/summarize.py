#!/usr/bin/env python3
"""Summarize actual paired collector records from one KDA attempt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics


def interval(rounds, seed=20261028):
    rng = random.Random(seed)
    samples = []
    for _ in range(2000):
        # Preserve paired observations and retain round-level variation.
        selected = [rng.choice(rounds) for _ in rounds]
        draws = [rng.choice(block) for block in selected for _ in block]
        samples.append(statistics.mean(draws))
    samples.sort()
    return [samples[49], samples[1949]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    groups = {}
    records = []
    for path in sorted(args.attempt.rglob('*.json')):
        try:
            result = json.loads(path.read_text())
        except (ValueError, UnicodeError):
            continue
        if not isinstance(result, dict) or result.get('schema') != 'cellerator.adaptive_fields.e7_portfolio_collector.v1':
            continue
        records.append(str(path.resolve()))
        fixture = result['fixture']['fixture']
        for candidate, row in result['candidates'].items():
            groups.setdefault(fixture, {}).setdefault(candidate, []).append(row)
    summary = {}
    for fixture, candidates in groups.items():
        summary[fixture] = {}
        for name, rows in candidates.items():
            blocks = [[pair['candidate']['wall_ms'] - pair['baseline']['wall_ms']
                       for pair in row['timing']['paired_samples']] for row in rows]
            candidate = [pair['candidate']['wall_ms'] for row in rows
                         for pair in row['timing']['paired_samples']]
            baseline = [pair['baseline']['wall_ms'] for row in rows
                        for pair in row['timing']['paired_samples']]
            summary[fixture][name] = {
                'rounds': len(blocks), 'pairs': len(candidate),
                'complete_protocol': len(blocks) == 4 and all(len(b) == 21 for b in blocks),
                'candidate_wall_ms_median': statistics.median(candidate),
                'paired_baseline_wall_ms_median': statistics.median(baseline),
                'median_speedup': statistics.median(baseline) / statistics.median(candidate),
                'mean_paired_delta_ms': statistics.mean(x for block in blocks for x in block),
                'mean_paired_delta_ms_bootstrap_95_interval': interval(blocks),
                'candidate_cold_setup_ms_median': statistics.median(row['setup']['candidate_cold_setup_ms'] for row in rows),
                'baseline_cold_setup_ms_median': statistics.median(row['setup']['baseline_cold_setup_ms'] for row in rows),
                'peak_memory_scope': 'joint comparison process; not isolated per-candidate peak',
                'raw_records': len(rows),
            }
    result = {'schema': 'cellerator.adaptive_fields.e7_paired_summary.v1',
              'fixtures': summary, 'source_collector_records': records,
              'uncertainty': 'Two-level bootstrap of rounds and paired observations, 2000 draws, fixed seed',
              'kda_single_collector_selection_is_not_a_numerical_winner': True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'fixtures': list(summary), 'collector_records': len(records)}))
