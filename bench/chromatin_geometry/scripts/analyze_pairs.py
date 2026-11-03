#!/usr/bin/env python3
"""Evaluate independent same-device paired end-to-end trials without inventing data."""
import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(sorted_values, fraction):
    location = fraction * (len(sorted_values) - 1)
    lo = int(location)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (location - lo)


def analyze(data, seed=20261102, bootstrap=10000):
    if data.get('schema') != 'cgs.paired-trials.v1':
        raise ValueError('unsupported paired trial schema')
    if data.get('validation_passed') is not True or data.get('interference_check_passed') is not True:
        raise ValueError('a correctness pass and documented interference check are prerequisites')
    if not isinstance(data.get('measurement_boundary'), str) or not data['measurement_boundary']:
        raise ValueError('measurement cost boundary is required')
    if not isinstance(data.get('strongest_matched_control'), str) or not data['strongest_matched_control']:
        raise ValueError('strongest matched control identity is required')
    cases = defaultdict(list)
    orders = defaultdict(list)
    seen = set()
    for row in data.get('pairs', []):
        key = (row.get('case'), row.get('repeat'), row.get('device_uuid'))
        if None in key or key in seen or not str(row['device_uuid']).startswith('GPU-'):
            raise ValueError('each pair needs a unique case/repeat/device identity')
        seen.add(key)
        if row.get('candidate_device_uuid', row['device_uuid']) != row['device_uuid'] or row.get('control_device_uuid', row['device_uuid']) != row['device_uuid']:
            raise ValueError('control and candidate must run on the same physical GPU')
        if row.get('order') not in ('control-first', 'candidate-first'):
            raise ValueError('each trial must declare its paired order')
        for field in ('control_us', 'candidate_us'):
            if not isinstance(row.get(field), (float, int)) or isinstance(row[field], bool) or not math.isfinite(row[field]) or row[field] <= 0:
                raise ValueError('paired elapsed times must be finite and strictly positive')
        cases[row['case']].append(math.log(row['control_us'] / row['candidate_us']))
        orders[row['case']].append(row['order'])
    if not cases:
        raise ValueError('no trial pairs supplied')
    rng = random.Random(seed)
    reports = []
    for case, values in sorted(cases.items()):
        order_count = orders[case].count('control-first')
        if abs(2 * order_count - len(values)) > 1:
            reports.append({'case': case, 'pairs': len(values),
                            'decision': 'inconclusive: paired trial order is not balanced'})
            continue
        if len(values) < 5:
            reports.append({'case': case, 'pairs': len(values), 'decision': 'inconclusive: fewer than five independent pairs'})
            continue
        point = math.exp(statistics.median(values))
        draws = sorted(math.exp(statistics.median(rng.choices(values, k=len(values)))) for _ in range(bootstrap))
        low, high = percentile(draws, 0.025), percentile(draws, 0.975)
        reports.append({'case': case, 'pairs': len(values), 'median_speedup': point,
                        'median_latency_reduction': 1.0 - 1.0 / point,
                        'paired_bootstrap_95_percent_interval': [low, high],
                        # The practical gate is strictly >5% latency reduction.
                        # Guard the boundary against roundoff from log/exp.
                        'decision': 'calculator-threshold-met' if point > (1.0 / 0.95 + 1e-12) and low > 1.0 else 'not established'})
    return {'schema': 'cgs.paired-analysis.v1', 'measurement_boundary': data['measurement_boundary'],
            'strongest_matched_control': data['strongest_matched_control'],
            'seed': seed, 'bootstrap_samples': bootstrap, 'cases': reports,
            'interpretation': 'Calculator only: caller data are not independently joined to source, workload, admission, correctness or cleanup evidence. A threshold result is not an eligibility or performance claim.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(json.loads(args.input.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'output': str(args.output), 'cases': len(result['cases'])}))


if __name__ == '__main__':
    main()
