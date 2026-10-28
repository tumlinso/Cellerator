#!/usr/bin/env python3
"""Admitted small numerical validation and one-pair protocol smoke; no speed claim."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sanitizer', choices=('none', 'memcheck', 'racecheck', 'synccheck'), default='none')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[20261027, 20261028, 20261029])
    args = parser.parse_args()
    if not os.environ.get('ADAPTIVE_FIELDS_RUN_KEY'):
        raise RuntimeError('Use the root-owned scheduler JobSpec')
    output = args.output_dir.resolve()
    if HERE not in output.parents:
        raise RuntimeError('Evidence must stay within the experiment')
    output.mkdir(parents=True, exist_ok=True)
    commands = []
    cases = [('reset_protocol', ['--validate-reset-protocol', '--candidate', 'scan_simt_full',
                                 '--seed', str(args.seeds[0])])]
    cases.extend((f'validation_seed{seed}', ['--validate-only', '--seed', str(seed)])
                 for seed in args.seeds)
    cases.append(('protocol_smoke', ['--benchmark', '--warmup', '1', '--samples', '1',
                                     '--rounds', '1', '--seed', str(args.seeds[0])]))
    for label, mode in cases:
        command = [sys.executable, str(HERE / 'gpu/e7_relation_runner.py'),
                   '--case', 'small', '--output', str(output / (label + '.json')), *mode]
        if args.sanitizer != 'none':
            command = ['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/compute-sanitizer/compute-sanitizer',
                       '--tool', args.sanitizer, '--error-exitcode', '99', *command]
        run = subprocess.run(command, capture_output=True, text=True, check=False, timeout=600)
        result_path = output / (label + '.json')
        decision = None
        if run.returncode == 0 and result_path.is_file():
            decision = json.loads(result_path.read_text()).get('decision')
        record = {'label': label, 'argv': command, 'returncode': run.returncode,
                  'stdout': run.stdout, 'stderr': run.stderr,
                  'decision': decision, 'performance_evidence': False}
        commands.append(record)
        (output / (label + '_process.json')).write_text(json.dumps(record, indent=2) + '\n')
        print(f'{label}/{args.sanitizer}: exit {run.returncode}', flush=True)
        if run.returncode or decision != 'supported':
            print(run.stdout, run.stderr, flush=True)
            break
    passed = len(commands) == len(cases) and all(row['returncode'] == 0 and row['decision'] == 'supported'
                                               for row in commands)
    (output / 'summary.json').write_text(json.dumps({
        'passed': passed, 'commands': commands, 'performance_evidence': False,
        'run_key': os.environ['ADAPTIVE_FIELDS_RUN_KEY'],
    }, indent=2) + '\n')
    raise SystemExit(0 if passed else 1)
