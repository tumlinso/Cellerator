#!/usr/bin/env python3
"""Run the three admitted E7 bridge/protocol sanitizer checks serially."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=HERE / 'evidence/e7_sanitizers')
    parser.add_argument('--validate-first', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('ADAPTIVE_FIELDS_RUN_KEY'):
        raise RuntimeError('Use the root-owned scheduler JobSpec')
    output = args.output_dir.resolve()
    if HERE not in output.parents:
        raise RuntimeError('Sanitizer evidence must stay inside the experiment')
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.validate_first:
        command = [sys.executable, str(HERE / 'e7_validate.py'),
                   '--output-dir', str(output / 'normal')]
        run = subprocess.run(command, capture_output=True, text=True, check=False, timeout=1200)
        (output / 'normal_process.json').write_text(json.dumps({
            'argv': command, 'returncode': run.returncode,
            'stdout': run.stdout, 'stderr': run.stderr,
        }, indent=2) + '\n')
        print(f'normal validation: exit {run.returncode}', flush=True)
        if run.returncode:
            (output / 'summary.json').write_text(json.dumps({
                'passed': False, 'normal_validation_returncode': run.returncode,
                'performance_evidence': False,
            }, indent=2) + '\n')
            print(run.stdout, run.stderr, flush=True)
            raise SystemExit(1)
    for tool in ('memcheck', 'racecheck', 'synccheck'):
        command = [sys.executable, str(HERE / 'e7_validate.py'),
                   '--sanitizer', tool, '--output-dir', str(output / tool)]
        run = subprocess.run(command, capture_output=True, text=True, check=False, timeout=1200)
        rows.append({'tool': tool, 'argv': command, 'returncode': run.returncode,
                     'stdout': run.stdout, 'stderr': run.stderr})
        print(f'{tool}: exit {run.returncode}', flush=True)
        if run.returncode:
            print(run.stdout, run.stderr, flush=True)
            break
    passed = len(rows) == 3 and all(row['returncode'] == 0 for row in rows)
    (output / 'summary.json').write_text(json.dumps({
        'passed': passed, 'checks': rows, 'performance_evidence': False,
        'run_key': os.environ['ADAPTIVE_FIELDS_RUN_KEY'],
    }, indent=2) + '\n')
    raise SystemExit(0 if passed else 1)
