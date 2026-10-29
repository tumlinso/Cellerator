#!/usr/bin/env python3
"""Measure the frozen NVCC/NVRTC region in a clean admitted interval."""
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RUNTIME = HERE / 'gpu/runtime'

if __name__ == '__main__':
    if not os.environ.get('ADAPTIVE_FIELDS_RUN_KEY'):
        raise RuntimeError('Use the root-owned scheduler JobSpec')
    output = HERE / 'evidence/runtime_clean'
    output.mkdir(parents=True, exist_ok=True)
    commands = [
        [sys.executable, str(RUNTIME / 'prepare_nvcc.py'), '--output-dir',
         str(output / 'artifacts'), '--arch', 'sm_70'],
        [sys.executable, str(RUNTIME / 'run_nvrtc_compare.py'), '--manifest',
         str(output / 'artifacts/nvcc_artifact_manifest.json'), '--output',
         str(output / 'comparison.json')],
    ]
    records = []
    for index, argv in enumerate(commands):
        run = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=240)
        record = {'argv': argv, 'returncode': run.returncode,
                  'stdout': run.stdout, 'stderr': run.stderr}
        records.append(record)
        (output / f'command_{index}.json').write_text(json.dumps(record, indent=2) + '\n')
        print(f'command {index}: exit {run.returncode}', flush=True)
        if run.returncode:
            print(run.stdout, run.stderr, flush=True)
            break
    passed = len(records) == 2 and all(row['returncode'] == 0 for row in records)
    (output / 'summary.json').write_text(json.dumps({
        'passed': passed, 'commands': records,
        'run_key': os.environ['ADAPTIVE_FIELDS_RUN_KEY'],
        'scope': 'Same experimental region; compiler/load overhead and execution, not application training speedup',
    }, indent=2) + '\n')
    raise SystemExit(0 if passed else 1)
