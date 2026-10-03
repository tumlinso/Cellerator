#!/usr/bin/env python3
"""Focused native host foundation gate; no GPU or installed-package claim."""
import argparse
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--build-dir', required=True)
args = parser.parse_args()
local = Path(__file__).resolve().parent
root = local.parents[2]
seed = json.loads((local.parent / 'machine/preservation.json').read_text())
expanded = json.loads((local / 'preservation.json').read_text())
assert expanded['records'][:len(seed['records'])] == seed['records'], 'sealed seed ledger changed'
plans = ['planning/baseplane_moonshot_parallel/cellerator.todo-plan.json',
         'planning/moonshot-parallel-v1/cellerator.todo-plan.json',
         'planning/learning-v2/machine/cellerator.todo-plan.json']
ids = {r['id'] for r in expanded['records']}
for plan in plans:
    for task in json.loads((root / plan).read_text())['tasks']:
        assert 'current:' + task['id'] in ids, task['id']
for entry in json.loads((local / 'source-destinations.json').read_text())['entries']:
    assert (root / entry['source']).exists(), entry['source']
for cmd in [['cmake', '-S', str(local), '-B', args.build_dir, '-DCMAKE_BUILD_TYPE=Release'],
            ['cmake', '--build', args.build_dir, '-j2'],
            ['ctest', '--test-dir', args.build_dir, '--output-on-failure']]:
    subprocess.run(cmd, check=True)
print(f'Native host foundation passed; preserved {len(seed["records"])} seed records and {len(expanded["records"])-len(seed["records"])} current records.')
