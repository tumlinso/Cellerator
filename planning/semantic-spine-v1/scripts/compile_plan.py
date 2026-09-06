#!/usr/bin/env python3
"""Emit this package's native schema-3 plan without contacting or mutating Todo.

Do NOT substitute project-control plan compile: its inspected generic preledger
compiler emits schema 2 and does not preserve this first-class lane graph.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]

def assemble(master: dict) -> dict:
    if master.get('format') != 'cellerator-semantic-spine-preledger':
        raise ValueError('not a Semantic Spine preledger')
    if master.get('authority_to_apply') is not False or master.get('native_schema_version') != 3:
        raise ValueError('invalid authority/schema metadata')
    tasks = [master['root_record']] + [t['native_record'] for t in master['tasks']]
    return {'schema_version': 3, 'project': master['project'], 'tasks': tasks,
            'invariants': master['invariants'], 'locks': master['locks'],
            'interfaces': master['interfaces'], 'barriers': master['barriers'],
            'resource_classes': master['resource_classes'], 'runs': [master['run']]}

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--package-root', type=Path, default=PACKAGE)
    ap.add_argument('--output', type=Path, help='Optional local output; no Todo mutation')
    ap.add_argument('--check', action='store_true', help='Check shipped native plan matches the richer catalog')
    a = ap.parse_args()
    root = a.package_root.resolve()
    master = json.loads((root/'machine/proposed_todos.json').read_text())
    plan = assemble(master)
    if a.check:
        existing = json.loads((root/'machine/semantic-spine-v1.todo-plan.json').read_text())
        if existing != plan:
            raise ValueError('native plan differs from master catalog; regenerate before validation')
        print(json.dumps({'status': 'native_projection_matches', 'schema_version': 3,
                          'task_count': len(plan['tasks']), 'lane_count': len(plan['runs'][0]['lanes']),
                          'authority_validation_performed': False}))
    elif a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(plan, indent=2, ensure_ascii=False)+'\n')
    else:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
    return 0

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError) as e:
        raise SystemExit(f'compile_plan: {e}')
