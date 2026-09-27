#!/usr/bin/env python3
"""Read-only ML2 package integrity/design lint, not the native Todo validator.

Python 3.10+, standard library only. Never invokes Project Control, changes
repository state or writes a Todo ledger. Refresh target-specific native plan
validation through the installed Project Control front door before adoption.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

class PackageError(ValueError):
    pass

def need(condition: bool, message: str) -> None:
    if not condition: raise PackageError(message)

def safe_relative(value: str) -> None:
    p = PurePosixPath(value)
    need(bool(value) and not p.is_absolute() and '..' not in p.parts
         and '\\' not in value, f'Unsafe relative path: {value!r}')

def native_digest(plan: dict) -> str:
    return hashlib.sha256(json.dumps(plan, sort_keys=True, indent=2).encode()).hexdigest()

def lint_plan(plan: dict, prefix: str, project_name: str) -> dict:
    need(plan.get('schema_version') == 3, 'Expected native plan schema 3')
    need(plan.get('project', {}).get('name') == project_name, 'Wrong plan target')
    tasks = plan.get('tasks', [])
    ids = [t.get('id') for t in tasks]
    need(bool(ids) and all(isinstance(x, str) for x in ids), 'Missing task identity')
    need(len(ids) == len(set(ids)), 'Duplicate task IDs')
    need(all(x.startswith(prefix+'-') for x in ids), 'Foreign task identity')
    by_id = {t['id']: t for t in tasks}
    epics = [t for t in tasks if t.get('kind') == 'epic']
    need(len(epics) == 1, 'Expected one closure-only aggregate')
    epic = epics[0]['id']
    invariant_ids = {x['id'] for x in plan.get('invariants', [])}
    graph, sizes = {}, []
    for t in tasks:
        tid = t['id']
        need(t.get('objective') and t.get('completion_contract', {}).get('required'),
             f'Missing intent/acceptance for {tid}')
        need('status' not in t and 'result' not in t, 'Bootstrap cannot claim completed work')
        need(not t.get('gates'), 'Do not ship invented future product-test gate commands')
        if tid != epic: need(t.get('parent_id') == epic, 'Wrong aggregate relationship')
        for k in ('exclusive_paths', 'read_paths', 'forbidden_paths'):
            for p in t.get('scope', {}).get(k, []): safe_relative(p)
        need(not set(t.get('scope', {}).get('exclusive_paths', []))
             & {'.', 'preprint', '.todo-orchestrator'}, 'Broad/protected ownership')
        need(set(t.get('invariants', [])) <= invariant_ids, 'Unknown invariant')
        graph[tid] = []
        for d in t.get('depends_on', []):
            need(d.get('type') == 'task' and d.get('task_id') in by_id,
                 'Foreign or unsupported native dependency')
            need(d['task_id'] != epic, 'A child cannot depend on its closure aggregate')
            graph[tid].append(d['task_id'])
        brief = {k:t[k] for k in ('objective','scope','completion_contract',
                  'delegated_choices','references','next_action') if k in t}
        sizes.append(len(json.dumps(brief, separators=(',', ':'), ensure_ascii=False).encode()))
    active, done = set(), set()
    def visit(x):
        need(x not in active, 'Dependency cycle')
        if x in done: return
        active.add(x)
        for dep in graph[x]: visit(dep)
        active.remove(x); done.add(x)
    for x in ids: visit(x)
    runs = plan.get('runs', [])
    need(len(runs) == 1, 'Expected one local run')
    run = runs[0]
    need(run.get('root_task_id') == epic and isinstance(run.get('charter'), dict), 'Bad run charter/root')
    lanes = run.get('lanes', [])
    need(len(lanes) == 1 and lanes[0].get('role') == 'implementer', 'Expected writable implementer default')
    need(lanes[0].get('workspace', {}).get('mode') == 'exclusive', 'Wrong default workspace')
    queue = lanes[0].get('tasks', [])
    need(len(queue) == len(ids) and set(queue) == set(ids), 'Task assignment mismatch')
    need(queue[-1] == epic, 'Closure must be last')
    pos = {x:i for i,x in enumerate(queue)}
    need(all(pos[d] < pos[t] for t, ds in graph.items() for d in ds), 'Dependency after consumer in default queue')
    need(max(sizes) < 6144, 'Task brief exceeds intended compact budget')
    return {'project':project_name, 'tasks':len(tasks), 'outcomes':len(tasks)-1,
            'run':run['id'], 'max_brief_bytes':max(sizes), 'plan_digest':native_digest(plan)}

def check(root: Path) -> dict:
    root = root.resolve()
    manifest = json.loads((root/'MANIFEST.json').read_text())
    need(manifest.get('format') == 'glasshelix-ml2-bootstrap', 'Wrong manifest')
    for name, digest in manifest['files'].items():
        safe_relative(name)
        file = root/name
        need(not file.is_symlink() and file.is_file()
             and file.resolve().is_relative_to(root), f'Unsafe/missing package file: {name}')
        need(hashlib.sha256(file.read_bytes()).hexdigest() == digest, f'Integrity mismatch: {name}')
    summaries = {}
    receipts = json.loads((root/'validation/native_preview.json').read_text())['plans']
    for name, prefix, project in [('glasshelix','GH-ML2','GlassHelix'), ('cellerator','CE-ML2','Cellerator')]:
        plan = json.loads((root/'machine'/f'{name}.todo-plan.json').read_text())
        summary = lint_plan(plan, prefix, project)
        need(summary['plan_digest'] == receipts[name]['plan_digest'], 'Preview/payload digest mismatch')
        for task in plan['tasks']:
            for ref in task.get('references', []):
                if ref.startswith('planning/learning-v2/'):
                    need((root/ref.removeprefix('planning/learning-v2/')).is_file(), f'Missing reference: {ref}')
        summaries[name] = summary
    return {'ok': True, 'scope':'Package integrity, design lint and exact preview payload correspondence only.',
            'hashed_files':len(manifest['files']), 'plans':summaries,
            'live_preview_limit':'Cellerator target preview failed; shared-schema validation is not a CE readiness check.',
            'no_mutation': True}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try: result = check(args.root)
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(json.dumps({'ok':False,'error':str(error)}, indent=2)); return 1
    print(json.dumps(result, indent=2)); return 0
if __name__ == '__main__': raise SystemExit(main())
