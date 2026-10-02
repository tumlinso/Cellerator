#!/usr/bin/env python3
"""Validate this research package locally; never writes or mutates a Todo authority."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load(root: Path, name: str) -> Any:
    return json.loads((root / name).read_text(encoding='utf-8'))


def safe_path(value: str) -> bool:
    p = PurePosixPath(value)
    return bool(value) and not p.is_absolute() and '..' not in p.parts and '\\' not in value


def check_plan(root: Path, repository: str, expected_count: int) -> dict[str, dict[str, Any]]:
    plan = load(root, f'machine/{repository}.todo-plan.json')
    require(plan['schema_version'] == 3, f'wrong plan schema: {repository}')
    require(plan['project']['name'] == repository, f'wrong authority: {repository}')
    tasks = {t['id']: t for t in plan['tasks']}
    require(len(tasks) == len(plan['tasks']) == expected_count, f'duplicate/missing tasks: {repository}')
    graph: dict[str, list[str]] = {}
    for ident, task in tasks.items():
        require(bool(task.get('objective')), f'empty objective: {ident}')
        if task.get('parent_id'):
            require(task['parent_id'] in tasks, f'foreign parent: {ident}')
            require((root / 'tasks' / f'{ident}.md').is_file(), f'missing task brief: {ident}')
        for key in ['exclusive_paths', 'read_paths', 'forbidden_paths']:
            require(all(safe_path(x) for x in task.get('scope', {}).get(key, [])), f'unsafe scope: {ident}')
        deps = task.get('depends_on', [])
        require(all(d['type'] == 'task' and d['task_id'] in tasks for d in deps), f'foreign dependency: {ident}')
        graph[ident] = [d['task_id'] for d in deps]
        require(task.get('parent_id') not in graph[ident], f'child depends on aggregate: {ident}')
    visiting: set[str] = set()
    done: set[str] = set()
    def visit(n: str) -> None:
        require(n not in visiting, f'dependency cycle at {n}')
        if n in done:
            return
        visiting.add(n)
        for dep in graph[n]:
            visit(dep)
        visiting.remove(n)
        done.add(n)
    for n in graph:
        visit(n)
    require(len(plan['runs']) == 1, f'expected one bootstrap run: {repository}')
    run = plan['runs'][0]
    require(run['root_task_id'] in tasks, f'missing run root: {repository}')
    require(len(run['lanes']) == 1, f'expected one default lane: {repository}')
    queue = run['lanes'][0]['tasks']
    require(len(queue) == len(set(queue)) and set(queue) == set(tasks), f'incomplete queue: {repository}')
    order = {x: i for i, x in enumerate(queue)}
    require(queue[-1] == run['root_task_id'], f'aggregate not last: {repository}')
    for ident, deps in graph.items():
        require(all(order[d] < order[ident] for d in deps), f'queue violates dependency: {ident}')
    require(len(json.dumps(run['charter']).encode()) <= 8192, f'oversized charter: {repository}')
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--verify-manifest', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    docs = [p for p in root.rglob('*.json') if 'build-host' not in p.parts]
    for path in docs:
        json.loads(path.read_text(encoding='utf-8'))
    bp = check_plan(root, 'baseplane', 16)
    ce = check_plan(root, 'cellerator', 7)
    tasks = {**bp, **ce}
    experiments = load(root, 'machine/experiments.json')['experiments']
    ids = {e['id'] for e in experiments}
    require(len(experiments) == 48 and ids == {f'E{i:02}' for i in range(1, 49)}, 'experiment catalogue incomplete')
    sources = {s['id'] for s in load(root, 'machine/sources.json')['sources']}
    for e in experiments:
        require(e['owner_task'] in tasks and e['sequence_owner_task'] in bp, f'owner mismatch: {e["id"]}')
        require(set(e['sources']) <= sources, f'unknown source: {e["id"]}')
        require((root / 'experiments' / (e['id'] + '.md')).is_file(), f'missing card: {e["id"]}')
        require(all(e.get(k) for k in ['question', 'mechanism', 'volta_mapping', 'semantic_limit', 'minimum_probe']), f'empty experiment: {e["id"]}')
    mapping = load(root, 'machine/old-to-new.json')['records']
    expected_old = {'BITOP-00', 'BP-BITOP-03'} | {f'BP-BITOP-{i}' for i in [10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 30, 31, 32, 33, 34, 35, 36, 50]} | {f'CE-BITOP-{i}' for i in range(40, 46)} | {'CS-BITOP-60'} | {f'STACK-BITOP-{i:02}' for i in [1, 2, 4, 51, 52, 53]}
    require(len(mapping) == 33 and {r['old_id'] for r in mapping} == expected_old, 'old backlog map incomplete')
    for row in mapping:
        if row['observed_status'] == 'done':
            require(row['disposition'] == 'preserve_completed', f'completed work not preserved: {row["old_id"]}')
    comps = load(root, 'machine/compositions.json')['compositions']
    require(len(comps) == 10, 'composition catalogue incomplete')
    for c in comps:
        require(set(c['experiments']) <= ids, f'unknown composition input: {c["id"]}')
    for link in load(root, 'machine/cross-authority.json')['links']:
        require(link['producer_task'] in ce and link['consumer_task'] in bp, 'cross-authority link invalid')
        require(set(link['primary_experiments']) <= ids, 'cross-authority experiment mismatch')
    # Relative markdown links must resolve; code-form source paths can refer to the inspected remote repository.
    broken = []
    for p in root.rglob('*.md'):
        for target in re.findall(r'(?<!!)\[[^\]]*\]\(([^)]+)\)', p.read_text(encoding='utf-8')):
            target = target.split('#', 1)[0]
            if not target or '://' in target or target.startswith(('mailto:', 'sandbox:')):
                continue
            if not (p.parent / target).exists():
                broken.append(str(p.relative_to(root)) + ' -> ' + target)
    require(not broken, 'broken local links: ' + '; '.join(broken))
    verified = 0
    if args.verify_manifest:
        manifest = root / 'MANIFEST.sha256'
        require(manifest.exists(), 'manifest missing')
        for line in manifest.read_text().splitlines():
            digest, name = line.split('  ', 1)
            require(safe_path(name), 'unsafe manifest path')
            require((root / name).is_file(), 'missing manifest file: ' + name)
            require(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, 'manifest mismatch: ' + name)
            verified += 1
        delivered = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name != 'MANIFEST.sha256' and 'build-host' not in p.parts and '__pycache__' not in p.parts}
        listed = {line.split('  ', 1)[1] for line in manifest.read_text().splitlines()}
        require(listed == delivered, 'manifest coverage mismatch')
    print(json.dumps({'status': 'package_structure_pass', 'experiments': 48, 'compositions': 10, 'baseplane_tasks': 16, 'cellerator_tasks': 7, 'old_tasks_mapped': 33, 'primary_sources': len(sources), 'json_files_parsed': len(docs), 'manifest_files_verified': verified, 'native_todo_validation': False, 'gpu_executed': False}))


if __name__ == '__main__':
    main()
