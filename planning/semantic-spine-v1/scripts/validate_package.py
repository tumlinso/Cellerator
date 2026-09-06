#!/usr/bin/env python3
"""Offline integrity and graph validation. This is NOT Todo's native validator."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
sys.dont_write_bytecode = True
from compile_plan import assemble

PACKAGE = Path(__file__).resolve().parents[1]
REQUIRED = ['machine/semantic-spine-v1.todo-plan.json', 'machine/proposed_todos.json',
            'machine/proposed_todos.csv', 'machine/dependency_edges.csv',
            'machine/checkpoints.csv', 'machine/interface_catalog.json', 'machine/barriers.json',
            'machine/lanes.json', 'machine/workstreams.json', 'machine/plan_summary.json',
            'machine/external_dependency_receipts.csv', 'machine/delivery_files.json',
            'evidence/live_snapshot.json', 'evidence/tooling_contract.json',
            '01_SCOPE_AND_DECISIONS.md', '02_SEMANTIC_AND_NATIVE_CONTRACT.md',
            '03_PARALLEL_LANES_AND_INTEGRATION.md', '04_VALIDATION_AND_DEMO.md',
            '05_MANUAL_BOOTSTRAP.md', '06_DEFERRED_WORK.md', '07_SOURCE_LEDGER.md',
            'scripts/compile_plan.py', 'scripts/todo_bootstrap.py', 'scripts/run_gate.py',
            'contracts/relation_semantics.hh', 'contracts/prepared_relation.hh']

def require(value: bool, message: str) -> None:
    if not value: raise ValueError(message)

def load(p: Path): return json.loads(p.read_text(encoding='utf-8'))
def rows(p: Path):
    with p.open(newline='', encoding='utf-8') as f: return list(csv.DictReader(f))
def digest(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()

def safe_path(path: str) -> bool:
    p = PurePosixPath(path)
    return bool(path) and not p.is_absolute() and '..' not in p.parts and '\\' not in path and not path.startswith('./')

def generations(ids: set[str], edges: set[tuple[str, str]]) -> list[list[str]]:
    incoming = {k: set() for k in ids}
    for first, second in edges:
        require(first in ids and second in ids, 'edge references unknown task')
        incoming[second].add(first)
    layers = []
    remaining = set(ids)
    while remaining:
        frontier = sorted(t for t in remaining if not (incoming[t] & remaining))
        require(bool(frontier), 'cycle in task/dependency/lane-queue graph')
        layers.append(frontier)
        remaining.difference_update(frontier)
    return layers

def validate(root: Path, source_root: Path | None = None, manifest: bool = True) -> dict:
    root = root.resolve()
    for p in REQUIRED: require((root/p).is_file(), f'missing package file: {p}')
    master = load(root/'machine/proposed_todos.json')
    plan = load(root/'machine/semantic-spine-v1.todo-plan.json')
    require(plan == assemble(master), 'native plan diverges from richer catalog')
    require(plan['schema_version'] == 3, 'first-class lanes require native schema 3')
    require(len(json.dumps(plan, separators=(',', ':')).encode()) < 256*1024, 'native proposal exceeds 256 KiB')
    for t in master['tasks']:
        n=t['native_record']
        require(n['id']==t['id'] and n['title']==t['title'], 'task label projection mismatch')
        require(n['scope']['exclusive_paths']==t['write_scope'], 'task scope projection mismatch')
        require([d['task_id'] for d in n.get('depends_on',[])]==t['depends_on'], 'task dependency projection mismatch')
        require(n['completion_contract']['required']==t['acceptance'], 'task acceptance projection mismatch')
    tasks = plan['tasks']; ids = {t['id'] for t in tasks}
    require(len(ids) == len(tasks), 'duplicate Todo ID')
    leaves = [t for t in tasks if t['kind'] == 'task']
    require(20 <= len(leaves) <= 40 and len(tasks) < 100, 'task budget violated')
    byid = {t['id']: t for t in tasks}
    require(all(t['status'] == 'planned' for t in tasks), 'fresh plan pre-completes work')
    require(sum(t['kind'] == 'epic' for t in tasks) == 1, 'unexpected hierarchy expansion')
    require(len(plan['runs']) == 1, 'expected one inactive-to-be-created run')
    run = plan['runs'][0]; lanes = run['lanes']
    require(run['root_task_id'] in ids, 'unknown run root')
    lane_ids = {l['id'] for l in lanes}
    require(len(lane_ids) == len(lanes), 'duplicate lane ID')
    require(sum('parent_lane_id' not in l for l in lanes) == 1, 'exactly one root lane required')
    owner = {}
    lane_edges = set()
    for lane in lanes:
        if 'parent_lane_id' in lane: require(lane['parent_lane_id'] in lane_ids, 'missing parent lane')
        require(lane['role'] in {'coordinator','implementer','validator','integrator','specialist'}, 'invalid lane role')
        require(lane['workspace']['mode'] in {'exclusive','read_shared','isolated_merge','contract_split'}, 'invalid workspace mode')
        for t in lane['tasks']:
            require(t in ids and t not in owner, 'task missing or assigned to multiple lanes')
            owner[t] = lane['id']
        lane_edges.update(zip(lane['tasks'], lane['tasks'][1:]))
    require(set(owner) == ids, 'not every task has a first-class lane')
    require(load(root/'machine/lanes.json')['lanes'] == lanes, 'lane projection mismatch')
    require(load(root/'machine/interface_catalog.json') == plan['interfaces'], 'interface projection mismatch')
    require(load(root/'machine/barriers.json') == plan['barriers'], 'barrier projection mismatch')
    interfaces = {x['id']: x for x in plan['interfaces']}
    checkpoints = {}
    edges = set()
    for t in tasks:
        if 'parent_id' in t: require(t['parent_id'] in ids, 'missing task parent')
        require((root/'proposed-todos'/f"{t['id'].lower()}.md").is_file(), 'missing detailed task document')
        for cp in t.get('checkpoints', []):
            require(cp['id'] not in checkpoints, 'duplicate checkpoint')
            checkpoints[cp['id']] = (t['id'], cp)
            require(cp.get('state') not in ('reached', 'done'), 'fresh plan pre-reaches checkpoint')
            for publication in cp.get('publishes_interfaces', []):
                require(publication['id'] in interfaces, 'unknown published interface')
                require(interfaces[publication['id']]['owner_task_id'] == t['id'], 'interface publication by non-owner')
        for d in t.get('depends_on', []):
            require(d['type'] == 'task', 'unexpected dependency form in this bounded plan')
            require(d['task_id'] in ids, 'unknown dependency')
            require(d['task_id'] != run['root_task_id'], 'leaf waits on root closure')
            edges.add((d['task_id'], t['id']))
        for c in t.get('consumes_interfaces', []):
            require(c['id'] in interfaces, 'unknown consumed interface')
            edges.add((interfaces[c['id']]['owner_task_id'], t['id']))
        for paths in t.get('scope', {}).values():
            if isinstance(paths, list):
                for p in paths: require(safe_path(p), 'unsafe scope path')
        for a in t.get('produced_artifacts', []): require(safe_path(a['path']), 'unsafe artifact path')
        for gate in t.get('gates', []):
            require(gate['type'] == 'command' and gate.get('argv'), 'missing executable gate')
            require('run_gate.py' in ' '.join(gate['argv']), 'unexpected gate executable')
    for i in interfaces.values():
        require(i['owner_task_id'] in ids and i['state'] == 'draft', 'bad interface ownership/state')
        for p in i['contract_paths']: require(safe_path(p), 'unsafe interface path')
    for barrier in plan['barriers']:
        for r in barrier['requirements']:
            require((r['type']=='task' and r['id'] in ids) or
                    (r['type']=='checkpoint' and r['id'] in checkpoints), 'bad barrier requirement')
    require({x['id']:x['task_ids'] for x in load(root/'machine/workstreams.json')} ==
            {'CE-SS1-WS-'+l['id'].split('-L-')[-1]:l['tasks'] for l in lanes}, 'workstream projection mismatch')
    summary=load(root/'machine/plan_summary.json')
    require(summary['todo_record_count']==len(tasks) and summary['leaf_task_count']==len(leaves) and summary['lane_count']==len(lanes), 'summary count mismatch')
    layers = generations(ids, edges | lane_edges)
    # Confirm the promised fan-out is actually possible, not only four differently named lanes.
    entry = {'CE-SS1-N01','CE-SS1-F01','CE-SS1-A01','CE-SS1-V01'}
    require(any(entry <= set(layer) for layer in layers), 'four-lane fan-out is accidentally serialized')
    indexed=rows(root/'machine/proposed_todos.csv')
    require(len(indexed)==len(tasks) and {r['id'] for r in indexed}==ids, 'task CSV mismatch')
    for r in indexed:
        n=byid[r['id']]
        require(r['kind']==n['kind'] and r['title']==n['title'] and r['lane']==owner[r['id']], 'task CSV metadata mismatch')
        require(r['depends_on']==';'.join(d['task_id'] for d in n.get('depends_on',[])), 'task CSV dependencies mismatch')
    actual_edges = {(r['depends_on_task_id'], r['task_id']) for r in rows(root/'machine/dependency_edges.csv')}
    explicit_edges = {(d['task_id'],t['id']) for t in tasks for d in t.get('depends_on', [])}
    require(actual_edges == explicit_edges, 'dependency CSV mismatch')
    require({r['id'] for r in rows(root/'machine/checkpoints.csv')} == set(checkpoints), 'checkpoint CSV mismatch')
    require(not rows(root/'machine/external_dependency_receipts.csv'), 'unexpected cross-authority work')
    # No simultaneous lanes may own overlapping paths except their serial integration handoff.
    def ancestor(a: str, b: str) -> bool:
        seen = {a}
        while True:
            new = {v for u,v in edges|lane_edges if u in seen} - seen
            if not new: break
            seen |= new
        return b in seen
    for i, a in enumerate(leaves):
        for b in leaves[i+1:]:
            if owner[a['id']] == owner[b['id']] or ancestor(a['id'],b['id']) or ancestor(b['id'],a['id']): continue
            for p in a['scope']['exclusive_paths']:
                for q in b['scope']['exclusive_paths']:
                    require(not (p==q or p.startswith(q+'/') or q.startswith(p+'/')),
                            f"parallel scope collision: {a['id']} / {b['id']} at {p} / {q}")
    if manifest:
        entries = (root/'MANIFEST.sha256').read_text().splitlines()
        declared = set()
        for line in entries:
            expected, rel = line.split('  ', 1)
            require(safe_path(rel) and rel not in declared, 'unsafe/duplicate manifest entry')
            declared.add(rel)
            require((root/rel).is_file() and digest(root/rel)==expected, f'checksum mismatch: {rel}')
        actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
                  and '__pycache__' not in p.parts and p.name!='MANIFEST.sha256'}
        require(actual == declared, 'manifest missing or unexpectedly added package files')
    demo_checked = False
    if source_root is not None:
        for entry in load(root/'machine/delivery_files.json')['files']:
            require(safe_path(entry['path']), 'unsafe delivery destination')
            f = source_root.resolve()/entry['path']
            require(f.is_file() and digest(f)==entry['sha256'], 'demo missing or modified: '+entry['path'])
        demo_checked = True
    return {'status':'package_valid','todo_records':len(tasks),'leaf_tasks':len(leaves),
            'first_class_lanes':len(lanes),'dependency_and_lane_graph_acyclic':True,
            'four_way_post_foundation_fanout':True,'manifest_checked':manifest,
            'placed_demo_checked':demo_checked,'native_authority_validation':'NOT_PERFORMED',
            'gpu_execution_validation':'NOT_PERFORMED','layers':layers}

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--package-root',type=Path,default=PACKAGE)
    ap.add_argument('--source-root',type=Path,help='Also check the supplied demo has been placed, byte for byte')
    ap.add_argument('--without-manifest',action='store_true',help='Development-only package check; never enough for manual bootstrap')
    a = ap.parse_args()
    print(json.dumps(validate(a.package_root,a.source_root,not a.without_manifest),indent=2))
    return 0
if __name__=='__main__':
    try: raise SystemExit(main())
    except (ValueError,KeyError,OSError) as e: raise SystemExit(f'package validation FAILED: {e}')
