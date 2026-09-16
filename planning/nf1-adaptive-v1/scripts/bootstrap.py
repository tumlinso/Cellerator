#!/usr/bin/env python3
"""Manual additive import through the installed Project Control mutation boundary.

Preview is read-only. Apply consumes exactly the saved, reviewed proposal once.
No operation launches workers, recovers claims, retires runs, or edits source.
"""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import importlib
import json
import sys
from pathlib import Path
sys.dont_write_bytecode = True
from common import PACKAGE, TERMINAL, canonical, external, host_context, load, require, rows, sha, sha_bytes, verify_manifest, write_new


def runtime_stamp() -> dict:
    files = {}
    for name in ('project_control.mutation', 'project_control.models', 'project_control.proposals',
                 'project_control.workflow_binding', 'todo_orchestrator.plan', 'todo_orchestrator.retirement'):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        files[name] = {'path': str(path), 'sha256': sha(path)}
    python = Path(sys.executable).resolve()
    return {'python': str(python), 'python_sha256': sha(python), 'modules': files}


def package_pair(roots: dict[str, Path]) -> dict:
    meta = load(PACKAGE / 'machine/package.json')
    require(PACKAGE == (roots[meta['project']] / meta['install_path']).resolve(),
            'Run the installed package in the registered authority checkout, not a donor/nested gitlink')
    result = {}
    for name, root in roots.items():
        package = root / meta['install_path']
        other = load(package / 'machine/package.json')
        require(other['pair_id'] == meta['pair_id'] and other['project'] == name, 'Wrong peer package identity')
        require(sha(package / 'machine/program.json') == sha(PACKAGE / 'machine/program.json'), 'Peer program differs')
        result[name] = {'path': str(package.resolve()), 'manifest_sha256': verify_manifest(package)}
    return result


def all_new_ids(plan: dict) -> set[str]:
    identifiers = {t['id'] for t in plan['tasks']}
    for field in ('invariants', 'interfaces', 'decisions', 'barriers', 'resource_classes'):
        identifiers.update(x['id'] for x in plan.get(field, []))
    for task in plan['tasks']:
        for field in ('gates', 'checkpoints'):
            identifiers.update(x['id'] for x in task.get(field, []))
    for run in plan['runs']:
        identifiers.add(run['id']); identifiers.update(x['id'] for x in run['lanes'])
    return identifiers


def verify_import(review: dict, snapshot: object, *, pristine: bool = True) -> dict:
    """Verify actual identities and preserved rows, not just an apply return code."""
    plan = review['proposal']['proposed_change']
    existing = rows(snapshot, 'tasks')
    for tid, digest in review['preserved_task_rows'].items():
        require(tid in existing and sha_bytes(canonical(existing[tid])) == digest,
                'Existing task changed during import: ' + tid)
    for task in plan['tasks']:
        actual = existing.get(task['id'], {})
        for key in ('id', 'parent_id', 'title', 'objective', 'kind'):
            require(actual.get(key) == task.get(key), 'Imported task differs: ' + task['id'] + ':' + key)
        if pristine:
            require(actual.get('status') == 'planned', 'Import unexpectedly advanced a task: ' + task['id'])
    runs = rows(snapshot, 'workflow_runs'); lanes = rows(snapshot, 'workflow_lanes')
    queue = snapshot.todo_tables.get('workflow_lane_tasks')
    require(isinstance(queue, list), 'Lane queue read coverage unavailable')
    for run in plan['runs']:
        actual = runs.get(run['id'], {})
        require(actual.get('root_task_id') == run['root_task_id'], 'Run root mismatch')
        for lane in run['lanes']:
            actual = lanes.get(lane['id'], {})
            require(actual.get('run_id') == run['id'] and actual.get('role') == lane['role'], 'Lane identity mismatch')
            require(actual.get('parent_lane_id') == lane.get('parent_lane_id'), 'Lane parent mismatch')
            require(actual.get('workspace_mode') == lane['workspace']['mode'], 'Lane workspace mode mismatch')
            items = sorted((q for q in queue if q['lane_id'] == lane['id']), key=lambda q: q['position'])
            require([q['task_id'] for q in items] == lane['tasks'], 'Lane queue differs')
            if pristine:
                require(all(q['state'] == 'queued' for q in items), 'Import unexpectedly dispatched a lane')
    gates = rows(snapshot, 'gates')
    for task in plan['tasks']:
        for expected in task.get('gates', []):
            actual = gates.get(expected['id'], {})
            require(actual.get('task_id') == task['id'], 'Gate owner mismatch')
            cfg = actual.get('config_json') or '{}'
            cfg = json.loads(cfg) if isinstance(cfg, str) else cfg
            require(cfg.get('argv') == expected['argv'], 'Gate executable/bindings differ')
    return {'status': 'verified', 'project_uuid': snapshot.project_uuid, 'revision': snapshot.todo_revision,
            'new_tasks': len(plan['tasks']), 'preserved_existing_tasks': len(review['preserved_task_rows']),
            'workers_started': False}


def preview(state: Path) -> dict:
    from project_control.mutation import validate_native_plan
    from project_control.models import ProposalEnvelope, ObservationPreconditions
    cfg, roots, snapshots = host_context(); pair = package_pair(roots)
    meta = load(PACKAGE / 'machine/package.json'); project = meta['project']; snap = snapshots[project]
    state = external(state, list(roots.values())); state.mkdir(parents=True, exist_ok=True)
    plan = copy.deepcopy(load(PACKAGE / meta['native_plan']))
    occupied = {str(r[k]) for table in snap.todo_tables.values() for r in table
                for k in ('id', 'name') if isinstance(r, dict) and r.get(k) is not None}
    require(not all_new_ids(plan) & occupied,
            'Successor IDs already exist; inspect/verify the previous import, do not blindly reapply: ' + repr(sorted(all_new_ids(plan) & occupied)))
    bindings = state / (project + '-execution-bindings.json')
    if not bindings.exists():
        write_new(bindings, {'format': 'nf1a-execution-bindings-v1', 'pair_id': meta['pair_id'],
                            'project': project, 'evidence_dir': str(state / 'evidence'), 'tasks': {},
                            'note': 'Unconfigured by design. ADOPT binds actual tests/runtime; empty bindings cannot pass.'})
    for task in plan['tasks']:
        for gate in task.get('gates', []):
            gate['argv'][0] = str(Path(sys.executable).resolve())
            gate['argv'] += ['--bindings', str(bindings)]
    validation = validate_native_plan(cfg, project, plan)
    expected = {t['id'] for t in plan['tasks']}
    require(validation.get('revision') == snap.todo_revision, 'Authority moved during preview; reobserve')
    require(validation.get('valid') is True, 'Native plan validation did not pass: ' + repr(validation))
    require(not validation.get('would_modify') and set(validation.get('would_add', [])) == expected,
            'Native diff is not exactly the intended additive import')
    stamp = runtime_stamp()
    proposal = ProposalEnvelope.create(intent='Import the reviewed NF1A successor only; no recovery, retirement, source edits or dispatch',
        proposed_change=plan,
        observation_preconditions=ObservationPreconditions.model_validate(validation['current_observation_preconditions']))
    now = dt.datetime.now(dt.timezone.utc)
    review = {'format': 'nf1a-import-review-v1', 'project': project, 'pair_id': meta['pair_id'],
              'created_at': now.isoformat(), 'expires_at': (now + dt.timedelta(hours=1)).isoformat(),
              'runtime': stamp, 'packages': pair, 'roots': {p: str(r) for p, r in roots.items()},
              'proposal': proposal.model_dump(mode='json'), 'native_validation': validation,
              'preserved_task_rows': {tid: sha_bytes(canonical(r)) for tid, r in rows(snap, 'tasks').items()},
              'preserved_successful_task_rows': {tid: sha_bytes(canonical(r)) for tid, r in rows(snap, 'tasks').items() if tid.startswith(meta['baseline']['old']+'-') and r.get('status') == 'done'},
              'execution_bindings': str(bindings)}
    path = state / (project + '-review-' + now.strftime('%Y%m%dT%H%M%S%fZ') + '.json')
    write_new(path, review)
    return {'status': 'review_ready', 'review': str(path), 'review_sha256': sha(path),
            'would_add': sorted(expected), 'would_modify': [], 'native_warnings': validation.get('warnings', []),
            'expires_at': review['expires_at'], 'workers_started': False}


def apply(review_path: Path, confirmation: str) -> dict:
    from project_control.mutation import apply_proposal
    digest = sha(review_path); require(confirmation == digest, 'Confirm the exact review SHA-256')
    review = load(review_path); meta = load(PACKAGE / 'machine/package.json')
    require(review.get('project') == meta['project'] and review.get('pair_id') == meta['pair_id'], 'Review/package identity mismatch')
    require(dt.datetime.now(dt.timezone.utc) <= dt.datetime.fromisoformat(review['expires_at']), 'Review expired; reobserve')
    cfg, roots, _ = host_context(); external(review_path, list(roots.values()))
    require(package_pair(roots) == review['packages'], 'Package bytes/locations changed after review')
    require(runtime_stamp() == review['runtime'], 'Installed runtime or interpreter changed after review')
    require({p: str(r) for p, r in roots.items()} == review['roots'], 'Registered roots changed')
    attempt = Path(str(review_path) + '.attempt.json')
    write_new(attempt, {'format': 'nf1a-import-attempt-v1', 'review_sha256': digest,
                        'state': 'reserved_before_mutation', 'automatic_retry_authorized': False})
    try:
        receipt = apply_proposal(cfg, review['project'], review['proposal'])
        _, _, snapshots = host_context()
        proof = verify_import(review, snapshots[review['project']])
        result = {'format': 'nf1a-import-receipt-v1', 'review_sha256': digest, 'receipt': receipt, 'verification': proof}
        destination = Path(str(review_path) + '.applied.json'); write_new(destination, result)
        return {'status': 'applied_and_verified', 'receipt': str(destination), 'receipt_sha256': sha(destination), **proof}
    except Exception as exc:
        write_new(Path(str(review_path) + '.uncertain.json'), {'state': 'failed_or_outcome_unknown',
            'review_sha256': digest, 'error': str(exc), 'automatic_retry_authorized': False})
        raise


def retirement_intent(output: Path) -> dict:
    _, roots, snapshots = host_context(); meta = load(PACKAGE / 'machine/package.json')
    snap = snapshots[meta['project']]; tasks = rows(snap, 'tasks')
    require(meta['new_run_id'] in rows(snap, 'workflow_runs'), 'Import successor before preparing retirement')
    mapping = load(PACKAGE / 'machine/legacy-disposition.json'); allowed = {m['legacy_task_id'] for m in mapping}
    lanes = {lid for lid, row in rows(snap, 'workflow_lanes').items() if row['run_id'] == meta['old_run_id']}
    require('workflow_lane_tasks' in snap.todo_tables, 'Missing old lane membership')
    members = {r['task_id'] for r in snap.todo_tables['workflow_lane_tasks'] if r['lane_id'] in lanes}
    require(members == allowed, 'Old run membership changed; review explicit mapping before retirement')
    for m in mapping:
        tid = m['legacy_task_id']; require(tid in tasks, 'Missing historical task: ' + tid)
        if m['observed_status'] == 'done':
            require(tasks[tid]['status'] == 'done', 'Previously completed task changed: ' + tid)
    selected = sorted(t for t in allowed if tasks[t]['status'] not in TERMINAL)
    require(selected, 'No nonterminal records remain; inspect retirement/run status instead')
    intent = {'source_run_id': meta['old_run_id'], 'successor_run_id': meta['new_run_id'], 'task_ids': selected,
              'dispositions': {t: 'superseded' for t in selected},
              'reason': 'NF1A outcome ownership replaces only unfinished NF1 procedure; source and successful evidence remain preserved.'}
    output = external(output, list(roots.values())); write_new(output, intent)
    return {'status': 'intent_only', 'path': str(output), 'sha256': sha(output), 'tasks': len(selected),
            'next': 'Canonical prepare-retire-run-batch, review, then root-authorized retirement after preservation and quiescence.'}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__); commands = p.add_subparsers(dest='command', required=True)
    q = commands.add_parser('preview'); q.add_argument('--state-dir', type=Path, required=True)
    q = commands.add_parser('apply'); q.add_argument('--review', type=Path, required=True); q.add_argument('--confirm', required=True)
    q = commands.add_parser('verify'); q.add_argument('--review', type=Path, required=True)
    q = commands.add_parser('retirement-intent'); q.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.command == 'preview': result = preview(args.state_dir)
    elif args.command == 'apply': result = apply(args.review, args.confirm)
    elif args.command == 'retirement-intent': result = retirement_intent(args.output)
    else:
        review = load(args.review); _, roots, snapshots = host_context()
        require(review['project'] == load(PACKAGE / 'machine/package.json')['project'], 'Wrong review project')
        external(args.review, list(roots.values()))
        result = verify_import(review, snapshots[review['project']])
    print(json.dumps(result, sort_keys=True))

if __name__ == '__main__':
    try: main()
    except Exception as exc:
        print(json.dumps({'status': 'stopped', 'error': str(exc), 'automatic_retry_authorized': False}), file=sys.stderr)
        raise SystemExit(1)
