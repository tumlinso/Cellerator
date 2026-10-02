#!/usr/bin/env python3
"""Capture a sanitized native inventory; approval writes require a controller flag."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import os
import tempfile
import subprocess
import sys

PROJECTS = {'cellerator': Path('/home/tumlinson/Cellerator'), 'baseplane': Path('/home/tumlinson/Baseplane'), 'glasshelix': Path('/home/tumlinson/GlassHelix')}
PROVIDER = Path('/home/tumlinson/.agents/skills/todo-orchestrator')
PACKAGE = 'planning/integrated-substrate-v1'
NEW_RUNS = {'cellerator': ('CE-IS1-RUN-V1', 'CE-IS1-000'), 'baseplane': ('BP-IS1-RUN-V1', 'BP-IS1-000'), 'glasshelix': ('GH-IS1-RUN-V1', 'GH-IS1-000')}
TERMINAL = {'done', 'superseded', 'cancelled'}
# Exact pre-administration identities from the preserved native inventory.
KNOWN_RUNS = {
    'cellerator': set('CE-AMP-RUN-V1 CE-BIOPREP-EXTRACTION-RUN CE-CCP1-RUN-V1 CE-DOCS-RUN-V1 CE-EXOP-RUN-V1 CE-GEO-RUN-V1 CE-JBC-RUN-V1 CE-ML2-RUN-V1 CE-MOON-RUN-1 CE-MOON-RUN-V1 CE-NF1-RUN-V1 CE-NF1A-RUN-V1 CE-POST-REMAP-RUN CE-PTR-RUN CE-RU1-RUN-V1 CE-SS1-RUN-V1 compat-v2'.split()),
    'baseplane': set('BP-CUDA-LAB-CLOSE-1 BP-CUDA-LAB-RUN-1 BP-DOCS-RUN-V1 BP-MOON-RUN-1 compat-v2'.split()),
    'glasshelix': set('GH-DOCS-RUN-V1 GH-ML2-RUN-V1 GH-MOON-RUN-V1 GH-NF1-RUN-V1 GH-NF1A-RUN-V1'.split()),
}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def select(row, fields):
    return {key: row[key] for key in fields.split() if key in row}

def git(root, *args):
    return subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True, check=True).stdout.strip()

def heads(expected):
    actual = {}
    for project, root in PROJECTS.items():
        require(Path(git(root, 'rev-parse', '--show-toplevel')).resolve() == root.resolve(), 'wrong root: '+project)
        require(not git(root, 'status', '--porcelain', '--untracked-files=no'), 'tracked Git dirty: '+project)
        actual[project] = git(root, 'rev-parse', 'HEAD')
        require(re.fullmatch('[0-9a-f]{40}', actual[project]), 'invalid Git identity')
    require(actual == expected, 'heads differ from explicit controller expectations')
    return actual

def sanitized_task(task):
    result = select(task, 'id parent_id kind raw_status effective_state terminal execution revision tags')
    result['dependencies'] = [select(x, 'type task_id checkpoint_id decision_id interface_id state value required') for x in task.get('dependencies', [])]
    result['scopes'] = [select(x, 'mode path') for x in task.get('scopes', [])]
    result['has_active_claim'] = bool(task.get('active_claim'))
    return result

def sanitized_run(run):
    result = select(run, 'id root_task_id status active_charter_version')
    result['lanes'] = []
    for lane in run['lanes']:
        item = select(lane, 'id parent_lane_id role state workspace_mode context_cursor')
        item['queue'] = [select(x, 'position task_id state') for x in lane['queue']]
        item['has_dispatch'] = lane.get('dispatch') is not None
        item['workspace'] = select(lane['workspace'], 'id repository_identity run_id lane_id mode base_commit worktree_path branch state integration_task_id artifact_kind artifact_ref diff_hash cleanup_eligible created_at updated_at') if lane.get('workspace') else None
        result['lanes'].append(item)
    return result

def validate_inventory(project, state, workflow, classifications, source_head):
    require(workflow.get('available') is True, 'native workflow unavailable: '+project)
    require(state['read_authority_fingerprint'] == workflow['read_authority_fingerprint'], 'authority changed between observations: '+project)
    require(not workflow['first_class_agents'], 'observable agents: '+project)
    require(not workflow['local_children'], 'local children: '+project)
    require(not workflow['recovery_needed'], 'recovery required: '+project)
    tasks = {x['id']: x for x in state['tasks']}
    require(len(tasks) == len(state['tasks']), 'ambiguous task IDs')
    require(not any(x.get('active_claim') for x in tasks.values()), 'active claims: '+project)
    require(not any(x.get('blocking') for x in workflow['blocking_messages']), 'blocking messages: '+project)
    known = {x['run_id']: x for x in classifications if x['project'] == project}
    require(set(known) == KNOWN_RUNS[project], 'reviewed original identities changed: '+project)
    require(len(known) == sum(x['project'] == project for x in classifications), 'duplicate run classifications')
    actual = {x['id']: x for x in workflow['runs']}
    new_id, new_root = NEW_RUNS[project]
    require(set(actual) in (set(known), set(known) | {new_id}), 'run inventory changed: '+project)
    rows = []
    for run_id, run in actual.items():
        root_id = run['root_task_id']
        require(root_id in tasks, 'missing run root: '+root_id)
        root_state = tasks[root_id]['effective_state']
        if run_id == new_id:
            require(root_id == new_root, 'unexpected IS1 root')
            require(root_state not in TERMINAL, 'IS1 root unexpectedly terminal')
            row = {'project': project, 'run_id': run_id, 'root_task_id': root_id, 'disposition': 'preserved_deferred', 'reason': 'Installed IS1 awaits explicit controller dispatch.', 'preservation_owner': 'root controller', 'trigger': 'Explicit IS1 implementation dispatch after fresh activation gates.'}
        else:
            row = select(known[run_id], 'project run_id root_task_id disposition reason preservation_owner trigger')
            require(row['root_task_id'] == root_id, 'reviewed root changed: '+run_id)
            disposition = row['disposition']
            require(disposition in {'predecessor', 'completed', 'historical_closed', 'permission_locked', 'preserved_deferred'}, 'invalid classification')
            if disposition in {'completed', 'predecessor'}:
                require(root_state == 'done', 'reviewed completed root not done: '+root_id)
            elif disposition == 'historical_closed':
                require(root_state in TERMINAL, 'historical root not terminal: '+root_id)
            else:
                require(project == 'cellerator' and ((root_id == 'CE-AMP-00' and disposition == 'permission_locked') or (root_id == 'CE-GEO-00' and disposition == 'preserved_deferred')), 'unexpected deferred nonterminal run: '+run_id)
                require(row.get('preservation_owner') and row.get('trigger'), 'missing deferred owner/trigger')
        require(row.get('reason'), 'missing classification reason')
        require(not any(lane.get('dispatch') is not None or any(x['state'] == 'active' for x in lane['queue']) for lane in run['lanes']), 'active dispatch or lane queue: '+run_id)
        rows.append(row)
    pending_checks = []
    for patch in workflow['pending_patches']:
        require(patch['kind'] == 'commit', 'unreviewed pending noncommit artifact')
        commit = patch['artifact_ref']
        require(re.fullmatch('[0-9a-f]{40}', commit), 'invalid pending commit identity')
        require(tasks[patch['task_id']]['effective_state'] == 'done', 'pending artifact owner not done')
        code = subprocess.run(['git', '-C', str(PROJECTS[project]), 'merge-base', '--is-ancestor', commit, source_head], capture_output=True).returncode
        require(code == 0, 'pending commit absent from accepted HEAD: '+commit)
        pending_checks.append({'patch_id': patch['id'], 'task_id': patch['task_id'], 'commit': commit, 'ancestor_of_head': True, 'owner_done': True})
    historical_queue = []
    for entry in workflow['integration_queue']:
        if entry['state'] != 'integrated':
            require(entry['run_id'] in {'CE-NF1-RUN-V1', 'GH-NF1-RUN-V1'} and tasks[entry['integration_task_id']]['effective_state'] in TERMINAL, 'unexpected unresolved integration queue')
            historical_queue.append(select(entry, 'id run_id integration_task_id state position'))
    return rows, pending_checks, historical_queue

def capture(classifications, expected):
    sys.path.insert(0, str(PROVIDER))
    from todo_orchestrator.semantic import SemanticReader
    accepted = heads(expected)
    predecessors = json.loads((PROJECTS['cellerator']/PACKAGE/'machine/predecessors.json').read_text())
    inventories, rows, fingerprints = {}, [], {}
    for project, root in PROJECTS.items():
        reader = SemanticReader(root)
        state, workflow = reader.state(), reader.workflow()
        classified, pending, historical = validate_inventory(project, state, workflow, classifications, accepted[project])
        tasks = {x['id']: x for x in state['tasks']}
        require(all(tasks[x]['effective_state'] == 'done' for x in predecessors[project]), 'predecessor not done: '+project)
        fingerprints[project] = state['read_authority_fingerprint']
        evidence = {'schema_version': 1, 'project': project, 'head': accepted[project], 'project_uuid': state['project_uuid'], 'revision': state['revision'], 'authority_fingerprint': fingerprints[project], 'tasks': [sanitized_task(x) for x in state['tasks']], 'runs': [sanitized_run(x) for x in workflow['runs']], 'run_inventory': classified, 'agents': [], 'claims': [], 'children': [], 'recovery_needed': [], 'pending_commit_checks': pending, 'preserved_historical_integration_queue': historical}
        for key in ('patch_artifacts', 'pending_patches'):
            evidence[key] = [select(x, 'id workspace_id run_id lane_id task_id kind artifact_ref content_hash base_commit state created_at') for x in workflow[key]]
        evidence['integration_queue'] = [select(x, 'id run_id integration_task_id integrator_lane_id position state') | {'has_conflict': bool(x.get('conflict'))} for x in workflow['integration_queue']]
        evidence['active_run_id'] = workflow.get('active_run_id')
        evidence['checkpoints'] = [select(x, 'id task_id effective_state raw_state revision reached_at revoked_at') for x in state.get('checkpoints', [])]
        evidence['gates'] = [select(x, 'id task_id type required effective_state raw_status raw_valid last_run_at') for x in state.get('gates', [])]
        evidence['messages'] = [select(x, 'id run_id author_lane_id task_id kind blocking state revision') for x in workflow['blocking_messages']]
        evidence['rendezvous'] = [select(x, 'id run_id barrier_id mode quorum join_task_id state') | {'arrivals': [select(a, 'lane_id task_id state context_version revision') for a in x['arrivals']]} for x in workflow['rendezvous']]
        inventories[project] = evidence
        rows.extend(classified)
    require(heads(expected) == accepted, 'source changed during capture')
    for project, root in PROJECTS.items():
        require(SemanticReader(root).state()['read_authority_fingerprint'] == fingerprints[project], 'authority changed during capture: '+project)
    return inventories, rows, accepted

def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True)+'\n').encode()

def write_private(path, data):
    require(not path.is_symlink(), 'refuse symlink output')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o600)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()

def invalidate_review(path):
    marker = {'schema_version': 1, 'record_kind': 'is1_activation_review', 'approved': False,
              'reason': 'Approval preparation has not completed successfully.'}
    try:
        write_private(path, encode(marker))
    except Exception:
        # If replacing fails, removing this generated approval also blocks gates.
        # Never follow an output symlink or touch its target.
        path.unlink(missing_ok=True)
        raise

def publish_activation(inventories, rows, accepted, inventory_path, reviewed_by, check_consistency):
    review_path = PROJECTS['cellerator']/PACKAGE/'results/activation-review.json'
    invalidate_review(review_path)
    try:
        captured = datetime.now(timezone.utc).isoformat()
        evidence_entries = []
        for project, inventory in inventories.items():
            inventory['captured_at'] = captured
            data = encode(inventory)
            relative = PACKAGE+'/results/authority-review.json'
            write_private(PROJECTS[project]/relative, data)
            evidence_entries.append({'project': project, 'path': relative, 'sha256': hashlib.sha256(data).hexdigest()})
        review = {'schema_version': 1, 'record_kind': 'is1_activation_review', 'approved': False,
                  'reviewed_by': reviewed_by, 'captured_at': captured, 'heads': accepted,
                  'inventory_complete': True, 'claims_and_pending_patches_reconciled': True,
                  'predecessor_scope_reviewed': True, 'run_inventory': rows, 'authority_evidence': evidence_entries,
                  'reviewed_inventory_sha256': hashlib.sha256(inventory_path.read_bytes()).hexdigest()}
        write_private(review_path, encode(review))
        config = {'schema_version': 1, 'repos': {p: str(r) for p,r in PROJECTS.items()},
                  'todo_provider': str(PROVIDER), 'activation_review': str(review_path),
                  'max_review_age_seconds': 600, 'artifact_roots': {}}
        for root in PROJECTS.values():
            write_private(root/PACKAGE/'local-config.json', encode(config))
        check_consistency()
        # This atomic replacement is the sole publication of approved=True.
        review['approved'] = True
        write_private(review_path, encode(review))
        return review_path
    except BaseException:
        invalidate_review(review_path)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviewed-inventory', required=True, type=Path, help='Controller-reviewed JSON list of exactly 27 original run classifications.')
    parser.add_argument('--expected-head', action='append', required=True, metavar='PROJECT=SHA')
    parser.add_argument('--reviewed-by')
    parser.add_argument('--approve-reviewed-inventory', action='store_true')
    args = parser.parse_args()
    if args.approve_reviewed_inventory:
        require(args.reviewed_by, 'controller identity required for approval')
        invalidate_review(PROJECTS['cellerator']/PACKAGE/'results/activation-review.json')
    classifications = json.loads(args.reviewed_inventory.read_text())
    require(isinstance(classifications, list) and len(classifications) == 27, 'exact original 27-run reviewed inventory required')
    require(all(x.get('project') in PROJECTS and x.get('run_id') != NEW_RUNS[x['project']][0] for x in classifications), 'invalid original inventory')
    expected = dict(x.split('=', 1) for x in args.expected_head)
    require(set(expected) == set(PROJECTS) and len(args.expected_head) == 3, 'exact expected heads for all repositories required')
    if args.approve_reviewed_inventory:
        require(args.reviewed_by, 'controller identity required for approval')
    inventories, rows, accepted = capture(classifications, expected)
    summary = {'status': 'passed', 'heads': accepted, 'run_count': len(rows), 'approved': args.approve_reviewed_inventory, 'authority_writes': False}
    if args.approve_reviewed_inventory:
        def check_consistency():
            require(heads(expected) == accepted, 'source changed while writing observations')
            sys.path.insert(0, str(PROVIDER))
            from todo_orchestrator.semantic import SemanticReader
            for project, root in PROJECTS.items():
                require(SemanticReader(root).state()['read_authority_fingerprint'] == inventories[project]['authority_fingerprint'], 'authority changed while writing observations: '+project)
        summary['activation_review'] = str(publish_activation(inventories, rows, accepted, args.reviewed_inventory, args.reviewed_by, check_consistency))
    print(json.dumps(summary, sort_keys=True))

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('Activation preparation blocked: '+str(exc), file=sys.stderr)
        sys.exit(1)
