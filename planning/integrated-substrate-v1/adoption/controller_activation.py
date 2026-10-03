#!/usr/bin/env python3
"""Refresh activation evidence while exact reviewed controller claims are running."""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys

BOOTSTRAP = Path(__file__).resolve().parents[2] / 'integrated-substrate-bootstrap-2026-10-02'
spec = importlib.util.spec_from_file_location('is1_bootstrap_activation', BOOTSTRAP / 'prepare_activation.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
require = base.require


def pairs(values, label):
    result = {}
    for value in values:
        project, item = value.split('=', 1)
        require(project in base.PROJECTS and item, 'invalid '+label)
        require(project not in result, 'duplicate '+label+': '+project)
        result[project] = item
    return result


def validate_current(project, state, workflow, allowed, sessions, claims, plan):
    """Validate exceptions before using the unchanged historical inventory validator."""
    require(not workflow['local_children'], 'local children: '+project)
    tasks = {x['id']: x for x in state['tasks']}
    requested = allowed.get(project)
    actual_claims = {x['task_id']: x for x in claims}
    require(len(actual_claims) == len(claims), 'duplicate live claims')
    expected = {requested} if requested else set()
    require(set(actual_claims) == expected, 'unexpected or missing controller claims: '+project)
    semantic_claims = {t['id']: t['active_claim'] for t in state['tasks'] if t.get('active_claim')}
    require(set(semantic_claims) == expected, 'semantic claims mismatch: '+project)
    run_id, root_id = base.NEW_RUNS[project]
    run = next((r for r in workflow['runs'] if r['id'] == run_id), None)
    require(run is not None and run['root_task_id'] == root_id, 'actual IS1 run required: '+project)
    bindings = []
    for r in workflow['runs']:
        for lane in r['lanes']:
            dispatch = lane.get('dispatch')
            active_queue = [q['task_id'] for q in lane['queue'] if q['state'] == 'active']
            if dispatch or active_queue:
                require(requested is not None and r['id'] == run_id, 'unreviewed dispatch or active queue')
                require(dispatch is not None and active_queue == [requested], 'controller queue binding mismatch')
                claim = actual_claims[requested]
                require(dispatch['task_id'] == requested and dispatch['claim_id'] == claim['id'] == semantic_claims[requested]['id'], 'controller claim binding mismatch')
                require(dispatch['session_id'] == claim['session_id'] == sessions[project], 'controller session mismatch')
                require(claim['state'] == 'active', 'inactive controller claim')
                bindings.append({'run_id': run_id, 'lane_id': lane['id'], **dispatch})
    require(len(bindings) == len(expected), 'controller dispatch missing or duplicated')
    require(all(any(a['dispatch_id'] == b['dispatch_id'] and a['claim_id'] == b['claim_id'] and a['session_id'] == b['session_id'] and a['task_id'] == b['task_id'] and a['run_id'] == b['run_id'] and a['lane_id'] == b['lane_id'] for b in bindings) for a in workflow['first_class_agents']), 'unreviewed observable agent')
    if requested:
        require(requested in tasks and tasks[requested]['effective_state'] == 'active', 'controller task not running')
        planned = next((t for t in plan['tasks'] if t['id'] == requested), None)
        require(planned is not None and requested.startswith(root_id.split('-000')[0]+'-'), 'task outside reviewed IS1 plan')
        scope = planned['scope']
        reviewed_scope = {(mode, path) for mode, key in [('exclusive','exclusive_paths'), ('read','read_paths'), ('forbidden','forbidden_paths')] for path in scope.get(key, [])}
        require({(s['mode'], s['path']) for s in tasks[requested]['scopes']} == reviewed_scope, 'controller task scope changed')
    return bindings


def read_claims(state, workflow):
    """Enrich actual semantic claims only with their unique public dispatch."""
    require(state['read_authority_fingerprint'] == workflow['read_authority_fingerprint'], 'authority changed before controller claim read')
    dispatches = [lane['dispatch'] for run in workflow['runs'] for lane in run['lanes'] if lane.get('dispatch')]
    claims = []
    for task in state['tasks']:
        claim = task.get('active_claim')
        if claim:
            matches = [d for d in dispatches if d['claim_id'] == claim['id'] and d['task_id'] == task['id']]
            require(len(matches) == 1, 'semantic claim has missing or duplicate dispatch: '+task['id'])
            item = base.select(claim, 'id state baseline_head baseline_revision')
            item.update(task_id=task['id'], session_id=matches[0]['session_id'])
            claims.append(item)
    return claims


def capture(classifications, expected, allowed, sessions):
    sys.path.insert(0, str(base.PROVIDER))
    from todo_orchestrator.semantic import SemanticReader
    accepted = base.heads(expected)
    predecessors = json.loads((base.PROJECTS['cellerator']/base.PACKAGE/'machine/predecessors.json').read_text())
    inventories, rows = {}, []
    for project, root in base.PROJECTS.items():
        reader = SemanticReader(root)
        state, workflow = reader.state(), reader.workflow()
        fingerprint = state['read_authority_fingerprint']
        require(fingerprint == workflow['read_authority_fingerprint'], 'authority changed between observations')
        claims = read_claims(state, workflow)
        plan = json.loads((root/'planning/integrated-substrate-bootstrap-2026-10-02'/f'{project}.todo-plan.json').read_text())
        bindings = validate_current(project, state, workflow, allowed, sessions, claims, plan)
        # Strip only the explicitly verified exceptions from copies supplied to
        # the historical validator; published evidence retains all actual activity.
        historical_state, historical_workflow = copy.deepcopy(state), copy.deepcopy(workflow)
        for task in historical_state['tasks']:
            if task['id'] == allowed.get(project):
                task['active_claim'] = None
        historical_workflow['first_class_agents'] = []
        for run in historical_workflow['runs']:
            for lane in run['lanes']:
                if lane.get('dispatch') and lane['dispatch']['task_id'] == allowed.get(project):
                    lane['dispatch'] = None
                    for q in lane['queue']:
                        if q['task_id'] == allowed[project] and q['state'] == 'active':
                            q['state'] = 'queued'
        classified, pending, historical = base.validate_inventory(project, historical_state, historical_workflow, classifications, accepted[project])
        tasks = {t['id']: t for t in state['tasks']}
        require(all(tasks[t]['effective_state'] == 'done' for t in predecessors[project]), 'predecessor not done: '+project)
        for row in classified:
            if row['run_id'] == base.NEW_RUNS[project][0]:
                row.update(reason='IS1 progression reviewed with exact controller claims and scopes.', trigger='Continue assigned IS1 tasks after fresh activation gates.')
        evidence = {'schema_version':1, 'project':project, 'head':accepted[project], 'project_uuid':state['project_uuid'], 'revision':state['revision'], 'authority_fingerprint':fingerprint, 'tasks':[base.sanitized_task(t) for t in state['tasks']], 'runs':[base.sanitized_run(r) for r in workflow['runs']], 'run_inventory':classified, 'agents':workflow['first_class_agents'], 'claims':claims, 'reviewed_controller_bindings':bindings, 'children':workflow['local_children'], 'recovery_needed':workflow['recovery_needed'], 'pending_commit_checks':pending, 'preserved_historical_integration_queue':historical, 'active_run_id':workflow.get('active_run_id')}
        for key in ('patch_artifacts','pending_patches'):
            evidence[key] = [base.select(x,'id workspace_id run_id lane_id task_id kind artifact_ref content_hash base_commit state created_at') for x in workflow[key]]
        evidence['integration_queue'] = [base.select(x,'id run_id integration_task_id integrator_lane_id position state') | {'has_conflict':bool(x.get('conflict'))} for x in workflow['integration_queue']]
        evidence['checkpoints'] = [base.select(x,'id task_id effective_state raw_state revision reached_at revoked_at') for x in state.get('checkpoints',[])]
        evidence['gates'] = [base.select(x,'id task_id type required effective_state raw_status raw_valid last_run_at') for x in state.get('gates',[])]
        evidence['messages'] = [base.select(x,'id run_id author_lane_id task_id kind blocking state revision') for x in workflow['blocking_messages']]
        inventories[project] = evidence
        rows.extend(classified)
    check_consistency(expected, accepted, inventories)
    return inventories, rows, accepted


def check_consistency(expected, accepted, inventories):
    from todo_orchestrator.semantic import SemanticReader
    require(base.heads(expected) == accepted, 'source changed during capture/publication')
    for project, root in base.PROJECTS.items():
        require(SemanticReader(root).state()['read_authority_fingerprint'] == inventories[project]['authority_fingerprint'], 'authority changed during capture/publication: '+project)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviewed-inventory', required=True, type=Path)
    parser.add_argument('--expected-head', action='append', required=True)
    parser.add_argument('--allowed-current-task', action='append', default=[])
    parser.add_argument('--controller-session', action='append', default=[])
    parser.add_argument('--reviewed-by', required=True)
    parser.add_argument('--approve-reviewed-inventory', action='store_true')
    args = parser.parse_args()
    review_path = base.PROJECTS['cellerator']/base.PACKAGE/'results/activation-review.json'
    if args.approve_reviewed_inventory:
        base.invalidate_review(review_path)
    classifications = json.loads(args.reviewed_inventory.read_text())
    require(isinstance(classifications,list) and len(classifications) == 27, 'exact original 27-run inventory required')
    expected = pairs(args.expected_head, 'expected head')
    require(set(expected) == set(base.PROJECTS), 'expected heads required for all repositories')
    allowed = pairs(args.allowed_current_task, 'allowed task')
    sessions = pairs(args.controller_session, 'controller session')
    require(set(sessions) == set(allowed), 'session required for each allowed task')
    require(bool(args.reviewed_by.strip()), 'reviewer required')
    inventories, rows, accepted = capture(classifications, expected, allowed, sessions)
    summary = {'status':'passed', 'heads':accepted, 'run_count':len(rows), 'claims':{p:len(e['claims']) for p,e in inventories.items()}, 'approved':args.approve_reviewed_inventory, 'authority_writes':False}
    if args.approve_reviewed_inventory:
        summary['activation_review'] = str(base.publish_activation(inventories, rows, accepted, args.reviewed_inventory, args.reviewed_by, lambda:check_consistency(expected,accepted,inventories)))
    print(json.dumps(summary,sort_keys=True))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('Controller activation blocked: '+str(exc), file=sys.stderr)
        sys.exit(1)
