#!/usr/bin/env python3
"""Read-only checks for the derived parallel plans and preservation boundary."""
from pathlib import Path, PurePosixPath
import hashlib
import json

root = Path(__file__).resolve().parent
source = root.parent / 'baseplane_moonshot_bootstrap' / 'machine'
results = []
for repo, prefix, count, family_count in [('baseplane', 'BP', 16, 12), ('cellerator', 'CE', 7, 4)]:
    path = root / f'{repo}.todo-plan.json'
    raw = path.read_text()
    original = json.loads((source / path.name).read_text().replace('planning/bp-moonshot-20261002', 'planning/baseplane_moonshot_bootstrap'))
    plan = json.loads(raw)
    assert 'planning/bp-moonshot-20261002' not in raw
    assert plan['schema_version'] == original['schema_version'] == 3
    assert plan['project'] == original['project']
    assert plan['tasks'] == original['tasks'], 'task records changed beyond path correction'
    assert len(plan['tasks']) == count
    run, oldrun = plan['runs'][0], original['runs'][0]
    assert {k:v for k,v in run.items() if k not in ['lanes','charter']} == {k:v for k,v in oldrun.items() if k not in ['lanes','charter']}
    assert {k:v for k,v in run['charter'].items() if k != 'delegated_judgment'} == {k:v for k,v in oldrun['charter'].items() if k != 'delegated_judgment'}
    tasks = {t['id']: t for t in plan['tasks']}
    lanes = {l['id']: l for l in run['lanes']}
    assigned = [t for l in lanes.values() for t in l['tasks']]
    assert len(lanes) == family_count + 2
    assert len(assigned) == len(set(assigned)) == count and set(assigned) == set(tasks)
    foundation = lanes[f'{prefix}-MOON-L-FOUNDATION']
    integration = lanes[f'{prefix}-MOON-L-INTEGRATION']
    assert foundation['tasks'] == [f'{prefix}-MOON-010'] and foundation['workspace']['mode'] == 'exclusive'
    assert integration['role'] == 'integrator' and integration['workspace']['mode'] == 'exclusive'
    assert integration['tasks'][-1] == run['root_task_id']
    families = [l for l in lanes.values() if l not in [foundation, integration]]
    paths = []
    for lane in families:
        assert lane['parent_lane_id'] == foundation['id']
        assert lane['workspace'] == {'mode':'isolated_merge', 'integration_task_id':integration['tasks'][0]}
        assert len(lane['tasks']) == 1
        task = tasks[lane['tasks'][0]]
        assert [d['task_id'] for d in task['depends_on']] == foundation['tasks']
        for value in task['scope']['exclusive_paths']:
            path_parts = PurePosixPath(value).parts
            assert all(not (path_parts[:len(p)] == p or p[:len(path_parts)] == path_parts) for p in paths), 'overlapping family writes'
            paths.append(path_parts)
    assert integration['parent_lane_id'] == foundation['id']
    assert {d['task_id'] for d in tasks[integration['tasks'][0]]['depends_on']} == {l['tasks'][0] for l in families}
    visiting, done = set(), set()
    def visit(task):
        assert task not in visiting, 'dependency cycle'
        if task in done:
            return
        visiting.add(task)
        for dep in tasks[task].get('depends_on', []):
            assert dep['type'] == 'task' and dep['task_id'] in tasks
            visit(dep['task_id'])
        visiting.remove(task)
        done.add(task)
    for task in tasks:
        visit(task)
    request = json.loads((root / 'evidence' / f'{repo}.native-request.json').read_text())
    assert request['proposal'] == plan, 'preview request differs from delivered plan'
    results.append({'repository':repo, 'tasks':count, 'lanes':len(lanes), 'family_lanes':family_count, 'sha256':hashlib.sha256(raw.encode()).hexdigest(), 'task_records_preserved':True, 'family_write_scopes_disjoint':True, 'dependencies_acyclic':True, 'preview_request_matches':True})
print(json.dumps({'status':'parallel_static_checks_pass', 'plans':results}, indent=2))
