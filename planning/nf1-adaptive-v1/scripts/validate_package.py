#!/usr/bin/env python3
"""Offline structural checks. These are not installed-kernel or CUDA qualification."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from common import PACKAGE, load, require, verify_manifest


def acyclic(graph: dict[str, set[str]]) -> list[str]:
    todo = {n:set(deps) for n,deps in graph.items()}; done=[]
    require(all(d in todo for ds in todo.values() for d in ds), 'Unknown dependency')
    while todo:
        ready=sorted(n for n,ds in todo.items() if not ds)
        require(ready, 'Dependency cycle: '+repr(sorted(todo)))
        done.extend(ready)
        for n in ready: del todo[n]
        for ds in todo.values(): ds.difference_update(ready)
    return done


def validate(package: Path, peer: Path | None=None, manifest: bool=True) -> dict:
    if manifest: verify_manifest(package)
    meta=load(package/'machine/package.json'); program=load(package/'machine/program.json')
    plan=load(package/'machine/native-plan.json'); outcomes=load(package/'machine/outcomes.json')
    mapping=load(package/'machine/legacy-disposition.json'); reqs=load(package/'machine/requirements.json')
    require(plan['schema_version']==3 and plan['project']['name']==meta['baseline']['label'], 'Wrong native plan identity')
    tasks={t['id']:t for t in plan['tasks']}; require(len(tasks)==len(plan['tasks'])==9, 'Unexpected duplicate/count')
    require(len(plan['runs'])==1,'Exactly one local successor run required')
    run=plan['runs'][0]; require(run['id']==meta['new_run_id'],'Run differs')
    lanes={l['id']:l for l in run['lanes']};require(len(lanes)==len(run['lanes'])==7,'Duplicate/lane count')
    roots=[l for l in lanes.values() if not l.get('parent_lane_id')];require(len(roots)==1,'Exactly one coordinator root')
    prefix=meta['baseline']['prefix'];require(roots[0]['tasks']==[prefix+'-ADOPT',prefix+'-CONTROL',prefix+'-0000'],'Coordinator/aggregate lifecycle changed')
    assigned=[t for l in lanes.values() for t in l['tasks']]
    require(len(assigned)==len(tasks) and set(assigned)==set(tasks),'Tasks must be assigned once')
    native_graph={t:set() for t in tasks}
    for task in tasks.values():
        require(task.get('title') and task.get('objective'),'Durable outcome needs intent')
        require('work_profile' not in task and 'model' not in task and 'reasoning_effort' not in task,'Do not invent unpersisted task/provider fields')
        for d in task.get('depends_on',[]):
            require(d['type']=='task' and d['task_id'] in tasks,'Native dependencies must be local and declared')
            require(d['task_id']!=run['root_task_id'],'Child cannot depend on aggregate')
            native_graph[task['id']].add(d['task_id'])
        for name in task.get('scope',{}).get('exclusive_paths',[]):
            require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe source scope')
        for gate in task.get('gates',[]):
            require(gate['required'] is True and gate['type']=='command' and gate['expected_exit_code']==0,'Gate contract changed')
            require('cuda_execution' not in gate,'Unsupported invented CUDA field')
            if outcomes[task['id']]['gate_kind']=='execution':require(gate.get('cuda',{}).get('gpus')==1,'Native foreground GPU lease missing')
    for lane in lanes.values():
        require(not lane.get('parent_lane_id') or lane['parent_lane_id'] in lanes,'Foreign lane parent')
        if lane['workspace']['mode']=='isolated_merge':
            target=lane['workspace'].get('integration_task_id')
            destination=[l for l in lanes.values() if target in l['tasks']]
            require(len(destination)==1 and destination[0]['role']=='integrator' and destination[0]['workspace']['mode']=='exclusive','Missing exclusive integration owner')
        for previous,current in zip(lane['tasks'],lane['tasks'][1:]):native_graph[current].add(previous)
    native_graph[run['root_task_id']].update(t for t,v in tasks.items() if v.get('parent_id')==run['root_task_id'])
    order=acyclic(native_graph)
    # Every direct source overlap must already be ordered at claim time, not just
    # at completion. Isolated branches are not an excuse for avoidable contests.
    def ancestors(t):
        result=set();pending=list(native_graph[t])
        while pending:
            n=pending.pop()
            if n not in result:result.add(n);pending+=list(native_graph[n])
        return result
    for a,ta in tasks.items():
        for b,tb in tasks.items():
            if a>=b:continue
            overlaps=[(x,y) for x in ta.get('scope',{}).get('exclusive_paths',[]) for y in tb.get('scope',{}).get('exclusive_paths',[])
                      if Path(x)==Path(y) or Path(x) in Path(y).parents or Path(y) in Path(x).parents]
            require(not overlaps or a in ancestors(b) or b in ancestors(a),'Unordered source ownership overlap: '+a+' / '+b)
    global_graph={n:set(o['completion_requires']) for n,o in outcomes.items()}
    for name,info in program['projects'].items():
        p=info['prefix']; global_graph[p+'-0000']={n for n in outcomes if n.startswith(p+'-')}
    acyclic(global_graph)
    ids={r['id'] for r in reqs};require(len(ids)==len(reqs)==30,'Requirement IDs/count')
    require(all(r['owner'] in outcomes or (r['owner']=='PAIR' and all(r['id'] in outcomes[p+'-ADOPT']['requirements'] for p in ('CE-NF1A','GH-NF1A'))) for r in reqs),'Requirement lacks owner')
    require(all(m['successor'] in outcomes or m['successor'].endswith('-0000') for m in mapping),'Legacy result lacks successor')
    require(all(m['requirements'] and set(m['requirements'])<=ids for m in mapping),'Legacy obligation dropped')
    require(len({m['legacy_task_id'] for m in mapping})==len(mapping),'Duplicate legacy mapping')
    done=[m for m in mapping if m['observed_status']=='done']
    require(all(m['disposition']=='preserve_completed_record' for m in done),'Completed work must remain historical')
    intent=load(package/'machine/retirement-intent.json')
    require(set(intent['task_ids'])=={m['legacy_task_id'] for m in mapping if m['observed_status']!='done'},'Retirement scope differs from observed unfinished mapping')
    require(set(tasks).isdisjoint(m['legacy_task_id'] for m in mapping),'Never redefine an old task in the successor')
    entry=(package/'handoff/START_CONTROLLER.md').read_text().rstrip('\n');require(len(entry)<4000,'Launch text too long')
    if peer:
        if manifest:verify_manifest(peer)
        other=load(peer/'machine/package.json');require(other['pair_id']==meta['pair_id'] and other['project']==meta['peer_project'],'Wrong paired package')
        require(load(peer/'machine/program.json')==program and load(peer/'machine/outcomes.json')==outcomes,'Pair plans disagree')
        otherplan=load(peer/'machine/native-plan.json')
        require(set(tasks).isdisjoint(t['id'] for t in otherplan['tasks']),'Cross-project task IDs collide')
    return {'status':'passed','qualification':'offline_package_structure_only','project':meta['project'],'records':len(tasks),
            'lanes':len(lanes),'requirements':len(reqs),'legacy_records':len(mapping),'preserved_completed':len(done),
            'conditional_retirement_count':len(intent['task_ids']),'controller_characters_excluding_final_newline':len(entry),
            'native_claim_topological_order':order,'paired_completion_graph_acyclic':True}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--peer-package',type=Path);p.add_argument('--structure-only',action='store_true');a=p.parse_args()
    try:print(json.dumps(validate(PACKAGE,a.peer_package,not a.structure_only),indent=2))
    except Exception as exc:print('Package rejected: '+str(exc),file=sys.stderr);raise SystemExit(1)
