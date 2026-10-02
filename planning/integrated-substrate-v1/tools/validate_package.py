#!/usr/bin/env python3
"""Static structural validation of this package; not native Todo validation."""
from __future__ import annotations
import argparse,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
REL='planning/integrated-substrate-v1/'
PROJECTS=('cellerator','baseplane','glasshelix')
def load(p):return json.loads(p.read_text())
def check(ok,msg):
    if not ok:raise ValueError(msg)
def overlaps(a,b):
    x=pathlib.PurePosixPath(a);y=pathlib.PurePosixPath(b)
    return x==y or x in y.parents or y in x.parents

def validate(root=ROOT):
    outcomes={x['id']:x for x in load(root/'machine/outcomes.json')}
    pre=load(root/'machine/predecessors.json');cross=load(root/'machine/cross-authority.json')
    tasks={};lanes={};graphs={};epics=[]
    for pr in PROJECTS:
        plan=load(root/f'machine/{pr}.todo-plan.json')
        check(plan['schema_version']==3 and plan['project']['name']==pr,'native plan header mismatch')
        inv={x['id'] for x in plan.get('invariants',[])}
        for task in plan['tasks']:
            ident=task['id'];check(ident not in tasks,'duplicate task ID '+ident);tasks[ident]=task
            if task['kind']=='epic':epics.append(ident);continue
            check(ident in outcomes and outcomes[ident]['project']==pr,'outcome mapping mismatch')
            check(set(task.get('invariants',[]))<=inv,'unknown invariant')
            check(task.get('completion_contract',{}).get('required'),'completion contract absent')
            for path in sum([task['scope'].get(k,[]) for k in ('exclusive_paths','read_paths','forbidden_paths')],[]):
                p=pathlib.PurePosixPath(path);check(not p.is_absolute() and '..' not in p.parts,'unsafe task scope')
            check('.todo-orchestrator' in task['scope']['forbidden_paths'],'generated authority not protected')
            for ref in task.get('references',[]):
                check(ref.startswith(REL) and (root/ref[len(REL):]).is_file(),'missing task reference '+ref)
            for dep in task.get('depends_on',[]):
                check(dep['type']=='task','unexpected native dependency')
                target=dep['task_id']
                check(target in outcomes or target in pre[pr],'unknown local prerequisite '+target)
                if target in outcomes:check(outcomes[target]['project']==pr,'cross-authority IDs must not be local task dependencies')
            graphs[ident]={d['task_id'] for d in task.get('depends_on',[]) if d['task_id'] in outcomes}
        check(len(plan['runs'])==1,'expected one successor run per authority')
        assigned=[]
        for lane in plan['runs'][0]['lanes']:
            check(lane['id'] not in lanes,'duplicate lane ID');lanes[lane['id']]=lane;assigned+=lane['tasks']
            check(all(x in tasks for x in lane['tasks']),'lane owns missing task')
        check(len(assigned)==len(set(assigned)) and set(assigned)=={t['id'] for t in plan['tasks']},'each local task needs exactly one lane')
    for link in cross:
        check(link['provider_task'] in outcomes and link['consumer_task'] in outcomes,'bad cross-authority link')
        check(outcomes[link['provider_task']]['project']==link['provider_project'],'provider authority mismatch')
        graphs[link['consumer_task']].add(link['provider_task'])
    check(set(graphs)==set(outcomes),'outcomes differ from native tasks')
    check(not any(set(epics)&deps for deps in graphs.values()),'child depends on closure epic')
    # Detect global cycles and compute ideal dependency waves (not runtime/resource promises).
    remaining={k:set(v) for k,v in graphs.items()};waves=[]
    while remaining:
        ready=sorted(k for k,v in remaining.items() if not v)
        check(bool(ready),'cyclic global outcome DAG')
        waves.append(ready)
        remaining={k:v-set(ready) for k,v in remaining.items() if k not in ready}
    ancestors={}
    for wave in waves:
        for task in wave:
            ancestors[task]=set(graphs[task])
            for dep in graphs[task]:ancestors[task]|=ancestors[dep]
    ids=list(outcomes)
    for i,a in enumerate(ids):
        for b in ids[i+1:]:
            if outcomes[a]['project']!=outcomes[b]['project'] or a in ancestors[b] or b in ancestors[a]:continue
            for x in tasks[a]['scope']['exclusive_paths']:
                for y in tasks[b]['scope']['exclusive_paths']:
                    check(not overlaps(x,y),f'unordered write-scope collision {a}:{x} / {b}:{y}')
    # A completion flag alone cannot transfer an isolated worktree to its child.
    for lane in lanes.values():
        w=lane['workspace']
        if w['mode']!='isolated_merge':continue
        target=w['integration_task_id'];check(target in outcomes,'missing integration target')
        for task in lane['tasks']:
            consumers={k for k,deps in graphs.items() if task in deps}
            check(consumers=={target},f'isolated patch has an unmerged dependent: {task} -> {consumers}, target {target}')
    records=load(root/'machine/preservation.json')['records'];record_ids={r['id'] for r in records}
    check({f'BP:E{i:02}' for i in range(1,49)}<=record_ids,'lost Baseplane mechanism card')
    check({f'CE:M{i:02}' for i in range(1,14)}<=record_ids,'lost Cellerator mechanism card')
    check({f'BP:Q{i:02}' for i in range(1,6)}<=record_ids,'lost qualification obligation')
    check({f'NEW:N{i:02}' for i in range(1,13)}<=record_ids,'lost new mathematical card')
    for r in records:
        owners=r.get('owners',[])+[r[k] for k in ('sequence_consumer_owner','general_math_owner','scientific_consumer') if k in r]
        check(owners and all(x in outcomes for x in owners),'unassigned preservation owner '+r['id'])
    check(len(record_ids)==len(records),'duplicate preservation record')
    return {'status':'passed','validation_kind':'package_static_not_native_authority','outcomes':len(outcomes),'closure_epics':len(epics),'lanes':len(lanes),'preservation_records':len(records),'ideal_dependency_waves':waves,'maximum_ready_wave':max(map(len,waves))}

def verify_manifest(root=ROOT):
    manifest=load(root/'MANIFEST.json')
    for rel,digest in manifest['sha256'].items():
        rp=pathlib.PurePosixPath(rel)
        check(not rp.is_absolute() and '..' not in rp.parts and str(rp)!='.' and '\\' not in rel,'unsafe manifest member')
        p=root/rel
        check(not any(x.is_symlink() for x in [p,*p.parents] if x!=root and root in x.parents),'symlink package path '+rel)
        check(not p.is_symlink() and p.is_file() and p.resolve().is_relative_to(root.resolve()),'missing/symlink package member '+rel)
        check(hashlib.sha256(p.read_bytes()).hexdigest()==digest,'changed package member '+rel)
    actual=set()
    for p in root.rglob('*'):
        if not p.is_file() and not p.is_symlink():continue
        rel=p.relative_to(root);parts=rel.parts
        if parts[0] in ('receipts','results','_build','.git','.pytest_cache') or '__pycache__' in parts or p.suffix=='.pyc' or str(rel) in ('MANIFEST.json','local-config.json'):continue
        actual.add(rel.as_posix())
    check(actual==set(manifest['sha256']),'unlisted or missing immutable package files')
    return len(manifest['sha256'])

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify-manifest',action='store_true');p.add_argument('--write-report',type=pathlib.Path)
    a=p.parse_args();report=validate()
    if a.verify_manifest:report['verified_manifest_files']=verify_manifest()
    text=json.dumps(report,indent=2)
    if a.write_report:a.write_report.write_text(text+'\n')
    print(text)
if __name__=='__main__':main()
