#!/usr/bin/env python3
"""Offline plan, queue, cross-authority graph, scope and delivery integrity checks."""
from pathlib import Path,PurePosixPath
import argparse,csv,hashlib,json,sys
sys.dont_write_bytecode=True
from compile_plan import assemble,load,PACKAGE

def demand(ok,msg):
    if not ok:raise ValueError(msg)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(x):return (json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)+'\n').encode()
def safe_rel(s):
    demand(isinstance(s,str) and bool(s),'empty/non-string path')
    p=PurePosixPath(s)
    demand(not p.is_absolute() and '..' not in p.parts and '.' not in s.split('/') and '\\' not in s and str(p)==s,'unsafe relative path: '+s)
    return p
def overlap(a,b):return a==b or a.startswith(b+'/') or b.startswith(a+'/')
def ancestors(parents):
    memo={};active=set()
    def visit(t):
        if t in memo:return memo[t]
        demand(t not in active,'Dependency/lane/cross-project cycle at '+t);active.add(t);out=set()
        for d in parents[t]:
            demand(d in parents,'Unknown prerequisite '+d);out.add(d);out|=visit(d)
        active.remove(t);memo[t]=out;return out
    for t in parents:visit(t)
    return memo

def graph(m):
    p=assemble(m);rows=p['tasks'];ids=[r['id'] for r in rows];by={r['id']:r for r in rows}
    demand(len(ids)==len(by),'duplicate task ID');pre=m['prefix'];root=pre+'-0000'
    demand(all(i.startswith(pre+'-') for i in ids),'foreign native task ID')
    demand([r['id'] for r in rows if r['kind']=='epic']==[root],'wrong epic')
    parents={i:set() for i in ids};explicit=[];owner={};lanes=m['run']['lanes'];laneids={l['id'] for l in lanes}
    demand(len(laneids)==len(lanes),'duplicate lane ID')
    coords=[l for l in lanes if l['role']=='coordinator'];demand(len(coords)==1 and coords[0]['tasks']==[root],'one local coordinator root required')
    demand(m['run']['root_task_id']==root and not {'status','active','activate'}&m['run'].keys(),'premature run-state command')
    for r in rows:
        demand(r['status']=='planned','premature task state')
        demand(r['kind']=='epic' or r.get('parent_id')==root,'wrong parent')
        safe_rel(r['notes'])
        for x in r.get('scope',{}).get('exclusive_paths',[]):safe_rel(x)
        for d in r.get('depends_on',[]):
            demand(d['type']=='task' and d['task_id'] in by and d['task_id']!=r['id'],'invalid native dependency')
            demand(d['task_id'] not in parents[r['id']],'duplicate prerequisite');parents[r['id']].add(d['task_id']);explicit.append((d['task_id'],r['id']))
    for l in lanes:
        demand(l['role'] in {'coordinator','implementer','validator','integrator','specialist'},'invalid lane role')
        demand(l['workspace']['mode'] in {'read_shared','isolated_merge','contract_split','exclusive'},'invalid workspace mode')
        demand(not l.get('parent_lane_id') or l['parent_lane_id'] in laneids,'cross-project/unknown native lane parent')
        for t in l['tasks']:
            demand(t in by and t not in owner,'unassigned/duplicate lane task');owner[t]=l['id']
        for a,b in zip(l['tasks'],l['tasks'][1:]):parents[b].add(a)
    demand(set(owner)==set(ids),'missing lane ownership')
    anc=ancestors(parents)
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            if a['id'] in anc[b['id']] or b['id'] in anc[a['id']]:continue
            for x in a.get('scope',{}).get('exclusive_paths',[]):
                for y in b.get('scope',{}).get('exclusive_paths',[]):
                    demand(not overlap(x,y),f'Unordered write conflict: {a["id"]}/{b["id"]}: {x}/{y}')
    cps={}
    for r in rows:
        for c in r.get('checkpoints',[]):
            demand(c['id'] not in cps and 'state' not in c,'duplicate/pre-reached checkpoint');cps[c['id']]=r['id']
    for b in m['barriers']:
        demand(b['mode']=='all','barrier mode')
        for q in b['requirements']:demand(q['type']=='checkpoint' and q['id'] in cps and q['state']=='reached','unknown barrier checkpoint')
    for i in m['interfaces']:
        demand(i['owner_task_id'] in by and i['state']=='draft' and i['version']=='1','invalid interface owner/state')
        for path in i['contract_paths']:safe_rel(path)
    for t in m['tasks']:
        demand(t['native_record']==by[t['id']],'native record mismatch')
        demand(t['title']==by[t['id']]['title'],'rich/native title drift')
        demand(t['depends_on']==[d['task_id'] for d in by[t['id']].get('depends_on',[])],'rich/native prerequisite drift')
        demand(t['write_scope']==by[t['id']]['scope']['exclusive_paths'],'scope drift')
        demand(len(t['mechanism'])>=70 and len(t['acceptance'])>=2,'task lacks directed substance')
    demand(len(json.dumps(p,indent=2).encode())<256*1024,'native plan exceeds observer budget')
    return p,parents,owner,cps,explicit

def validate(package=PACKAGE,peer_package=None,integrity=True):
    package=Path(package).resolve();m=load(package/'machine/proposed_todos.json');p,parents,owner,cps,edges=graph(m)
    demand(p==load(package/'machine'/m['native_plan_file']),'native projection drift')
    for f,key in [('barriers.json','barriers'),('interface_catalog.json','interfaces')]:demand(load(package/'machine'/f)==m[key],f+' drift')
    demand(load(package/'machine/lanes.json')['lanes']==m['run']['lanes'],'lane projection drift')
    def csvrows(f):
        with (package/'machine'/f).open(newline='',encoding='utf-8') as h:return list(csv.DictReader(h))
    demand(csvrows('dependency_edges.csv')==[dict(depends_on_task_id=a,task_id=b) for a,b in edges],'dependency CSV drift')
    demand(csvrows('checkpoints.csv')==[dict(id=k,owner_task_id=v,title=next(c['title'] for c in next(r for r in p['tasks'] if r['id']==v)['checkpoints'] if c['id']==k)) for k,v in cps.items()],'checkpoint CSV drift')
    flat=[dict(id=r['id'],kind=r['kind'],title=r['title'],lane=owner[r['id']],depends_on=';'.join(d['task_id'] for d in r.get('depends_on',[])),specification=r['notes']) for r in p['tasks']]
    demand(csvrows('proposed_todos.csv')==flat,'task CSV drift')
    summaries=load(package/'machine/workstreams.json');demand({s['lane_id'] for s in summaries}=={l['id'] for l in m['run']['lanes'] if l['role']!='coordinator'},'workstream coverage drift')
    summary=load(package/'machine/plan_summary.json')
    demand(summary['native_plan_canonical_sha256']==hashlib.sha256(canonical(p)).hexdigest(),'summary plan digest drift')
    demand(summary['todo_record_count']==len(p['tasks']) and summary['lane_count']==len(m['run']['lanes']),'summary drift')
    accept=load(package/'machine/acceptance_matrix.json')
    demand({t['task_id'] for t in accept['task_acceptance']}=={t['id'] for t in m['tasks']},'acceptance coverage mismatch')
    for t in m['tasks']:
        row=next(a for a in accept['task_acceptance'] if a['task_id']==t['id'])
        demand(row['conditions']==t['acceptance'],'acceptance projection drift')
        text=(package/Path(t['native_record']['notes']).relative_to(m['package_path'])).read_text()
        demand(t['id'] in text and t['mechanism'] in text,'missing/stale task sheet')
    for l in m['run']['lanes']:demand((package/'handoff'/f'{l["id"].lower()}.md').is_file(),'missing lane handoff')
    for name,g in accept['gate_groups'].items():
        demand(g['ctest_names'] and len(g['ctest_names'])==len(set(g['ctest_names'])),'empty/duplicate gate inventory')
        demand(g['owner_task_id'] in parents,'gate group foreign owner')
        if g.get('scope')=='integrated-current-source':
            local_anc=ancestors(parents)[g['owner_task_id']]
            needed={name for t in m['tasks'] if t['id'] in local_anc for name in t['required_ctest_names']}
            demand(needed<=set(g['ctest_names']),'milestone dropped prerequisite test coverage')
    cross=load(package/'machine/cross_repository_contracts.json')
    demand(len({e['id'] for e in cross['edges']})==len(cross['edges']),'duplicate cross edge')
    demand(all(e.get('status')=='planned' and not e.get('receipt') for e in cross['edges']),'fabricated producer receipt')
    extern=csvrows('external_dependency_receipts.csv')
    demand({x['id'] for x in extern}=={e['id'] for e in cross['edges'] if e['consumer_project']==m['project']['workspace']},'external CSV drift')
    if peer_package:
        peer=Path(peer_package).resolve();pm=load(peer/'machine/proposed_todos.json');_,pp,_,_,_=graph(pm)
        demand(pm['project']['workspace']!=m['project']['workspace'],'peer is same project')
        demand(load(peer/'machine/cross_repository_contracts.json')==cross,'cross-authority contract drift')
        joint={**parents,**pp};demand(len(joint)==len(parents)+len(pp),'cross-project ID collision')
        for e in cross['edges']:
            demand(e['consumer_task'] in joint and e['producer_task'] in joint,'missing cross task')
            joint[e['consumer_task']].add(e['producer_task'])
        ancestors(joint)
        trace=load(package/'machine/requirements_traceability.json')['requirements']
        demand(len(trace)==35 and all(t in joint for r in trace for t in r['tasks']),'user requirement coverage/reference mismatch')
        demand(load(peer/'machine/joint_program.json')==load(package/'machine/joint_program.json'),'joint controller manifest drift')
    if integrity:
        entries={}
        demand((package/'MANIFEST.sha256').is_file() and not (package/'MANIFEST.sha256').is_symlink(),'manifest missing or symlinked')
        for line in (package/'MANIFEST.sha256').read_text().splitlines():
            sha,name=line.split('  ',1);safe_rel(name);demand(name not in entries,'duplicate manifest entry');entries[name]=sha
        files={f.relative_to(package).as_posix() for f in package.rglob('*') if f.is_file() and f!=package/'MANIFEST.sha256'}
        demand(set(entries)==files,'manifest membership mismatch')
        for name,sha in entries.items():
            f=package/name;demand(not f.is_symlink() and digest(f)==sha,'integrity failure: '+name)
        repo=package.parents[1]
        delivery=load(package/'machine/delivery_files.json')
        expected_delivery={e['path'] for e in delivery['files']};actual_delivery=set()
        for relative_root in delivery['roots']:
            safe_rel(relative_root);root=repo/relative_root;demand(root.is_dir() and not root.is_symlink(),'invalid delivery root')
            actual_delivery|={f.relative_to(repo).as_posix() for f in root.rglob('*') if f.is_file()}
        demand(expected_delivery==actual_delivery,'delivery membership mismatch')
        for e in delivery['files']:
            safe_rel(e['path']);f=repo/e['path'];demand(f.is_file() and not f.is_symlink() and digest(f)==e['sha256'],'delivery mismatch: '+e['path'])
    return dict(status='offline_validated',workspace=m['project']['workspace'],todo_records=len(p['tasks']),leaf_tasks=len(m['tasks']),lanes=len(m['run']['lanes']),explicit_edges=len(edges),native_schema=3,queue_and_local_scope_checked=True,cross_repository_graph_checked=bool(peer_package),live_native_plan_validation=False,applied=False)

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--package',type=Path,default=PACKAGE);a.add_argument('--peer-package',type=Path);o=a.parse_args();print(json.dumps(validate(o.package,o.peer_package),indent=2))
if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,OSError) as e:raise SystemExit('validate_package: '+str(e))
