#!/usr/bin/env python3
"""Offline package/graph/integrity validation. Never reads or mutates Todo."""
from pathlib import Path,PurePosixPath
import argparse,csv,hashlib,json,sys
sys.dont_write_bytecode=True
from compile_plan import assemble,PACKAGE,PLAN

def demand(test,message):
    if not test:raise ValueError(message)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def safe_rel(s):
    demand(isinstance(s,str) and bool(s),'empty/non-string path')
    p=PurePosixPath(s)
    demand(not p.is_absolute() and '..' not in p.parts and '.' not in s.split('/') and '\\' not in s and str(p)==s,'unsafe relative path: '+s)
    return p
def load(p):return json.loads(p.read_text())
def overlap(a,b):return a==b or a.startswith(b+'/') or b.startswith(a+'/')
def validate_graph(m):
    p=assemble(m);rows=p['tasks'];ids=[r['id'] for r in rows]
    demand(len(ids)==len(set(ids)),'duplicate task ID')
    demand(all(x.startswith('CE-RU1-') for x in ids),'foreign task ID')
    by={r['id']:r for r in rows};roots=[r for r in rows if r['kind']=='epic']
    demand(len(roots)==1 and roots[0]['id']=='CE-RU1-0000','wrong root')
    root=roots[0]['id'];specs=m['tasks']
    demand(len(specs)==len(rows)-1,'wrong leaf count')
    for r in rows:
        demand(r['status']=='planned','premature task state')
        if r['id']!=root:demand(r['parent_id']==root and r['kind']=='task','wrong parent/kind')
        safe_rel(r['notes'])
    run=m['run'];demand(run['id']=='CE-RU1-RUN-V1' and run['root_task_id']==root,'wrong run')
    demand('status' not in run and 'active' not in run,'run activation field forbidden')
    parents={i:set() for i in ids};edges=[]
    for r in rows:
        ds=r.get('depends_on',[])
        demand(len(ds)==len({d['task_id'] for d in ds}),'duplicate dependency')
        for d in ds:
            demand(d.get('type')=='task' and d['task_id'] in by and d['task_id']!=r['id'],'unknown/self dependency')
            parents[r['id']].add(d['task_id']);edges.append((d['task_id'],r['id'],'explicit'))
    assigned=[];coordinators=[];laneids=[l['id'] for l in run['lanes']]
    demand(len(laneids)==len(set(laneids)),'duplicate lane')
    for l in run['lanes']:
        demand(l['role'] in {'coordinator','implementer','integrator','validator','specialist'},'bad lane role')
        demand(l['workspace']['mode'] in {'read_shared','contract_split','isolated_merge','exclusive'},'bad workspace mode')
        if l['role']=='coordinator':coordinators.append(l);demand(l['tasks']==[root],'coordinator must own root only')
        else:demand(l.get('parent_lane_id')=='CE-RU1-L-COORD','wrong parent lane')
        demand(all(t in by for t in l['tasks']),'unknown queued task');assigned+=l['tasks']
        for a,b in zip(l['tasks'],l['tasks'][1:]):parents[b].add(a);edges.append((a,b,'queue'))
    demand(len(coordinators)==1 and sorted(assigned)==sorted(ids),'lane assignment missing or duplicated')
    visiting=set();anc={}
    def walk(i):
        if i in anc:return anc[i]
        demand(i not in visiting,'dependency/queue cycle at '+i);visiting.add(i);a=set()
        for d in parents[i]:a|={d}|walk(d)
        visiting.remove(i);anc[i]=a;return a
    for i in ids:walk(i)
    for a in ids:
        for b in ids:
            if a>=b or a in anc[b] or b in anc[a]:continue
            for x in by[a].get('scope',{}).get('exclusive_paths',[]):
                for y in by[b].get('scope',{}).get('exclusive_paths',[]):
                    demand(not overlap(x,y),f'unordered write conflict {a}/{b}: {x} / {y}')
    checkpoints={}
    for r in rows:
        for c in r.get('checkpoints',[]):
            demand(c['id'] not in checkpoints,'duplicate checkpoint');checkpoints[c['id']]=r['id']
            demand('state' not in c,'pre-reached checkpoint')
    for b in m['barriers']:
        demand(b['mode']=='all','unexpected barrier mode')
        for q in b['requirements']:demand(q['type']=='checkpoint' and q['id'] in checkpoints and q['state']=='reached','unknown barrier checkpoint')
    for i in m['interfaces']:
        demand(i['owner_task_id'] in by and i['state']=='draft' and i['version']=='1','invalid interface')
        for path in i['contract_paths']:safe_rel(path)
    for s in specs:
        n=by[s['id']]
        demand(s['native_record']==n,'native task mismatch')
        demand(n['title']==s['title'] and n['scope']['exclusive_paths']==s['write_scope'],'scope/title projection drift')
        demand([x['task_id'] for x in n['depends_on']]==s['depends_on'],'dependency projection drift')
        demand(len(s['acceptance'])>=2 and len(s['mechanism'])>150,'task lacks substantive acceptance/mechanism')
        for path in s['write_scope']:safe_rel(path)
    demand(len(json.dumps(p,sort_keys=True,indent=2).encode())<256*1024,'native proposal exceeds bounded observer input')
    return {'leaf_tasks':len(specs),'todo_records':len(rows),'lanes':len(run['lanes']),
            'explicit_edges':sum(x[2]=='explicit' for x in edges),'queue_edges':sum(x[2]=='queue' for x in edges),
            'topological_order':sorted(ids,key=lambda x:(len(anc[x]),x))}

def validate(package=PACKAGE,source_root=None,integrity=True):
    package=package.resolve();source_root=(source_root or package.parents[1]).resolve()
    m=load(package/'machine/proposed_todos.json');g=validate_graph(m);p=assemble(m)
    demand(load(package/'machine'/PLAN)==p,'native projection drift')
    native_receipt=load(package/'evidence/native_plan_validation.json')
    observer_digest=hashlib.sha256(json.dumps(p,sort_keys=True,indent=2).encode()).hexdigest()
    demand(native_receipt.get('valid') is True and native_receipt.get('plan_digest')==observer_digest,'archived final native receipt is not bound to this plan')
    demand(sorted(native_receipt.get('would_add',[]))==sorted(t['id'] for t in p['tasks']) and not native_receipt.get('would_modify'),'archived additive diff differs')

    summary=load(package/'machine/plan_summary.json')
    demand((summary['leaf_task_count'],summary['todo_record_count'],summary['lane_count'])==(g['leaf_tasks'],g['todo_records'],g['lanes']),'summary count drift')
    for file,key in [('barriers.json','barriers'),('interface_catalog.json','interfaces')]:demand(load(package/'machine'/file)==m[key],file+' differs')
    demand(load(package/'machine/lanes.json')['lanes']==m['run']['lanes'],'lane view drift')
    want_ws=[{'id':'CE-RU1-WS-'+l['id'].split('-L-')[-1],'lane_id':l['id'],'task_ids':l['tasks']} for l in m['run']['lanes']]
    demand(load(package/'machine/workstreams.json')==want_ws,'workstream drift')
    def csvrows(name):
        with (package/'machine'/name).open(newline='') as f:return list(csv.DictReader(f))
    edges=[dict(depends_on_task_id=d['task_id'],task_id=r['id']) for r in p['tasks'] for d in r.get('depends_on',[])]
    demand(csvrows('dependency_edges.csv')==edges,'dependency CSV drift')
    cps=[dict(id=c['id'],owner_task_id=r['id'],title=c['title']) for r in p['tasks'] for c in r.get('checkpoints',[])]
    demand(csvrows('checkpoints.csv')==cps,'checkpoint CSV drift')
    owners={t:l['id'] for l in m['run']['lanes'] for t in l['tasks']}
    taskrows=[dict(id=n['id'],kind=n['kind'],title=n['title'],lane=owners[n['id']],depends_on=';'.join(d['task_id'] for d in n.get('depends_on',[])),specification=n['notes']) for n in p['tasks']]
    demand(csvrows('proposed_todos.csv')==taskrows,'task CSV drift')
    demand(csvrows('external_dependency_receipts.csv')==[],'unexpected cross-authority dependency')
    matrix=load(package/'machine/acceptance_matrix.json')
    demand({t['task_id'] for t in matrix['task_acceptance']}=={t['id'] for t in m['tasks']},'acceptance tasks differ')
    for row in matrix['task_acceptance']:
        spec=next(t for t in m['tasks'] if t['id']==row['task_id'])
        demand(row['conditions']==spec['acceptance'] and row['cases']==spec['tests'],'acceptance view drift')
    for row in p['tasks']:
        rel=Path(row['notes']).relative_to('planning/relation-update-spine-v1');text=(package/rel).read_text()
        demand(row['id'] in text,'missing task sheet')
    if integrity:
        listed={}
        for line in (package/'MANIFEST.sha256').read_text().splitlines():
            sha,path=line.split('  ',1);safe_rel(path);demand(path not in listed,'duplicate manifest entry');listed[path]=sha
        actual={str(f.relative_to(package)) for f in package.rglob('*') if f.is_file() and f.name!='MANIFEST.sha256'}
        demand(set(listed)==actual,'manifest is not complete')
        for f in package.rglob('*'):demand(not f.is_symlink(),'symlink forbidden in package')
        for rel,sha in listed.items():demand(digest(package/rel)==sha,'package content changed: '+rel)
        delivery=load(package/'machine/delivery_files.json')['files']
        demand(len(delivery)==len({d['path'] for d in delivery}),'duplicate delivery')
        for d in delivery:
            safe_rel(d['path']);f=source_root/d['path']
            demand(f.is_file() and not f.is_symlink() and f.resolve().is_relative_to(source_root),'unsafe/missing delivery '+d['path'])
            demand(digest(f)==d['sha256'],'delivery content changed: '+d['path'])
    return {'status':'offline_validated',**g,'integrity_checked':integrity,'native_authority_validation':False,'gpu_executed':False}
def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--package-root',type=Path,default=PACKAGE);a.add_argument('--source-root',type=Path)
    o=a.parse_args();print(json.dumps(validate(o.package_root,o.source_root),indent=2))
if __name__=='__main__':
    try:main()
    except (ValueError,OSError,KeyError,TypeError) as e:raise SystemExit('validate_package: '+str(e))
