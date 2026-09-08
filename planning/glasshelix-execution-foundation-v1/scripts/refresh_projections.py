#!/usr/bin/env python3
"""Regenerate local planning projections after an explicitly reviewed catalog edit. No Todo access."""
from pathlib import Path
import argparse,csv,hashlib,json,sys
sys.dont_write_bytecode=True
from compile_plan import PACKAGE,load,assemble

def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def csvout(p,fields,rows):
    with p.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--confirm',required=True);o=a.parse_args()
    if o.confirm!='REFRESH-NF1-PROJECTIONS':raise ValueError('Wrong confirmation')
    m=load(PACKAGE/'machine/proposed_todos.json')
    for t in m['tasks']:
        t['native_record']['title']=t['title'];t['native_record']['scope']['exclusive_paths']=t['write_scope']
        t['native_record']['depends_on']=[{'type':'task','task_id':d} for d in t['depends_on']]
    write(PACKAGE/'machine/proposed_todos.json',m);p=assemble(m)
    write(PACKAGE/'machine'/m['native_plan_file'],p)
    for f,key in [('interface_catalog.json','interfaces'),('barriers.json','barriers')]:write(PACKAGE/'machine'/f,m[key])
    write(PACKAGE/'machine/lanes.json',{'lanes':m['run']['lanes']})
    owners={t:l['id'] for l in m['run']['lanes'] for t in l['tasks']}
    csvout(PACKAGE/'machine/proposed_todos.csv',['id','kind','title','lane','depends_on','specification'],[dict(id=t['id'],kind=t['kind'],title=t['title'],lane=owners[t['id']],depends_on=';'.join(x['task_id'] for x in t.get('depends_on',[])),specification=t['notes']) for t in p['tasks']])
    edges=[dict(depends_on_task_id=d['task_id'],task_id=t['id']) for t in p['tasks'] for d in t.get('depends_on',[])]
    csvout(PACKAGE/'machine/dependency_edges.csv',['depends_on_task_id','task_id'],edges)
    cps=[dict(id=c['id'],owner_task_id=t['id'],title=c['title']) for t in p['tasks'] for c in t.get('checkpoints',[])]
    csvout(PACKAGE/'machine/checkpoints.csv',['id','owner_task_id','title'],cps)
    csvout(PACKAGE/'machine/scope_ownership.csv',['task_id','lane_id','mode','path'],[dict(task_id=t['id'],lane_id=owners[t['id']],mode='exclusive',path=x) for t in m['tasks'] for x in t['write_scope']])
    summary=load(PACKAGE/'machine/plan_summary.json')
    summary.update(todo_record_count=len(p['tasks']),leaf_task_count=len(m['tasks']),lane_count=len(m['run']['lanes']),explicit_edges=len(edges),checkpoint_count=len(cps),barrier_count=len(m['barriers']),interface_count=len(m['interfaces']),native_plan_canonical_sha256=hashlib.sha256((json.dumps(p,sort_keys=True,separators=(',',':'),ensure_ascii=False)+'\n').encode()).hexdigest())
    write(PACKAGE/'machine/plan_summary.json',summary)
    nv=load(PACKAGE/'evidence/native_plan_validation.json');nv.update(status='requires_fresh_native_preview',exact_plan_validated_live=False,plan_canonical_sha256=summary['native_plan_canonical_sha256']);write(PACKAGE/'evidence/native_plan_validation.json',nv)
    print(json.dumps({'projections_refreshed':True,'native_authority_mutated':False,'still_required':'Update authored acceptance/workstream/cross/requirement views, task sheets and handoffs consistently; validate both packages; reseal; fresh native preview.'}))
if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,OSError) as e:raise SystemExit(str(e))
