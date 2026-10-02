#!/usr/bin/env python3
"""Read-only activation/provider checks; never apply plans or run workloads.

Known task completion is re-read through the inspected SemanticReader API.
Inventory completeness/claim reconciliation remain explicit controller-reviewed
assertions backed by current hashed authority evidence, not facts inferred from
Git or this package's historical watchlist. A changed provider API fails closed.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib,json,pathlib,re,subprocess,sys
from typing import Callable

ROOT=pathlib.Path(__file__).resolve().parents[1]
PROJECTS=('cellerator','baseplane','glasshelix')

def require(ok:bool,message:str) -> None:
    if not ok: raise ValueError(message)

def load(path:pathlib.Path):
    return json.loads(path.read_text(encoding='utf-8'))

def safe_file(root:pathlib.Path,relative:str) -> pathlib.Path:
    require(isinstance(relative,str) and bool(relative),'missing evidence path')
    p=pathlib.PurePosixPath(relative)
    require(not p.is_absolute() and '..' not in p.parts and '\\' not in relative and str(p)!='.', 'unsafe evidence path')
    root=root.resolve(strict=True);path=root
    for part in p.parts:
        path=path/part
        require(not path.is_symlink(),'symlink evidence is not accepted')
    path=path.resolve(strict=True)
    require(path.is_relative_to(root) and path.is_file(),'evidence is not a regular contained file')
    return path

def check_blob(entry:dict,roots:dict[str,pathlib.Path]) -> pathlib.Path:
    root_key=entry.get('project',entry.get('root'))
    require(root_key in roots,'unknown artifact/evidence root')
    p=safe_file(roots[root_key],entry.get('path',entry.get('evidence_path','')))
    require(hashlib.sha256(p.read_bytes()).hexdigest()==entry.get('sha256'),'evidence hash mismatch: '+str(p))
    return p

def git(root:pathlib.Path,*args:str) -> str:
    return subprocess.run(['git','-C',str(root),*args],check=True,text=True,capture_output=True).stdout.strip()

def get_heads(roots:dict[str,pathlib.Path], projects=PROJECTS) -> dict[str,str]:
    heads={}
    for pr in projects:
        require(pr in PROJECTS,'unknown source project')
        root=roots[pr]
        require(pathlib.Path(git(root,'rev-parse','--show-toplevel')).resolve()==root.resolve(),'config must name repository root: '+pr)
        require(not git(root,'status','--porcelain','--untracked-files=no'),'tracked source is dirty: '+pr)
        head=git(root,'rev-parse','HEAD');require(bool(re.fullmatch('[0-9a-f]{40}',head)),'unexpected source identity')
        heads[pr]=head
    return heads

def validate_activation(review:dict,config:dict,heads:dict[str,str],
                        state:Callable[[str,str],str],roots:dict[str,pathlib.Path],
                        now:datetime|None=None) -> None:
    require(review.get('schema_version')==1 and review.get('record_kind')=='is1_activation_review','wrong activation record')
    require(review.get('approved') is True and bool(review.get('reviewed_by')),'activation review is not approved')
    for key in ('inventory_complete','claims_and_pending_patches_reconciled','predecessor_scope_reviewed'):
        require(review.get(key) is True,'unresolved activation requirement: '+key)
    captured=datetime.fromisoformat(review.get('captured_at','').replace('Z','+00:00'))
    require(captured.tzinfo is not None,'activation time must have timezone')
    now=now or datetime.now(timezone.utc)
    age=(now-captured).total_seconds();limit=config.get('max_review_age_seconds',600)
    require(isinstance(limit,(int,float)) and 0<limit<=3600,'invalid freshness limit')
    require(-60<=age<=limit,'activation observation is stale or in the future')
    require(review.get('heads')==heads,'source changed since activation review')
    evidence=review.get('authority_evidence',[])
    require({e.get('project') for e in evidence}>=set(PROJECTS),'current authority evidence required for every project')
    for e in evidence:check_blob(e,roots)
    rows=review.get('run_inventory',[])
    require(isinstance(rows,list),'run inventory must be a list')
    keys=[(r.get('project'),r.get('run_id')) for r in rows]
    require(len(set(keys))==len(keys),'duplicate run-inventory row')
    watch=load(ROOT/'machine/run-watchlist.json')
    require({(r['project'],r['run_id']) for r in watch}<=set(keys),'historically active runs still unclassified')
    required={(pr,t) for pr,ids in load(ROOT/'machine/predecessors.json').items() for t in ids}
    for row in rows:
        pr=row.get('project');task=row.get('root_task_id');disp=row.get('disposition')
        require(pr in PROJECTS and bool(task) and bool(row.get('reason')),'incomplete run classification')
        require(disp in ('predecessor','completed','historical_closed','preserved_deferred','permission_locked'),'unclassified run')
        if disp in ('predecessor','completed'):required.add((pr,task))
        if disp=='historical_closed':
            require(state(pr,task) in ('done','superseded','cancelled'),'historical root is not terminal: '+task)
        if disp in ('preserved_deferred','permission_locked'):
            require(bool(row.get('preservation_owner')) and bool(row.get('trigger')),'deferred work has no owner/trigger')
    for pr,task in sorted(required):
        require(state(pr,task)=='done',f'predecessor not effectively done: {pr}:{task}')

def validate_provider(receipt:dict,link:dict,heads:dict[str,str],
                      state:Callable[[str,str],str],roots:dict[str,pathlib.Path]) -> None:
    pr=link['provider_project'];task=link['provider_task']
    require(receipt.get('schema_version')==1 and receipt.get('record_kind')=='is1_provider_receipt','wrong provider record')
    require(receipt.get('project')==pr and receipt.get('task_id')==task,'provider/task mismatch')
    require(state(pr,task)=='done','provider task is not effectively done')
    require(receipt.get('source_commit')==heads[pr],'provider source changed; refresh qualified receipt')
    require(bool(receipt.get('reviewed_by')) and bool(receipt.get('reviewed_at')),'provider review absent')
    caps=receipt.get('capabilities',[]);names=[c.get('name') for c in caps]
    require(len(set(names))==len(names),'duplicate capability record')
    by_name={c['name']:c for c in caps}
    for name in link['required']:
        c=by_name.get(name,{})
        require(c.get('status')=='implemented' and bool(c.get('call')) and bool(c.get('shapes')) and bool(c.get('numerical_policy')),'unsupported/unqualified provider capability: '+name)
    artifacts=receipt.get('build_artifacts',[])
    require(bool(artifacts),'installed build artifact evidence absent')
    for artifact in artifacts:check_blob(artifact,roots)
    checks=receipt.get('checks',[])
    expected={'native_correctness','installed_consumer','derivatives'} if pr=='cellerator' else {'sequence_correctness','installed_consumer'}
    if 'framework' in link['required']:expected.add('framework_consumer')
    require({c.get('kind') for c in checks}>=expected,'required actual check kinds missing')
    for c in checks:
        require(c.get('execution')=='real' and c.get('status')=='passed' and type(c.get('exit_code')) is int and c['exit_code']==0,'mock/unperformed/failing check cannot qualify provider')
        require(isinstance(c.get('argv'),list) and bool(c['argv']) and all(isinstance(x,str) for x in c['argv']),'actual check command absent')
        source=c.get('source_commits',{})
        require(source.get(pr)==heads[pr],'check is not bound to provider source')
        require(all(p in heads and source[p]==heads[p] for p in source),'check bound to stale sibling source')
        check_blob(c,roots)
    if pr=='cellerator':
        require(any(c.get('kind')=='native_correctness' and c.get('backend')=='cuda_sm70' for c in checks),'SM70 native evidence absent')

def configuration(path:pathlib.Path):
    config=load(path)
    require(config.get('schema_version')==1,'wrong config schema')
    repos=config.get('repos',{})
    require(set(repos)==set(PROJECTS),'explicit paths for all three repositories required')
    require(not (set(config.get('artifact_roots',{})) & set(PROJECTS)), 'artifact roots cannot override repository roots')
    roots={}
    for name,value in {**repos,**config.get('artifact_roots',{})}.items():
        require(isinstance(value,str) and pathlib.Path(value).expanduser().is_absolute(),'absolute explicit path required: '+name)
        roots[name]=pathlib.Path(value).expanduser().resolve(strict=True)
    provider=config.get('todo_provider')
    require(isinstance(provider,str) and pathlib.Path(provider).expanduser().is_absolute(),'explicit installed todo_provider path required')
    sys.path.insert(0,str(pathlib.Path(provider).expanduser().resolve(strict=True)))
    from todo_orchestrator.semantic import SemanticReader
    readers={pr:SemanticReader(roots[pr]) for pr in PROJECTS}
    def state(pr:str,task:str) -> str:
        result=readers[pr].state(task_id=task)
        rows=[r for r in result.get('tasks',[]) if r.get('id')==task]
        require(len(rows)==1,'missing/ambiguous live task state: '+pr+':'+task)
        return rows[0].get('effective_state','unknown')
    return config,roots,state

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('activation','provider'))
    p.add_argument('--config',required=True,type=pathlib.Path)
    p.add_argument('--consumer');p.add_argument('--provider',choices=PROJECTS)
    args=p.parse_args();config,roots,state=configuration(args.config)
    if args.mode=='activation':
        heads=get_heads(roots)
        path=config.get('activation_review')
        require(isinstance(path,str) and pathlib.Path(path).is_absolute(),'explicit activation review path required')
        validate_activation(load(pathlib.Path(path)),config,heads,state,roots)
    else:
        links=[x for x in load(ROOT/'machine/cross-authority.json') if x['consumer_task']==args.consumer and x['provider_project']==args.provider]
        require(len(links)==1,'unknown cross-authority link')
        link=links[0];receipt=load(safe_file(roots[link['provider_project']],link['receipt']))
        relevant={link['provider_project']}
        for check in receipt.get('checks',[]):relevant.update(check.get('source_commits',{}))
        heads=get_heads(roots,sorted(relevant))
        validate_provider(receipt,link,heads,state,roots)
    require(get_heads(roots,sorted(heads))==heads,'source changed during gate; refresh before dispatch')
    print(json.dumps({'status':'passed','mode':args.mode,'heads':heads,'writes':False,'workloads_run':False}))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('IS1 gate blocked: '+str(exc),file=sys.stderr);sys.exit(1)
