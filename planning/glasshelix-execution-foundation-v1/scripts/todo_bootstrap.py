#!/usr/bin/env python3
"""Validate, review, explicitly apply and verify one NF1 native Todo plan.
Run import may create an 'active' run record; this script never claims or dispatches.
"""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time
sys.dont_write_bytecode=True
from validate_package import PACKAGE,validate,demand,digest,load,canonical

def call(argv,cwd=None,timeout=180):
    r=subprocess.run([str(x) for x in argv],cwd=cwd,text=True,capture_output=True,timeout=timeout)
    if r.returncode:raise RuntimeError(f'Command failed ({r.returncode}): {argv[0]}\n{r.stderr}\n{r.stdout}')
    return r.stdout

def git(root,*args):return call(['git','-C',root,*args]).strip()
def snapshot(root,review_head,ack_dirty):
    root=Path(root).resolve();demand(root==Path(git(root,'rev-parse','--show-toplevel')).resolve(),'not repository root')
    head=git(root,'rev-parse','HEAD');demand(review_head==head and len(head)==40,'reviewed HEAD must equal full current HEAD')
    status=git(root,'status','--porcelain=v1','--untracked-files=all');demand(not status or ack_dirty,'dirty workspace: inspect/preserve it and pass --acknowledge-dirty; never auto-clean')
    raw=subprocess.run(['git','-C',str(root),'diff','--binary','HEAD'],capture_output=True,check=True).stdout
    untracked=subprocess.run(['git','-C',str(root),'ls-files','--others','--exclude-standard','-z'],capture_output=True,check=True).stdout
    hashes={}
    for name in untracked.decode().split('\0'):
        if name:
            p=root/name
            if p.is_symlink():hashes[name]='symlink:'+os.readlink(p)
            elif p.is_file():hashes[name]=digest(p)
    return {'root':str(root),'head':head,'status':status,'tracked_diff_sha256':hashlib.sha256(raw).hexdigest(),'untracked_files':hashes}

def check_native(v,plan,uuid):
    expected=sorted(t['id'] for t in plan['tasks'])
    demand(v.get('valid') is True and v.get('status')=='validated','native validator did not accept plan')
    demand(v.get('project_uuid')==uuid and isinstance(v.get('revision'),int),'wrong or missing authority identity')
    demand(sorted(v.get('would_add',[]))==expected and not v.get('would_modify'),'collision, partial import or existing tasks would change')
    demand(not v.get('warnings'),'native warnings require review')
    demand(v.get('plan_digest')==hashlib.sha256(canonical(plan)).hexdigest(),'native canonical digest mismatch')
    c=v.get('current_observation_preconditions',{})
    demand(c.get('project_uuid')==uuid and c.get('todo_revision')==v['revision'],'native authority cursor mismatch')
    demand(c.get('workflow_revision')==v['revision'] and c.get('workflow_authority_fingerprint') and c.get('workflow_authority_fingerprint')==c.get('todo_semantic_authority_fingerprint'),'incoherent native workflow cursor')
    return c

def check_preview(old,source,peer,fresh,plan,uuid,now=None):
    demand(old.get('kind')=='nf1-reviewed-preview-v1','wrong preview kind')
    age=(time.time() if now is None else now)-old.get('created_unix',0)
    demand(0<=age<=3600,'preview expired or future-dated')
    demand(old.get('source')==source and old.get('peer_source')==peer,'source or peer changed after review')
    demand(old.get('manifest_sha256')==digest(PACKAGE/'MANIFEST.sha256'),'package changed after review')
    prev=old['native_response'];check_native(prev,plan,uuid);check_native(fresh,plan,uuid)
    for k in ['project_uuid','revision','plan_digest','would_add','would_modify','runtime_identity']:demand(prev.get(k)==fresh.get(k),'native preview changed: '+k)
    a=prev['current_observation_preconditions'];b=fresh['current_observation_preconditions']
    for k in set(a)|set(b):
        if k not in {'observed_at','provider_skew'}:demand(a.get(k)==b.get(k),'native precondition changed: '+k)

def outside(path,roots):
    p=Path(path).absolute();demand(p.parent.is_dir(),'receipt parent must already exist')
    demand(not p.exists() and not p.is_symlink(),'receipt exists; never overwrite evidence')
    for r in roots:demand(not p.resolve().is_relative_to(Path(r).resolve()),'receipt must be outside both repositories')
    return p

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('command',choices=['validate','preview','apply','verify','inspect-runtime'])
    a.add_argument('--repo',type=Path,default=PACKAGE.parents[1]);a.add_argument('--peer-repo',type=Path);a.add_argument('--peer-package',type=Path)
    a.add_argument('--review-head');a.add_argument('--peer-review-head');a.add_argument('--acknowledge-dirty',action='store_true');a.add_argument('--runtime-python',type=Path)
    a.add_argument('--receipt',type=Path);a.add_argument('--preview',type=Path);a.add_argument('--applied-receipt',type=Path);a.add_argument('--confirm');o=a.parse_args()
    meta=load(PACKAGE/'machine/proposed_todos.json');repo=o.repo.resolve();project=meta['project']['workspace'];plan=load(PACKAGE/'machine'/meta['native_plan_file']);uuid=meta['project_uuid']
    valid=validate(PACKAGE,o.peer_package)
    if o.command=='validate':print(json.dumps(valid,indent=2));return
    demand(PACKAGE==(repo/meta['package_path']).resolve(),'install at documented repository-relative planning path')
    demand(o.runtime_python and o.runtime_python.is_file(),'explicit configured Project Control Python required')
    bridge=[o.runtime_python,PACKAGE/'scripts/native_bridge.py']
    if o.command=='inspect-runtime':print(call(bridge+['inspect-runtime']));return
    demand(o.receipt and o.peer_repo,'new external receipt and explicit peer repository required')
    output=outside(o.receipt,[repo,o.peer_repo])
    if o.command=='verify':
        demand(o.applied_receipt and o.applied_receipt.is_file(),'--applied-receipt required')
        applied=load(o.applied_receipt);demand(applied.get('status')=='applied','not a successful apply receipt')
        demand(applied.get('workspace')==project and applied.get('native_response',{}).get('plan_digest')==hashlib.sha256(canonical(plan)).hexdigest(),'applied receipt belongs to another plan')
        result=json.loads(call(bridge+['observe']));demand(result['project_uuid']==uuid,'wrong project')
        ids={r.get('id') for r in result['tables']['tasks']};want={r['id'] for r in plan['tasks']};demand(want<=ids,'imported tasks absent')
        expected=set(plan['runs'][0]['lanes'][i]['id'] for i in range(len(plan['runs'][0]['lanes'])))
        available={r.get('id',r.get('lane_id')) for r in result['tables']['lanes']}
        if available:demand(expected<=available,'imported lanes absent')
        local_runs=[r for r in result['tables']['runs'] if r.get('id')==meta['run']['id']]
        if result['tables']['runs']:demand(bool(local_runs),'imported run absent')
        expected_interfaces={i['id'] for i in plan['interfaces']};actual_interfaces={i.get('id') for i in result['tables']['interfaces']}
        demand(expected_interfaces<=actual_interfaces,'imported interfaces absent')
        expected_cp={c['id'] for t in plan['tasks'] for c in t.get('checkpoints',[])}
        demand(expected_cp<={c.get('id') for c in result['tables']['checkpoints']},'imported checkpoints absent')
        c=result['current_observation_preconditions'];demand(not (set(c.get('task_ids',[]))&want),'NF1 task already claimed during bootstrap; inspect before executing')
        verification_status='verified' if available and local_runs else 'partially_verified_requires_run_lane_inspection'
        record=dict(kind='nf1-import-verification-v1',workspace=project,plan_sha256=hashlib.sha256(canonical(plan)).hexdigest(),task_ids_verified=sorted(want),lane_membership_verified=bool(available),observed=result,workers_dispatched_by_script=False,status=verification_status)
        with output.open('x') as f:json.dump(record,f,indent=2)
        print(json.dumps({'status':verification_status,'receipt':str(output),'lanes_checked':bool(available),'note':'Inspect run/lanes explicitly if tables were not exposed; no dispatch performed.'}));return
    if o.command=='apply':demand(o.confirm=='APPLY-'+meta['run']['id'] and o.preview and o.preview.is_file(),'exact confirmation and reviewed preview required')
    demand(o.review_head and o.peer_review_head,'explicit reviewed HEADs for both repositories required')
    src=snapshot(repo,o.review_head,o.acknowledge_dirty);peer=snapshot(o.peer_repo,o.peer_review_head,o.acknowledge_dirty)
    src['package_manifest']=digest(PACKAGE/'MANIFEST.sha256')
    peer_info=load(PACKAGE/'machine/joint_program.json')['projects'];other=next(x for x in peer_info if x['workspace']!=project)
    pp=o.peer_repo/other['package_path'];demand(pp.is_dir(),'peer package not installed');validate(pp,PACKAGE);peer['package_manifest']=digest(pp/'MANIFEST.sha256')
    fresh=json.loads(call(bridge+['validate','--plan',PACKAGE/'machine'/meta['native_plan_file']]))
    c=check_native(fresh,plan,uuid);demand(c['repository_commits'].get(project)==o.review_head,'native source view differs from reviewed HEAD')
    if o.command=='preview':
        record=dict(kind='nf1-reviewed-preview-v1',workspace=project,created_unix=time.time(),source=src,peer_source=peer,manifest_sha256=digest(PACKAGE/'MANIFEST.sha256'),native_response=fresh,authority_to_apply=False)
        with output.open('x') as f:json.dump(record,f,indent=2)
        print(json.dumps({'status':'preview_saved','tasks_to_add':len(fresh['would_add']),'receipt':str(output),'applied':False}));return
    check_preview(load(o.preview),src,peer,fresh,plan,uuid)
    with output.open('x') as f:
        record=dict(kind='nf1-manual-apply-receipt-v1',workspace=project,status='attempt_starting',preview_sha256=digest(o.preview),started_unix=time.time(),workers_dispatched_by_script=False)
        json.dump(record,f,indent=2);f.flush();os.fsync(f.fileno())
        try:
            demand(snapshot(repo,o.review_head,o.acknowledge_dirty)=={k:v for k,v in src.items() if k!='package_manifest'},'source changed before apply')
            demand(snapshot(o.peer_repo,o.peer_review_head,o.acknowledge_dirty)=={k:v for k,v in peer.items() if k!='package_manifest'},'peer source changed before apply')
            result=json.loads(call(bridge+['apply','--plan',PACKAGE/'machine'/meta['native_plan_file'],'--approved-preview',o.preview,'--confirm',o.confirm]))
            demand(result.get('status')=='applied' and result.get('plan_digest')==fresh['plan_digest'] and result.get('before_revision')==fresh['revision'],'ambiguous/native apply result; inspect before retry')
            demand(sorted(result.get('applied_add',[]))==sorted(t['id'] for t in plan['tasks']) and not result.get('applied_modify'),'unexpected applied diff')
            record.update(status='applied',native_response=result)
        except Exception as e:
            record.update(status='failed_or_outcome_requires_inspection',error=str(e));f.seek(0);f.truncate();json.dump(record,f,indent=2);f.flush();os.fsync(f.fileno())
            raise RuntimeError('STOP: do not retry automatically; inspect authority and receipt. '+str(e)) from e
        f.seek(0);f.truncate();json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps({'status':'applied','receipt':str(output),'workers_started':False,'run_may_be_active':True}))
if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,OSError,RuntimeError,subprocess.SubprocessError) as e:raise SystemExit('todo_bootstrap: '+str(e))
