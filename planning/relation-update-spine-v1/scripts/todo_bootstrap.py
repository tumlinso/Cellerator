#!/usr/bin/env python3
"""Guarded manual bootstrap: validate, preview, deliberate apply. Never dispatch tasks."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time
sys.dont_write_bytecode=True
from validate_package import validate,demand,digest,load,PACKAGE,PLAN
CONFIRM='APPLY-CE-RU1-RUN-V1'
UUID='0ccaac37-dbbf-448e-a5f8-def197a70aba'
BASE='45b965123d1d7ae4d10b0ac59e1714fcb7990d02'

def call(argv,cwd=None,timeout=120):
    r=subprocess.run([str(x) for x in argv],cwd=cwd,text=True,capture_output=True,timeout=timeout)
    if r.returncode:raise RuntimeError(f'{argv[0]} failed ({r.returncode}): {r.stderr}\n{r.stdout}')
    return r.stdout

def git(repo,*args):return call(['git','-C',repo,*args]).strip()
def source_snapshot(repo,head):
    demand(repo.resolve()==Path(git(repo,'rev-parse','--show-toplevel')).resolve(),'not repository root')
    demand(git(repo,'status','--porcelain=v1','--untracked-files=all')=='','repository must be clean; inspect and commit package separately')
    actual=git(repo,'rev-parse','HEAD');demand(len(head)==40 and actual==head,'explicit --review-head must match HEAD')
    call(['git','-C',repo,'merge-base','--is-ancestor',BASE,'HEAD'])
    changed=call(['git','-C',repo,'diff','--name-only','-z',BASE,'HEAD']).split('\0')
    allowed=('planning/relation-update-spine-v1/','examples/relation_update_spine_v1/')
    demand(all(not x or x.startswith(allowed) for x in changed),'source changed outside package/demo since baseline; re-review/rebase before apply')
    return {'repo':str(repo.resolve()),'head':actual,'baseline':BASE,'manifest_sha256':digest(PACKAGE/'MANIFEST.sha256')}
def native_validate(runtime_python):
    s=call([runtime_python,PACKAGE/'scripts/native_bridge.py','validate','--plan',PACKAGE/'machine'/PLAN])
    return json.loads(s)
def check_native(v,plan):
    expected=sorted(t['id'] for t in plan['tasks'])
    demand(v.get('valid') is True and v.get('status')=='validated','native validation did not pass')
    demand(v.get('project_uuid')==UUID and isinstance(v.get('revision'),int),'wrong/missing authority identity')
    demand(sorted(v.get('would_add',[]))==expected and not v.get('would_modify'),'collision, partial reapply or existing work would be changed')
    demand(not v.get('warnings'),'native warnings require review, not an automatic override')
    canonical=(json.dumps(plan,sort_keys=True,separators=(',',':'),ensure_ascii=False)+'\n').encode()
    demand(v.get('plan_digest')==hashlib.sha256(canonical).hexdigest(),'native digest mismatch')
    c=v.get('current_observation_preconditions',{})
    demand(c.get('project_uuid')==UUID and c.get('todo_revision')==v['revision'],'incoherent native cursor')
    demand(c.get('workflow_revision')==v['revision'] and c.get('todo_semantic_authority_fingerprint')==c.get('workflow_authority_fingerprint') and c.get('workflow_authority_fingerprint'),'incoherent workflow authority')
    demand(c.get('repository_commits',{}).get('cellerator') is not None,'missing source observation')
    return c

def check_preview(old,snapshot,fresh,plan,now=None):
    demand(old.get('kind')=='ce-ru1-reviewed-preview-v1','wrong preview format')
    age=(time.time() if now is None else now)-old.get('created_unix',0)
    demand(0<=age<=3600,'preview expired or future-dated')
    demand(old.get('source')==snapshot,'source/package changed since preview')
    prev=old['native_response'];check_native(prev,plan);check_native(fresh,plan)
    for k in ['project_uuid','revision','plan_digest','would_add','would_modify','runtime_identity']:
        demand(prev.get(k)==fresh.get(k),'native review changed: '+k)
    a=prev['current_observation_preconditions'];b=fresh['current_observation_preconditions']
    for k in set(a)|set(b):
        if k in {'observed_at','provider_skew'}:continue
        demand(a.get(k)==b.get(k),'native precondition changed: '+k)

def outside_output(path,repo):
    path=path.absolute();demand(path.parent.is_dir(),'receipt parent directory must exist')
    demand(not path.resolve().is_relative_to(repo.resolve()),'receipts must be outside repository to preserve source preconditions')
    demand(not path.exists() and not path.is_symlink(),'receipt already exists; never overwrite review/apply evidence')
    return path

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('command',choices=['validate','preview','apply'])
    a.add_argument('--repo',type=Path,default=PACKAGE.parents[1]);a.add_argument('--review-head')
    a.add_argument('--runtime-python',type=Path);a.add_argument('--receipt',type=Path)
    a.add_argument('--preview',type=Path);a.add_argument('--confirm')
    o=a.parse_args();repo=o.repo.resolve()
    result=validate(PACKAGE,repo)
    if o.command=='validate':print(json.dumps(result,indent=2));return
    demand(PACKAGE==(repo/'planning/relation-update-spine-v1').resolve(),'install package at recorded repository-relative path before native review')
    demand(o.runtime_python and o.runtime_python.is_file(),'--runtime-python must be the configured Project Control Python interpreter')
    demand(o.review_head and o.receipt,'explicit reviewed HEAD and new receipt path required')
    if o.command=='apply':demand(o.confirm==CONFIRM and o.preview and o.preview.is_file(),'manual apply requires exact confirmation and a reviewed preview')
    snapshot=source_snapshot(repo,o.review_head);plan=load(PACKAGE/'machine'/PLAN)
    fresh=native_validate(o.runtime_python);c=check_native(fresh,plan)
    demand(c['repository_commits']['cellerator']==o.review_head,'native source view differs from reviewed HEAD')
    output=outside_output(o.receipt,repo)
    if o.command=='preview':
        record={'kind':'ce-ru1-reviewed-preview-v1','created_unix':time.time(),'authority_to_apply':False,
                'source':snapshot,'native_response':fresh}
        with output.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
        print(json.dumps({'status':'preview_saved','path':str(output),'would_add_count':len(fresh['would_add']),'applied':False}));return
    old=load(o.preview);check_preview(old,snapshot,fresh,plan)
    # Reserve the receipt BEFORE mutation. Save the full outcome even on an
    # ambiguous error; never automatically retry a potentially applied plan.
    with output.open('x') as f:
        record={'kind':'ce-ru1-manual-apply-receipt-v1','preview_sha256':digest(o.preview),'started_unix':time.time(),'task_execution_requested':False}
        json.dump({**record,'status':'attempt_starting'},f,indent=2);f.flush();os.fsync(f.fileno())
        try:
            source_snapshot(repo,o.review_head);validate(PACKAGE,repo)
            result=json.loads(call([o.runtime_python,PACKAGE/'scripts/native_bridge.py','apply','--plan',PACKAGE/'machine'/PLAN,
                                   '--approved-preview',o.preview,'--confirm',CONFIRM]))
            demand(result.get('status')=='applied','apply did not return applied; inspect authority before retry')
            demand(result.get('plan_digest')==fresh['plan_digest'] and result.get('before_revision')==fresh['revision'],'apply receipt differs from approved plan')
            record.update(status='applied',native_response=result)
        except Exception as e:
            record.update(status='failed_or_outcome_requires_inspection',error=str(e))
            f.seek(0);f.truncate();json.dump(record,f,indent=2);f.flush();os.fsync(f.fileno())
            raise RuntimeError('STOP: inspect current authority and this receipt before any retry. '+str(e)) from e
        f.seek(0);f.truncate();json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps({'status':'applied_plan','receipt':str(output),'task_execution_requested':False}))
if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,OSError,RuntimeError,subprocess.SubprocessError) as e:raise SystemExit('todo_bootstrap: '+str(e))
