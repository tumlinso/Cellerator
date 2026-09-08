#!/usr/bin/env python3
"""Pinned native Todo front door, generalized from the inspected RU1 bootstrap.
Never dispatches workers. 'apply' mutates one project's Todo authority only.
"""
from pathlib import Path
import argparse,hashlib,json,sys
sys.dont_write_bytecode=True
PACKAGE=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def runtime(require_pin=True):
    import project_control.mutation as mutation
    import project_control.models as models
    import project_control.config as config
    import project_control.proposals as proposals
    policy=load(PACKAGE/'evidence/tooling_contract.json')
    modules=[mutation,models,config,proposals]
    ident={m.__name__:{'path':str(Path(m.__file__).resolve()),'sha256':sha(m.__file__)} for m in modules}
    ident['python']={'path':str(Path(sys.executable).resolve()),'sha256':sha(Path(sys.executable).resolve())}
    if require_pin and ident['project_control.mutation']['sha256']!=policy['installed_mutation_pin_sha256']:
        raise ValueError('Installed Project Control mutation API differs from the verified predecessor pin. Reinspect installed code and update the reviewed tooling contract; do not bypass validation or guess a CLI.')
    return mutation,models,config,ident

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('command',choices=['inspect-runtime','validate','apply','observe']);a.add_argument('--plan',type=Path);a.add_argument('--approved-preview',type=Path);a.add_argument('--confirm');o=a.parse_args()
    meta=load(PACKAGE/'machine/proposed_todos.json');project=meta['project']['workspace'];confirm='APPLY-'+meta['run']['id']
    if o.command=='apply' and (o.confirm!=confirm or not o.approved_preview):raise ValueError('Explicit manual apply confirmation and preview required')
    mutation,models,config,ident=runtime(require_pin=o.command!='inspect-runtime')
    if o.command=='inspect-runtime':
        policy=load(PACKAGE/'evidence/tooling_contract.json')
        print(json.dumps({'runtime_identity':ident,'installed_pin_matches':ident['project_control.mutation']['sha256']==policy['installed_mutation_pin_sha256'],'native_authority_accessed':False}));return
    conf=config.load_config()
    if o.command=='observe':
        snapshot=mutation.build_mutation_snapshot(conf,project)
        import project_control.proposals as proposals
        result={'project_uuid':snapshot.project_uuid,'revision':snapshot.todo_revision,'current_observation_preconditions':proposals.observation_preconditions(snapshot).model_dump(mode='json'),'tables':{k:snapshot.todo_tables.get(k,[]) for k in ['tasks','checkpoints','interfaces','runs','lanes']},'runtime_identity':ident}
        print(json.dumps(result,sort_keys=True,separators=(',',':')));return
    if not o.plan:raise ValueError('--plan required')
    plan=load(o.plan)
    if plan.get('schema_version')!=3 or plan.get('project',{}).get('workspace')!=project:raise ValueError('Wrong native plan schema/project')
    if o.command=='validate':result=mutation.validate_native_plan(conf,project,plan)
    else:
        approved=load(o.approved_preview)
        if approved.get('kind')!='nf1-reviewed-preview-v1' or approved.get('workspace')!=project:raise ValueError('Wrong preview')
        v=approved['native_response']
        if v['runtime_identity']!=ident or v['plan_digest']!=mutation.plan_digest(plan):raise ValueError('Runtime/plan changed since reviewed preview')
        conditions=models.ObservationPreconditions.model_validate(v['current_observation_preconditions'])
        envelope=models.ProposalEnvelope.create(intent='Manually import the reviewed '+meta['run']['id']+' task graph only; no worker dispatch or source execution',proposed_change=plan,observation_preconditions=conditions,created_at=conditions.observed_at)
        result=mutation.apply_proposal(conf,project,envelope)
    result['runtime_identity']=ident;print(json.dumps(result,sort_keys=True,separators=(',',':')))
if __name__=='__main__':
    try:main()
    except Exception as e:
        print(json.dumps({'status':'failed','error':str(e),'code':getattr(e,'code',None),'details':getattr(e,'details',{})}),file=sys.stderr);raise SystemExit(1)
