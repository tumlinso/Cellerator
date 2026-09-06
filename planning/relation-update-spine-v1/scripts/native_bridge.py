#!/usr/bin/env python3
"""Use the verified installed Project Control Python API. No task dispatch."""
from pathlib import Path
import argparse,hashlib,json,sys
sys.dont_write_bytecode=True
CONFIRM='APPLY-CE-RU1-RUN-V1'
PIN='3d8558670bb66a3d03614126236582f27649d91b16589a615e379fa4f74282a9'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def runtime():
    import project_control.mutation as mutation
    import project_control.models as models
    import project_control.config as config
    import project_control.proposals as proposals
    identity={m.__name__:{'path':str(Path(m.__file__).resolve()),'sha256':sha(m.__file__)} for m in [mutation,models,config,proposals]}
    if identity['project_control.mutation']['sha256']!=PIN:
        raise ValueError('Installed mutation API differs from reviewed source. Re-review/rebase the bridge; do not bypass the pin.')
    identity['python']={'path':str(Path(sys.executable).resolve()),'sha256':sha(Path(sys.executable).resolve())}
    return mutation,models,config,identity

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('command',choices=['inspect-runtime','validate','apply'])
    a.add_argument('--plan',type=Path);a.add_argument('--approved-preview',type=Path);a.add_argument('--confirm')
    o=a.parse_args()
    # Check mutation authorization before importing config or opening authority.
    if o.command=='apply' and (o.confirm!=CONFIRM or not o.approved_preview):
        raise ValueError('Explicit confirmation and reviewed preview required for manual application')
    mutation,models,config,identity=runtime()
    if o.command=='inspect-runtime':print(json.dumps({'runtime_identity':identity}));return
    if not o.plan:raise ValueError('--plan required')
    plan=json.loads(o.plan.read_text())
    if plan.get('schema_version')!=3:raise ValueError('RU1 requires native schema 3')
    conf=config.load_config()
    if o.command=='validate':result=mutation.validate_native_plan(conf,'cellerator',plan)
    else:
        approved=json.loads(o.approved_preview.read_text())
        if approved.get('kind')!='ce-ru1-reviewed-preview-v1':raise ValueError('wrong preview kind')
        if approved['native_response']['runtime_identity']!=identity:raise ValueError('runtime changed since preview')
        if approved['native_response']['plan_digest']!=mutation.plan_digest(plan):raise ValueError('plan changed since preview')
        # Use THE REVIEWED preconditions, not a new snapshot. Native apply checks
        # freshness again and checks expected revision inside BEGIN IMMEDIATE.
        conditions=models.ObservationPreconditions.model_validate(approved['native_response']['current_observation_preconditions'])
        proposal=models.ProposalEnvelope.create(intent='Manually apply the reviewed CE-RU1 plan only; do not dispatch tasks',
            proposed_change=plan,observation_preconditions=conditions,created_at=conditions.observed_at)
        result=mutation.apply_proposal(conf,'cellerator',proposal)
    result['runtime_identity']=identity;print(json.dumps(result,sort_keys=True,separators=(',',':')))
if __name__=='__main__':
    try:main()
    except Exception as e:
        print(json.dumps({'status':'failed','error':str(e),'code':getattr(e,'code',None),'details':getattr(e,'details',{})}),file=sys.stderr)
        raise SystemExit(1)
