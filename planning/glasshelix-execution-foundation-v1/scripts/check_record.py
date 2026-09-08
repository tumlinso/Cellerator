#!/usr/bin/env python3
"""Check a governance milestone record. This is not an accelerator-test gate."""
from pathlib import Path
import argparse,json,os,sys
sys.dont_write_bytecode=True
from validate_package import PACKAGE,load,demand,digest,safe_rel

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--milestone',required=True);o=a.parse_args()
    m=load(PACKAGE/'machine/proposed_todos.json');matrix=load(PACKAGE/'machine/acceptance_matrix.json');spec=matrix['governance_records'].get(o.milestone);demand(spec,'unknown governance record')
    p=PACKAGE.parents[1]/spec['record_path'];r=load(p)
    demand(r.get('kind')=='nf1-governance-receipt-v1' and r.get('task_id')==o.milestone,'wrong governance record')
    demand(r.get('reviewed') is True and r.get('source_commit') and r.get('reviewer_lane'),'review fields absent')
    demand(set(spec['required_topics'])<=set(r.get('topics',[])),'required review topics absent')
    demand(r.get('artifacts'),'evidence artifacts absent')
    for a in r['artifacts']:
        safe_rel(a['path']);demand(digest(PACKAGE.parents[1]/a['path'])==a['sha256'],'artifact drift')
    print(json.dumps({'status':'governance_record_checked','milestone':o.milestone,'gpu_acceptance':False}))
if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,OSError) as e:raise SystemExit('check_record: '+str(e))
