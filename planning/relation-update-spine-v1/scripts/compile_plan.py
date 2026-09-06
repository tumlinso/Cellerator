#!/usr/bin/env python3
"""Pure, deterministic schema-3 projection. Never contacts Todo authority."""
from pathlib import Path
import argparse,json,sys
sys.dont_write_bytecode=True
PACKAGE=Path(__file__).resolve().parents[1]
PLAN='relation-update-spine-v1.todo-plan.json'
def assemble(m):
    if m.get('format')!='cellerator-relation-update-preledger' or m.get('authority_to_apply') is not False or m.get('native_schema_version')!=3:
        raise ValueError('wrong preledger format, authority or schema')
    return {'schema_version':3,'project':m['project'],'tasks':[m['root_record']]+[t['native_record'] for t in m['tasks']],
            'invariants':m['invariants'],'locks':m['locks'],'interfaces':m['interfaces'],'barriers':m['barriers'],
            'resource_classes':m['resource_classes'],'runs':[m['run']]}
def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--package-root',type=Path,default=PACKAGE)
    a.add_argument('--check',action='store_true');a.add_argument('--output',type=Path);o=a.parse_args()
    p=assemble(json.loads((o.package_root/'machine/proposed_todos.json').read_text()))
    if o.check:
        if p!=json.loads((o.package_root/'machine'/PLAN).read_text()):raise ValueError('native projection differs')
        print(json.dumps({'native_projection_matches':True,'native_authority_validation':False}));return
    s=json.dumps(p,indent=2,ensure_ascii=False)+'\n'
    if o.output:o.output.write_text(s)
    else:print(s,end='')
if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,OSError) as e:raise SystemExit('compile_plan: '+str(e))
