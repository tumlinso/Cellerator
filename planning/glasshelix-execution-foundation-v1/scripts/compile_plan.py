#!/usr/bin/env python3
"""Deterministic native schema-3 projection. No authority access."""
from pathlib import Path
import argparse,json,sys
sys.dont_write_bytecode=True
PACKAGE=Path(__file__).resolve().parents[1]
def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))
def assemble(m):
    if m.get('format')!='glasshelix-paired-foundation-preledger-v1' or m.get('authority_to_apply') is not False:
        raise ValueError('Not the paired non-authoritative preledger')
    if m.get('native_schema_version')!=3:raise ValueError('Native schema 3 required')
    return dict(schema_version=3,project=m['project'],tasks=[m['root_record']]+[t['native_record'] for t in m['tasks']],invariants=m['invariants'],locks=m['locks'],interfaces=m['interfaces'],barriers=m['barriers'],resource_classes=m['resource_classes'],runs=[m['run']])
def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--package',type=Path,default=PACKAGE);a.add_argument('--check',action='store_true');a.add_argument('--output',type=Path)
    o=a.parse_args();m=load(o.package/'machine/proposed_todos.json');p=assemble(m)
    if o.check:
        if p!=load(o.package/'machine'/m['native_plan_file']):raise ValueError('Native plan differs from rich catalog')
        print(json.dumps({'native_projection_matches':True,'authority_validation':False}));return
    text=json.dumps(p,indent=2,ensure_ascii=False)+'\n'
    if o.output:
        with o.output.open('x',encoding='utf-8') as f:f.write(text)
    else:print(text,end='')
if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,OSError) as e:raise SystemExit('compile_plan: '+str(e))
