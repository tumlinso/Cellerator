#!/usr/bin/env python3
"""Run real post-epic CTest gates; no tests, disabled tests or skipped tests fail."""
from pathlib import Path
import argparse,json,subprocess,tempfile,re,sys,xml.etree.ElementTree as ET
sys.dont_write_bytecode=True
PACKAGE=Path(__file__).resolve().parents[1]
def require(ok,msg):
    if not ok:raise ValueError(msg)
def select_inventory(data,names):
    ts={t['name']:t for t in data.get('tests',[])}
    require(set(names)<=set(ts),'Required CTest tests missing: '+str(set(names)-set(ts)))
    for name in names:
        props={p['name']:p.get('value') for p in ts[name].get('properties',[])}
        require(not props.get('DISABLED'),'disabled test: '+name)
        require('SKIP_RETURN_CODE' not in props and 'SKIP_REGULAR_EXPRESSION' not in props,'skip-as-success forbidden: '+name)
        require(bool(ts[name].get('command')),'test has no executable command')
    return ts

def check_junit(path,names):
    root=ET.parse(path).getroot();cases=list(root.iter('testcase'))
    require(len(cases)==len(names) and {c.attrib.get('name') for c in cases}==set(names),'JUnit testcase set differs')
    for c in cases:
        require(not any(c.find(k) is not None for k in ['skipped','failure','error']),'JUnit skipped/failed/error: '+str(c.attrib))
        require(c.attrib.get('status','run') not in {'notrun','disabled'},'JUnit test did not run')
def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--build-dir',type=Path,required=True)
    a.add_argument('--group',choices=['host','gpu','demo','all'],default='all');a.add_argument('--receipt',type=Path)
    o=a.parse_args();matrix=json.loads((PACKAGE/'machine/acceptance_matrix.json').read_text())
    names=[x['ctest_name'] for x in matrix['tests'] if o.group=='all' or x['group']==o.group]
    require(bool(names),'empty test set');require((o.build_dir/'CMakeCache.txt').is_file(),'missing configured build')
    if o.receipt:require(not o.receipt.exists(),'receipt path exists')
    cmd=['ctest','--test-dir',str(o.build_dir.resolve())]
    inv=subprocess.run(cmd+['--show-only=json-v1'],capture_output=True,text=True,check=True)
    select_inventory(json.loads(inv.stdout),names)
    with tempfile.TemporaryDirectory(prefix='ce-ru1-gate-') as d:
        junit=Path(d)/'result.xml';pattern='^('+ '|'.join(re.escape(n) for n in names)+')$'
        argv=cmd+['-R',pattern,'--output-on-failure','--no-tests=error','--output-junit',str(junit)]
        r=subprocess.run(argv,capture_output=True,text=True,timeout=1800)
        print(r.stdout,end='');print(r.stderr,end='',file=sys.stderr)
        record={'command':argv,'exit_code':r.returncode,'expected_tests':names,'stdout':r.stdout,'stderr':r.stderr,
                'junit':junit.read_text() if junit.exists() else None,'passed':False}
        try:require(r.returncode==0,'CTest failed');check_junit(junit,names);record['passed']=True
        finally:
            if o.receipt:
                with o.receipt.open('x') as f:json.dump(record,f,indent=2)
        require(record['passed'],'gate did not pass')
if __name__=='__main__':
    try:main()
    except (ValueError,OSError,subprocess.SubprocessError) as e:raise SystemExit('run_gate: '+str(e))
