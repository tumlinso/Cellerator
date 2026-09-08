#!/usr/bin/env python3
"""Execute exact post-epic CTest inventories, retaining evidence and rejecting skips.
GPU gates require the actual local lease verifier AND a shared host file lock.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,os,re,subprocess,sys,tempfile,xml.etree.ElementTree as ET
sys.dont_write_bytecode=True
from validate_package import PACKAGE,load,demand,digest

def select_inventory(data,names):
    tests=data.get('tests',[]);by={t['name']:t for t in tests}
    demand(len(by)==len(tests),'duplicate CTest names')
    demand(set(names)<=set(by),'missing tests: '+str(sorted(set(names)-set(by))))
    for name in names:
        t=by[name];props={p['name']:p.get('value') for p in t.get('properties',[])}
        demand(not props.get('DISABLED'),'disabled test: '+name)
        demand('SKIP_RETURN_CODE' not in props and 'SKIP_REGULAR_EXPRESSION' not in props,'skip-as-success forbidden: '+name)
        demand(bool(t.get('command')),'test lacks executable: '+name)
    return by

def check_junit(path,names):
    cases=list(ET.parse(path).getroot().iter('testcase'))
    demand(len(cases)==len(names) and {c.attrib.get('name') for c in cases}==set(names),'JUnit inventory differs')
    for c in cases:
        demand(not any(c.find(k) is not None for k in ['skipped','failure','error']),'test skipped or failed: '+str(c.attrib))
        demand(c.attrib.get('status','run') not in {'notrun','disabled'},'test did not execute')

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--group',required=True);a.add_argument('--build-dir',type=Path);a.add_argument('--bindings',type=Path);a.add_argument('--receipt',type=Path);o=a.parse_args()
    bindings_path=o.bindings or (Path(os.environ['NF1_EXECUTION_BINDINGS']) if 'NF1_EXECUTION_BINDINGS' in os.environ else None)
    bindings=load(bindings_path) if bindings_path else {}
    m=load(PACKAGE/'machine/proposed_todos.json');matrix=load(PACKAGE/'machine/acceptance_matrix.json')
    demand(o.group in matrix['gate_groups'],'unknown gate group')
    group=matrix['gate_groups'][o.group];names=group['ctest_names'];demand(names,'empty test group')
    build=o.build_dir or (Path(bindings['build_dir']) if bindings.get('build_dir') else None)
    demand(build and (build/'CMakeCache.txt').is_file(),'configured build directory required')
    build=build.resolve();repo=PACKAGE.parents[1]
    out=o.receipt
    if not out:
        demand(bindings.get('evidence_dir'),'--receipt or evidence_dir required')
        root=Path(bindings['evidence_dir']);demand(root.is_dir(),'evidence directory must exist')
        out=root/(m['prefix'].lower()+'-'+o.group+'-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    out=out.absolute();demand(out.parent.is_dir() and not out.exists(),'new external receipt required')
    demand(not out.resolve().is_relative_to(repo),'runtime evidence must be outside source/package')
    hardware={};lock=None
    if group.get('requires_gpu'):
        demand(bindings.get('gpu_lease_receipt') and Path(bindings['gpu_lease_receipt']).is_file(),'actual GPU lease receipt required')
        verifier=bindings.get('gpu_lease_verifier_argv')
        demand(isinstance(verifier,list) and verifier and all(isinstance(x,str) for x in verifier),'configure an inspected native GPU lease verifier; no guessed command')
        cmd=[x.replace('{lease_receipt}',bindings['gpu_lease_receipt']) for x in verifier]
        check=subprocess.run(cmd,text=True,capture_output=True,timeout=60)
        demand(check.returncode==0,'native GPU lease verifier failed')
        hardware['lease_verifier']={'argv':cmd,'stdout':check.stdout,'stderr':check.stderr,'lease_sha256':digest(bindings['gpu_lease_receipt'])}
        demand(bindings.get('shared_gpu_lock_file'),'one shared lock path for BOTH projects is required')
        import fcntl
        lock=open(bindings['shared_gpu_lock_file'],'a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        smi=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,compute_cap,driver_version','--format=csv,noheader'],text=True,capture_output=True,timeout=30)
        demand(smi.returncode==0,'GPU identity probe failed');hardware['inventory']=smi.stdout
    cmd=['ctest','--test-dir',str(build)]
    inv=subprocess.run(cmd+['--show-only=json-v1'],text=True,capture_output=True,check=True)
    by=select_inventory(json.loads(inv.stdout),names)
    executables={}
    for n in names:
        p=Path(by[n]['command'][0]);p=p if p.is_absolute() else build/p
        demand(p.is_file(),'CTest executable unavailable: '+str(p));executables[n]={'command':by[n]['command'],'executable_sha256':digest(p)}
    # Read only committed/clean source for capability receipts; local development may use non-publishing checks.
    head=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],text=True,capture_output=True,check=True).stdout.strip()
    dirty=subprocess.run(['git','-C',str(repo),'status','--porcelain=v1','--untracked-files=all'],text=True,capture_output=True,check=True).stdout
    tracked_diff=subprocess.run(['git','-C',str(repo),'diff','--binary','HEAD'],capture_output=True,check=True).stdout
    source_fingerprint=hashlib.sha256(dirty.encode()+tracked_diff).hexdigest()
    record=dict(kind='nf1-test-receipt-v1',workspace=m['project']['workspace'],group=o.group,source_commit=head,source_status=dirty,source_fingerprint=source_fingerprint,build_dir=str(build),cmake_cache_sha256=digest(build/'CMakeCache.txt'),expected_tests=names,executables=executables,hardware=hardware,passed=False,started_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    # Reserve the durable result before executing. A killed gate leaves a pending record, not a fake pass.
    with out.open('x') as evidence:
        json.dump(record,evidence,indent=2);evidence.flush();os.fsync(evidence.fileno())
        with tempfile.TemporaryDirectory(prefix='nf1-ctest-') as td:
            junit=Path(td)/'result.xml';pattern='^('+'|'.join(re.escape(n) for n in names)+')$'
            argv=cmd+['-R',pattern,'--output-on-failure','--no-tests=error','--output-junit',str(junit),'-j','1' if group.get('requires_gpu') else str(max(1,int(bindings.get('ctest_jobs',1))))]
            try:
                r=subprocess.run(argv,text=True,capture_output=True,timeout=int(bindings.get('test_timeout_seconds',1800)))
                record.update(argv=argv,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr,junit=junit.read_text() if junit.exists() else None)
                demand(r.returncode==0,'CTest failed');check_junit(junit,names)
                end=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],text=True,capture_output=True,check=True).stdout.strip()
                end_status=subprocess.run(['git','-C',str(repo),'status','--porcelain=v1','--untracked-files=all'],text=True,capture_output=True,check=True).stdout
                end_diff=subprocess.run(['git','-C',str(repo),'diff','--binary','HEAD'],capture_output=True,check=True).stdout
                demand(end==head and hashlib.sha256(end_status.encode()+end_diff).hexdigest()==source_fingerprint,'source changed during evidence capture')
                demand(digest(build/'CMakeCache.txt')==record['cmake_cache_sha256'],'build configuration changed during evidence capture')
                record['passed']=True
            except Exception as e:record['error']=str(e)
            finally:
                record['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();evidence.seek(0);evidence.truncate();json.dump(record,evidence,indent=2);evidence.write('\n');evidence.flush();os.fsync(evidence.fileno())
    if lock:lock.close()
    print(json.dumps({'passed':record['passed'],'receipt':str(out),'tests':names}));demand(record['passed'],record.get('error','gate failed'))
if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,OSError,subprocess.SubprocessError) as e:raise SystemExit('run_gate: '+str(e))
