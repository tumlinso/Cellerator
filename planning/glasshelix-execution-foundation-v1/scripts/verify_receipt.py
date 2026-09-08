#!/usr/bin/env python3
"""Verify a peer capability plus fresh source/authority evidence. Never marks tasks done."""
from pathlib import Path
import argparse,datetime,json,os,subprocess,sys
sys.dont_write_bytecode=True
from validate_package import PACKAGE,load,demand,digest,safe_rel

def walk(x):
    if isinstance(x,dict):
        yield x
        for v in x.values():yield from walk(v)
    elif isinstance(x,list):
        for v in x:yield from walk(v)

def verify(edge,receipt,authority,peer,now=None):
    demand(receipt.get('kind')=='nf1-capability-receipt-v1' and receipt.get('status')=='qualified','not a qualified capability receipt')
    for k in ['producer_project','producer_task','capability']:demand(receipt.get(k)==edge[k],'wrong producer '+k)
    demand(receipt.get('edge_id')==edge['id'],'wrong cross-repository edge')
    joint=load(PACKAGE/'machine/joint_program.json');pr=next(p for p in joint['projects'] if p['workspace']==edge['producer_project'])
    demand(receipt.get('producer_uuid')==pr['project_uuid'] and receipt.get('producer_run')==pr['run_id'],'wrong producer authority/run')
    commit=receipt.get('source_commit','');demand(len(commit)==40,'missing exact source commit')
    r=subprocess.run(['git','-C',str(peer),'merge-base','--is-ancestor',commit,'HEAD'],capture_output=True);demand(r.returncode==0,'producer commit not in consumed checkout')
    files=receipt.get('source_files',[]);demand(files,'source dependency fingerprints required')
    for f in files:
        safe_rel(f['path']);p=peer/f['path'];demand(p.is_file() and digest(p)==f['sha256'],'changed producer dependency: '+f['path'])
    interfaces=receipt.get('interfaces',[]);demand(interfaces,'interface fingerprints required')
    for i in interfaces:
        demand(i.get('id') and i.get('version') and i.get('files'),'incomplete interface receipt')
        for f in i['files']:
            safe_rel(f['path']);demand(digest(peer/f['path'])==f['sha256'],'interface drift: '+f['path'])
    tests=receipt.get('test_receipts',[])
    producer_package=peer/pr['package_path']
    matrix=load(producer_package/'machine/acceptance_matrix.json')
    group=edge['producer_task'].split('-')[-1].lower()
    required=matrix['gate_groups'].get(group)
    covered=set();hardware_qualified=False
    if edge['capability']!='developmental-contracts':demand(tests,'executed evidence required')
    for t in tests:
        p=Path(t['path']);demand(p.is_file() and digest(p)==t['sha256'],'test receipt hash mismatch')
        data=load(p);demand(data.get('kind')=='nf1-test-receipt-v1' and data.get('passed') is True and data.get('expected_tests'),'test receipt not executed/pass')
        demand(data.get('source_commit')==commit,'test source does not match producer receipt')
        demand(not data.get('source_status'),'published capability tests must use clean committed source')
        demand(data.get('junit') and data.get('executables'),'raw test and binary evidence absent')
        import xml.etree.ElementTree as ET
        cases=list(ET.fromstring(data['junit']).iter('testcase'))
        demand({c.attrib.get('name') for c in cases}==set(data['expected_tests']) and len(cases)==len(data['expected_tests']),'raw JUnit differs from claimed inventory')
        demand(not any(c.find(k) is not None for c in cases for k in ['skipped','failure','error']),'raw JUnit contains skipped/failed tests')
        demand(set(data['expected_tests'])<=set(data['executables']),'missing binary fingerprints')
        covered.update(data['expected_tests']);hardware_qualified|=bool(data.get('hardware',{}).get('lease_verifier'))
    if required:
        demand(set(required['ctest_names'])<=covered,'required producer milestone tests are absent from receipt')
        if required.get('requires_gpu'):demand(hardware_qualified,'GPU milestone has only host evidence')
    demand(authority.get('kind')=='nf1-authority-observation-v1','fresh normalized authority observation required')
    demand(authority.get('workspace')==edge['producer_project'] and authority.get('project_uuid')==pr['project_uuid'],'authority identity mismatch')
    demand(authority.get('task_id')==edge['producer_task'] and authority.get('effective_state')=='done','producer task not authoritatively done')
    cp=edge['producer_task'].replace('-M','-CP-M')
    demand(authority.get('checkpoint_id')==cp and authority.get('checkpoint_state')=='reached','producer checkpoint not reached')
    dt=datetime.datetime.fromisoformat(authority['observed_at'].replace('Z','+00:00'))
    age=((now or datetime.datetime.now(datetime.timezone.utc))-dt).total_seconds();demand(0<=age<=3600,'authority observation stale or future-dated')
    raw=authority.get('raw_observations',[]);demand(raw,'raw authoritative tool responses required')
    nodes=list(walk(raw))
    demand(any(n.get('id')==edge['producer_task'] and n.get('effective_state')=='done' for n in nodes),'raw source does not support completed task')
    demand(any(n.get('id')==cp and (n.get('effective_state')=='reached' or n.get('raw_state')=='reached') and not n.get('revoked_at') for n in nodes),'raw source does not support reached checkpoint')
    demand(any(n.get('project_uuid')==pr['project_uuid'] for n in nodes),'raw authority UUID absent')
    return {'status':'verified','edge_id':edge['id'],'producer_commit':commit,'authority_revision':authority.get('revision'),'marks_task_done':False}

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--edge',required=True);a.add_argument('--peer-root',type=Path);a.add_argument('--receipt',type=Path);a.add_argument('--authority',type=Path);o=a.parse_args()
    edges=load(PACKAGE/'machine/cross_repository_contracts.json')['edges'];edge=next((e for e in edges if e['id']==o.edge),None);demand(edge,'unknown edge')
    cfg=load(Path(os.environ['NF1_EXECUTION_BINDINGS'])) if os.environ.get('NF1_EXECUTION_BINDINGS') else {}
    peer=o.peer_root or (Path(cfg['peer_root']) if cfg.get('peer_root') else None);demand(peer and peer.is_dir(),'peer root required')
    bind=cfg.get('external_receipts',{}).get(o.edge,{})
    receipt=o.receipt or (Path(bind['receipt']) if bind.get('receipt') else None);authority=o.authority or (Path(bind['authority']) if bind.get('authority') else None)
    demand(receipt and authority,'receipt and fresh authority observation required; templates do not qualify')
    print(json.dumps(verify(edge,load(receipt),load(authority),peer.resolve()),indent=2))
if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,OSError,subprocess.SubprocessError) as e:raise SystemExit('verify_receipt: '+str(e))
