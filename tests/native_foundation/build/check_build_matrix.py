#!/usr/bin/env python3
"""Reject stale source, changed binaries and incomplete NF1 build evidence."""
import hashlib, json, pathlib, subprocess, sys

def verify(receipt, root):
    if receipt.get('kind')!='nf1-build-matrix-v1' or receipt.get('passed') is not True: raise ValueError('matrix not qualified')
    head=subprocess.check_output(['git','-C',root,'rev-parse','HEAD'],text=True).strip()
    if receipt['source_commit']!=head: raise ValueError('stale matrix source')
    if receipt['architecture']!='70' or 'release 12.' not in receipt['cuda_version_output']: raise ValueError('matrix lacks CUDA12 sm70 qualification')
    if not receipt['commands'] or any(c['exit_code']!=0 for c in receipt['commands']): raise ValueError('matrix command failure')
    expected={'lib'+t+'.a' for t in ['cellerator_prepared_relation_cuda','cellerator_relation_algebra','cellerator_runtime','cellerator_operation_schema_v2','cellerator_prepared_program_v2','cellerator_relation_semantics','cellerator_relation_calculus','cellerator_segment_host','cellerator_gate_validation']}
    if not expected <= {pathlib.Path(x['path']).name for x in receipt['artifacts']}: raise ValueError('compiled target missing')
    for item in [*receipt['artifacts'],receipt['dependency_manifest']]:
        path=pathlib.Path(item['path'])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']: raise ValueError('artifact drift')
    manifest=json.loads(pathlib.Path(receipt['dependency_manifest']['path']).read_text())
    if manifest['source_commit']!=head: raise ValueError('stale configure manifest')
    for item in manifest['files']:
        if hashlib.sha256((pathlib.Path(root)/item['path']).read_bytes()).hexdigest()!=item['sha256']: raise ValueError('source dependency drift')

if __name__=='__main__':
    receipt=json.loads(pathlib.Path(sys.argv[1]).read_text());verify(receipt,sys.argv[2])
    altered=dict(receipt,source_commit='0'*40)
    try: verify(altered,sys.argv[2])
    except ValueError: pass
    else: raise ValueError('stale source negative control failed')
    print('Qualified actual host/CUDA build matrix; stale-source negative control rejected.')
