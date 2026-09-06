#!/usr/bin/env python3
"""Post-implementation executable gates. Never apply Todo or launch implementation workers.

These commands are future acceptance, not proof that today's Cellerator already
implements the proposed targets. No target/device/sanitizer failure is skipped.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

GROUPS={
 'core':['ceSpineCoreTest'],
 'conformance':['ceSpineCoreTest','ceSpineAlgebraTest','ceSpineFrontendTest','ceSpineNativeTest','ceSpineCrossOriginTest'],
 'demo':['ceSemanticSpineDemo'],
 'final':['ceSpineCoreTest','ceSpineAlgebraTest','ceSpineFrontendTest','ceSpineNativeTest','ceSpineCrossOriginTest','ceSemanticSpineDemo']}
GPU_TARGETS={'ceSpineNativeTest','ceSpineCrossOriginTest','ceSemanticSpineDemo'}

def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--group',choices=GROUPS,required=True)
    ap.add_argument('--source-root',type=Path,default=Path('.'))
    ap.add_argument('--build-dir',type=Path,help='Default from CELLERATOR_SS1_BUILD_DIR or a temp/cache directory')
    ap.add_argument('--jobs',type=int,default=int(os.environ.get('CELLERATOR_SS1_BUILD_JOBS','4')))
    a=ap.parse_args();root=a.source_root.resolve()
    default=Path(os.environ.get('CELLERATOR_SS1_BUILD_DIR',str(Path.home()/'.cache/cellerator/semantic-spine-v1')))
    build=(a.build_dir or default).expanduser().resolve()
    if a.jobs<1: raise RuntimeError('jobs must be positive')
    cmake=shutil.which('cmake')
    if not cmake:raise RuntimeError('cmake unavailable')
    calls=[]
    def call(argv: list[str]) -> str:
        start=time.time()
        r=subprocess.run(argv,cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=False)
        calls.append({'argv':argv,'returncode':r.returncode,'elapsed_seconds':time.time()-start,
                      'stdout_sha256':hashlib.sha256(r.stdout.encode()).hexdigest()})
        print(r.stdout,end='')
        if r.returncode:raise RuntimeError(f'gate command failed ({r.returncode}): {argv}')
        return r.stdout
    head=call(['git','rev-parse','HEAD']).strip()
    targets=GROUPS[a.group]
    uses_gpu=bool(GPU_TARGETS & set(targets))
    if uses_gpu:
        # A record from the canonical resource/coordination service, not a homemade physical inventory.
        # This is a documented precondition rather than a claim that an env var itself leases a GPU.
        if not os.environ.get('CELLERATOR_SS1_GPU_LEASE_RECEIPT'):
            raise RuntimeError('acquire the existing canonical GPU lease, then set CELLERATOR_SS1_GPU_LEASE_RECEIPT to its local receipt path')
        lease=Path(os.environ['CELLERATOR_SS1_GPU_LEASE_RECEIPT']).expanduser().resolve()
        if not lease.is_file(): raise RuntimeError('GPU lease receipt is absent')
        print('GPU lease evidence:',lease)
    config=[cmake,'-S',str(root),'-B',str(build),'-DCELLERATOR_BUILD_TESTS=ON',
            '-DCELLERATOR_BUILD_SEMANTIC_SPINE_V1=ON','-DCELLERATOR_ENABLE_CUDA=ON',
            '-DCMAKE_CUDA_ARCHITECTURES=70','-DCELLERATOR_AUTO_DETECT_CUDA_ARCHITECTURES=OFF']
    call(config)
    call([cmake,'--build',str(build),'--parallel',str(a.jobs),'--target',*targets])
    for target in targets:
        exe=build/target
        if not exe.is_file():raise RuntimeError('missing real test executable: '+str(exe))
        argv=[str(exe)]
        if target in GPU_TARGETS:argv+=['--require-sm70']
        call(argv)
    if a.group in ('demo','final'):
        sanitizer=shutil.which('compute-sanitizer')
        if not sanitizer:raise RuntimeError('Compute Sanitizer is required; cannot report a skipped success')
        call([sanitizer,'--tool','memcheck','--error-exitcode','99',str(build/'ceSemanticSpineDemo'),'--require-sm70'])
    output=build/'semantic-spine-receipts';output.mkdir(parents=True,exist_ok=True)
    receipt={'status':'passed','group':a.group,'source_head':head,'source_root':str(root),
             'created_unix':time.time(),'gpu_targets_executed':sorted(GPU_TARGETS & set(targets)),
             'calls':calls,'performance_promotion':False,'full_cell_compilation_complete':False}
    dest=output/f'{a.group}-{time.time_ns()}.json';dest.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'gate':'passed','receipt':str(dest)}));return 0
if __name__=='__main__':
    try:raise SystemExit(main())
    except (RuntimeError,OSError,ValueError) as e:raise SystemExit(f'GATE FAILED: {e}')
