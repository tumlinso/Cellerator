#!/usr/bin/env python3
"""Negative controls for the real integrated executable; absence is failure, not skip."""
import argparse
import os
from pathlib import Path
import subprocess

p=argparse.ArgumentParser();p.add_argument('--build-dir',type=Path,required=True);a=p.parse_args()
exe=a.build_dir.resolve()/'ceSpineCrossOriginTest'
if not exe.is_file():raise SystemExit('missing integrated implementation executable')
for name,args,env,expected_text in [
    ('missing-device',[],dict(os.environ,CUDA_VISIBLE_DEVICES=''),'device'),
    ('wrong-output',['--require-sm70','--inject-wrong-output'],dict(os.environ),'differs from logical-edge oracle')]:
    r=subprocess.run([str(exe),*args],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(name,'exit',r.returncode,r.stdout)
    if r.returncode==0 or expected_text not in r.stdout:
        raise SystemExit(name+' did not fail for expected reason')
print('integrated negative controls passed')
