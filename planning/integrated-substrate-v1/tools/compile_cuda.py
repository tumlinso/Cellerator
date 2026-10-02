#!/usr/bin/env python3
"""Compile the authored SM70 seed only. Never selects or launches a GPU."""
import argparse,hashlib,json,pathlib,re,subprocess
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--nvcc',required=True,type=pathlib.Path)
p.add_argument('--out',required=True,type=pathlib.Path)
a=p.parse_args(); compiler=a.nvcc.resolve(strict=True)
v=subprocess.run([str(compiler),'--version'],check=True,text=True,capture_output=True).stdout
if not re.search(r'release 12\.',v):raise SystemExit('An explicitly selected CUDA 12.x compiler is required for this SM70 bootstrap.')
a.out.mkdir(parents=True,exist_ok=True)
source=pathlib.Path(__file__).resolve().parents[1]/'seed/cuda/hybrid_scalar_sm70.cu'
records=[]
for flag,suffix in [('-c','o'),('--ptx','ptx')]:
    dest=a.out/('hybrid_scalar_sm70.'+suffix)
    cmd=[str(compiler),'-std=c++17','-arch=sm_70',flag,str(source),'-o',str(dest)]
    r=subprocess.run(cmd,check=True,text=True,capture_output=True)
    records.append({'argv':cmd,'stdout':r.stdout,'stderr':r.stderr,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
print(json.dumps({'status':'compiled_not_executed','compiler':v,'records':records},indent=2))
