#!/usr/bin/env python3
"""Illustrative host costs, not a throughput or winning-strategy qualification."""
import argparse
from pathlib import Path
import json
import subprocess
p=argparse.ArgumentParser()
p.add_argument('--build-dir',required=True)
p.add_argument('--sdk-prefix',default='/tmp/ce-is1-sdk-a')
p.add_argument('--output',required=True)
a=p.parse_args()
root=Path(__file__).resolve().parents[3]
for argv in [['cmake','-S',str(Path(__file__).resolve().parent),'-B',a.build_dir,'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_PREFIX_PATH='+a.sdk_prefix],
             ['cmake','--build',a.build_dir,'--parallel','2']]:
    subprocess.run(argv,check=True)
result=subprocess.run([str(Path(a.build_dir)/'ce_strategy_host_sample')],check=True,text=True,capture_output=True)
data=json.loads(result.stdout)
data['source_base']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
data['sdk_prefix']=a.sdk_prefix
data['statistic']='average nanoseconds per invocation, 512 repetitions after 16 warmups; one toy fixture'
Path(a.output).write_text(json.dumps(data,indent=2)+'\n')
print('Host sample saved to '+a.output)
