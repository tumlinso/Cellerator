#!/usr/bin/env python3
import argparse
from pathlib import Path
import subprocess
p=argparse.ArgumentParser()
p.add_argument('--build-dir',required=True)
p.add_argument('--sdk-prefix',default='/tmp/ce-is1-sdk-a')
a=p.parse_args()
for argv in [['cmake','-S',str(Path(__file__).resolve().parent),'-B',a.build_dir,'-DCMAKE_PREFIX_PATH='+a.sdk_prefix],
             ['cmake','--build',a.build_dir,'--parallel','2'],
             ['ctest','--test-dir',a.build_dir,'--output-on-failure']]:
    subprocess.run(argv,check=True)
