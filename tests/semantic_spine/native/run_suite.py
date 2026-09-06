#!/usr/bin/env python3
"""Run built native witnesses; use this as a CUDA controller argv, under its lease."""
import argparse
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--build', type=Path, required=True)
parser.add_argument('--sanitize', action='store_true')
args = parser.parse_args()
build = args.build.resolve()
if args.sanitize:
    sanitizer = Path('/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/compute-sanitizer/compute-sanitizer')
    for tool in ('memcheck', 'initcheck'):
        for name in ('spine_lifecycle_test', 'spine_transpose_test', 'spine_regulatory_reuse'):
            subprocess.run([str(sanitizer), '--tool', tool, '--error-exitcode', '99', str(build / name)], check=True)
else:
    for name in ('adapter', 'preparation', 'generation', 'forward', 'transpose', 'lifecycle'):
        subprocess.run([str(build / f'spine_{name}_test')], check=True)
    subprocess.run([str(build / 'spine_regulatory_reuse'), '--require-sm70'], check=True)
