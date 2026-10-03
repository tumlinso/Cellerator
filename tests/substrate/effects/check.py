#!/usr/bin/env python3
"""Build/run native host effect composition and preserved numerical providers."""
import argparse
from pathlib import Path
import subprocess
parser = argparse.ArgumentParser()
parser.add_argument('--build-dir', required=True)
args = parser.parse_args()
source = Path(__file__).resolve().parent
for command in [['cmake', '-S', str(source), '-B', args.build_dir, '-DCMAKE_BUILD_TYPE=Release'],
                ['cmake', '--build', args.build_dir, '-j2'],
                ['ctest', '--test-dir', args.build_dir, '--output-on-failure']]:
    subprocess.run(command, check=True)
