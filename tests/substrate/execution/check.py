#!/usr/bin/env python3
"""Build and execute the actual host prepared-program consumer; no GPU launch."""
import argparse
from pathlib import Path
import subprocess
parser = argparse.ArgumentParser()
parser.add_argument('--build-dir', required=True)
args = parser.parse_args()
source = Path(__file__).resolve().parent
for argv in (["cmake", "-S", str(source), "-B", args.build_dir],
             ["cmake", "--build", args.build_dir, "--parallel", "2"],
             ["ctest", "--test-dir", args.build_dir, "--output-on-failure"]):
    subprocess.run(argv, check=True)
