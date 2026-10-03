#!/usr/bin/env python3
"""Compile bounded native adaptive consumer against the installed host SDK."""
import argparse
import pathlib
import subprocess
import tempfile

p = argparse.ArgumentParser()
p.add_argument('--sdk', type=pathlib.Path, required=True)
p.add_argument('--sanitize', action='store_true')
args = p.parse_args()
root = pathlib.Path(__file__).resolve().parents[3]
with tempfile.TemporaryDirectory(prefix='ce-adapt-native-') as tmp:
    exe = pathlib.Path(tmp) / 'reuse_test'
    command = ['c++', '-std=c++20', '-Wall', '-Wextra', '-Werror', '-pedantic',
               '-I' + str(root / 'include'), '-I' + str(args.sdk / 'include'),
               str(root / 'src/math/adaptive/reuse.cc'),
               str(root / 'tests/substrate/adaptive/reuse_test.cc'),
               str(args.sdk / 'lib/libcellerator_substrate_structured_state.a'),
               '-o', str(exe)]
    if args.sanitize:
        command[1:1] = ['-fsanitize=address,undefined', '-fno-omit-frame-pointer', '-g']
    print(' '.join(command), flush=True)
    subprocess.run(command, check=True)
    subprocess.run([str(exe)], check=True)
