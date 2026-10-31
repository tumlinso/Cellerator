#!/usr/bin/env python3
"""Run a finite CUDA correctness suite inside a Device Scheduler job."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tool', choices=['none', 'all', 'memcheck', 'racecheck', 'initcheck', 'synccheck'], default='none')
    parser.add_argument('--sanitizer', default='/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/compute-sanitizer/compute-sanitizer')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.tool == 'all':
        records = []
        failed = False
        for tool in ['none', 'memcheck', 'racecheck', 'initcheck', 'synccheck']:
            target = args.output / tool
            command = [sys.executable, str(Path(__file__).resolve()), '--build', str(args.build),
                       '--cases', *args.cases, '--tool', tool, '--sanitizer', args.sanitizer,
                       '--output', str(target)]
            result = subprocess.run(command, timeout=300)
            failed = failed or result.returncode != 0
            result_file = target / 'results.json'
            if result_file.exists():
                records.extend(json.loads(result_file.read_text()))
        (args.output / 'results.json').write_text(json.dumps(records, indent=2) + '\n')
        return int(failed)
    records = []
    for case in args.cases:
        if case == 'bindings':
            snippet = (
                'import cellerator, unittest, sys; '
                f'cellerator.__path__.append({str(args.build)!r}); '
                'suite=unittest.defaultTestLoader.discover("tests/framework/cuda", pattern="test_runtime_binding.py"); '
                'r=unittest.TextTestRunner(verbosity=2).run(suite); sys.exit(not r.wasSuccessful())'
            )
            command = [sys.executable, '-c', snippet]
            binary = args.build / '_modeling_cuda.cpython-313-x86_64-linux-gnu.so'
        else:
            binary = args.build / f'cellerator_framework_cuda_{case}_smoke'
            command = [str(binary)]
            if case == 'attention':
                command += ['--device']
            elif case == 'grouped_linear':
                command += ['--gpu']
        if args.tool != 'none':
            command = [args.sanitizer, '--tool', args.tool, '--error-exitcode', '99', *command]
        env = dict(os.environ, CELLERATOR_TEST_CUDA='1', CELLERATOR_TEST_CUDA_DEVICE='0',
                   PYTHONPATH=str(Path.cwd() / 'python'), PYTHONDONTWRITEBYTECODE='1')
        started = time.time()
        result = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180)
        log = args.output / f'{case}-{args.tool}.log'
        log.write_text(result.stdout)
        record = {'case': case, 'tool': args.tool, 'argv': command, 'returncode': result.returncode,
                  'seconds': time.time() - started, 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
                  'log': str(log), 'visible_devices': env.get('CUDA_VISIBLE_DEVICES')}
        records.append(record)
        print(json.dumps(record), flush=True)
        print(result.stdout[-6000:], flush=True)
    (args.output / 'results.json').write_text(json.dumps(records, indent=2) + '\n')
    return int(any(item['returncode'] != 0 for item in records))


if __name__ == '__main__':
    raise SystemExit(main())
