#!/usr/bin/env python3
"""Run the bounded coherence CUDA checks inside an admitted scheduler job."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--installed-package', type=Path, required=True)
    parser.add_argument('--sanitizer', choices=('memcheck', 'racecheck', 'synccheck'))
    parser.add_argument('--sanitizer-executable', type=Path)
    args = parser.parse_args()
    if os.environ.get('CELLERATOR_RUN_CUDA_TESTS') != '1':
        parser.error('requires CELLERATOR_RUN_CUDA_TESTS=1 from the admitted job')
    if args.sanitizer and args.sanitizer_executable is None:
        parser.error('--sanitizer-executable is required for sanitizer runs')
    repo = Path(__file__).resolve().parents[2]
    package = args.installed_package.resolve()
    extensions = tuple((package / 'cellerator' / 'torch').glob('_torch*.so'))
    if len(extensions) != 1:
        parser.error('the frozen installed package must contain exactly one Torch extension')
    environment = os.environ.copy()
    environment['PYTHONPATH'] = os.pathsep.join(map(str, (
        package, repo / 'tests' / 'sparse_support', repo / 'tests' / 'prepared_routing')))
    commands = (
        [sys.executable, '-m', 'unittest', '-v', 'test_cuda_training',
         'test_transition_routing_tail', 'test_sparse_module_prepared_route'],
        [sys.executable, str(repo / 'tests' / 'prepared_routing' / 'run_cuda_correctness.py'),
         '--device-index', '0', '--torch-module-file', str(extensions[0])],
    )
    for command in commands:
        if args.sanitizer:
            command = [str(args.sanitizer_executable), '--tool', args.sanitizer,
                       '--error-exitcode', '99', *command]
        print('Running:', repr(command), flush=True)
        result = subprocess.run(command, env=environment, cwd=repo)
        if result.returncode:
            return result.returncode if result.returncode > 0 else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
