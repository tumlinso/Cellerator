#!/usr/bin/env python3
"""Write a finite Cellerator Device Scheduler JobSpec; never submit or execute."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CUDA = Path('/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--package-root', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=3300)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not argv or not args.gpu_uuid.startswith('GPU-'):
        parser.error('an argument-vector command and a GPU UUID are required')
    if not 0 < args.timeout <= 10800:
        parser.error('timeout must be positive and at most three hours')
    output = args.output.resolve()
    if HERE not in output.parents:
        parser.error('job specification must stay inside this experiment')
    package = args.package_root.resolve(strict=True)
    run = str(uuid.uuid4())
    spec = {
        'schema_version': 1, 'owner_id': 'cellerator',
        'idempotency_key': f'adaptive-fields-{args.stage}-{run}',
        'argv': argv, 'working_directory': str(ROOT),
        'environment': {
            'PATH': f'{CUDA}/bin:/home/tumlinson/Software/miniconda3/bin:/usr/local/bin:/usr/bin:/bin',
            'LD_LIBRARY_PATH': str(CUDA / 'lib64'),
            'PYTHONPATH': f'/home/tumlinson/kda:{package}',
            'CELLERATOR_SPARSE_PACKAGE_ROOT': str(package),
            'CELLERATOR_SPARSE_SUPPORT_BENCH_ADMITTED': '1',
            'ADAPTIVE_FIELDS_RUN_KEY': f'adaptive-fields-{args.stage}-{run}',
            'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
            'PYTHONUNBUFFERED': '1',
        },
        'resources': [{
            'schema_version': 1, 'kind': 'gpu', 'mode': 'exclusive',
            'quantities': {'memory_bytes': 6 * 1024**3},
            'selectors': {'resource_id': f'gpu:testlsserver:{args.gpu_uuid}'},
            'same_host': True, 'topology_constraints': {},
        }],
        'preconditions': [{
            'name': 'gpu.quiescence', 'schema_version': 1,
            'parameters': {'timeout_seconds': 60, 'interval_seconds': 1,
                           'consecutive_samples': 3, 'max_utilization_percent': 0},
        }],
        'timeout_seconds': args.timeout, 'max_attempts': 1,
        'interruption': {'cancellation_supported': True, 'checkpoint_supported': False,
                         'cooperative_yield': False, 'restart_safe': False},
        'tags': ['cellerator.adaptive_fields', args.stage, 'gpu.exclusive'],
        'secret_refs': [], 'dependencies': [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(spec, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'job_spec': str(output), 'stage': args.stage,
                      'prepared_at': dt.datetime.now(dt.timezone.utc).isoformat(),
                      'submitted': False}))
