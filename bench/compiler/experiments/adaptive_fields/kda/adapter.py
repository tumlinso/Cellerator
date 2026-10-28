#!/usr/bin/env python3
"""KDA protocol adapter for a trusted six-way paired portfolio collector.

The single registered collector is not a competing numerical implementation.
All numerical comparisons are the six paired rows in the retained child output;
KDA's single-candidate preferred state is not used as evidence of a winner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
DRIVER = BASE / 'gpu/e7_relation_runner.py'
EXPECTED = {f'{indexer}_{backend}' for indexer in ('scan', 'ordered_postings')
            for backend in ('simt_full', 'simt_union', 'cegeo_union')}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(request, request_path, case, validation=False):
    output = request_path.parent / (request_path.stem + '-' + case['id'] + '-portfolio.json')
    timing = request.get('timing', {})
    argv = [sys.executable, str(DRIVER), '--case', case['id'], '--output', str(output),
            '--seed', str(case['input']['seed']), '--rounds', '1',
            '--warmup', str(1 if validation else timing['warmup']),
            '--samples', str(1 if validation else timing['repetitions']),
            '--validate-only' if validation else '--benchmark']
    process = subprocess.run(argv, cwd=BASE.parents[3], capture_output=True,
                             text=True, check=False, timeout=1500)
    (output.with_suffix('.process.json')).write_text(json.dumps({
        'argv': argv, 'returncode': process.returncode,
        'stdout': process.stdout, 'stderr': process.stderr,
    }, indent=2) + '\n')
    if process.returncode:
        raise RuntimeError(f'Portfolio failed: {process.stderr[-5000:]}')
    result = read(output)
    if result.get('decision') != 'supported':
        raise RuntimeError(f'Portfolio did not validate: {output}')
    return result, output


def execute(operation, request, request_path):
    envelope = {'schema_version': 1, 'request_id': request['request_id'], 'status': 'ok'}
    if operation == 'build':
        source = Path(request['candidate']['source_dir']) / 'config.json'
        if read(source) != {'collector': 'six-way-paired-v1'}:
            raise ValueError('Unknown collector configuration')
        directory = Path(request['build']['directory'])
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, directory / 'config.json')
        return envelope | {'artifacts': [{'path': 'config.json',
                                         'sha256': sha(directory / 'config.json')}]}
    if os.environ.get('CELLERATOR_SPARSE_SUPPORT_BENCH_ADMITTED') != '1':
        raise RuntimeError('GPU operations require the externally admitted scheduler job')
    if operation == 'probe':
        import torch
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        driver = subprocess.run(['nvidia-smi', '--id=' + os.environ['CUDA_VISIBLE_DEVICES'],
                                 '--query-gpu=driver_version', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, check=True, timeout=15).stdout.strip()
        return envelope | {'environment': {
            'kind': 'cuda', 'architecture': f'sm_{props.major}{props.minor}',
            'compute_capability': f'{props.major}.{props.minor}',
            'device_uuid': os.environ['CUDA_VISIBLE_DEVICES'],
            'runtime_version': str(torch.version.cuda),
            'driver_version': driver,
            'device': {'name': props.name, 'total_memory_bytes': props.total_memory},
            'tools': {'python': sys.version.split()[0], 'torch': str(torch.__version__)},
        }}
    if operation == 'validate':
        rows = []
        for case in request['cases']:
            result, output = collect(request, request_path, case, validation=True)
            candidates = result['candidates']
            if len(candidates) != 6 or any(row['status'] != 'pass' for row in candidates):
                raise RuntimeError('Expected six successful candidate validations')
            rows.append({'case_id': case['id'], 'workload_sha256': case['workload_sha256'],
                         'status': 'pass', 'portfolio_evidence': str(output)})
        return envelope | {'cases': rows}
    result, output = collect(request, request_path, request['case'])
    if set(result['candidates']) != EXPECTED:
        raise RuntimeError('Collector did not execute exactly the bounded six candidates')
    rows = result['candidates']['scan_simt_full']['timing']['paired_samples']
    samples = [row['candidate']['wall_ms'] for row in rows]
    timing = request['timing']
    if len(samples) != timing['repetitions']:
        raise RuntimeError('KDA sample count differs from measured collector output')
    return envelope | {
        'case_id': request['case']['id'],
        'workload_sha256': request['case']['workload_sha256'],
        'timing_scope': timing['scope'], 'warmup': timing['warmup'],
        'repetitions': timing['repetitions'], 'metric': timing['metric'],
        'unit': timing['unit'], 'samples': samples,
        'portfolio_evidence': str(output), 'portfolio_evidence_sha256': sha(output),
        'interpretation': 'Metric is collector baseline only; compare the six measured paired rows.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('build', 'probe', 'validate', 'benchmark'))
    parser.add_argument('request', type=Path)
    parser.add_argument('result', type=Path)
    args = parser.parse_args()
    request = read(args.request)
    try:
        result = execute(args.operation, request, args.request)
    except Exception as error:
        result = {'schema_version': 1, 'request_id': request.get('request_id'),
                  'status': 'error', 'error': f'{type(error).__name__}: {error}'}
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result['status'] == 'ok' else 1)
