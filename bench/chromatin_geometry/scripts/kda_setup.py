#!/usr/bin/env python3
"""Prepare KDA candidate identities and cached artifacts without using a GPU."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True, help='verified extracted Cellerator source root')
    parser.add_argument('--kda-root', type=Path, required=True, help='verified extracted KDA source root')
    parser.add_argument('--binary', type=Path, required=True, help='CPU-built cgs_study binary')
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    kda = args.kda_root.resolve(strict=True)
    binary = args.binary.resolve(strict=True)
    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text())
    if config.get('schema') != 'cgs.kda-config.v1':
        raise RuntimeError('invalid KDA experiment configuration')
    choices = {'scalar', 'bitmap', 'direct', 'cohort-direct', 'cohort-direct-two-tile',
               'cohort-shared', 'library', 'library-batched'}
    baseline = config['incumbent']
    candidates = config['candidates']
    if baseline not in choices or not isinstance(candidates, list) or not candidates or baseline in candidates or len(set(candidates)) != len(candidates) or not set(candidates) <= choices:
        raise RuntimeError('invalid KDA candidate set')
    for case in config.get('cases', []):
        expected = case.get('input', {}).get('campaign_workload_sha256')
        if not isinstance(expected, str) or len(expected) != 64:
            raise RuntimeError('each case needs an independently frozen campaign workload SHA-256')
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=False)
    study = source / 'bench/chromatin_geometry'
    driver = study / 'scripts/kda_driver.py'
    backend = study / 'backends'
    task_file = workspace / 'task.json'
    engine_workspace = workspace / 'engine'
    cmd = lambda mode: [sys.executable, str(driver), mode, '{request}', '{result}']
    inputs = [{'id': 'driver', 'role': 'harness', 'path': str(driver)},
              {'id': 'reference', 'role': 'reference', 'path': str(study / 'reference')},
              {'id': 'workloads', 'role': 'harness', 'path': str(study / 'workloads.cc')},
              {'id': 'workload_header', 'role': 'harness', 'path': str(study / 'workloads.hh')},
              {'id': 'runner', 'role': 'harness', 'path': str(study / 'campaign_runner.cu')},
              {'id': 'config', 'role': 'other', 'path': str(config_path)},
              {'id': 'binary', 'role': 'other', 'path': str(binary)},
              {'id': 'source_manifest', 'role': 'other', 'path': str(source.parent / 'SOURCE-MANIFEST.json')}]
    interference = config_path.parent / 'interference-evidence.json'
    if not interference.is_file() or digest(interference) != config.get('interference_attestation', {}).get('sha256'):
        raise RuntimeError('KDA task requires frozen isolated-versus-concurrent interference evidence')
    inputs.append({'id': 'interference_evidence', 'role': 'other', 'path': str(interference)})
    contract = config_path.parent / 'contract-freeze.json'
    if not contract.is_file() or digest(contract) != config.get('contract_freeze_sha256'):
        raise RuntimeError('KDA task requires frozen workload and validator identity')
    inputs.append({'id': 'workload_validator_freeze', 'role': 'other', 'path': str(contract)})
    replay_root = source.parent / 'replay'
    for i, item in enumerate(config.get('trusted_replay_inputs', [])):
        replay = (replay_root / item).resolve(strict=True)
        if not replay.is_relative_to(replay_root):
            raise RuntimeError('replay inputs must come from the extracted source closure')
        inputs.append({'id': f'replay_{i}', 'role': 'other', 'path': str(replay)})
    for case in config['cases']:
        replay_name = case.get('parameters', {}).get('replay')
        if replay_name:
            replay = (replay_root / replay_name).resolve(strict=True)
            if not replay.is_relative_to(replay_root):
                raise RuntimeError('case replay path escaped extracted closure')
            if replay_name not in config.get('trusted_replay_inputs', []) or digest(replay) != case['input'].get('replay_sha256'):
                raise RuntimeError('case replay must be a declared, hashed trusted input')
            case['parameters']['replay'] = str(replay)
    task = {'schema_version': 1, 'task_id': config['task_id'], 'cases': config['cases'],
            'commands': {'build': {'argv': [sys.executable, str(driver), 'build', str(binary), str(backend), '{request}', '{result}'], 'cpu_only': True, 'timeout_seconds': 60},
                         'validate': {'argv': cmd('validate'), 'timeout_seconds': 180},
                         'benchmark': {'argv': cmd('benchmark'), 'timeout_seconds': 180}},
            'trusted_inputs': inputs,
            'numerical_requirements': {'description': 'independent frozen scalar references, bit-exact gate/workset identities and staged numerical bounds; first and last complete steps'},
            'probe': {'argv': cmd('probe'), 'timeout_seconds': 20},
            'timing': {'metric': 'complete_operational_horizon', 'unit': 'us', 'direction': 'minimize',
                       'warmup': 1, 'repetitions': 1, 'rounds': 2, 'confirmation_rounds': 2,
                       'seed': 20261102,
                       'scope': 'cold execution preparation plus complete recurrent horizon and output recovery; controlled fixture and oracle excluded'},
            'promotion': {'min_improvement': 0.0, 'max_regression': 0.999, 'confidence': 0.95, 'bootstrap_samples': 10000},
            'target': {'kind': 'cuda', 'architecture': 'sm_70'},
            'toolchain': {'compiler_target_probe_argv': ['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc', '--list-gpu-code']},
            'execution': {'mode': 'direct'},
            'budgets': {'max_candidates': len(candidates) + 1, 'max_build_jobs': 1, 'max_seconds': 500}}
    task_file.write_text(json.dumps(task, indent=2, sort_keys=True) + '\n')
    env = dict(os.environ, PYTHONPATH=str(kda), CGS_KDA_CPU_PREPARE='1', CUDA_VISIBLE_DEVICES='')
    base = [sys.executable, '-m', 'kda_experiments', '--task', str(task_file), '--workspace', str(engine_workspace)]
    for name in [baseline, *candidates]:
        path = workspace / 'candidate-inputs' / name
        path.mkdir(parents=True)
        (path / 'backend.json').write_text(json.dumps({'backend': name}) + '\n')
        for filename in ('consumers.cu', 'conditions.cu', 'grouped_linear_library.cu'):
            shutil.copy2(backend / filename, path / filename)
        manifest = workspace / f'{name}.manifest.json'
        manifest.write_text(json.dumps({'schema_version': 1,
                                       'files': ['backend.json', 'consumers.cu', 'conditions.cu', 'grouped_linear_library.cu'],
                                       'dependencies': []}) + '\n')
        argv = base + ['register', name, str(path), '--manifest', str(manifest),
                       '--hypothesis', f'compare complete operational cost of {name}',
                       '--change', f'select the {name} implementation in the common frozen binary']
        if name == baseline:
            argv.append('--baseline')
        else:
            argv.extend(['--parent', baseline])
        subprocess.run(argv, env=env, check=True)
    subprocess.run(base + ['prepare', baseline, *candidates, '--jobs', '1'], env=env, check=True)
    print(json.dumps({'task': str(task_file), 'workspace': str(engine_workspace),
                      'binary_sha256': digest(binary), 'candidate_ids': [baseline, *candidates],
                      'task_sha256': digest(task_file),
                      'selection_status': 'preliminary; apply study 5-percent latency and independent evidence gates'}, sort_keys=True))


if __name__ == '__main__':
    main()
