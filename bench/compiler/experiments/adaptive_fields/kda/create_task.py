#!/usr/bin/env python3
"""Prepare KDA records for the bounded trusted portfolio collector (CPU-only)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[3]
PYTHON = '/home/tumlinson/Software/venvs/cellerator-ml2-py313-cu126/bin/python'
PACKAGE = ROOT / 'build-sparse-support/evidence/torch-v14-optional-policies/package'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    files = [
        ('adapter', HERE / 'adapter.py', 'harness'),
        ('portfolio_driver', BASE / 'gpu/e7_relation_runner.py', 'benchmark'),
        ('fixture_reader', BASE / 'gpu/e7_fixtures.py', 'reference'),
        ('indexer_binding', BASE / 'gpu/indexer/ctypes_binding.py', 'harness'),
        ('indexer_source', BASE / 'gpu/indexer/indexer_bridge.cu', 'other'),
        ('indexer_header', BASE / 'gpu/indexer/indexer_bridge.h', 'other'),
        ('indexer_library', BASE / 'gpu/indexer/artifacts/libadaptive_fields_indexer.so', 'toolchain'),
        ('indexer_manifest', BASE / 'gpu/indexer/artifacts/build_manifest.json', 'toolchain'),
        ('sparse_adapter', PACKAGE / 'cellerator/torch/sparse_support.py', 'reference'),
        ('mechanism_adapter', PACKAGE / 'cellerator/torch/mechanism.py', 'reference'),
        ('native_extension', PACKAGE / 'cellerator/torch/_torch.cpython-313-x86_64-linux-gnu.so', 'toolchain'),
        ('package_identity', PACKAGE.parent / 'identity.json', 'toolchain'),
    ]
    fixture_root = Path('/tmp/cellerator_scaled_sparse_fixtures_887437fc')
    cases = []
    for name, directory in [('small', 'small'), ('large_events', 'large-events'),
                            ('large_ranked', 'large-ranked')]:
        manifest = fixture_root / directory / 'manifest.json'
        manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
        cases.append({'id': name, 'input': {'seed': 20261027,
                      'fixture_manifest': str(manifest),
                      'fixture_manifest_sha256': manifest_hash},
                      'parameters': {'width': 64, 'collector_candidates': 6}, 'weight': 1.0})
        files.append((f'fixture_{name}', manifest, 'reference'))
    missing = [str(path) for _, path, _ in files if not path.is_file()]
    if missing:
        raise SystemExit('Missing inputs: ' + ', '.join(missing))
    commands = {operation: {'argv': [PYTHON, str(HERE / 'adapter.py'), operation,
                                   '{request}', '{result}'],
                            'timeout_seconds': 1800}
                for operation in ('probe', 'build', 'validate', 'benchmark')}
    commands['build']['cpu_only'] = True
    task = {
        'schema_version': 1, 'task_id': 'adaptive-fields-bounded-portfolio-v1',
        'cases': cases, 'commands': commands,
        'trusted_inputs': [{'id': name, 'path': str(path), 'role': role}
                           for name, path, role in files],
        'numerical_requirements': {
            'exact': 'Candidate IDs, ordinals, gate indices, worksets, epochs and owner identity',
            'arithmetic': 'Preserve existing operation-specific tolerances in the trusted driver',
            'validation': 'All six real indexer/numerical paths must pass; no fixture shrinking or skipped cells',
            'scope': 'Relation-only workload, excludes the unresolved recurrent transition contract',
        },
        'probe': commands['probe'],
        'timing': {'direction': 'minimize', 'metric': 'complete_training_step_wall_ms',
                   'unit': 'ms',
                   'scope': 'Native discovery + conversions + scoring/worksets + values + forward + backward + guarded Adam; synchronized wall',
                   'warmup': 5, 'repetitions': 21, 'rounds': 2,
                   'confirmation_rounds': 2, 'iterations': 1, 'seed': 20261027},
        'promotion': {'min_improvement': 0.0, 'max_regression': 0.0,
                      'confidence': 0.95, 'bootstrap_samples': 1000},
        'target': {'kind': 'cuda', 'architecture': 'sm_70', 'compiler_target': 'sm_70'},
        'toolchain': {'runtime': 'Frozen Cellerator v14 Python/v13 native extension; Torch 2.7 cu126',
                      'compiler_target_probe_argv': ['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc', '--list-gpu-code']},
        'execution': {'mode': 'direct'},
        'budgets': {'max_seconds': 3300, 'max_candidates': 1, 'max_build_jobs': 1},
    }
    write(HERE / 'task.json', task)
    write(HERE / 'collector/config.json', {'collector': 'six-way-paired-v1'})
    write(HERE / 'collector/manifest.json', {'schema_version': 1, 'files': ['config.json'],
          'dependencies': [{'name': 'target', 'revision': 'sm_70'}]})
    print(HERE / 'task.json')
