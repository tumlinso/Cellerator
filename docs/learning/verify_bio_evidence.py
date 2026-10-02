#!/usr/bin/env python3
"""Prepare, execute under a controller lease, and verify source-bound CE-ML2-BIO evidence.

Preparation and verification never launch GPU work. The run subcommand executes
only the commands in the pre-recorded manifest, under the root's assigned lease.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/learning/bio-evidence'
LIB = Path('/home/tumlinson/Software/cellerator-ml2-python/lib/libcellera_torch_mechanism.so')
PYTHON = '/home/tumlinson/Software/venvs/cellerator-ml2-py313-cu126/bin/python'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def historical_hash():
    receipt = read(ROOT / 'docs/learning/capability-receipt.json')
    return next(a['sha256'] for a in receipt['artifacts'] if a['path'] == str(LIB))


def snapshot(paths):
    return {str(Path(p).resolve()): digest(p) for p in paths}


def verify_inputs(manifest):
    for path, expected in manifest['input_sha256'].items():
        require(digest(path) == expected, f'input changed since preparation: {path}')
    require(digest(LIB) == historical_hash(), 'native library differs from qualified historical dependency')
    require(manifest['native_library']['sha256'] == historical_hash(), 'manifest library is unqualified')
    require(importlib.metadata.version('celleratorch') == manifest['environment']['installed_version'], 'installed version changed')


def prepare(args):
    import celleratorch
    import torch
    installed = Path(celleratorch.__file__).resolve().parent
    sources = [ROOT / 'components/CelleraTorch/python/celleratorch' / name
               for name in ('biology.py', '__init__.py', 'mechanism.py')]
    sources += [ROOT / args.test_path, ROOT / 'bench/learning/shared_support_lifecycle.py',
                ROOT / 'components/CelleraTorch/src/mechanism.cc',
                ROOT / 'docs/learning/capability-receipt.json', Path(__file__), LIB]
    sources += sorted(installed.rglob('*.py'))
    for name in ('biology.py', '__init__.py', 'mechanism.py'):
        require(digest(installed / name) == digest(ROOT / 'components/CelleraTorch/python/celleratorch' / name), 'installed/source mismatch: ' + name)
    wheel = Path('/tmp/ce-bio-wheel-20261002/celleratorch-0.1.0-py3-none-any.whl')
    if wheel.is_file():
        sources.append(wheel)
    commands = [
        [sys.executable, '-m', 'pytest', '-q', str(ROOT / args.test_path),
         '--junitxml=' + str(OUT / 'pytest.xml')],
        [sys.executable, str(ROOT / 'bench/learning/shared_support_lifecycle.py'),
         '--run-cuda', '--output', str(OUT / 'benchmark.json'), '--repetitions', '5'],
    ]
    manifest = {'schema_version': 1, 'task': 'CE-ML2-BIO',
                'prepared_at_unix': time.time(),
                'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'input_sha256': snapshot(sources), 'commands': commands,
                'cwd': str(ROOT), 'environment': {'python': platform.python_version(),
                    'python_executable': sys.executable, 'torch': torch.__version__,
                    'torch_cuda': torch.version.cuda, 'installed_package': str(installed),
                    'installed_version': importlib.metadata.version('celleratorch')},
                'native_library': {'path': str(LIB), 'sha256': digest(LIB),
                                   'qualification': 'unchanged dependency qualified by CE-ML2-TRAIN'},
                'required_environment': {'CELLERATORCH_NATIVE_LIBRARY': str(LIB), 'CELLERATORCH_REQUIRE_NATIVE': '1'}}
    verify_inputs(manifest)
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / 'manifest.json', manifest)
    print('Prepared source-bound command manifest. Run it only under the CUDA controller lease.')


def run(args):
    manifest = read(OUT / 'manifest.json')
    verify_inputs(manifest)
    require(not os.environ.get('PYTHONPATH'), 'installed qualification requires no PYTHONPATH override')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, ''), 'controller-assigned CUDA_VISIBLE_DEVICES required')
    for key, value in manifest['required_environment'].items():
        require(os.environ.get(key) == value, f'{key} must match manifest')
    results = []
    for index, argv in enumerate(manifest['commands']):
        started = time.monotonic()
        stdout, stderr = OUT / f'command-{index}.stdout.txt', OUT / f'command-{index}.stderr.txt'
        with stdout.open('w') as so, stderr.open('w') as se:
            completed = subprocess.run(argv, cwd=manifest['cwd'], stdout=so, stderr=se)
        results.append({'argv': argv, 'returncode': completed.returncode,
                        'elapsed_seconds': time.monotonic() - started,
                        'stdout': str(stdout), 'stderr': str(stderr)})
        print(f'command {index}: returncode={completed.returncode}', flush=True)
        if completed.returncode:
            write(OUT / 'run.json', {'commands': results, 'status': 'failed'})
            raise SystemExit(completed.returncode)
    verify_inputs(manifest)
    write(OUT / 'run.json', {'commands': results, 'status': 'passed',
                            'manifest_sha256': digest(OUT / 'manifest.json'),
                            'source_unchanged_after_run': True})


def validate_results(manifest):
    run_result = read(OUT / 'run.json')
    require(run_result.get('manifest_sha256') == digest(OUT / 'manifest.json'), 'run manifest mismatch')
    require(run_result.get('source_unchanged_after_run') is True, 'post-run inputs were not verified')
    require(run_result['status'] == 'passed', 'execution failed')
    require([r['argv'] for r in run_result['commands']] == manifest['commands'], 'executed commands differ')
    require(all(r['returncode'] == 0 for r in run_result['commands']), 'command failure')
    tree = ET.parse(OUT / 'pytest.xml').getroot()
    suites = [tree] if tree.tag == 'testsuite' else list(tree.iter('testsuite'))
    counts = {key: sum(int(s.attrib.get(key, 0)) for s in suites)
              for key in ('tests', 'failures', 'errors', 'skipped')}
    require(counts['tests'] > 0 and all(counts[k] == 0 for k in ('failures', 'errors', 'skipped')),
            f'pytest requires positive pass count and no skips/errors/failures: {counts}')
    benchmark = read(OUT / 'benchmark.json')
    require(benchmark['native_library']['sha256'] == manifest['native_library']['sha256'], 'benchmark native hash mismatch')
    for rel, expected in benchmark['source_sha256'].items():
        require(manifest['input_sha256'].get(str(ROOT / rel)) == expected, 'benchmark source hash mismatch: ' + rel)
    require(benchmark['disposition'] == 'evaluated_not_promoted', 'benchmark promotion claim unsupported')
    require(len(benchmark['results']) >= 2, 'shared batch and counter regime required')
    for case in benchmark['results']:
        require(set(case['paths']) == {'native_composition', 'torch_materialized'}, 'both measured paths required')
        for result in case['paths'].values():
            require(len(result['samples']) >= 5, 'at least five complete iterations required')
            cold_keys = ('support_identity_packing_ms', 'initial_coefficients_upload_ms',
                         'program_or_module_preparation_ms', 'optimizer_setup_ms', 'complete_setup_ms')
            for key in cold_keys:
                value = result['cold'][key]
                require(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0, 'invalid setup cost ' + key)
            keys = ('input_target_upload_ms', 'forward_ms', 'loss_ms', 'backward_ms',
                    'adam_and_publication_ms', 'complete_iteration_ms')
            for key in keys:
                values = [s[key] for s in result['samples']]
                require(all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in values), 'invalid cost ' + key)
                require(math.isclose(result['median_ms'][key], statistics.median(values), rel_tol=1e-12), 'median mismatch ' + key)
    return counts, benchmark


def finalize(args):
    manifest = read(OUT / 'manifest.json')
    verify_inputs(manifest)
    counts, benchmark = validate_results(manifest)
    controller = read(args.controller)
    require(controller.get('ok') is True and controller.get('returncode') == 0
            and controller.get('status') == 'succeeded', 'CUDA controller did not succeed')
    evidence = {}
    for key, name in (('stdout_path', 'controller.stdout.txt'), ('stderr_path', 'controller.stderr.txt')):
        shutil.copyfile(controller[key], OUT / name)
        evidence[name] = digest(OUT / name)
    lease = read(controller['lease_receipt'])
    # Preserve only nonsecret resource admission facts; hash the exact original lease.
    public_lease = {k: lease[k] for k in ('format', 'owner_id', 'project_root', 'pid', 'resource_ids', 'observed_unix', 'state') if k in lease}
    require(public_lease.get('resource_ids'), 'lease lacks assigned resources')
    write(OUT / 'lease-public.json', public_lease)
    write(OUT / 'controller-public.json', {k: controller[k] for k in ('classification', 'evidence_id', 'ok', 'returncode', 'status') if k in controller})
    evidence.update({p.name: digest(p) for p in OUT.iterdir()
                     if p.is_file() and p.name not in ('receipt.json', 'README.md')})
    receipt = {'schema_version': 1, 'task': 'CE-ML2-BIO', 'status': 'passed',
               'source_commit_at_preparation': manifest['source_commit'],
               'controller_evidence_id': controller['evidence_id'],
               'original_lease_sha256': digest(controller['lease_receipt']),
               'evidence_sha256': evidence, 'pytest': counts,
               'benchmark_disposition': benchmark['disposition'],
               'limits': ['Prior native qualification reused only for unchanged native library.',
                          'Torch allocation peaks exclude raw native CUDA allocations.',
                          'No performance victory, biological fit, or checkpoint cost claimed.']}
    write(OUT / 'receipt.json', receipt)
    verify(args)


def verify(args):
    manifest, receipt = read(OUT / 'manifest.json'), read(OUT / 'receipt.json')
    verify_inputs(manifest)
    counts, benchmark = validate_results(manifest)
    require(receipt['status'] == 'passed' and counts == receipt['pytest'], 'receipt mismatch')
    for name, expected in receipt['evidence_sha256'].items():
        require(Path(name).name == name and digest(OUT / name) == expected, 'evidence changed: ' + name)
    controller = read(OUT / 'controller-public.json')
    require(controller['ok'] is True and controller['returncode'] == 0 and controller['status'] == 'succeeded', 'controller failure')
    require(controller['evidence_id'] == receipt['controller_evidence_id'], 'controller identity mismatch')
    print(json.dumps({'task': 'CE-ML2-BIO', 'status': 'passed', 'pytest': counts,
                      'benchmark': benchmark['disposition']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--test-path', default='components/CelleraTorch/tests/test_biology.py')
    sub.add_parser('run')
    final = sub.add_parser('finalize')
    final.add_argument('--controller', type=Path, required=True)
    sub.add_parser('verify')
    args = parser.parse_args()
    globals()[args.mode](args)


if __name__ == '__main__':
    main()
