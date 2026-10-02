#!/usr/bin/env python3
"""Source-bound product2 build/run evidence; GPU runs are root-controller owned."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import tempfile

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / 'experiments/moonshot-parallel-v1/integration'
EVIDENCE = HERE / 'evidence'
PYTHON = '/home/tumlinson/Software/venvs/cellerator-ml2-py313-cu126/bin/python'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as output:
            output.write(json.dumps(value, indent=2, sort_keys=True) + '\n')
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def source_hashes():
    paths = list((ROOT / 'include/Cellerator/compute/operation/product2').glob('*'))
    paths += list((ROOT / 'src/compute/operation/product2').glob('*'))
    paths += [ROOT / p for p in ('include/Cellerator/execution/program/program_v2.h',
        'src/execution/program/program_v2.cc', 'CMakeLists.txt', 'cmake/Product2.cmake',
        'components/CelleraTorch/python/celleratorch/moonshot_product.py')]
    paths += [HERE / p for p in ('native_product2_test.cc', 'product2_cuda_smoke.cu',
                                'test_product_adapter.py', 'native_product2_consumer.cc', 'verify_product_evidence.py')]
    # Include transitive repository headers, e.g. biological identity declarations.
    found = set(paths)
    while paths:
        path = paths.pop()
        for name in re.findall(r'#\s*include\s*[<"](Cellerator/[^>"]+)[>"]', path.read_text()):
            include = ROOT / 'include' / name
            if include.is_file() and include not in found:
                found.add(include)
                paths.append(include)
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(found) if p.is_file()}


def check_sources(expected):
    require(source_hashes() == expected, 'source or transitive header changed; fresh build required')


def check_hashes(mapping):
    for path, expected in mapping.items():
        require(sha(path) == expected, 'artifact changed: ' + path)


def execute(argv, prefix, env=None):
    started = time.monotonic()
    stdout, stderr = EVIDENCE / (prefix + '.stdout.txt'), EVIDENCE / (prefix + '.stderr.txt')
    with stdout.open('w') as so, stderr.open('w') as se:
        result = subprocess.run(argv, cwd=ROOT, env=env, stdout=so, stderr=se)
    return {'argv': argv, 'cwd': str(ROOT), 'returncode': result.returncode,
            'elapsed_seconds': time.monotonic() - started,
            'stdout_path': str(stdout), 'stderr_path': str(stderr),
            'stdout_sha256': sha(stdout), 'stderr_sha256': sha(stderr)}


def build(args):
    """Execute owner-provided configure/build commands into an absent build dir."""
    fresh = args.fresh_dir.resolve()
    require(not fresh.exists(), 'fresh build directory must be absent; preserve existing build work')
    commands = read(args.commands)
    require(isinstance(commands, list) and commands and all(isinstance(a, list) and a for a in commands),
            'commands must be a nonempty JSON array of argv arrays')
    require(any(str(fresh) in a for a in commands), 'fresh build directory must appear in command argv')
    require(not (EVIDENCE / (args.name + '-build.json')).exists(), 'build record already exists; choose fresh name')
    before = source_hashes()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    results = []
    for index, argv in enumerate(commands):
        results.append(execute(argv, f'{args.name}-build-{index}'))
        if results[-1]['returncode']:
            write(EVIDENCE / (args.name + '-build.failed.json'), {'commands': results, 'source_before': before})
            raise SystemExit(results[-1]['returncode'])
    check_sources(before)
    require(fresh.is_dir(), 'fresh build directory was not created')
    outputs = {}
    for output in args.outputs:
        path = output.resolve()
        require(path.is_relative_to(fresh) and path.is_file(), 'build output must be inside fresh directory')
        outputs[str(path)] = sha(path)
    require(outputs, 'compiled outputs required')
    record = {'schema_version': 1, 'task': 'CE-MOON-INTEGRATE', 'fresh_dir': str(fresh),
              'fresh_dir_absent_before': True, 'source_before': before, 'source_after': source_hashes(),
              'commands': results, 'output_sha256': outputs,
              'git_head_at_build': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    write(EVIDENCE / (args.name + '-build.json'), record)
    print(json.dumps({'build_record': str(EVIDENCE / (args.name + '-build.json')), 'status': 'built'}))


def build_outputs(record):
    return record.get('output_sha256', record.get('binary_sha256', {}))


def build_sources(record):
    return record.get('source_after', record.get('source_sha256_after', {}))


def verify_build(record):
    fresh = record.get('fresh_dir_absent_before', record.get('directory_created_empty'))
    require(fresh is True, 'no fresh build origin')
    before = record.get('source_before', record.get('source_sha256_before', {}))
    after = build_sources(record)
    require(before and before == after, 'inputs changed during build')
    for relative, expected in after.items():
        require(not Path(relative).is_absolute() and '..' not in Path(relative).parts,
                'build source must be repository-relative')
        require(sha(ROOT / relative) == expected, 'build input changed: ' + relative)
    check_hashes(build_outputs(record))
    require(record['commands'] and all(r['returncode'] == 0 for r in record['commands']), 'build failed')
    for result in record['commands']:
        require(result.get('argv'), 'build argv missing')
        if 'stdout_path' in result:
            check_hashes({result['stdout_path']: result['stdout_sha256'], result['stderr_path']: result['stderr_sha256']})
        elif 'log' in result:
            check_hashes({result['log']: result['log_sha256']})
        else:
            require(isinstance(result.get('stdout'), str) and isinstance(result.get('stderr'), str),
                    'embedded build stdout/stderr logs required')


def loaded_product_library(executable, loader_path):
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = loader_path
    result = subprocess.run(['ldd', str(executable)], cwd=ROOT, text=True, capture_output=True, env=env)
    require(result.returncode == 0, 'dynamic library inspection failed')
    match = re.search(r'libcellerator_product2\.so\s+=>\s+(\S+)', result.stdout)
    require(match, 'actual product2 shared-library linkage missing')
    path = Path(match.group(1)).resolve(strict=True)
    return {'path': str(path), 'sha256': sha(path), 'loader_path': loader_path, 'ldd_argv': ['ldd', str(executable)],
            'ldd_stdout': result.stdout, 'ldd_stderr': result.stderr}


def prepare(args):
    import torch
    records = [p.resolve() for p in args.build_record]
    outputs = {}
    for path in records:
        record = read(path)
        verify_build(record)
        outputs.update(build_outputs(record))
    covered = set().union(*(set(build_sources(read(p))) for p in records))
    required_compiled = {str(p.relative_to(ROOT)) for p in (ROOT / 'src/compute/operation/product2').glob('*')}
    required_compiled.update(('include/Cellerator/compute/operation/product2/c_api.h', 'include/Cellerator/compute/operation/product2/product2.hh', 'src/execution/program/program_v2.cc', 'include/Cellerator/execution/program/program_v2.h', 'experiments/moonshot-parallel-v1/integration/native_product2_test.cc', 'experiments/moonshot-parallel-v1/integration/product2_cuda_smoke.cu'))
    require(required_compiled <= covered, 'build provenance omits compiled source/test/header inputs')
    for path in (args.cpu, args.library, args.cuda, args.consumer, args.cuda_library):
        require(str(path.resolve()) in outputs, 'test binary/library missing from source-bound fresh build record: ' + str(path))
    cpu_loader = str(args.library.resolve().parent)
    loaders = {str(args.cpu.resolve()): cpu_loader, str(args.cuda.resolve()): args.cuda_loader_path, str(args.consumer.resolve()): cpu_loader}
    linkage = {p: loaded_product_library(p, loader) for p, loader in loaders.items()}
    for p, link in linkage.items():
        require(link['path'] in outputs, 'dynamic product library lacks clean-build provenance: ' + link['path'])
    require(linkage[str(args.cuda.resolve())]['path'] == str(args.cuda_library.resolve()), 'CUDA smoke resolves another shared library')
    manifest = {'schema_version': 1, 'task': 'CE-MOON-INTEGRATE', 'source_sha256': source_hashes(),
                'dynamic_linkage': linkage,
                'command_environment': [{'LD_LIBRARY_PATH': cpu_loader}, {'LD_LIBRARY_PATH': cpu_loader, 'CUDA_VISIBLE_DEVICES': ''}, {'LD_LIBRARY_PATH': args.cuda_loader_path}, {'LD_LIBRARY_PATH': cpu_loader}],
                'build_records': {str(p): sha(p) for p in records}, 'compiled_outputs_sha256': outputs,
                'python': sys.executable, 'torch': torch.__version__,
                'commands': [[str(args.cpu.resolve())],
                    [sys.executable, str(HERE / 'test_product_adapter.py')], [str(args.cuda.resolve())], [str(args.consumer.resolve())]],
                'required_environment': {'CELLERATOR_PRODUCT2_LIBRARY': str(args.library.resolve()), 'LD_LIBRARY_PATH': args.cuda_loader_path},
                'git_head_at_preparation': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    write(EVIDENCE / 'manifest.json', manifest)
    print('Prepared manifest for root controller run.')


def verify_manifest(manifest):
    check_sources(manifest['source_sha256'])
    check_hashes(manifest['build_records'])
    check_hashes(manifest['compiled_outputs_sha256'])
    for path in manifest['build_records']:
        verify_build(read(path))
    for executable, expected in manifest['dynamic_linkage'].items():
        actual = loaded_product_library(executable, expected['loader_path'])
        require(actual['path'] == expected['path'] and actual['sha256'] == expected['sha256'], 'dynamic product library resolution changed')


def run(args):
    manifest = read(EVIDENCE / 'manifest.json')
    verify_manifest(manifest)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, ''), 'controller-assigned CUDA device required')
    for key, value in manifest['required_environment'].items():
        require(os.environ.get(key) == value, key + ' must match manifest')
    print('manifest_sha256=' + sha(EVIDENCE / 'manifest.json'), flush=True)
    results = []
    for index, argv in enumerate(manifest['commands']):
        env = os.environ.copy()
        env.update(manifest['command_environment'][index])
        result = execute(argv, 'run-' + str(index), env)
        results.append(result)
        print(f'command {index}: returncode={result["returncode"]}', flush=True)
        if result['returncode']:
            write(EVIDENCE / 'run.failed.json', {'commands': results})
            raise SystemExit(result['returncode'])
    verify_manifest(manifest)
    write(EVIDENCE / 'run.json', {'manifest_sha256': sha(EVIDENCE / 'manifest.json'),
                                'sources_and_binaries_unchanged_after': True, 'commands': results})
    print('run_sha256=' + sha(EVIDENCE / 'run.json'), flush=True)


def validate_run(manifest):
    result = read(EVIDENCE / 'run.json')
    require(result['manifest_sha256'] == sha(EVIDENCE / 'manifest.json'), 'manifest identity mismatch')
    require(result['sources_and_binaries_unchanged_after'] is True, 'post-run source check missing')
    require([r['argv'] for r in result['commands']] == manifest['commands'], 'executed argv mismatch')
    require(all(r['returncode'] == 0 for r in result['commands']), 'test failed')
    for item in result['commands']:
        check_hashes({item['stdout_path']: item['stdout_sha256'], item['stderr_path']: item['stderr_sha256']})
    cpu = Path(result['commands'][0]['stdout_path']).read_text()
    cuda = Path(result['commands'][2]['stdout_path']).read_text()
    adapter = Path(result['commands'][1]['stderr_path']).read_text()
    require('native product2 CPU' in cpu and 'PASS' in cpu, 'native CPU success marker missing')
    require('product2 native CUDA' in cuda and 'prepared CUDA stage admission PASS' in cuda, 'CUDA success marker missing')
    count = re.search(r'Ran (\d+) tests?', adapter)
    require(count and int(count.group(1)) >= 10 and re.search(r'^OK\s*$', adapter, re.M)
            and 'skipped=' not in adapter, 'adapter requires at least ten tests, no skips, OK')
    consumer = Path(result['commands'][3]['stdout_path']).read_text()
    require('installed Cellerator::product2 public C++ prepared-stage consumer PASS' in consumer, 'installed exported consumer failed')
    return int(count.group(1))


def publish_capability(manifest, controller, count):
    run_result = read(EVIDENCE / 'run.json')
    checks = []
    for kind, index in (('native_correctness', 0), ('framework_consumer', 1), ('derivatives', 2), ('native_correctness', 3)):
        result = run_result['commands'][index]
        log = Path(result['stderr_path'] if index == 1 else result['stdout_path'])
        checks.append({'kind': kind, 'project': 'cellerator', 'argv': result['argv'],
            'exit_code': result['returncode'], 'source_commit_at_preparation': manifest['git_head_at_preparation'],
            'evidence_path': str(log.relative_to(ROOT)), 'sha256': sha(log)})
    capability = {'schema_version': 1, 'record_kind': 'moonshot_product2_producer_capability',
        'task': 'CE-MOON-INTEGRATE', 'status': 'tested',
        'git_head_at_preparation': manifest['git_head_at_preparation'],
        'source_sha256': manifest['source_sha256'],
        'controller_evidence_id': controller['evidence_id'],
        'capability': {
            'native_call': 'ce_product2_{create,forward,vjp,jvp,destroy}; ce_product2_cuda_{create,forward,vjp,jvp,destroy}; cellerator::compute::product2::{prepared_owner,make_stage,make_cuda_stage} with execute_prepared_program_v2',
            'framework_call': 'celleratorch.moonshot_product.{product2,product2_jvp,Product2Module}; CELLERATOR_PRODUCT2_LIBRARY points at standalone libcellerator_product2.so',
            'expression': 'y[i] = k[i] * x[a[i]] * x[b[i]]',
            'supported_shapes': {'contract': 'rank-1 x[n], k[m], a[m], b[m]; signed int64 topology indices in [0,n); one coefficient per product; n and m may be zero subject to valid indices',
                'native_cpu_tested': {'n': [0,3], 'm': [0,4]},
                'native_cuda_tested': {'n': [3], 'm': [0,1,5,129]},
                'framework': 'CPU contiguous rank-1 float32 tensors; repeated indices and empty support tested'},
            'precision': {'storage': 'FP32', 'arithmetic': 'FP32', 'accumulation': 'FP32',
                          'derivative_convention': 'ordinary first-order derivatives of FP32 product expression, including coefficient directions; CUDA repeated-input VJP uses atomic summation'},
            'capabilities': {'forward': 'implemented', 'input_vjp': 'implemented', 'parameter_vjp': 'implemented', 'jvp': 'implemented', 'second_order': 'unsupported'},
            'unsupported': ['CUDA tensors in Torch adapter', 'FP16/BF16/mixed precision', 'second-order Torch differentiation', 'torch.func/vmap/autocast', 'native optimizer ownership', 'other moonshot operators through this native product2 ABI'],
            'admission': 'explicit count/index/alignment/alias/generation checks; typed prepared stages bind biological axes and copied topology; CUDA owner/context exact topology matching'},
        'checks': checks, 'framework_tests_passed': count,
        'limits': ['Verification depends on retained managed build workspace and compiled outputs; it is not portable offline qualification from a clean checkout.', 'No performance result or biological fit claimed.', 'Standalone new adapter qualification does not requalify or change old indexed-mechanism adapters.', 'CUDA smoke is correctness qualification; no sanitizer run claimed.']}
    write(HERE / 'capability.json', capability)
    (HERE / 'README.md').write_text(
        '# Native product2 integration\n\n'
        'The native operation evaluates `y[i] = k[i] * x[a[i]] * x[b[i]]` in FP32. '
        'Direct CPU and CUDA calls and typed `prepared_program_v2` stages passed the retained tests. '
        'An installed C++ consumer links exported `Cellerator::product2` without compiling numerical source. '
        'Input and coefficient VJPs and a JVP with both input and coefficient directions are implemented. '
        'The tests exercise zeros, repeated inputs, empty support, generation checks, aliases, indices, '
        'typed axis admission and CUDA owner/context topology agreement.\n\n'
        f'The standalone Torch adapter passed {count} CPU tests. It uses the new native product library '
        'through `CELLERATOR_PRODUCT2_LIBRARY`; CUDA Torch tensors, mixed precision and higher-order '
        'differentiation remain unsupported. Existing indexed-mechanism files and libraries are separate dependencies.\n\n'
        f'Actual controller evidence: `{controller["evidence_id"]}`. The manifest binds source and transitive '
        'headers to retained clean-build command logs, output hashes, and actual execution evidence. '
        'Inputs and outputs were checked before and after execution. Actual dynamically resolved product libraries are hashed. '
        'Pure verification depends on the retained managed build workspace and binaries; a clean checkout needs fresh qualification. '
        '`capability.json` supplies callable paths, '
        'shape and precision policy, derivative support and hashed test evidence for the GH consumer. '
        'Consumer acceptance must separately bind the final delivered CE commit and live task completion.\n\n'
        'No performance victory, biological fit, native optimizer ownership or sanitizer qualification is claimed.\n\n'
        'Pure gate: `python experiments/moonshot-parallel-v1/integration/verify_product_evidence.py verify`.\n')


def verify_controller_binding():
    stdout = (EVIDENCE / 'controller.stdout.txt').read_text()
    expected = ('manifest_sha256=' + sha(EVIDENCE / 'manifest.json'), 'run_sha256=' + sha(EVIDENCE / 'run.json'))
    for marker in expected:
        require(stdout.splitlines().count(marker) == 1, 'controller stdout not bound to exact manifest/run: ' + marker)


def finalize(args):
    manifest = read(EVIDENCE / 'manifest.json')
    verify_manifest(manifest)
    count = validate_run(manifest)
    controller = read(args.controller)
    require(controller.get('ok') is True and controller.get('returncode') == 0
            and controller.get('status') == 'succeeded', 'controller did not succeed')
    for key, name in (('stdout_path', 'controller.stdout.txt'), ('stderr_path', 'controller.stderr.txt')):
        shutil.copyfile(controller[key], EVIDENCE / name)
    verify_controller_binding()
    lease = read(controller['lease_receipt'])
    require(lease.get('resource_ids'), 'lease lacks assigned resources')
    require(Path(lease.get('project_root', '')).resolve() == ROOT, 'controller lease belongs to another workspace')
    public = {k: lease[k] for k in ('format', 'owner_id', 'project_root', 'pid', 'resource_ids', 'observed_unix', 'state') if k in lease}
    write(EVIDENCE / 'lease-public.json', public)
    write(EVIDENCE / 'controller-public.json', {k: controller[k] for k in ('classification', 'evidence_id', 'ok', 'returncode', 'status') if k in controller})
    publish_capability(manifest, controller, count)
    receipt = {'schema_version': 1, 'task': 'CE-MOON-INTEGRATE', 'status': 'passed',
        'controller_evidence_id': controller['evidence_id'], 'original_lease_sha256': sha(controller['lease_receipt']),
        'capability_and_docs_sha256': {str(HERE / p): sha(HERE / p) for p in ('capability.json', 'README.md')},
        'lease_project_root': str(ROOT), 'adapter_tests_passed': count, 'manifest_sha256': sha(EVIDENCE / 'manifest.json'),
        'evidence_sha256': {str(p): sha(p) for p in EVIDENCE.iterdir() if p.is_file() and p.name != 'receipt.json'}}
    write(EVIDENCE / 'receipt.json', receipt)
    verify(args)


def verify(args):
    manifest, receipt = read(EVIDENCE / 'manifest.json'), read(EVIDENCE / 'receipt.json')
    verify_manifest(manifest)
    require(validate_run(manifest) == receipt['adapter_tests_passed'], 'adapter receipt mismatch')
    check_hashes(receipt['evidence_sha256'])
    check_hashes(receipt['capability_and_docs_sha256'])
    verify_controller_binding()
    lease = read(EVIDENCE / 'lease-public.json')
    require(lease.get('project_root') == receipt['lease_project_root'] and lease.get('resource_ids'), 'controller lease mismatch')
    controller = read(EVIDENCE / 'controller-public.json')
    require(controller['ok'] is True and controller['returncode'] == 0 and controller['status'] == 'succeeded', 'controller failed')
    require(controller['evidence_id'] == receipt['controller_evidence_id'], 'controller identity mismatch')
    print(json.dumps({'task': 'CE-MOON-INTEGRATE', 'status': 'passed', 'adapter_tests': receipt['adapter_tests_passed']}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest='mode', required=True)
    b = subs.add_parser('build')
    b.add_argument('--commands', type=Path, required=True)
    b.add_argument('--fresh-dir', type=Path, required=True)
    b.add_argument('--name', required=True)
    b.add_argument('--outputs', type=Path, nargs='+', required=True)
    prep = subs.add_parser('prepare')
    prep.add_argument('--build-record', type=Path, nargs='+', required=True)
    for option in ('cpu', 'library', 'cuda', 'consumer', 'cuda-library'):
        prep.add_argument('--' + option, type=Path, required=True)
    prep.add_argument('--cuda-loader-path', required=True)
    subs.add_parser('run')
    final = subs.add_parser('finalize')
    final.add_argument('--controller', type=Path, required=True)
    subs.add_parser('verify')
    args = p.parse_args()
    globals()[args.mode](args)


if __name__ == '__main__':
    main()
