#!/usr/bin/env python3
"""Compile the bounded circuit probe and record its actual source dependencies.

This helper configures and compiles only. It never executes the CUDA binary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

CUDA_ROOT = Path('/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9')
PROTOTYPE = Path('bench/factorized_interactions/prototypes/shared_product_circuit')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def entries(root: Path, paths: list[Path]) -> list[dict]:
    return [{'path': p.relative_to(root).as_posix(), 'bytes': p.stat().st_size,
             'sha256': digest(p)} for p in sorted(set(paths))]


def aggregate(records: list[dict]) -> str:
    return hashlib.sha256(json.dumps(records, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def run(command: list[str], cwd: Path, log: Path) -> None:
    with log.open('a') as stream:
        stream.write(json.dumps({'argv': command}) + '\n')
        stream.flush()
        subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                       check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--parallel', type=int, default=2)
    parser.add_argument('--cuda-root', type=Path, default=CUDA_ROOT)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    build = args.build.resolve()
    if build.is_relative_to(source):
        parser.error('--build must be outside the source tree')
    if build.exists() and any(build.iterdir()):
        parser.error('--build must be new or empty')
    if args.parallel < 1 or args.parallel > 4:
        parser.error('--parallel must be in [1,4]')
    build.mkdir(parents=True, exist_ok=True)
    cuda = args.cuda_root.resolve(strict=True)
    nvcc = cuda / 'bin/nvcc'
    cxx = Path('/usr/bin/g++-12').resolve(strict=True)
    if not nvcc.is_file():
        parser.error('NVCC is absent')

    # Guard the possible repository dependency set before compiling. Only the
    # actual compiler-reported dependencies enter the final source closure.
    possible = list((source / 'include').rglob('*'))
    possible += list((source / 'src/layout').glob('factorized_interaction*'))
    possible += list((source / 'src/modeling/cuda').glob('*factorized_interaction*'))
    possible += [source / 'src/memory/cuda/resources.cu',
                 source / 'bench/factorized_interactions/wire_input.hh']
    possible += [source / PROTOTYPE / name for name in
                 ('CMakeLists.txt', 'build_runner.py', 'circuit_runner.cu',
                  'circuit_kernels.cuh', 'PROVENANCE.json', 'docs/draft.md')]
    possible = [p for p in possible if p.is_file()]
    before = {p: digest(p) for p in possible}
    configure = ['cmake', '-S', str(source / PROTOTYPE), '-B', str(build / 'cmake'),
                 '-DCMAKE_BUILD_TYPE=Release', f'-DCMAKE_CXX_COMPILER={cxx}',
                 f'-DCMAKE_CUDA_COMPILER={nvcc}', f'-DCMAKE_CUDA_HOST_COMPILER={cxx}',
                 f'-DCUDAToolkit_ROOT={cuda}', '-DCMAKE_CUDA_ARCHITECTURES=70',
                 f'-DCELLERATOR_SOURCE_ROOT={source}']
    compile_command = ['cmake', '--build', str(build / 'cmake'), '--parallel',
                       str(args.parallel), '--target', 'shared_product_circuit_runner']
    log = build / 'build.log'
    run(configure, source, log)
    run(compile_command, source, log)
    runner = build / 'cmake/shared_product_circuit_runner'
    if not runner.is_file():
        raise RuntimeError('compiler produced no circuit runner')
    actual = set()
    depfiles = list((build / 'cmake').rglob('*.o.d'))
    if not depfiles:
        raise RuntimeError('compiler produced no dependency files')
    for depfile in depfiles:
        text = depfile.read_text().replace('\\\n', ' ')
        _, dependencies = text.split(':', 1)
        for name in shlex.split(dependencies):
            p = Path(name)
            p = p.resolve() if p.is_absolute() else (build / 'cmake' / p).resolve()
            if p.is_relative_to(source):
                if not p.is_file():
                    raise RuntimeError(f'compiled dependency is missing: {p}')
                actual.add(p)
    actual.update(source / PROTOTYPE / name for name in
                  ('CMakeLists.txt', 'build_runner.py', 'PROVENANCE.json', 'docs/draft.md'))
    for p in actual:
        if p not in before or before[p] != digest(p):
            raise RuntimeError(f'compiled source changed or escaped preflight closure: {p}')
    records = entries(source, list(actual))
    toolchain = {}
    for label, command in [('nvcc', [str(nvcc), '--version']),
                            ('host_cxx', [str(cxx), '--version']),
                            ('cmake', ['cmake', '--version'])]:
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, check=True)
        executable = Path(command[0])
        if not executable.is_absolute():
            import shutil
            executable = Path(shutil.which(command[0])).resolve()
        toolchain[label] = {'path': str(executable), 'sha256': digest(executable),
                            'version': result.stdout}
    metadata = {'schema': 'cellerator.shared_product_circuit.build.v1',
                'source_root': str(source), 'source_files': records,
                'source_closure_sha256': aggregate(records),
                'runner': runner.relative_to(build).as_posix(),
                'runner_binary_sha256': digest(runner),
                'runner_bytes': runner.stat().st_size,
                'architecture': 'sm_70', 'fast_math': False,
                'toolchain': toolchain, 'configure_argv': configure,
                'build_argv': compile_command, 'build_log_sha256': digest(log),
                'execution': 'compiled only; CUDA binary was not executed'}
    manifest = build / 'build_manifest.json'
    manifest.write_text(json.dumps(metadata, sort_keys=True, indent=2) + '\n')
    print(json.dumps({'runner': str(runner), 'build_manifest': str(manifest),
                      'build_manifest_sha256': digest(manifest),
                      'runner_binary_sha256': metadata['runner_binary_sha256'],
                      'source_closure_sha256': metadata['source_closure_sha256'],
                      'source_files': len(records)}, sort_keys=True))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f'circuit build failed: {error}', file=sys.stderr)
        raise SystemExit(1)
