#!/usr/bin/env python3
"""Collect bounded fresh-process CPU precision measurements; GPU jobs use the scheduler separately."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ('half', 'allhalf', 'fp32', 'fp64', 'legacy')
METHODS = ('steady', 'stochastic')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--input', type=Path, default=ROOT/'benchmarks/artifacts/stochastic-canonical.bin')
    p.add_argument('--cpu', type=int, default=min(os.sched_getaffinity(0)))
    p.add_argument('--build-dir', type=Path, help='build directory whose CMakeCache records compiler and flags')
    p.add_argument('--anndata-prefix', type=Path, help='installed dependency prefix to fingerprint')
    p.add_argument('--profiles', nargs='+', choices=PROFILES, default=list(PROFILES))
    p.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    p.add_argument('--runs', type=int, choices=[1, 3], default=3)
    p.add_argument('--controls', action='store_true', help='run bounded shape and format controls, one sample each')
    p.add_argument('--output', type=Path, required=True, help='new directory for config, events and results')
    args = p.parse_args()
    binary, fixture = args.binary.resolve(strict=True), args.input.resolve(strict=True)
    with fixture.open('rb') as f:
        header = f.read(40)
    if len(header) != 40 or struct.unpack('<2Q', header[:16]) != (4096, 2048):
        raise RuntimeError('canonical paired input must have shape 4096 x 2048')
    old = json.loads((ROOT/'benchmarks/results/milestone-2.json').read_text())
    if sha(fixture) != old['workload_sha256']:
        raise RuntimeError('canonical fixture hash differs from the milestone-2 workload')
    args.output.mkdir(parents=True, exist_ok=False)
    events = args.output/'events.jsonl'
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    source_paths = sorted(x for x in (ROOT/'src').rglob('*') if x.is_file()
                          and 'cuda' not in x.relative_to(ROOT/'src').parts
                          and x.name not in ('cuda.cpp', 'velocity.cuda.cppm'))
    config = dict(schema='velocity.precision-cpu-run.v1', utc=datetime.now(timezone.utc).isoformat(),
        binary=str(binary), binary_sha256=sha(binary), fixture=str(fixture), fixture_sha256=sha(fixture),
        source_sha256={str(x.relative_to(ROOT)): sha(x) for x in source_paths},
        harness_sha256=sha(ROOT/'benchmarks/precision_qualification.cpp'), runner_sha256=sha(Path(__file__)),
        generator_sha256=sha(ROOT/'tools/benchmark_reference.py'),
        seed=20261103, generator=dict(n=4096, g=2048, k=32, density=.01),
        profiles=args.profiles, methods=args.methods, samples=args.runs, cpu=args.cpu, threads=1,
        precision_tuples={'half':'S/U=f16,W=f32,compute=f32,output=f32', 'allhalf':'S/U/W=f16,compute=f32,output=f32',
                          'fp32':'S/U/W=f32,compute=f32,output=f32', 'fp64':'S/U/W=f64,compute=f64,output=f64',
                          'legacy':'S/U/W=f32,compute=f64,output=f32'},
        cmake_cache_sha256=sha(args.build_dir/'CMakeCache.txt') if args.build_dir else None,
        anndata_sha256={str(x.relative_to(args.anndata_prefix)):sha(x) for x in sorted(args.anndata_prefix.rglob('*')) if x.is_file()} if args.anndata_prefix else None,
        compiler=subprocess.run(['clang++-18','--version'],capture_output=True,text=True,check=True).stdout.splitlines()[0],
        kernel=platform.release(), cpu_model=next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')), 'unknown'),
        options=dict(min_r2=-100, min_ratio=-100, block_genes=64, stochastic_second_residual=True),
        timeout_s=30, virtual_memory_limit_bytes=2*1024**3,
        allocation_scope='global C++ new calls and requested bytes, includes fixture load; excludes malloc and library allocations not using operator new',
        peak_scope='process VmHWM includes input load, packing, preparation, workspace, execution and block sink; incremental is HWM minus RSS after fixture load',
        sink_scope='checksum traversal included in execute time; numerical parity is tested separately')
    (args.output/'run_config.json').write_text(json.dumps(config, indent=2)+'\n')

    def limits():
        resource.setrlimit(resource.RLIMIT_AS, (2*1024**3, 2*1024**3))
        os.sched_setaffinity(0, {args.cpu})

    def invoke(label, command):
        try:
            proc = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True,
                                  timeout=30, preexec_fn=limits)
            event = dict(label=label, command=command, exit_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
        except subprocess.TimeoutExpired as exc:
            event = dict(label=label, command=command, timeout_s=30, stdout=str(exc.stdout or ''), stderr=str(exc.stderr or ''))
            with events.open('a') as f: f.write(json.dumps(event)+'\n')
            raise RuntimeError(f'{label} exceeded 30 seconds') from exc
        with events.open('a') as f: f.write(json.dumps(event)+'\n')
        if proc.returncode:
            raise RuntimeError(f'{label} failed ({proc.returncode}): {proc.stderr}')
        row = json.loads(proc.stdout)
        if not row.get('finite'):
            raise RuntimeError(f'{label} failed finiteness check')
        return row

    rows=[]
    for method in args.methods:
        for profile in args.profiles:
            for sample in range(args.runs):
                # Alternate order to reduce systematic order bias; each worker is a fresh process.
                backends = ('scalar','avx2') if sample % 2 == 0 else ('avx2','scalar')
                for backend in backends:
                    cmd = [str(binary), '--input', str(fixture), '--profile', profile, '--method', method, '--backend', backend]
                    rows.append(dict(sample=sample, **invoke(f'{method}-{profile}-{backend}-{sample}', cmd)))
    controls=[]
    if args.controls:
        fixtures = [(256,256),(256,8192),(4096,256),(8192,256)]
        for n,g in fixtures:
            path=ROOT/f'benchmarks/artifacts/stochastic-scaling-n{n}-g{g}.bin'
            if not path.exists(): raise RuntimeError(f'missing preserved scaling fixture {path}')
            cmd=[str(binary),'--input',str(path),'--profile','fp32','--method','stochastic','--backend','scalar']
            controls.append(dict(fixture_sha256=sha(path), **invoke(f'shape-{n}-{g}',cmd)))
        for expression,graph in [('csr','csc'),('csc','csr'),('dense','csr')]:
            path=ROOT/'benchmarks/artifacts/stochastic-scaling-n256-g256.bin'
            cmd=[str(binary),'--input',str(path),'--profile','fp32','--method','stochastic','--backend','scalar','--expression',expression,'--graph',graph]
            controls.append(dict(fixture_sha256=sha(path), **invoke(f'format-{expression}-{graph}',cmd)))
    groups=[]
    for method in args.methods:
        for profile in args.profiles:
            group={'method':method,'profile':profile}
            for backend in ('scalar','avx2'):
                x=[r['end_to_end_s'] for r in rows if r['method']==method and r['profile']==profile and r['backend']==backend]
                group[backend]={'median_s':statistics.median(x),'min_s':min(x),'max_s':max(x)}
            groups.append(group)
    stable = (sha(binary)==config['binary_sha256'] and sha(fixture)==config['fixture_sha256'] and
        all(sha(ROOT/path)==digest for path,digest in config['source_sha256'].items()) and
        sha(ROOT/'benchmarks/precision_qualification.cpp')==config['harness_sha256'] and sha(Path(__file__))==config['runner_sha256'] and
        (args.anndata_prefix is None or all(sha(args.anndata_prefix/path)==digest for path,digest in config['anndata_sha256'].items())))
    result=dict(schema='velocity.precision-cpu-results.v1', samples=rows, controls=controls, summary=groups, identity_stable=stable,
        full_selection=all(r['selected_genes']==2048 for r in rows),
        full_refinement=all(r['stochastic_status_counts'][0]==2048 for r in rows if r['method']=='stochastic'),
        note='Three samples support a bounded comparison only; checksum finiteness is not numerical parity. No GPU measurements are included.')
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.output/'summary.json').write_text(json.dumps(groups,indent=2)+'\n')
    (args.output/'summary.txt').write_text('\n'.join(json.dumps(x) for x in groups)+'\n')
    print(json.dumps(groups, indent=2))
    if not stable: raise RuntimeError('source, dependency, fixture, or binary changed during measurement')

if __name__ == '__main__':
    main()
