#!/usr/bin/env python3
"""Paired CPU/CUDA measurements; execute only inside an admitted Scheduler job."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cpu', required=True, type=Path)
    p.add_argument('--cuda', required=True, type=Path)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--repetitions', type=int, choices=(1, 3), default=3)
    p.add_argument('--control', action='append', type=Path, default=[])
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    identities = {str(f): digest(f) for f in (args.cpu, args.cuda, args.input)}
    root = Path(__file__).resolve().parents[1]
    source = {str(f.relative_to(root)): digest(f) for f in (root/'src').rglob('*') if f.is_file()}
    samples = []

    def run(backend, profile, method, repetition, fixture):
        binary = args.cpu if backend == 'cpu' else args.cuda
        command = [str(binary), '--input', str(fixture), '--profile', profile,
                   '--method', method, '--block', '64']
        if backend == 'cpu':
            command += ['--backend', 'scalar']
        name = f'{len(samples):03d}-{backend}-{profile}-{method}-{repetition}'
        started = time.monotonic()
        proc = subprocess.run(command, capture_output=True, text=True, timeout=30)
        (args.output/f'{name}.stdout').write_text(proc.stdout)
        (args.output/f'{name}.stderr').write_text(proc.stderr)
        if proc.returncode:
            raise RuntimeError(f'{name} failed: {proc.stderr}')
        data = json.loads(proc.stdout)
        if not data.get('finite'):
            raise RuntimeError(f'{name}: nonfinite output')
        row = dict(backend=backend, profile=profile, method=method,
                   repetition=repetition, command=command, measurement=data,
                   process_wall_s=time.monotonic()-started, log=name,
                   input_sha256=digest(fixture))
        samples.append(row)
        print(json.dumps(dict(sample=name, wall_s=row['process_wall_s'])), flush=True)
        return data

    checks = []
    for profile in ('half', 'allhalf', 'fp32', 'fp64', 'legacy'):
        for method in ('steady', 'stochastic'):
            for repetition in range(args.repetitions):
                results = {}
                order = ('cpu', 'cuda') if repetition % 2 == 0 else ('cuda', 'cpu')
                for backend in order:
                    results[backend] = run(backend, profile, method, repetition, args.input)
                cpu, gpu = results['cpu'], results['cuda']
                selected = cpu['selected_genes'] == gpu['selected_genes'] == 2048
                refined = method == 'steady' or (cpu['stochastic_status_counts'][0] ==
                    gpu['successful_genes'] == 2048)
                tolerance = 1e-9 if profile in ('fp64', 'legacy') else 2e-4
                checksum = abs(cpu['velocity_checksum']-gpu['checksums']['velocity_sum']) <= \
                    tolerance*max(1, gpu['checksums']['velocity_absolute_sum'])
                checks.append(dict(profile=profile, method=method, repetition=repetition,
                                   full_population=selected and refined,
                                   checksum_sanity=checksum))
                if not selected or not refined or not checksum:
                    raise RuntimeError(f'Population/checksum mismatch: {checks[-1]}')
    for fixture in args.control:
        run('cuda', 'fp32', 'stochastic', 0, fixture)

    summary = []
    for profile in ('half', 'allhalf', 'fp32', 'fp64', 'legacy'):
        for method in ('steady', 'stochastic'):
            paired = [r for r in samples if r['profile'] == profile and
                      r['method'] == method and r['input_sha256'] == identities[str(args.input)]]
            cpu = [r['measurement']['end_to_end_s'] for r in paired if r['backend'] == 'cpu']
            gpu = [r['measurement']['phases_s']['end_to_end'] for r in paired if r['backend'] == 'cuda']
            resident = [r['measurement']['phases_s']['resident_compute_device_events']
                        for r in paired if r['backend'] == 'cuda']
            summary.append(dict(profile=profile, method=method,
                cpu_end_to_end_s=cpu, cuda_end_to_end_s=gpu,
                cuda_resident_s=resident,
                median_end_to_end_ratio=statistics.median(cpu)/statistics.median(gpu)))
    stable = all(digest(Path(f)) == value for f, value in identities.items()) and \
        all(digest(root/f) == value for f, value in source.items())
    result = dict(schema='velocity.precision-cuda-paired.v1', identities=identities,
        source_sha256=source, identities_stable=stable, repetitions=args.repetitions,
        worker_timeout_s=30, summaries=summary, samples=samples, checks=checks,
        all_checks_pass=stable and all(c['full_population'] and c['checksum_sanity'] for c in checks),
        limits=['Exclusive Scheduler GPU job; host CPU/RAM cgroup grants unavailable.',
                'Input fixtures and allocation checks bound host memory; no CUDA RLIMIT_AS.',
                'Resident timing is an additional evaluation after cold end-to-end.',
                'cudaMalloc/free counters omit driver-internal, managed and async allocations.',
                'Original input preparation and output consumption are included in end-to-end.'])
    (args.output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(all_checks_pass=result['all_checks_pass'], summaries=summary)), flush=True)
    return 0 if result['all_checks_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
