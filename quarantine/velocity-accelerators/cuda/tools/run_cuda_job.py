#!/usr/bin/env python3
"""Run one finite CUDA check through Device Scheduler and retain its receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import uuid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--timeout', type=int, default=300)
    p.add_argument('--cpu-cores', type=float, default=0)
    p.add_argument('--host-memory-mib', type=int, default=0)
    p.add_argument('--output', type=Path, default=Path('tests/artifacts/device-precision'))
    p.add_argument('argv', nargs=argparse.REMAINDER)
    args = p.parse_args()
    argv = args.argv[1:] if args.argv[:1] == ['--'] else args.argv
    if not argv or not 0 < args.timeout <= 300:
        p.error('supply a command after -- and a timeout from 1 through 300 seconds')
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve()/f'{args.name}-{uuid.uuid4().hex[:12]}'
    out.mkdir(parents=True)
    inventory = json.loads(subprocess.check_output(
        ['device-scheduler', 'resource', 'list'], text=True, timeout=30))
    (out/'inventory.json').write_text(json.dumps(inventory, indent=2)+'\n')
    resources = [e['resource'] for e in inventory['resources']
                 if e['resource'].get('attributes', {}).get('uuid') == args.gpu_uuid]
    if len(resources) != 1:
        raise RuntimeError('The requested GPU UUID is not unique in Scheduler inventory')
    spec = dict(owner_id='operator', argv=argv, working_directory=str(root),
        environment={'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1'},
        resources=[dict(kind='gpu', quantities={'memory_bytes': 2*1024**3},
                 selectors={'resource_id': resources[0]['resource_id']}, mode='exclusive')],
        tags=['gpu.exclusive'], timeout_seconds=args.timeout,
        kill_timeout_seconds=args.timeout, termination_grace_seconds=5,
        max_attempts=1, preemption_action='abandon', flexible=False,
        idempotency_key=f'bio-cppm:{out.name}',
        preconditions=[dict(name='gpu.quiescence', schema_version=1,
            parameters=dict(timeout_seconds=30, interval_seconds=1,
                consecutive_samples=3, max_utilization_percent=0))])
    if args.cpu_cores:
        spec['resources'].append(dict(kind='cpu', quantities={'cores': args.cpu_cores}, mode='shared'))
    if args.host_memory_mib:
        spec['resources'].append(dict(kind='memory', quantities={'bytes': args.host_memory_mib*1024**2}, mode='shared'))
    spec_file = out/'spec.json'
    spec_file.write_text(json.dumps(spec, indent=2)+'\n')
    identity = {}
    for value in argv:
        f = Path(value)
        if f.is_absolute() and f.is_file():
            identity[value] = hashlib.sha256(f.read_bytes()).hexdigest()
    (out/'command-files.json').write_text(json.dumps(identity, indent=2)+'\n')

    def call(label, command):
        result = subprocess.run(['device-scheduler', *command], text=True,
            capture_output=True, timeout=45)
        (out/f'{label}.json').write_text(result.stdout)
        if result.stderr:
            (out/f'{label}.stderr').write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(f'Scheduler {label} failed: {result.stdout or result.stderr}')
        return json.loads(result.stdout)

    response = call('submit', ['job', 'run', '--spec-file', str(spec_file), '--wait-seconds', '1'])
    job = response.get('job', response)
    job_id = response.get('job_id') or job.get('job_id') or job.get('id')
    if not job_id:
        raise RuntimeError(f'Scheduler response has no job ID; inspect {out}')
    print(json.dumps(dict(job_id=job_id, receipts=str(out))), flush=True)
    started = time.monotonic()
    terminal = {'succeeded', 'failed', 'cancelled', 'rejected', 'lost'}
    index = 0
    while True:
        job = call(f'state-{index}', ['job', 'get', job_id])
        job = job.get('job', job)
        state = job.get('state')
        print(json.dumps(dict(job_id=job_id, state=state,
                             cleanup_verified=job.get('cleanup_verified'))), flush=True)
        if state in terminal:
            break
        if time.monotonic()-started > args.timeout+90:
            call('cancel', ['job', 'control', job_id, 'cancel'])
            raise RuntimeError(f'Job exceeded bounded wait; verify cleanup using receipts in {out}')
        call(f'wait-{index}', ['job', 'wait', job_id, '--timeout-seconds', '30'])
        index += 1
    logs = call('logs', ['job', 'logs', job_id])
    print(json.dumps(logs), flush=True)
    summary = dict(job_id=job_id, state=state,
        cleanup_verified=job.get('cleanup_verified'), receipts=str(out),
        success=state == 'succeeded' and job.get('cleanup_verified') is True)
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)
    return 0 if summary['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
