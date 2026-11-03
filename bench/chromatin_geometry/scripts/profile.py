#!/usr/bin/env python3
"""Run a bounded diagnostic profiler inside an admitted study packet."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

NCU = '/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/profilers/12.9/Nsight_Compute/ncu'
NSYS = '/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/profilers/12.9/Nsight_Systems_2025.3/target-linux-x64/nsys'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not os.environ.get('AUTOBENCH_SCHEDULER_JOB_ID') or not uuid.startswith('GPU-'):
        raise RuntimeError('profiling requires an admitted single-GPU packet')
    c = json.loads(args.config.read_text())
    if c.get('schema') != 'cgs.profile.v1' or c.get('tool') not in ('nsys', 'ncu'):
        raise RuntimeError('invalid profile configuration')
    args.output.mkdir(parents=True, exist_ok=True)
    executable = NCU if c['tool'] == 'ncu' else NSYS
    version = subprocess.check_output([executable, '--version'], text=True)
    if c['tool'] == 'ncu' and '2025.2.' not in version:
        raise RuntimeError('Volta profiling requires Nsight Compute 2025.2')
    campaign = [str(args.binary), '--mode', 'campaign', '--case', c['case'], '--backend', c['backend'],
                '--gate', c['gate'], '--body', c['body'], '--precision', c['precision'],
                '--condition-update', c['condition_update'], '--tile', str(c['tile']), '--stages', str(c['stages']),
                '--warmups', '0', '--repeats', '1', '--output', str(args.output / 'campaign.json')]
    if c['tool'] == 'ncu':
        kernel = c.get('kernel_name_regex')
        if not isinstance(kernel, str) or not kernel:
            raise RuntimeError('NCU requires a kernel selected from a prior timeline')
        command = [executable, '--set', 'full', '--clock-control', 'none', '--target-processes', 'all',
                   '--kernel-name', 'regex:' + kernel, '--launch-count', '1', '--force-overwrite',
                   '--export', str(args.output / 'kernel')] + campaign
        report = args.output / 'kernel.ncu-rep'
    else:
        command = [executable, 'profile', '--trace=cuda,nvtx,osrt', '--sample=none', '--force-overwrite=true',
                   '--output', str(args.output / 'timeline')] + campaign
        report = args.output / 'timeline.nsys-rep'
    result = subprocess.run(command, text=True, capture_output=True)
    (args.output / 'profiler.stdout').write_text(result.stdout)
    (args.output / 'profiler.stderr').write_text(result.stderr)
    if result.returncode or not report.is_file():
        raise RuntimeError(f'profile collection failed ({result.returncode}): {result.stderr[-3000:]}')
    campaign_result = json.loads((args.output / 'campaign.json').read_text())
    if campaign_result.get('status') != 'ok' or campaign_result.get('device_uuid') != uuid:
        raise RuntimeError('profiled campaign failed validation or device identity')
    summary = {'schema': 'cgs.profile-result.v1', 'tool': c['tool'], 'device_uuid': uuid,
               'profiler_version': version, 'profiler_sha256': sha(Path(executable).resolve()),
               'report_sha256': sha(report), 'argv': command, 'workload_sha256': campaign_result['workload_sha256'],
               'scope': 'diagnostic profile; replay timing is not throughput evidence'}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()
