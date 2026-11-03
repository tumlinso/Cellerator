#!/usr/bin/env python3
"""Register a triage packet using the exact retained cgs_study binary."""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import campaign


def triage_packet_argv(uuid: str) -> list[str]:
    return [
        'python3', '{snapshot}/scripts/triage_binary.py', 'run', '--build', '{build}',
        '--config', '{snapshot}/experiment.json', '--uuid', uuid,
        '--output', '{output}/triage-packet',
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--lane-root', type=Path, default=campaign.LANES)
    parser.add_argument('--lane', type=int, required=True)
    parser.add_argument('--uuid', required=True)
    parser.add_argument('--resource-id', required=True)
    parser.add_argument('--revision', type=int, default=1)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--archive-sha256', required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--expected', type=int, default=300)
    parser.add_argument('--kill', type=int, default=600)
    args = parser.parse_args()

    campaign.LANES = campaign.validated_lane_root(args.lane_root)
    lane = campaign.lane_path(args.lane)
    original_paths = (args.archive, args.binary, args.receipt, args.config)
    if any(value.is_symlink() for value in original_paths):
        raise RuntimeError('archive, binary, receipt and config must not be symlinks')
    archive, binary, receipt_path, config_path = (value.resolve(strict=True) for value in original_paths)
    if any(not value.is_file() for value in (archive, binary, receipt_path, config_path)):
        raise RuntimeError('archive, binary, receipt and config must be regular non-symlink files')
    if args.revision < 1 or args.kill < 2 or args.kill > 600 or args.expected >= args.kill:
        raise RuntimeError('registration revision or finite packet timeout is invalid')
    resource_parts = args.resource_id.split(':')
    if (not args.uuid.startswith('GPU-') or len(resource_parts) != 3 or resource_parts[0] != 'gpu' or
            resource_parts[2] != args.uuid):
        raise RuntimeError('GPU resource ID must bind the exact Scheduler UUID')
    if campaign.file_sha256(archive) != args.archive_sha256:
        raise RuntimeError('archive digest differs from requested source identity')
    build = json.loads(receipt_path.read_text())
    config = json.loads(config_path.read_text())
    if (build.get('schema') != 'cgs.build.v1' or build.get('target') != 'cgs_study' or
            build.get('archive_sha256') != args.archive_sha256 or
            campaign.file_sha256(binary) != build.get('binary_sha256')):
        raise RuntimeError('retained binary and build receipt do not match the source archive')
    if config.get('schema') != 'cgs.triage.v1' or not config.get('runs'):
        raise RuntimeError('triage config must contain a nonempty frozen cgs.triage.v1 matrix')
    with tarfile.open(archive, 'r:gz') as tar:
        manifest_file = tar.extractfile('SOURCE-MANIFEST.json')
        if manifest_file is None:
            raise RuntimeError('frozen source archive has no source manifest')
        manifest = json.load(manifest_file).get('files', {})
    frozen_triage = manifest.get('Cellerator/bench/chromatin_geometry/scripts/triage.py')
    if not frozen_triage:
        raise RuntimeError('frozen archive does not contain the original triage implementation')

    with (campaign.LEDGER.parent / 'campaign-budget.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = campaign.load()
        if any(campaign.same_lane(row, args.lane) and row['state'] in ('reserving', 'unknown', 'submitted')
               for row in ledger['reservations']):
            raise RuntimeError('lane has an unresolved shared-budget reservation')
        status = campaign.command(['autobench', 'status', '--project', str(lane)])
        enrolled = json.loads((lane / '.autobench/config.json').read_text())
        if enrolled.get('active') is not False or status.get('paused'):
            raise RuntimeError('triage registration requires an inactive, unpaused lane')
        attempts = status.get('attempts', [])
        if attempts and args.revision < 2:
            raise RuntimeError('used lane requires an explicit new registration revision')
        for attempt in attempts:
            reservations = [row for row in ledger['reservations'] if row.get('attempt_id') == attempt['id']]
            if (len(reservations) != 1 or reservations[0]['state'] not in ('cpu_build', 'settled') or
                    not attempt.get('job_id')):
                raise RuntimeError('previous attempt lacks an attributable settled reservation')
            result = asyncio.run(campaign.get_job(enrolled, attempt['job_id']))
            job = result.get('job', result)
            if (job.get('state') not in ('succeeded', 'completed', 'failed', 'cancelled', 'canceled',
                                          'timed_out', 'abandoned') or not job.get('cleanup_verified')):
                raise RuntimeError('previous lane job lacks terminal state and verified cleanup')

        campaign.preserve_archive(archive)
        data = lane / '.autobench/data'
        data.mkdir(parents=True, exist_ok=True)
        scripts = lane / 'scripts'
        scripts.mkdir(exist_ok=True)
        shutil.copy2(Path(__file__).with_name('triage_binary.py'), scripts / 'triage_binary.py')
        shutil.copyfile(archive, data / 'closure.tar.gz')
        shutil.copyfile(binary, data / 'frozen-binary')
        shutil.copyfile(receipt_path, data / 'frozen-build.json')
        shutil.copyfile(config_path, lane / 'experiment.json')

        build_command = [
            'python3', '{snapshot}/scripts/triage_binary.py', 'build',
            '--archive', '{snapshot}/.autobench/data/closure.tar.gz', '--archive-sha256', args.archive_sha256,
            '--binary', '{snapshot}/.autobench/data/frozen-binary',
            '--receipt', '{snapshot}/.autobench/data/frozen-build.json', '--build', '{build}',
        ]
        run_command = triage_packet_argv(args.uuid)
        array = lambda values: '[' + ', '.join(json.dumps(value) for value in values) + ']'
        name = f'cgs-ste-{args.lane}'
        definition = f'''[benchmark]
name = {json.dumps(name)}
revision = {args.revision}
[source]
include = ["scripts/triage_binary.py", "experiment.json"]
[data]
paths = [".autobench/data/closure.tar.gz", ".autobench/data/frozen-binary", ".autobench/data/frozen-build.json"]
[build]
argv = {array(build_command)}
expected_seconds = 60
kill_seconds = 240
resources = {{ cpu_cores = 2, host_memory_bytes = 2147483648 }}
[[packet]]
name = "full-chain-triage"
kind = "correctness"
argv = {array(run_command)}
expected_seconds = {args.expected}
kill_seconds = {args.kill}
resources = {{ tags = ["gpu.exclusive"], scheduler = [{{kind = "cpu", quantities = {{cores = 2}}, mode = "shared"}}, {{kind = "memory", quantities = {{bytes = 2147483648}}, mode = "shared"}}, {{kind = "gpu", quantities = {{memory_bytes = 2147483648}}, mode = "exclusive", selectors = {{resource_id = {json.dumps(args.resource_id)}}}}}] }}
'''
        path = lane / 'registration.toml'
        path.write_text(definition)
        command = ['autobench', 'register', '--project', str(lane), '--file', str(path)]
        if args.revision > 1:
            command.append('--replace-revision')
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(f'Autobench registration failed: {result.stderr}\n{result.stdout}')
        print(json.dumps({'lane': args.lane, 'lane_root': str(campaign.LANES),
                          'revision': args.revision, 'registration': result.stdout.strip()}))


if __name__ == '__main__':
    main()
