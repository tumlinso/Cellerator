#!/usr/bin/env python3
"""Register one frozen grouping-probe packet in an inactive study lane."""
import argparse
import fcntl
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path
import campaign


def sha(path):
    return campaign.file_sha256(path)


def register(args):
    campaign.LANES = campaign.validated_lane_root(args.lane_root)
    lane = campaign.lane_path(args.lane)
    values = [args.archive, args.binary, args.receipt, args.qualification_binary, args.qualification_receipt]
    if args.phase != 'qualification':
        if not args.qualification_proof:
            raise RuntimeError('timing requires the passed qualification proof')
        values.append(args.qualification_proof)
    elif args.qualification_proof:
        raise RuntimeError('qualification cannot depend on a later proof')
    if any(p.is_symlink() or not p.is_file() for p in values):
        raise RuntimeError('registration inputs must be regular files')
    if sha(args.archive) != args.archive_sha256:
        raise RuntimeError('archive differs from declared identity')
    for binary, receipt in ((args.binary, args.receipt), (args.qualification_binary, args.qualification_receipt)):
        data = json.loads(receipt.read_text())
        if data.get('schema') != 'cgs.build.v1' or data.get('archive_sha256') != args.archive_sha256 or data.get('binary_sha256') != sha(binary):
            raise RuntimeError('build receipt does not bind archive and binary')
    if not args.uuid.startswith('GPU-') or not args.resource_id.endswith(':' + args.uuid):
        raise RuntimeError('Scheduler UUID and resource do not match')
    with tarfile.open(args.archive, 'r:gz') as tar:
        manifest = json.load(tar.extractfile('SOURCE-MANIFEST.json'))['files']
    adapter = campaign.STUDY / 'scripts/probe_packet.py'
    if manifest.get('Cellerator/bench/chromatin_geometry/scripts/probe_packet.py', {}).get('sha256') != sha(adapter):
        raise RuntimeError('live adapter is not the reviewed archived adapter')
    with (campaign.LEDGER.parent / 'campaign-budget.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = campaign.load()
        if any(campaign.same_lane(x, args.lane) and x['state'] in ('reserving', 'unknown', 'submitted') for x in ledger['reservations']):
            raise RuntimeError('lane has an unresolved admission')
        cfg = json.loads((lane / '.autobench/config.json').read_text())
        status = campaign.command(['autobench', 'status', '--project', str(lane)])
        if cfg.get('active') is not False or status.get('paused') or status.get('attempts'):
            raise RuntimeError('probe registration requires a fresh inactive and unpaused lane')
        data = lane / '.autobench/data'
        data.mkdir(parents=True, exist_ok=True)
        scripts = lane / 'scripts'
        scripts.mkdir(exist_ok=True)
        shutil.copy2(adapter, scripts / 'probe_packet.py')
        for source, name in ((args.archive, 'closure.tar.gz'), (args.binary, 'frozen-binary'),
                             (args.receipt, 'frozen-build.json'), (args.qualification_binary, 'qualification-binary'),
                             (args.qualification_receipt, 'qualification-build.json')):
            shutil.copyfile(source, data / name)
        if args.qualification_proof:
            shutil.copyfile(args.qualification_proof, data / 'qualification-proof.json')
        build = ['python3', '{snapshot}/scripts/probe_packet.py', 'build',
                 '--archive', '{snapshot}/.autobench/data/closure.tar.gz', '--archive-sha256', args.archive_sha256,
                 '--binary', '{snapshot}/.autobench/data/frozen-binary', '--receipt', '{snapshot}/.autobench/data/frozen-build.json',
                 '--qualification-binary', '{snapshot}/.autobench/data/qualification-binary',
                 '--qualification-receipt', '{snapshot}/.autobench/data/qualification-build.json', '--build', '{build}']
        run = ['python3', '{snapshot}/scripts/probe_packet.py', 'run', '--phase', args.phase,
               '--build', '{build}', '--uuid', args.uuid, '--output', '{output}/probe-packet']
        paths = ['.autobench/data/closure.tar.gz', '.autobench/data/frozen-binary', '.autobench/data/frozen-build.json',
                 '.autobench/data/qualification-binary', '.autobench/data/qualification-build.json']
        if args.phase != 'qualification':
            paths.append('.autobench/data/qualification-proof.json')
            run.extend(['--qualification-proof', '{snapshot}/.autobench/data/qualification-proof.json', '--suite', 'kda'])
        arr = lambda v: '[' + ', '.join(json.dumps(x) for x in v) + ']'
        kill, expected = (240, 200) if args.phase == 'qualification' else (90, 80)
        definition = f'''[benchmark]
name = "cgs-ste-{args.lane}"
revision = 1
[source]
include = ["scripts/probe_packet.py"]
[data]
paths = {arr(paths)}
[build]
argv = {arr(build)}
expected_seconds = 60
kill_seconds = 240
resources = {{ cpu_cores = 2, host_memory_bytes = 2147483648 }}
[[packet]]
name = "grouping-{args.phase}"
kind = "correctness"
argv = {arr(run)}
expected_seconds = {expected}
kill_seconds = {kill}
resources = {{ tags = ["gpu.exclusive"], scheduler = [{{kind = "cpu", quantities = {{cores = 2}}, mode = "shared"}}, {{kind = "memory", quantities = {{bytes = 2147483648}}, mode = "shared"}}, {{kind = "gpu", quantities = {{memory_bytes = 2147483648}}, mode = "exclusive", selectors = {{resource_id = {json.dumps(args.resource_id)}}}}}] }}
'''
        target = lane / 'registration.toml'
        target.write_text(definition)
        result = subprocess.run(['autobench', 'register', '--project', str(lane), '--file', str(target)], text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(f'registration failed: {result.stderr}\n{result.stdout}')
        print(json.dumps({'lane': args.lane, 'phase': args.phase, 'registration': result.stdout.strip(),
                          'adapter_sha256': sha(adapter), 'qualification_proof_sha256': sha(args.qualification_proof) if args.qualification_proof else None}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lane-root', type=Path, required=True)
    p.add_argument('--lane', type=int, required=True)
    p.add_argument('--phase', choices=('qualification','r1','r16'), required=True)
    p.add_argument('--uuid', required=True)
    p.add_argument('--resource-id', required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--archive-sha256', required=True)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--qualification-binary', type=Path, required=True)
    p.add_argument('--qualification-receipt', type=Path, required=True)
    p.add_argument('--qualification-proof', type=Path)
    register(p.parse_args())


if __name__ == '__main__':
    main()
