#!/usr/bin/env python3
"""Register one complete-chain sanitizer packet in an inactive study lane."""
import argparse
import fcntl
import json
import shutil
import subprocess
from pathlib import Path

import campaign


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lane', type=int, required=True)
    p.add_argument('--uuid', required=True)
    p.add_argument('--resource-id', required=True)
    p.add_argument('--mode', required=True, choices=('memcheck', 'racecheck', 'initcheck', 'synccheck'))
    p.add_argument('--backend', choices=('cohort-shared', 'cohort-direct-two-tile'), default='cohort-shared')
    p.add_argument('--revision', type=int, required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--archive-sha256', required=True)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    lane = campaign.lane_path(a.lane)
    with (campaign.LEDGER.parent / 'campaign-budget.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = campaign.load()
        if any(r['lane'] == a.lane and r['state'] in ('reserving', 'unknown', 'submitted') for r in ledger['reservations']):
            raise RuntimeError('lane has an unresolved GPU reservation')
        if not a.uuid.startswith('GPU-') or not a.resource_id.endswith(':' + a.uuid):
            raise RuntimeError('GPU UUID and resource identity disagree')
        receipt = json.loads(a.receipt.read_text())
        if (campaign.file_sha256(a.archive) != a.archive_sha256 or receipt.get('archive_sha256') != a.archive_sha256 or
                campaign.file_sha256(a.binary) != receipt.get('binary_sha256')):
            raise RuntimeError('qualified binary, build receipt and source archive disagree')
        config = json.loads((lane / '.autobench/config.json').read_text())
        if config.get('active') is not False:
            raise RuntimeError('study lane is not inactive')
        campaign.preserve_archive(a.archive)
        data = lane / '.autobench/data'
        data.mkdir(parents=True, exist_ok=True)
        (lane / 'scripts').mkdir(exist_ok=True)
        shutil.copy2(campaign.STUDY / 'scripts/sanitize_chain.py', lane / 'scripts/sanitize_chain.py')
        shutil.copyfile(a.archive, data / 'closure.tar.gz')
        shutil.copyfile(a.binary, data / 'frozen-binary')
        shutil.copyfile(a.receipt, data / 'frozen-build.json')
        build = ['python3', '{snapshot}/scripts/sanitize_chain.py', 'build',
                 '--binary', '{snapshot}/.autobench/data/frozen-binary', '--receipt', '{snapshot}/.autobench/data/frozen-build.json',
                 '--archive', '{snapshot}/.autobench/data/closure.tar.gz', '--archive-sha256', a.archive_sha256, '--build', '{build}']
        run = ['python3', '{snapshot}/scripts/sanitize_chain.py', 'run', '--build', '{build}', '--uuid', a.uuid,
               '--sanitizer', a.mode, '--backend', a.backend, '--output', '{output}/chain']
        arr = lambda x: '[' + ', '.join(json.dumps(y) for y in x) + ']'
        definition = f'''[benchmark]
name = "cgs-ste-{a.lane}"
revision = {a.revision}
[source]
include = ["scripts/sanitize_chain.py"]
[data]
paths = [".autobench/data/closure.tar.gz", ".autobench/data/frozen-binary", ".autobench/data/frozen-build.json"]
[build]
argv = {arr(build)}
expected_seconds = 30
kill_seconds = 300
resources = {{ cpu_cores = 2, host_memory_bytes = 2147483648 }}
[[packet]]
name = "chain-{a.mode}"
kind = "correctness"
argv = {arr(run)}
expected_seconds = 300
kill_seconds = 600
resources = {{ tags = ["gpu.exclusive"], scheduler = [{{kind = "cpu", quantities = {{cores = 1}}, mode = "shared"}}, {{kind = "memory", quantities = {{bytes = 2147483648}}, mode = "shared"}}, {{kind = "gpu", quantities = {{memory_bytes = 2147483648}}, mode = "exclusive", selectors = {{resource_id = {json.dumps(a.resource_id)}}}}}] }}
'''
        path = lane / 'registration.toml'
        path.write_text(definition)
        subprocess.run(['autobench', 'register', '--project', str(lane), '--file', str(path), '--replace-revision'], check=True)


if __name__ == '__main__':
    main()
