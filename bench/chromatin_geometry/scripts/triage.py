#!/usr/bin/env python3
"""Cross-arm full-chain correctness triage; emitted timing is not evidence."""
import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--replay-root', type=Path)
    args = p.parse_args()
    uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not os.environ.get('AUTOBENCH_SCHEDULER_JOB_ID') or not uuid.startswith('GPU-'):
        raise RuntimeError('triage requires an admitted single-GPU study packet')
    config = json.loads(args.config.read_text())
    if config.get('schema') != 'cgs.triage.v1' or not config.get('runs'):
        raise RuntimeError('invalid triage input')
    if any(count < 2 for count in Counter(x.get('match_group') for x in config['runs']).values()):
        raise RuntimeError('every triage group requires at least two matched arms')
    args.output.mkdir(parents=True, exist_ok=True)
    groups = {}
    records = []
    for i, run in enumerate(config['runs']):
        result = args.output / f'run-{i:02d}.json'
        argv = [str(args.binary), '--mode', 'campaign', '--case', run['case'],
                '--backend', run['backend'], '--gate', run['gate'], '--body', run['body'],
                '--precision', run['precision'], '--condition-update', run['condition_update'],
                '--tile', str(run.get('tile', 32)), '--stages', str(run.get('stages', 1)),
                '--warmups', '0', '--repeats', '1', '--output', str(result)]
        if run.get('replay'):
            if args.replay_root is None:
                raise RuntimeError('replay triage lacks an extracted replay root')
            root = args.replay_root.resolve(strict=True)
            replay = (root / run['replay']).resolve(strict=True)
            if not replay.is_relative_to(root) or not replay.is_file():
                raise RuntimeError('replay input escaped the frozen closure')
            hasher = hashlib.sha256()
            with replay.open('rb') as f:
                for block in iter(lambda: f.read(1024 * 1024), b''):
                    hasher.update(block)
            if hasher.hexdigest() != run.get('replay_sha256'):
                raise RuntimeError('replay input hash differs from frozen triage configuration')
            argv.extend(['--replay', str(replay)])
        proc = subprocess.run(argv, text=True, capture_output=True)
        (args.output / f'run-{i:02d}.stdout').write_text(proc.stdout)
        (args.output / f'run-{i:02d}.stderr').write_text(proc.stderr)
        if proc.returncode:
            raise RuntimeError(f"triage {i} ({run['backend']}, {run['gate']}, {run['body']}) failed: {proc.stderr[-3000:]}")
        data = json.loads(result.read_text())
        if (data.get('schema') != 'cgs.campaign.v1' or data.get('status') != 'ok' or
                data.get('accuracy_accepted') is not True or
                data.get('validation_scope') != 'first-and-last-complete-steps' or
                data.get('device_uuid') != uuid):
            raise RuntimeError(f'triage {i} has an invalid validation or device identity')
        for key in ('case', 'backend', 'gate', 'body', 'precision', 'condition_update'):
            if data.get(key) != run[key]:
                raise RuntimeError(f'triage {i} identity mismatch: {key}')
        digest = data.get('workload_sha256')
        if not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError(f'triage {i} missing workload digest')
        group = run['match_group']
        if group in groups and groups[group] != digest:
            raise RuntimeError(f'triage group {group} has unmatched inputs')
        groups[group] = digest
        records.append({'run': i, 'backend': run['backend'], 'match_group': group,
                        'workload_sha256': digest, 'validation': 'pass',
                        'replay': run.get('replay'), 'replay_sha256': run.get('replay_sha256')})
    summary = {'schema': 'cgs.triage-result.v1', 'status': 'pass', 'device_uuid': uuid,
               'scope': 'full-chain correctness triage only; per-run timings are not performance evidence',
               'records': records}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()
