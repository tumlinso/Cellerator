#!/usr/bin/env python3
"""Create a bounded interference experiment from a passed triage arm."""
import argparse
import hashlib
import json
import math
import tarfile
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--triage-job', required=True)
    p.add_argument('--match-group', required=True)
    p.add_argument('--backend', required=True)
    p.add_argument('--primary-uuid', required=True)
    p.add_argument('--partner-uuid', required=True)
    p.add_argument('--contract-freeze', type=Path, required=True)
    p.add_argument('--repetitions', type=int, default=5)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.primary_uuid == args.partner_uuid or not all(x.startswith('GPU-') for x in (args.primary_uuid, args.partner_uuid)):
        raise RuntimeError('interference uses two distinct Scheduler GPU UUIDs')
    if not 2 <= args.repetitions <= 32:
        raise RuntimeError('interference repetitions must be 2 through 32')
    packet = STUDY / 'runs/packets' / args.triage_job
    metrics = json.loads((packet / 'metrics.json').read_text())
    summary = json.loads((packet / 'triage/summary.json').read_text())
    response = json.loads((STUDY / 'runs/receipts' / (args.triage_job + '.json')).read_text())
    job = response.get('job', response)
    if (metrics.get('scheduler_job_id') != args.triage_job or metrics.get('suite') != 'triage' or
            metrics.get('success') is not True or summary.get('status') != 'pass' or
            job.get('id') != args.triage_job or not job.get('cleanup_verified') or
            job.get('attempt', {}).get('attempt_id') != metrics.get('scheduler_attempt_id')):
        raise RuntimeError('interference requires a passed, receipt-bound triage')
    rows = [x for x in summary['records'] if x['match_group'] == args.match_group and x['backend'] == args.backend]
    if len(rows) != 1:
        raise RuntimeError('triage does not uniquely qualify the requested case and backend')
    triage = rows[0]
    raw = json.loads((packet / 'triage' / f"run-{triage['run']:02d}.json").read_text())
    if (raw.get('workload_sha256') != triage.get('workload_sha256') or raw.get('accuracy_accepted') is not True or
            not math.isclose(raw['cold_execution_preparation_ms'],
                             raw['replay_input_preparation_ms'] + raw['host_layout_preparation_ms'] + raw['cold_device_preparation_ms'],
                             rel_tol=1e-9, abs_tol=1e-8)):
        raise RuntimeError('triage lacks the complete current cost boundary')
    archive_sha, binary_sha = metrics['archive_sha256'], metrics['binary_sha256']
    archive = STUDY / 'runs/closures' / (archive_sha + '.tar.gz')
    binary = STUDY / 'runs/builds' / binary_sha / 'binary'
    if sha(archive) != archive_sha or sha(binary) != binary_sha:
        raise RuntimeError('triage closure or binary integrity failed')
    frozen = json.loads(args.contract_freeze.read_text())
    if frozen.get('format') != 'cgs.contract-freeze.v1':
        raise RuntimeError('workload and validator contract is not frozen')
    with tarfile.open(archive, 'r:gz') as tar:
        files = json.load(tar.extractfile('SOURCE-MANIFEST.json'))['files']
    for item in frozen['source_files']:
        path = item['path']
        if path.startswith('bench/chromatin_geometry/data/cache/'):
            continue
        archived = files.get('Cellerator/' + path)
        if not archived or archived['sha256'] != item['sha256']:
            raise RuntimeError(f'qualified source differs from frozen workload contract: {path}')
    config = {'schema': 'cgs.interference-config.v1', 'source_archive_sha256': archive_sha,
              'binary_sha256': binary_sha, 'workload_sha256': raw['workload_sha256'],
              **{k: raw[k] for k in ('case', 'backend', 'gate', 'body', 'precision',
                                      'condition_update', 'seed', 'tile', 'stages', 'input')},
              'primary_uuid': args.primary_uuid, 'concurrent_uuids': [args.primary_uuid, args.partner_uuid],
              'independent_repetitions': args.repetitions, 'warmups': 1,
              'contract_freeze_sha256': sha(args.contract_freeze),
              'triage_job_id': args.triage_job}
    if raw['input'] != 'synthetic':
        if not triage.get('replay') or not triage.get('replay_sha256'):
            raise RuntimeError('replay triage lacks its captured input identity')
        config.update(replay_path='replay/' + triage['replay'], replay_sha256=triage['replay_sha256'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'output': str(args.output), 'sha256': sha(args.output), 'repetitions': args.repetitions}))


if __name__ == '__main__':
    main()
