#!/usr/bin/env python3
"""Export compact, verified joins of packet identities and Scheduler receipts."""
import argparse
import hashlib
import json
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ledger = json.loads((STUDY / 'runs/campaign-budget.json').read_text())
    settled = {r['job_id']: r for r in ledger['reservations'] if r['state'] == 'settled'}
    records = []
    failures = {}
    for metrics_file in sorted((STUDY / 'runs/packets').glob('*/metrics.json')):
        metrics = json.loads(metrics_file.read_text())
        if metrics.get('schema') != 'cgs.packet.v1':
            continue
        if metrics.get('suite') in ('kda', 'profile', 'interference'):
            continue
        job_id = metrics['scheduler_job_id']
        if job_id not in settled:
            raise RuntimeError(f'packet lacks settled budget reservation: {job_id}')
        row = settled[job_id]
        receipt_file = STUDY / row['receipt']
        response = json.loads(receipt_file.read_text())
        job = response.get('job', response)
        attempt = job.get('attempt', {})
        if (job.get('id') != job_id or not job.get('cleanup_verified') or
                attempt.get('attempt_id') != metrics['scheduler_attempt_id'] or
                row['gpu_resource_id'].split(':')[-1] != metrics['uuid']):
            raise RuntimeError(f'identity or cleanup mismatch: {job_id}')
        output = metrics_file.parent / 'test.stdout'
        worker_result = metrics_file.parent / 'result.json'
        if not output.is_file() or not worker_result.is_file():
            raise RuntimeError(f'packet log or worker result missing: {job_id}')
        raw = output.read_text()
        worker = json.loads(worker_result.read_text())
        clean_summary = metrics['sanitizer'] == 'none' or '0 errors' in raw or '0 hazards displayed (0 errors' in raw
        passed = (metrics.get('success') is True and metrics.get('returncode') == 0 and
                  job.get('state') == 'succeeded' and worker.get('status') == 'succeeded' and clean_summary)
        if not passed:
            stderr = metrics_file.parent / 'test.stderr'
            failures[job_id] = {'test': metrics.get('target'), 'source_archive_sha256': metrics.get('archive_sha256'),
                                'binary_sha256': metrics.get('binary_sha256'), 'uuid': metrics.get('uuid'),
                                'scheduler_job_id': job_id, 'returncode': metrics.get('returncode'),
                                'worker_status': worker.get('status'), 'clean_sanitizer_summary': clean_summary,
                                'stderr_sha256': sha(stderr) if stderr.is_file() else None,
                                'receipt_sha256': sha(receipt_file), 'raw_evidence': str(metrics_file.parent.relative_to(STUDY)),
                                'charged_seconds': row['charged_seconds']}
            continue
        build_root = STUDY / 'runs/builds' / metrics['binary_sha256']
        binary = build_root / 'binary'
        if not binary.is_file() or sha(binary) != metrics['binary_sha256']:
            raise RuntimeError(f'preserved binary identity unavailable: {job_id}')
        matching = []
        for candidate in build_root.glob('*.json'):
            build = json.loads(candidate.read_text())
            if (candidate.stem == sha(candidate) and build.get('archive_sha256') == metrics['archive_sha256']
                    and build.get('binary_sha256') == metrics['binary_sha256']):
                matching.append(candidate)
        if not matching:
            raise RuntimeError(f'preserved build receipt unavailable: {job_id}')
        build_file = sorted(matching)[0]
        archive = STUDY / 'runs/closures' / (metrics['archive_sha256'] + '.tar.gz')
        if not archive.is_file() or sha(archive) != metrics['archive_sha256']:
            raise RuntimeError(f'preserved source archive unavailable: {job_id}')
        records.append({'test': metrics.get('target', 'standalone-ste'), 'sanitizer': metrics['sanitizer'],
                        'fixture': 'embedded deterministic synthetic', 'source_archive_sha256': metrics['archive_sha256'],
                        'binary_sha256': metrics['binary_sha256'], 'uuid': metrics['uuid'],
                        'scheduler_job_id': job_id, 'scheduler_attempt_id': metrics['scheduler_attempt_id'],
                        'scheduler_receipt_sha256': sha(receipt_file), 'worker_result_sha256': sha(worker_result),
                        'build_receipt_sha256': sha(build_file),
                        'raw_build_receipt': str(build_file.relative_to(STUDY)),
                        'raw_binary': str(binary.relative_to(STUDY)),
                        'raw_source_archive': str(archive.relative_to(STUDY)),
                        'test_stdout_sha256': sha(output), 'cleanup_verified': True,
                        'charged_seconds': row['charged_seconds'],
                        'raw_evidence': str(metrics_file.parent.relative_to(STUDY)),
                        'raw_receipt': row['receipt']})
    for path in sorted((STUDY / 'runs/failures').glob('*/metrics.json')):
        metrics = json.loads(path.read_text())
        row = settled.get(metrics.get('scheduler_job_id'))
        if not row or metrics.get('success') is not False:
            raise RuntimeError('failed packet missing settled reservation or failure status')
        receipt = STUDY / row['receipt']
        response = json.loads(receipt.read_text())
        job = response.get('job', response)
        if not job.get('cleanup_verified') or job.get('attempt', {}).get('attempt_id') != metrics.get('scheduler_attempt_id'):
            raise RuntimeError('failed packet receipt does not establish cleanup and attempt identity')
        failures[metrics['scheduler_job_id']] = {'test': metrics.get('target'), 'source_archive_sha256': metrics['archive_sha256'],
                         'binary_sha256': metrics['binary_sha256'], 'uuid': metrics['uuid'],
                         'scheduler_job_id': metrics['scheduler_job_id'], 'returncode': metrics['returncode'],
                         'worker_status': 'failed', 'stderr_sha256': sha(path.parent / 'test.stderr'),
                         'receipt_sha256': sha(receipt), 'raw_evidence': str(path.parent.relative_to(STUDY)),
                         'charged_seconds': row['charged_seconds']}
    for folder in sorted((STUDY / 'runs/packets').glob('*')):
        if not folder.is_dir() or (folder / 'metrics.json').exists():
            continue
        job_id = folder.name
        row = settled.get(job_id)
        result_file = folder / 'result.json'
        if not row or not result_file.is_file():
            raise RuntimeError('unclassified packet lacks a settled receipt or worker result')
        worker = json.loads(result_file.read_text())
        response = json.loads((STUDY / row['receipt']).read_text())
        job = response.get('job', response)
        if not job.get('cleanup_verified') or job.get('id') != job_id or worker.get('status') == 'succeeded':
            raise RuntimeError('packet without metrics has inconsistent worker or cleanup state')
        failures[job_id] = {'scheduler_job_id': job_id, 'worker_status': worker.get('status'),
                            'returncode': worker.get('returncode'), 'raw_evidence': str(folder.relative_to(STUDY)),
                            'receipt_sha256': sha(STUDY / row['receipt']), 'charged_seconds': row['charged_seconds']}
    result = {'schema': 'cgs.correctness-evidence.v1',
              'scope': 'only explicitly listed correctness fixtures; not performance or biological validation',
              'budget_seconds': ledger['budget_seconds'],
              'settled_gpu_seconds': sum(r['charged_seconds'] for r in ledger['reservations'] if r['state'] == 'settled'),
              'unsettled_reserved_gpu_seconds': sum(r['reserved_seconds'] for r in ledger['reservations']
                                                    if r['state'] in ('reserving', 'submitted', 'unknown')),
              'records': records, 'failures': list(failures.values())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'output': str(args.output), 'records': len(records),
                      'sha256': sha(args.output)}, sort_keys=True))


if __name__ == '__main__':
    main()
