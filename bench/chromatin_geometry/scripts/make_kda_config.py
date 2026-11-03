#!/usr/bin/env python3
"""Bind a KDA experiment to a passed full-chain triage and interference record."""
import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def receipt(job_id, required_uuid=None):
    path = STUDY / 'runs/receipts' / (job_id + '.json')
    data = json.loads(path.read_text())
    job = data.get('job', data)
    if job.get('id') != job_id or not job.get('cleanup_verified'):
        raise RuntimeError('job identity or verified cleanup is missing')
    attempt = job.get('attempt', {})
    alloc = [a.get('resource_id', '').split(':')[-1]
             for a in attempt.get('execution_context', {}).get('allocations', [])
             if a.get('resource_id', '').startswith('gpu:')]
    if len(alloc) != 1 or (required_uuid is not None and alloc[0] != required_uuid):
        raise RuntimeError('receipt does not bind the expected single GPU UUID')
    return sha(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--triage-job', required=True)
    p.add_argument('--match-group', required=True)
    p.add_argument('--incumbent', required=True)
    p.add_argument('--candidate', action='append', required=True)
    p.add_argument('--interference-evidence', type=Path, required=True)
    p.add_argument('--serialized-policy', type=Path)
    p.add_argument('--contract-freeze', type=Path, required=True)
    p.add_argument('--compare-updates', action='store_true', help='collect a same-packet rebuild/incremental ablation')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    packet = STUDY / 'runs/packets' / args.triage_job
    metrics = json.loads((packet / 'metrics.json').read_text())
    summary = json.loads((packet / 'triage/summary.json').read_text())
    if (metrics.get('scheduler_job_id') != args.triage_job or metrics.get('suite') != 'triage' or
            metrics.get('success') is not True or summary.get('status') != 'pass' or
            summary.get('device_uuid') != metrics.get('uuid')):
        raise RuntimeError('triage packet is not a passed, identity-bound full-chain check')
    triage_receipt = receipt(args.triage_job, metrics['uuid'])
    archive_sha = metrics['archive_sha256']
    archive = STUDY / 'runs/closures' / (archive_sha + '.tar.gz')
    binary = STUDY / 'runs/builds' / metrics['binary_sha256'] / 'binary'
    if not archive.is_file() or sha(archive) != archive_sha or not binary.is_file() or sha(binary) != metrics['binary_sha256']:
        raise RuntimeError('triage source archive or binary is unavailable')
    frozen = json.loads(args.contract_freeze.read_text())
    if frozen.get('format') != 'cgs.contract-freeze.v1':
        raise RuntimeError('workload and validator freeze has the wrong format')
    with tarfile.open(archive, 'r:gz') as tar:
        manifest = json.load(tar.extractfile('SOURCE-MANIFEST.json'))['files']
    for item in frozen['source_files']:
        path = item['path']
        if path.startswith('bench/chromatin_geometry/data/cache/'):
            if path.endswith('/manifest.json'):
                continue  # replay manifest is verified separately for replay cases below
            raise RuntimeError('unexpected frozen cache dependency')
        archived = manifest.get('Cellerator/' + path)
        if not archived or archived['sha256'] != item['sha256']:
            raise RuntimeError(f'qualified archive differs from frozen contract source: {path}')
    rows = [x for x in summary['records'] if x['match_group'] == args.match_group]
    arms = {x['backend'] for x in rows}
    if not set([args.incumbent, *args.candidate]) <= arms:
        raise RuntimeError('requested backends did not all pass this matched triage group')
    runs = [json.loads((packet / 'triage' / f"run-{x['run']:02d}.json").read_text()) for x in rows]
    digests = {x['workload_sha256'] for x in runs}
    if len(digests) != 1 or any(x.get('accuracy_accepted') is not True for x in runs):
        raise RuntimeError('triage arms have unmatched workloads or validation')
    first = runs[0]
    if args.compare_updates and (first['gate'] == 'ste' or first['condition_update'] not in ('incremental', 'rebuild')):
        raise RuntimeError('paired condition-update ablation requires a non-STE changed-input scenario')
    fields = ('case', 'gate', 'body', 'precision', 'condition_update', 'tile', 'stages', 'input', 'seed')
    if any(any(row[k] != first[k] for k in fields) for row in runs):
        raise RuntimeError('triage scenario metadata differs across matched arms')
    evidence = json.loads(args.interference_evidence.read_text())
    if evidence.get("schema") == "cgs.serialized-fallback.v1":
        device = evidence.get("device_uuid")
        expected = {"source_archive_sha256": archive_sha, "binary_sha256": metrics["binary_sha256"],
                    "workload_sha256": next(iter(digests)), "triage_job_id": args.triage_job,
                    "controller_serialization_required": True, "all_device_telemetry_required": True}
        if device != metrics["uuid"] or any(evidence.get(k) != v for k, v in expected.items()):
            raise RuntimeError("serialized fallback request differs from the qualified workload")
        evidence_receipts = []
    else:
        device = evidence.get('device_uuid')
        if (evidence.get('schema') != 'cgs.interference.v1' or evidence.get('status') not in ('passed', 'serialized') or
                evidence.get('source_archive_sha256') != archive_sha or evidence.get('binary_sha256') != metrics['binary_sha256'] or
                evidence.get('workload_sha256') != next(iter(digests)) or not isinstance(device, str)):
            raise RuntimeError('interference evidence does not bind the qualified source and device')
        identity = evidence.get('case_identity', {})
        if any(identity.get(k) != first[k] for k in ('case', 'gate', 'body', 'precision', 'condition_update', 'tile', 'stages', 'seed', 'input')):
            raise RuntimeError('interference evidence belongs to another workload or cost boundary')
        evidence_receipts = []
        all_ids = []
        for key in ('isolated_receipts', 'concurrent_receipts'):
            ids = evidence.get(key)
            if not isinstance(ids, list) or len(ids) < 2:
                raise RuntimeError('interference evidence requires repeated isolated and concurrent receipts')
            evidence_receipts.extend({'job_id': x, 'receipt_sha256': receipt(x, device)} for x in ids)
            all_ids.extend(ids)
        if len(set(all_ids)) != len(all_ids):
            raise RuntimeError('interference receipt IDs must be unique across both conditions')
        partner_ids = evidence.get('concurrent_partner_receipts')
        if not isinstance(partner_ids, list) or len(partner_ids) != len(evidence['concurrent_receipts']):
            raise RuntimeError('concurrent interference must include one distinct partner receipt per repetition')
        if len(set(all_ids + partner_ids)) != len(all_ids) + len(partner_ids):
            raise RuntimeError('interference receipt IDs are duplicated across participant sets')
        for job in partner_ids:
            evidence_receipts.append({'job_id': job, 'receipt_sha256': receipt(job)})
        # Recompute the analysis from raw packets and the analyzer frozen inside
        # the qualified closure. A hand-authored status flag is not sufficient.
        isolated = [STUDY / 'runs/packets' / job / 'interference/interference-run.json'
                    for job in evidence['isolated_receipts']]
        concurrent = [STUDY / 'runs/packets' / job / 'interference/interference-run.json'
                      for job in evidence['concurrent_receipts'] + partner_ids]
        config_path = STUDY / 'runs/packets' / evidence['isolated_receipts'][0] / 'experiment.json'
        if not config_path.is_file() or sha(config_path) != evidence.get('config_sha256'):
            raise RuntimeError('retained interference config does not match analysis identity')
        with tempfile.TemporaryDirectory(prefix='cgs-interference-recheck-') as tmp:
            tmp = Path(tmp)
            with tarfile.open(archive, 'r:gz') as tar:
                member = tar.getmember('Cellerator/bench/chromatin_geometry/scripts/interference.py')
                if not member.isfile():
                    raise RuntimeError('frozen interference analyzer is not a regular source file')
                (tmp / 'interference.py').write_bytes(tar.extractfile(member).read())
            output = tmp / 'analysis.json'
            command = [sys.executable, str(tmp / 'interference.py'), 'analyze', '--config', str(config_path),
                       '--receipts-root', str(STUDY / 'runs/receipts'), '--closures-root', str(STUDY / 'runs/closures'),
                       '--builds-root', str(STUDY / 'runs/builds'), '--output', str(output)]
            for path in isolated:
                command += ['--isolated', str(path)]
            for path in concurrent:
                command += ['--concurrent', str(path)]
            if evidence['status'] == 'serialized':
                if args.serialized_policy is None:
                    raise RuntimeError('serialized interference requires its original policy for independent recomputation')
                command += ['--serialized-policy', str(args.serialized_policy)]
            proc = subprocess.run(command, text=True, capture_output=True)
            if proc.returncode or not output.is_file() or json.loads(output.read_text()) != evidence:
                raise RuntimeError(f'interference evidence failed independent reconstruction: {proc.stderr[-1200:]}')
    case = {'id': first['case'], 'weight': 1.0,
            'input': {'campaign_workload_sha256': next(iter(digests)), 'seed': first['seed'],
                      'source_archive_sha256': archive_sha, 'triage_job_id': args.triage_job},
            'parameters': {k: first[k] for k in ('gate', 'body', 'precision', 'condition_update', 'tile', 'stages')}}
    replay_inputs = []
    if first['input'] != 'synthetic':
        replay_names = {x.get('replay') for x in rows}
        replay_hashes = {x.get('replay_sha256') for x in rows}
        if len(replay_names) != 1 or len(replay_hashes) != 1 or None in replay_names or None in replay_hashes:
            raise RuntimeError('replay triage lacks a consistent input path and SHA-256')
        replay_name, replay_sha = next(iter(replay_names)), next(iter(replay_hashes))
        replay_inputs = [replay_name, 'pbmc-atac-cgatac01/cells.tsv',
                         'pbmc-atac-cgatac01/features.tsv', 'pbmc-atac-cgatac01/manifest.json',
                         'pbmc_unsorted_10k_filtered_feature_bc_matrix.h5']
        with tarfile.open(archive, 'r:gz') as tar:
            manifest = json.load(tar.extractfile('SOURCE-MANIFEST.json'))
        entries = manifest['files']
        if any('replay/' + name not in entries for name in replay_inputs) or entries['replay/' + replay_name]['sha256'] != replay_sha:
            raise RuntimeError('source archive does not bind all replay inputs and sidecars')
        case['parameters']['replay'] = replay_name
        case['input']['replay_sha256'] = replay_sha
    result = {'schema': 'cgs.kda-config.v1', 'task_id': 'cgs-' + args.match_group,
              'source_archive_sha256': archive_sha, 'binary_sha256': metrics['binary_sha256'], 'incumbent': args.incumbent,
              'candidates': args.candidate, 'cases': [case], 'trusted_replay_inputs': replay_inputs,
              'contract_freeze_sha256': sha(args.contract_freeze),
              'interference_attestation': {'sha256': sha(args.interference_evidence), 'device_uuid': device,
                                           'receipt_identities': evidence_receipts},
              'triage': {'job_id': args.triage_job, 'receipt_sha256': triage_receipt,
                         'packet_metrics_sha256': sha(packet / 'metrics.json'),
                         'source_archive_sha256': archive_sha, 'binary_sha256': metrics['binary_sha256']},
              'selection_status': 'preliminary; strongest control and >5 percent latency gate remain independently checked'}
    if args.compare_updates:
        result['supplemental_updates'] = ['rebuild', 'incremental']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'output': str(args.output), 'sha256': sha(args.output)}))


if __name__ == '__main__':
    main()
