#!/usr/bin/env python3
"""Explicit one-shot Autobench submissions with a shared fail-closed GPU budget."""
import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import time
import tomllib
import uuid
from datetime import datetime
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
LANES = STUDY / '.campaign-lanes'
LEDGER = STUDY / 'runs/campaign-budget.json'
BUDGET = 14400


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def command(argv):
    p = subprocess.run(argv, text=True, capture_output=True)
    if p.returncode:
        raise RuntimeError(f'command failed: {argv!r}\n{p.stderr}\n{p.stdout}')
    return json.loads(p.stdout)


def save(ledger):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    tmp = LEDGER.with_name(LEDGER.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(ledger, f, indent=2, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, LEDGER)


def load():
    if LEDGER.exists():
        ledger = json.loads(LEDGER.read_text())
        if ledger.get('budget_seconds') != BUDGET or ledger.get('schema') != 'cgs.budget.v1':
            raise RuntimeError('incompatible budget ledger')
        return ledger
    return {'schema': 'cgs.budget.v1', 'budget_seconds': BUDGET, 'reservations': []}


def charged(ledger):
    return sum(r['charged_seconds'] if 'charged_seconds' in r else r['reserved_seconds']
               for r in ledger['reservations'])


def lane_path(index):
    if index not in range(4):
        raise RuntimeError('lane must be 0 through 3')
    return LANES / str(index)


def validated_lane_root(path):
    """Keep alternate enrollments inside the study, including through symlinks."""
    root = Path(path).resolve()
    study = STUDY.resolve()
    if root == study or not root.is_relative_to(study):
        raise RuntimeError('lane root must be a directory inside the study')
    return root


def same_lane(row, index):
    # Historical rows predate lane-root tracking. Conservatively treat them as
    # belonging to any root with that lane until they are settled.
    return row['lane'] == index and row.get('lane_root', str(LANES)) == str(LANES)


def initialize(args):
    lane = lane_path(args.lane)
    lane.mkdir(parents=True, exist_ok=True)
    config = json.loads((STUDY / '.autobench/config.json').read_text())
    out = command(['autobench', 'init', '--project', str(lane), '--no-activate',
                   '--agent-instructions', 'skip', '--scheduler-url', config['scheduler_url'],
                   '--token-file', config['token_file'], '--retained-cap-gib', '4'])
    if out.get('active') is not False or not out.get('scheduler_access', {}).get('ready'):
        raise RuntimeError('lane enrollment not inactive and ready')
    scripts = lane / 'scripts'
    scripts.mkdir(exist_ok=True)
    shutil.copy2(STUDY / 'scripts/run_packet.py', scripts / 'run_packet.py')
    closure = lane / '.autobench/data/closure.tar.gz'
    closure.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.archive, closure)
    preserve_archive(Path(args.archive))
    print(json.dumps({'lane': args.lane, 'lane_root': str(LANES), 'inactive': True, 'archive': str(closure)}))


def register(args):
    lane = lane_path(args.lane)
    status = command(['autobench', 'status', '--project', str(lane)])
    if status['attempts']:
        if args.revision < 2 or not args.archive:
            raise RuntimeError('an existing lane requires a new revision and source archive')
        ledger = load()
        config = json.loads((lane / '.autobench/config.json').read_text())
        for attempt in status['attempts']:
            found = [r for r in ledger['reservations'] if r.get('attempt_id') == attempt['id']]
            if len(found) != 1 or found[0]['state'] not in ('cpu_build', 'settled') or not attempt['job_id']:
                raise RuntimeError('previous attempt lacks an attributable settled reservation')
            response = asyncio.run(get_job(config, attempt['job_id']))
            job = response.get('job', response)
            if job.get('state') not in ('succeeded', 'completed', 'failed', 'cancelled', 'canceled', 'timed_out', 'abandoned') or not job.get('cleanup_verified'):
                raise RuntimeError('previous lane job has not completed with verified cleanup')
        if file_sha256(args.archive) != args.archive_sha256:
            raise RuntimeError('new archive hash does not match its declared identity')
        preserve_archive(Path(args.archive))
        lane_archive = lane / '.autobench/data/closure.tar.gz'
        if args.reuse_inputs:
            if file_sha256(lane_archive) != args.archive_sha256:
                raise RuntimeError('existing lane input differs from requested source archive')
        else:
            shutil.copyfile(args.archive, lane_archive)
    # Input reuse never reuses a stale execution wrapper. Bind the live wrapper
    # to the archived reviewed bytes before copying it into the lane snapshot.
    lane_archive = lane / '.autobench/data/closure.tar.gz'
    with tarfile.open(lane_archive, 'r:gz') as tar:
        manifest = json.load(tar.extractfile('SOURCE-MANIFEST.json'))['files']
    wrapper = STUDY / 'scripts/run_packet.py'
    archived = manifest.get('Cellerator/bench/chromatin_geometry/scripts/run_packet.py', {})
    if archived.get('sha256') != file_sha256(wrapper):
        raise RuntimeError('live execution wrapper differs from the frozen source archive')
    shutil.copy2(wrapper, lane / 'scripts/run_packet.py')
    if args.kill < 2 or args.kill > 600 or args.expected >= args.kill:
        raise RuntimeError('packet timing outside host or study limits')
    if not args.resource_id.endswith(':' + args.uuid) or not args.uuid.startswith('GPU-'):
        raise RuntimeError('resource ID and UUID disagree')
    if args.suite in ('kda', 'triage', 'profile', 'interference'):
        if not args.config or args.target != 'cgs_study' or args.sanitizer != 'none':
            raise RuntimeError('campaign packets require a frozen config, cgs_study target and no sanitizer')
        shutil.copy2(args.config, lane / 'experiment.json')
    if args.suite in ('interference', 'kda', 'profile'):
        if not args.frozen_binary or not args.frozen_build_receipt:
            raise RuntimeError('measured packets require a qualified binary and build receipt')
        if args.suite == 'interference' and not args.interference_mode:
            raise RuntimeError('interference packet requires its mode')
        build = json.loads(Path(args.frozen_build_receipt).read_text())
        config = json.loads(Path(args.config).read_text())
        if (build.get('archive_sha256') != args.archive_sha256 or
                file_sha256(args.frozen_binary) != build.get('binary_sha256') or
                config.get('source_archive_sha256') != args.archive_sha256 or
                config.get('binary_sha256') != build.get('binary_sha256')):
            raise RuntimeError('interference config, qualified binary and source archive disagree')
        if args.suite == 'interference' and args.interference_mode == 'concurrent' and not args.barrier_id:
            raise RuntimeError('concurrent interference requires a shared barrier id')
        shutil.copy2(args.frozen_binary, lane / '.autobench/data/frozen-binary')
        shutil.copy2(args.frozen_build_receipt, lane / '.autobench/data/frozen-build.json')
    if args.suite == 'kda':
        if not args.interference_evidence or not args.contract_freeze:
            raise RuntimeError('KDA confirmation requires frozen workload/validator and interference evidence')
        config = json.loads(Path(args.config).read_text())
        if config.get('source_archive_sha256') != args.archive_sha256:
            raise RuntimeError('KDA configuration must bind the qualified source archive')
        if file_sha256(args.interference_evidence) != config.get('interference_attestation', {}).get('sha256'):
            raise RuntimeError('interference evidence does not match experiment configuration')
        if file_sha256(args.contract_freeze) != config.get('contract_freeze_sha256'):
            raise RuntimeError('workload and validator freeze identity changed')
        shutil.copy2(args.interference_evidence, lane / 'interference-evidence.json')
        shutil.copy2(args.contract_freeze, lane / 'contract-freeze.json')
    # A single packet per lane means each reconcile can submit at most one GPU job.
    def arr(xs):
        return '[' + ', '.join(json.dumps(x) for x in xs) + ']'
    common = ['--archive', '{snapshot}/.autobench/data/closure.tar.gz',
              '--archive-sha256', args.archive_sha256, '--build', '{build}',
              '--suite', args.suite, '--target', args.target]
    if args.suite == 'kda':
        common += ['--kda-workspace', '{project}/runs/kda-r' + str(args.revision),
                   '--config', '{snapshot}/experiment.json']
    elif args.suite in ('triage', 'profile', 'interference'):
        common += ['--config', '{snapshot}/experiment.json']
    if args.suite in ('interference', 'kda', 'profile'):
        common += ['--frozen-binary', '{snapshot}/.autobench/data/frozen-binary',
                   '--frozen-build-receipt', '{snapshot}/.autobench/data/frozen-build.json']
        if args.suite == 'interference':
            common += ['--interference-mode', args.interference_mode, '--repetition', str(args.repetition)]
        if args.suite == 'interference' and args.interference_mode == 'concurrent':
            common += ['--shared-study-root', str(STUDY), '--barrier-id', args.barrier_id]
    build = ['python3', '{snapshot}/scripts/run_packet.py', 'build'] + common
    packet = ['python3', '{snapshot}/scripts/run_packet.py', 'ste-test'] + common + [
        '--output', '{output}', '--uuid', args.uuid, '--sanitizer', args.sanitizer]
    text = f'''[benchmark]
name = "cgs-ste-{args.lane}"
revision = {args.revision}
[source]
include = ["scripts/run_packet.py"{', "experiment.json"' if args.suite in ('kda', 'triage', 'profile', 'interference') else ''}{', "interference-evidence.json", "contract-freeze.json"' if args.suite == 'kda' else ''}]
[data]
paths = [".autobench/data/closure.tar.gz"{', ".autobench/data/frozen-binary", ".autobench/data/frozen-build.json"' if args.suite in ('interference', 'kda', 'profile') else ''}]
[build]
argv = {arr(build)}
expected_seconds = 120
kill_seconds = 300
resources = {{ cpu_cores = 2, host_memory_bytes = 2147483648 }}
[[packet]]
name = "ste-{args.sanitizer}"
kind = "correctness"
argv = {arr(packet)}
expected_seconds = {args.expected}
kill_seconds = {args.kill}
resources = {{ tags = ["gpu.exclusive"], scheduler = [{{kind = "cpu", quantities = {{cores = {2 if args.suite == 'kda' else 1}}}, mode = "shared"}}, {{kind = "memory", quantities = {{bytes = 2147483648}}, mode = "shared"}}, {{kind = "gpu", quantities = {{memory_bytes = 2147483648}}, mode = "exclusive", selectors = {{resource_id = {json.dumps(args.resource_id)}}}}}] }}
'''
    definition = lane / 'registration.toml'
    definition.write_text(text)
    argv = ['autobench', 'register', '--project', str(lane), '--file', str(definition)]
    if args.revision > 1:
        argv.append('--replace-revision')
    result = subprocess.run(argv,
                            text=True, capture_output=True, check=True)
    print(json.dumps({'registration': result.stdout.strip()}))


def validate_serial_admission(ledger, serialized):
    """Fail closed while any study confirmation owns or may own a GPU."""
    unresolved = [r for r in ledger['reservations']
                  if r['state'] in ('reserving', 'unknown', 'submitted')]
    if any(r.get('serialized_confirmation') for r in unresolved):
        raise RuntimeError('serialized confirmation has an unresolved study GPU reservation')
    if serialized and unresolved:
        raise RuntimeError('serialized confirmation requires all study GPU reservations to be settled')


def step(args):
    lane = lane_path(args.lane)
    config = json.loads((lane / '.autobench/config.json').read_text())
    if config.get('active') is not False:
        raise RuntimeError('lane is not explicitly inactive')
    ledger = load()
    if any(same_lane(r, args.lane) and r['state'] in ('reserving', 'unknown', 'submitted')
           for r in ledger['reservations']):
        raise RuntimeError('lane has an unresolved reservation; settle before another reconcile')
    before = command(['autobench', 'status', '--project', str(lane)])
    if before.get('paused'):
        raise RuntimeError('lane is paused')
    definitions = list((lane / '.autobench/registry').glob('*.toml'))
    if len(definitions) != 1:
        raise RuntimeError('one and only one lane registration is required')
    definition = tomllib.loads(definitions[0].read_text())
    packets = definition.get('packet', [])
    if len(packets) != 1:
        raise RuntimeError('one and only one lane GPU packet is required')
    packet = packets[0]
    argv = packet.get('argv', [])
    serialized = any(argv[i:i + 2] == ['--suite', 'kda'] for i in range(len(argv) - 1))
    validate_serial_admission(ledger, serialized)
    required = packet.get('resources', {}).get('scheduler', [])
    gpus = [x for x in required if x.get('kind') == 'gpu']
    if len(gpus) != 1 or gpus[0].get('mode') != 'exclusive' or not gpus[0].get('selectors', {}).get('resource_id', '').startswith('gpu:'):
        raise RuntimeError('packet must bind exactly one exclusive Scheduler GPU resource')
    kill = packet.get('kill_seconds')
    if type(kill) is not int or not 1 < kill <= 600:
        raise RuntimeError('unsupported packet timeout')
    build_resources = definition.get('build', {}).get('resources', {})
    if build_resources.get('gpu_memory_bytes') or any(x.get('kind') == 'gpu' for x in build_resources.get('scheduler', [])):
        raise RuntimeError('build must not request GPU resources')
    reserve = kill + 60
    if reserve > 900 or charged(ledger) + reserve > BUDGET:
        raise RuntimeError('GPU budget cannot cover this submission')
    row = {'id': str(uuid.uuid4()), 'lane': args.lane, 'lane_root': str(LANES), 'state': 'reserving',
           'reserved_seconds': reserve, 'created': time.time(),
           'expected_gpu_resource_id': gpus[0]['selectors']['resource_id'],
           'serialized_confirmation': serialized}
    ledger['reservations'].append(row)
    save(ledger)
    try:
        result = command(['autobench', 'reconcile', '--project', str(lane)])
        after = command(['autobench', 'status', '--project', str(lane)])
        added = [x for x in after['attempts'] if x['id'] not in {a['id'] for a in before['attempts']}]
        row['reconcile'] = result
        if result.get('submitted') == 0 and not added:
            row.update(state='no_submission', charged_seconds=0)
        elif result.get('submitted') == 1 and len(added) == 1 and added[0].get('job_id'):
            row.update(state='submitted', attempt_id=added[0]['id'], job_id=added[0]['job_id'],
                       kind=result.get('kind'))
            if result.get('kind') == 'build':
                row.update(state='cpu_build', charged_seconds=0)
        else:
            row.update(state='unknown', attempts=added)
    except Exception as exc:
        row.update(state='unknown', error=str(exc))
        save(ledger)
        raise
    save(ledger)
    print(json.dumps(row, sort_keys=True))


def stamp(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()


async def get_job(config, job_id):
    from device_scheduler_mcp.client import DeviceSchedulerClient
    client = DeviceSchedulerClient(config['scheduler_url'], config['token_file'])
    return await client.call('jobs.get', {'job_id': job_id})


def preserve_archive(path):
    digest = file_sha256(path)
    destination = STUDY / 'runs/closures' / (digest + '.tar.gz')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copyfile(path, destination)
    if file_sha256(destination) != digest:
        raise RuntimeError('preserved source archive failed integrity check')


def preserve_build(staged):
    metrics_file = staged / 'metrics.json'
    if not metrics_file.is_file():
        metrics_file = staged / 'chain/metrics.json'
    if not metrics_file.is_file():
        return None
    metrics = json.loads(metrics_file.read_text())
    explicit_binary = metrics.get('binary_path')
    binaries = ([Path(explicit_binary)] if explicit_binary else
                [Path(a) for a in metrics.get('argv', []) if '/cgs-' in a or '/cgs_' in a])
    if len(binaries) != 1 or not binaries[0].is_file() or file_sha256(binaries[0]) != metrics.get('binary_sha256'):
        raise RuntimeError('cannot preserve packet binary with its declared identity')
    binary = binaries[0]
    root = binary.parent.parent if binary.parent.name == 'cmake' else binary.parent
    receipt = root / 'build.json'
    build = json.loads(receipt.read_text())
    if build.get('archive_sha256') != metrics.get('archive_sha256') or build.get('binary_sha256') != metrics['binary_sha256']:
        raise RuntimeError('build receipt differs from packet identities')
    destination = STUDY / 'runs/builds' / metrics['binary_sha256']
    destination.mkdir(parents=True, exist_ok=True)
    saved_binary = destination / 'binary'
    if not saved_binary.exists():
        shutil.copy2(binary, saved_binary)
    if file_sha256(saved_binary) != metrics['binary_sha256']:
        raise RuntimeError('preserved binary failed integrity check')
    os.chmod(saved_binary, binary.stat().st_mode & 0o777)
    receipt_destination = destination / (file_sha256(receipt) + '.json')
    if not receipt_destination.exists():
        shutil.copyfile(receipt, receipt_destination)
    return str(receipt_destination.relative_to(STUDY))


def settle(args):
    ledger = load()
    rows = [r for r in ledger['reservations'] if same_lane(r, args.lane) and r['state'] == 'submitted']
    if len(rows) != 1:
        raise RuntimeError('lane has no uniquely identifiable submitted GPU job')
    row = rows[0]
    config = json.loads((lane_path(args.lane) / '.autobench/config.json').read_text())
    response = asyncio.run(get_job(config, row['job_id']))
    job = response.get('job', response)
    receipt = STUDY / 'runs/receipts' / (row['job_id'] + '.json')
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(response, indent=2) + '\n')
    if job.get('state') not in ('succeeded', 'completed', 'failed', 'cancelled', 'canceled', 'timed_out', 'abandoned') or not job.get('cleanup_verified'):
        print(json.dumps({'state': 'awaiting_terminal_verified_cleanup', 'job_id': row['job_id']}))
        return
    if (job.get('id') == row['job_id'] and job.get('state') in ('cancelled', 'canceled') and
            all(job.get(key) is None for key in ('attempt', 'attempt_id', 'lease_id', 'lease_binding', 'reservation_id'))):
        # The Scheduler confirms cancellation before any attempt or lease;
        # retain that exact receipt and packet command rather than inventing
        # a worker result or claiming a GPU allocation occurred.
        preserved = STUDY / 'runs/packets' / row['job_id']
        preserved.mkdir(parents=True, exist_ok=True)
        manifest = lane_path(args.lane) / '.autobench/state/packets' / (row['attempt_id'] + '.json')
        if not manifest.is_file():
            raise RuntimeError('unstarted cancellation lacks its registered packet command')
        shutil.copy2(manifest, preserved / 'registered-command.json')
        (preserved / 'cancelled-before-start.json').write_text(json.dumps(
            {'schema': 'cgs.cancelled-before-start.v1', 'job_id': row['job_id'],
             'receipt': str(receipt.relative_to(STUDY)), 'gpu_allocation': None,
             'cleanup_verified': True}, indent=2) + '\n')
        row.update(state='settled', charged_seconds=0, terminal_state=job['state'],
                   worker_status='not_started', packet_evidence=str(preserved.relative_to(STUDY)),
                   receipt=str(receipt.relative_to(STUDY)))
        save(ledger)
        print(json.dumps(row, sort_keys=True))
        return
    attempt = job.get('attempt', {})
    if not attempt.get('created_at') or not attempt.get('finished_at'):
        raise RuntimeError('terminal receipt lacks bounded attempt timestamps; retain reservation')
    elapsed = max(0.0, stamp(attempt['finished_at']) - stamp(attempt['created_at']))
    if elapsed > row['reserved_seconds']:
        raise RuntimeError('observed elapsed time exceeds reserved bound; retain reservation and stop')
    allocations = attempt.get('execution_context', {}).get('allocations', [])
    gpu_ids = [a.get('resource_id') for a in allocations if str(a.get('resource_id', '')).startswith('gpu:')]
    if len(gpu_ids) != 1 or gpu_ids[0] != row['expected_gpu_resource_id']:
        raise RuntimeError('cannot prove exactly one allocated GPU; retain reservation')
    # Autobench removes scratch after import and can evict older successful
    # evidence. Preserve every packet before any later reconciliation.
    staged = lane_path(args.lane) / '.autobench/scratch' / row['attempt_id']
    result_file = staged / 'result.json'
    if not result_file.is_file():
        raise RuntimeError('terminal job lacks its worker result; retain reservation')
    worker = json.loads(result_file.read_text())
    row['worker_status'] = worker.get('status')
    preserved = STUDY / 'runs/packets' / row['job_id']
    preserved.parent.mkdir(parents=True, exist_ok=True)
    if not preserved.exists():
        shutil.copytree(staged, preserved)
    packet_manifest = lane_path(args.lane) / '.autobench/state/packets' / (row['attempt_id'] + '.json')
    packet_details = json.loads(packet_manifest.read_text()) if packet_manifest.is_file() else {}
    adapters = ('sanitize_chain.py', 'triage_binary.py', 'probe_packet.py')
    adapter_names = [name for name in adapters if any(str(x).endswith('/' + name) for x in packet_details.get('argv', []))]
    if len(adapter_names) > 1:
        raise RuntimeError('packet ambiguously identifies multiple private adapters')
    adapter_name = adapter_names[0] if adapter_names else 'run_packet.py'
    if row.get('serialized_confirmation') or adapter_names:
        if not packet_manifest.is_file():
            raise RuntimeError('serialized job lacks its actual Autobench packet manifest')
        manifest = packet_details
        snapshot = Path(manifest['working_directory']).resolve(strict=True)
        snapshot_root = (lane_path(args.lane) / '.autobench/state/snapshots').resolve(strict=True)
        wrapper = snapshot / 'scripts' / adapter_name
        if snapshot.parent != snapshot_root or Path(manifest['argv'][1]).resolve() != wrapper or not wrapper.is_file():
            raise RuntimeError('serialized job wrapper is not bound to its immutable snapshot')
        shutil.copy2(packet_manifest, preserved / 'registered-command.json')
        shutil.copy2(wrapper, preserved / 'runtime-wrapper.py')
        files = {}
        for item in sorted(snapshot.rglob('*')):
            if item.is_symlink():
                raise RuntimeError('serialized snapshot contains a symlink')
            if item.is_file():
                files[item.relative_to(snapshot).as_posix()] = file_sha256(item)
        fingerprint = hashlib.sha256('\n'.join(f'{name}\0{files[name]}' for name in sorted(files)).encode()).hexdigest()
        if fingerprint != snapshot.name:
            raise RuntimeError('independently reconstructed snapshot fingerprint differs from its identity')
        source_manifest = preserved / 'snapshot-manifest.json'
        source_manifest.write_text(json.dumps({'schema': 'cgs.reconstructed-snapshot.v1',
                                               'snapshot_id': fingerprint, 'files': files,
                                               'origin': 'independently reconstructed from the retained immutable Autobench snapshot'}, indent=2) + '\n')
        shutil.copy2(lane_path(args.lane) / 'registration.toml', preserved / 'registration.toml')
        identity = {'schema': 'cgs.runtime-wrapper.v1', 'job_id': row['job_id'],
                    'attempt_id': row['attempt_id'], 'snapshot_id': snapshot.name,
                    'wrapper_sha256': file_sha256(wrapper),
                    'packet_manifest_sha256': file_sha256(packet_manifest),
                    'snapshot_manifest_sha256': file_sha256(source_manifest),
                    'snapshot_inputs': files}
        (preserved / 'runtime-wrapper-identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    if adapter_name == 'triage_binary.py':
        # Autobench writes worker logs in the scratch root before starting the
        # adapter. Keep its original clean subdirectory as raw evidence while
        # exposing the established triage packet layout to the KDA join.
        triage_packet = preserved / 'triage-packet'
        if triage_packet.is_dir():
            aliases = []
            for name in ('metrics.json', 'experiment.json', 'triage'):
                source = triage_packet / name
                destination = preserved / name
                if source.exists():
                    if destination.exists():
                        raise RuntimeError('triage packet normalization would overwrite evidence')
                    if source.is_dir():
                        shutil.copytree(source, destination)
                    else:
                        shutil.copy2(source, destination)
                    items = sorted(source.rglob('*')) if source.is_dir() else [source]
                    for item in items:
                        if item.is_file():
                            target = destination / item.relative_to(source) if source.is_dir() else destination
                            digest = file_sha256(item)
                            if file_sha256(target) != digest:
                                raise RuntimeError('triage evidence alias differs from its original')
                            aliases.append({'original': item.relative_to(preserved).as_posix(),
                                            'alias': target.relative_to(preserved).as_posix(), 'sha256': digest})
            (preserved / 'triage-aliases.json').write_text(json.dumps(
                {'schema': 'cgs.triage-aliases.v1', 'files': aliases}, indent=2) + '\n')
    if adapter_name == 'probe_packet.py' and (preserved / 'probe-packet').is_dir():
        # Preserve the raw packet at its original nested location. Its direct
        # metrics copy is only an index; no source artifacts are rewritten.
        source = preserved / 'probe-packet' / 'metrics.json'
        if source.is_file():
            destination = preserved / 'metrics.json'
            if destination.exists():
                raise RuntimeError('probe metrics alias would overwrite evidence')
            shutil.copy2(source, destination)
            digest = file_sha256(source)
            if file_sha256(destination) != digest:
                raise RuntimeError('probe metrics alias differs from its original')
            (preserved / 'probe-aliases.json').write_text(json.dumps(
                {'schema': 'cgs.probe-aliases.v1', 'files': [
                    {'original': 'probe-packet/metrics.json', 'alias': 'metrics.json', 'sha256': digest}]}, indent=2) + '\n')
    row['packet_evidence'] = str(preserved.relative_to(STUDY))
    nested = {'triage_binary.py': 'triage-packet', 'probe_packet.py': 'probe-packet'}.get(adapter_name)
    row['build_evidence'] = preserve_build(staged / nested if nested and (staged / nested).is_dir() else staged)
    row.update(state='settled', charged_seconds=elapsed, gpu_resource_id=gpu_ids[0],
               terminal_state=job['state'], receipt=str(receipt.relative_to(STUDY)))
    save(ledger)
    print(json.dumps(row, sort_keys=True))


def main():
    global LANES
    p = argparse.ArgumentParser()
    p.add_argument('--lane-root', default=str(LANES), help='study-contained enrollment root; shares the same budget ledger')
    sub = p.add_subparsers(dest='action', required=True)
    init = sub.add_parser('init-lane')
    init.add_argument('--lane', type=int, required=True)
    init.add_argument('--archive', required=True)
    reg = sub.add_parser('register-ste')
    reg.add_argument('--lane', type=int, required=True)
    reg.add_argument('--uuid', required=True)
    reg.add_argument('--resource-id', required=True)
    reg.add_argument('--archive-sha256', required=True)
    reg.add_argument('--archive')
    reg.add_argument('--revision', type=int, default=1)
    reg.add_argument('--reuse-inputs', action='store_true')
    reg.add_argument('--suite', choices=['ste', 'combined', 'kda', 'triage', 'profile', 'interference'], default='ste')
    reg.add_argument('--target', choices=['cgs-ste-gpu-tests', 'cgs_learning_tests', 'cgs_condition_tests',
                                          'cgs_consumer_tests', 'cgs_linear_library_tests', 'cgs_study'], default='cgs-ste-gpu-tests')
    reg.add_argument('--config')
    reg.add_argument('--interference-evidence')
    reg.add_argument('--contract-freeze')
    reg.add_argument('--frozen-binary')
    reg.add_argument('--frozen-build-receipt')
    reg.add_argument('--interference-mode', choices=['isolated', 'concurrent'])
    reg.add_argument('--repetition', type=int, default=0)
    reg.add_argument('--barrier-id')
    reg.add_argument('--sanitizer', choices=['none', 'memcheck', 'racecheck', 'initcheck', 'synccheck'], required=True)
    reg.add_argument('--expected', type=int, default=60)
    reg.add_argument('--kill', type=int, default=300)
    step_p = sub.add_parser('step')
    step_p.add_argument('--lane', type=int, required=True)
    settle_p = sub.add_parser('settle')
    settle_p.add_argument('--lane', type=int, required=True)
    sub.add_parser('status')
    args = p.parse_args()
    LANES = validated_lane_root(args.lane_root)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with (LEDGER.parent / 'campaign-budget.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.action == 'init-lane': initialize(args)
        elif args.action == 'register-ste': register(args)
        elif args.action == 'step': step(args)
        elif args.action == 'settle': settle(args)
        else:
            ledger = load()
            print(json.dumps({'lane_root': str(LANES), 'charged_and_reserved_seconds': charged(ledger), **ledger}, indent=2))


if __name__ == '__main__':
    main()
