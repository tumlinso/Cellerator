#!/usr/bin/env python3
"""Fail-closed local admission for the bounded chromatin execution probe.

The study campaign controller remains the only Scheduler submitter. This
wrapper records its own small reservation before invoking that controller,
using the same lock as the aggregate study ledger. It never edits the shared
ledger. Unknown controller outcomes keep the full local reservation charged.
"""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import tomllib
import uuid
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
CAMPAIGN = STUDY / 'scripts/campaign.py'
GLOBAL_LEDGER = STUDY / 'runs/campaign-budget.json'
LOCK = GLOBAL_LEDGER.parent / 'campaign-budget.lock'
LOCAL_BUDGET = 600
MAX_GPU_JOBS = 3
PHASE_LIMITS = {'qualification': 300, 'r1': 150, 'r16': 150}
PHASE_LANES = {'qualification': 0, 'r1': 1, 'r16': 2}
CLEANUP_MARGIN = 60
PHASES = tuple(PHASE_LIMITS)
GPU_STATES = {'reserving', 'submitted', 'unknown', 'passed', 'failed'}
CONTENT_SHA256 = 'c735b9f73b23be0e91554a1b17932b6150846d0b4bd750246068cda3683e8f03'
CASE_INPUTS = {
    'r1': {
        'case': 'probe-grouping-coherent-hard-frozen-r1-change0',
        'workload_sha256': '3547f22a38de21cd8c5fe2b762ec3d4d352fd3a3d7172314fd7d305cdeb711c6',
        'content_sha256': CONTENT_SHA256,
    },
    'r16': {
        'case': 'probe-grouping-coherent-hard-frozen-r16-change0',
        'workload_sha256': '2d5de22ae75633558f820c9366e0cdc40a0cc25bf48619093b1da63338733c4f',
        'content_sha256': CONTENT_SHA256,
    },
}
TERMINAL_STATES = {
    'succeeded', 'completed', 'failed', 'cancelled', 'canceled',
    'timed_out', 'abandoned',
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f'expected JSON object at {path}')
    return value


def study_path(path: str | Path) -> Path:
    candidate = Path(path).resolve()
    root = STUDY.resolve()
    if candidate == root or not candidate.is_relative_to(root):
        raise RuntimeError('probe state and lane root must remain inside the study')
    return candidate


def lock_context():
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    return LOCK.open('a+')


def local_load(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError('probe budget has not been initialized')
    data = read_json(path)
    if (data.get('schema') != 'cgs.probe-budget.v1' or
            data.get('budget_seconds') != LOCAL_BUDGET or
            data.get('max_gpu_jobs') != MAX_GPU_JOBS or
            data.get('phases') != list(PHASES) or
            data.get('phase_limits_seconds') != PHASE_LIMITS or
            data.get('phase_lanes') != PHASE_LANES or
            data.get('case_inputs') != CASE_INPUTS):
        raise RuntimeError('probe budget file has an incompatible schema or limit')
    if not isinstance(data.get('jobs'), list) or not isinstance(data.get('preparations'), list):
        raise RuntimeError('probe budget records are malformed')
    return data


def local_save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def global_rows() -> list[dict]:
    ledger = read_json(GLOBAL_LEDGER)
    if ledger.get('schema') != 'cgs.budget.v1' or ledger.get('budget_seconds') != 14400:
        raise RuntimeError('aggregate study ledger schema changed')
    rows = ledger.get('reservations')
    if not isinstance(rows, list):
        raise RuntimeError('aggregate study ledger reservations are malformed')
    return rows


def used_seconds(data: dict) -> float:
    total = 0.0
    for item in data['jobs']:
        if item['state'] in ('reserving', 'submitted', 'unknown'):
            total += item['reserved_seconds']
        else:
            total += item.get('charged_seconds', 0.0)
    return total


def gpu_job_count(data: dict) -> int:
    return sum(item['state'] in GPU_STATES for item in data['jobs'])


def phase_gpu_job(data: dict, phase: str) -> dict | None:
    matches = [item for item in data['jobs']
               if item.get('phase') == phase and item.get('state') in GPU_STATES]
    if len(matches) > 1:
        raise RuntimeError(f'{phase} has multiple GPU submissions; refusing to continue')
    return matches[0] if matches else None


def next_phase(data: dict, phase: str) -> None:
    if phase not in PHASE_LIMITS:
        raise RuntimeError('phase must be qualification, r1, or r16')
    for earlier in PHASES[:PHASES.index(phase)]:
        row = phase_gpu_job(data, earlier)
        if row is None or row['state'] != 'passed':
            raise RuntimeError(f'{phase} requires the earlier {earlier} phase to pass')


def lane_registration(data: dict, phase: str) -> tuple[Path, dict, int, str, str]:
    lane_index = PHASE_LANES[phase]
    lane = Path(data['lane_root']) / str(lane_index)
    if not lane.is_dir():
        raise RuntimeError('probe lane does not exist')
    status = subprocess.run(['autobench', 'status', '--project', str(lane)],
                            capture_output=True, text=True)
    if status.returncode:
        raise RuntimeError(f'cannot read probe lane status: {status.stderr[-800:]}')
    status_obj = json.loads(status.stdout)
    config = read_json(lane / '.autobench/config.json')
    if config.get('active') is not False or status_obj.get('paused'):
        raise RuntimeError('probe lane must be inactive and unpaused')
    registries = list((lane / '.autobench/registry').glob('*.toml'))
    if len(registries) != 1:
        raise RuntimeError('probe lane must have exactly one registered benchmark')
    definition_path = registries[0]
    definition = tomllib.loads(definition_path.read_text())
    packets = definition.get('packet', [])
    if len(packets) != 1:
        raise RuntimeError('probe benchmark must have exactly one packet')
    packet = packets[0]
    kill = packet.get('kill_seconds')
    if type(kill) is not int:
        raise RuntimeError('packet kill_seconds must be an integer')
    resources = packet.get('resources', {}).get('scheduler', [])
    gpus = [item for item in resources if item.get('kind') == 'gpu']
    if (len(gpus) != 1 or gpus[0].get('mode') != 'exclusive' or
            not gpus[0].get('selectors', {}).get('resource_id', '').startswith('gpu:')):
        raise RuntimeError('probe packet must request exactly one exclusive GPU by resource ID')
    build_resources = definition.get('build', {}).get('resources', {})
    if (build_resources.get('gpu_memory_bytes') or
            any(item.get('kind') == 'gpu' for item in build_resources.get('scheduler', []))):
        raise RuntimeError('CPU preparation must not request GPU resources')
    return lane, packet, kill, sha(definition_path), gpus[0]['selectors']['resource_id']


def check_lane_identities(lane: Path, data: dict) -> None:
    files = {
        'source_archive_sha256': lane / '.autobench/data/closure.tar.gz',
        'binary_sha256': lane / '.autobench/data/frozen-binary',
    }
    for identity, path in files.items():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f'probe lane {identity} input is missing or is a symlink')
        if sha(path) != data[identity]:
            raise RuntimeError(f'probe lane {identity} differs from the frozen local identity')


def _scheduler_job(lane: Path, job_id: str) -> dict:
    """Read exact CPU-build status through the enrolled Scheduler client."""
    config_path = lane / '.autobench/config.json'
    python = Path.home() / '.local/state/autobench/current/venv/bin/python'
    if not python.is_file():
        python = Path(sys.executable)
    script = (
        'import asyncio,json,sys\n'
        'from device_scheduler_mcp.client import DeviceSchedulerClient\n'
        'c=json.load(open(sys.argv[1]))\n'
        'x=DeviceSchedulerClient(c["scheduler_url"],c["token_file"])\n'
        'print(json.dumps(asyncio.run(x.call("jobs.get", {"job_id":sys.argv[2]}))))\n'
    )
    result = subprocess.run([str(python), '-c', script, str(config_path), job_id],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('cannot verify prior CPU-build cleanup through Scheduler')
    return json.loads(result.stdout)


def verify_cpu_builds(data: dict, phase: str, lane: Path, state_path: Path) -> None:
    """Permit an explicit later reconcile only after each CPU build is clean."""
    if any(item.get('phase') == phase and item.get('state') == 'cpu_build_failed'
           for item in data['preparations']):
        raise RuntimeError('prior CPU build failed; no retry is allowed')
    changed = False
    failed = False
    for item in data['preparations']:
        if item.get('phase') != phase or item.get('state') != 'cpu_build':
            continue
        matches = [row for row in global_rows()
                   if row.get('id') == item.get('global_reservation_id')]
        if (len(matches) != 1 or matches[0].get('state') != 'cpu_build' or
                matches[0].get('kind') != 'build' or matches[0].get('job_id') != item.get('job_id') or
                matches[0].get('lane') != item.get('lane') or
                matches[0].get('lane_root') != data['lane_root']):
            raise RuntimeError('prior CPU build no longer matches its shared campaign ledger row')
        build_lane = Path(data['lane_root']) / str(item['lane'])
        response = _scheduler_job(build_lane, item['job_id'])
        job = response.get('job', response)
        if (job.get('id') != item['job_id'] or job.get('cleanup_verified') is not True or
                job.get('state') not in TERMINAL_STATES):
            raise RuntimeError('prior CPU build lacks terminal verified cleanup; no reconcile allowed')
        item['terminal_state'] = job['state']
        item['cleanup_verified'] = True
        if job['state'] not in ('succeeded', 'completed'):
            item['state'] = 'cpu_build_failed'
            item['failure'] = 'CPU build failed; phase cannot be retried'
            changed = True
            failed = True
            continue
        item['state'] = 'cpu_build_clean'
        changed = True
    if changed:
        # Caller holds the shared lock; retain exact Scheduler cleanup evidence.
        local_save(state_path, data)
    if failed:
        raise RuntimeError('prior CPU build failed; no retry is allowed')


def _controller_step(lane_root: Path, lane_index: int) -> tuple[int, str, str]:
    """Call the established one-shot controller after the local reservation."""
    env = dict(os.environ)
    env['AUTOBENCH_HOST_SCRATCH_LIMIT_BYTES'] = str(536870912)
    env['AUTOBENCH_HOST_OUTPUT_LIMIT_BYTES'] = str(268435456)
    command = [sys.executable, str(CAMPAIGN), '--lane-root', str(lane_root),
               'step', '--lane', str(lane_index)]
    result = subprocess.run(command, text=True, capture_output=True, env=env)
    return result.returncode, result.stdout, result.stderr


def reconcile(args: argparse.Namespace) -> dict:
    path = study_path(args.state)
    phase = args.phase
    with lock_context() as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = local_load(path)
        if path.with_name(path.name + '.rebound').exists():
            raise RuntimeError('this binding was superseded; use its recorded successor')
        next_phase(data, phase)
        if phase_gpu_job(data, phase) is not None:
            raise RuntimeError(f'{phase} already has a GPU submission; retries are disabled')
        lane, packet, kill, registration_sha, resource_id = lane_registration(data, phase)
        verify_cpu_builds(data, phase, lane, path)
        if data.get('expected_gpu_resource_id') not in (None, resource_id):
            raise RuntimeError('all probe phases must use the same exact Scheduler GPU resource ID')
        data['expected_gpu_resource_id'] = resource_id
        local_save(path, data)
        check_lane_identities(lane, data)
        phase_bound = PHASE_LIMITS[phase]
        if kill + CLEANUP_MARGIN != phase_bound:
            raise RuntimeError(f'{phase} packet kill_seconds must be exactly {phase_bound - CLEANUP_MARGIN}')
        if gpu_job_count(data) >= MAX_GPU_JOBS:
            raise RuntimeError('probe GPU job limit is exhausted')
        if used_seconds(data) + phase_bound > LOCAL_BUDGET:
            raise RuntimeError('local probe GPU budget cannot cover this phase')
        before_ids = {row.get('id') for row in global_rows()}
        local_job = {
            'id': str(uuid.uuid4()), 'phase': phase, 'state': 'reserving',
            'reserved_seconds': phase_bound, 'charged_seconds': 0.0,
            'created': time.time(), 'global_reservation_id': None, 'job_id': None,
            'lane': PHASE_LANES[phase], 'lane_root': data['lane_root'],
            'registration_sha256': registration_sha,
            'expected_gpu_resource_id': resource_id,
            'expected_gpu_uuid': resource_id.rsplit(':', 1)[-1],
            'source_archive_sha256': data['source_archive_sha256'],
            'binary_sha256': data['binary_sha256'],
            'case_identities': CASE_INPUTS,
        }
        if phase in CASE_INPUTS:
            local_job.update(CASE_INPUTS[phase])
        data['jobs'].append(local_job)
        local_save(path, data)

    # The campaign controller takes this same lock while changing the shared
    # ledger. Never hold it across the scheduler-submit-capable subprocess.
    try:
        returncode, stdout, stderr = _controller_step(Path(data['lane_root']), PHASE_LANES[phase])
    except Exception as exc:
        returncode, stdout, stderr = -1, '', str(exc)

    with lock_context() as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = local_load(path)
        local_job = next(item for item in data['jobs'] if item['id'] == local_job['id'])
        new_rows = [row for row in global_rows() if row.get('id') not in before_ids]
        local_job['controller_returncode'] = returncode
        local_job['controller_stderr_tail'] = stderr[-1200:]
        if stdout.strip():
            try:
                local_job['controller_result'] = json.loads(stdout)
            except json.JSONDecodeError:
                local_job['controller_stdout_sha256'] = hashlib.sha256(stdout.encode()).hexdigest()
        if len(new_rows) != 1:
            local_job.update(state='unknown', error='controller did not create exactly one attributable global reservation')
            local_save(path, data)
            raise RuntimeError('campaign outcome is unknown; full local reservation remains charged')
        row = new_rows[0]
        if (row.get('lane_root') != data['lane_root'] or row.get('lane') != PHASE_LANES[phase] or
                row.get('reserved_seconds') != PHASE_LIMITS[phase] or
                row.get('expected_gpu_resource_id') != local_job['expected_gpu_resource_id']):
            local_job.update(state='unknown', error='global reservation lane or phase budget identity mismatch',
                             candidate_global_reservation_id=row.get('id'))
            local_save(path, data)
            raise RuntimeError('global reservation does not match local probe; full reservation remains charged')
        local_job['global_reservation_id'] = row.get('id')
        local_job['job_id'] = row.get('job_id')
        global_state = row.get('state')
        if global_state == 'no_submission' and not row.get('job_id'):
            local_job.update(state='no_submission', reserved_seconds=0, charged_seconds=0,
                             no_gpu_submission=True)
            data['preparations'].append({
                'phase': phase, 'state': 'no_submission', 'global_reservation_id': row.get('id'),
                'lane': PHASE_LANES[phase], 'lane_root': data['lane_root'],
                'source_archive_sha256': data['source_archive_sha256'],
                'binary_sha256': data['binary_sha256'], 'case_identities': CASE_INPUTS,
            })
        elif global_state == 'cpu_build' and row.get('job_id') and row.get('kind') == 'build':
            local_job.update(state='cpu_build', reserved_seconds=0, charged_seconds=0,
                             cpu_only=True)
            data['preparations'].append({
                'phase': phase, 'state': 'cpu_build', 'global_reservation_id': row.get('id'),
                'job_id': row['job_id'], 'lane': PHASE_LANES[phase], 'lane_root': data['lane_root'],
                'source_archive_sha256': data['source_archive_sha256'],
                'binary_sha256': data['binary_sha256'], 'case_identities': CASE_INPUTS,
            })
        elif (global_state == 'submitted' and row.get('job_id') and
              row.get('kind') == 'correctness'):
            local_job.update(state='submitted')
        else:
            local_job.update(state='unknown', error='global reservation is not a definite no-GPU, CPU-build, or GPU outcome')
            local_save(path, data)
            raise RuntimeError('campaign outcome is ambiguous; full local reservation remains charged')
        local_save(path, data)
        return local_job


def _resolve_packet_file(packet: Path, relative: str) -> Path:
    source = packet / relative
    resolved = source.resolve()
    if source.is_symlink() or not resolved.is_relative_to(packet.resolve()) or not resolved.is_file():
        raise RuntimeError('retained packet artifact path escapes or is missing')
    return resolved


def _expected_orders() -> list[list[str]]:
    arms = ['masked-direct-two-tile', 'cohort-direct-two-tile']
    return [order for _ in range(3) for order in (list(arms), list(reversed(arms)))]


def paired_summary(ratios: list[float]) -> dict:
    if len(ratios) != 6 or any(not math.isfinite(x) or x <= 0 for x in ratios):
        raise RuntimeError('paired analysis requires six positive finite ratios')
    point = statistics.median(ratios)
    rng = random.Random(20261102)
    draws = sorted(statistics.median(rng.choices(ratios, k=6)) for _ in range(10000))
    def percentile(p):
        pos = (len(draws) - 1) * p
        low = int(pos)
        high = min(low + 1, len(draws) - 1)
        return draws[low] + (draws[high] - draws[low]) * (pos - low)
    low, high = percentile(.025), percentile(.975)
    practical = point < .95 - 1e-12
    supported = high < 1.0
    return {'paired_ratios_cohort_over_masked': ratios,
            'median_paired_ratio_cohort_over_masked': point,
            'median_paired_latency_reduction': 1 - point,
            'paired_bootstrap_95_ratio_interval': [low, high],
            'paired_bootstrap_95_latency_reduction_interval': [1 - high, 1 - low],
            'paired_bootstrap_seed': 20261102, 'paired_bootstrap_samples': 10000,
            'practical_latency_reduction_over_5_percent': practical,
            'paired_95_interval_supports_improvement': supported,
            'grouping_beneficial_for_probe': practical and supported}


def _validate_phase_result(path: Path, phase: str, data: dict, row: dict) -> dict:
    packet = path / 'probe-packet'
    metrics = read_json(packet / 'metrics.json')
    expected_attempt = row.get('scheduler_attempt_id')
    expected_uuid = job_gpu_uuid(row)
    checks = {
        'schema': 'cgs.packet.v1', 'status': 'pass', 'success': True, 'phase': phase,
        'scheduler_job_id': row['job_id'], 'scheduler_attempt_id': expected_attempt,
        'uuid': expected_uuid,
        'archive_sha256': data['source_archive_sha256'],
        'binary_sha256': data['binary_sha256'],
    }
    for key, value in checks.items():
        if value is None or metrics.get(key) != value:
            raise RuntimeError(f'retained packet metrics do not bind expected {key}')
    binary_path = metrics.get('binary_path')
    if not isinstance(binary_path, str) or not Path(binary_path).is_absolute() or Path(binary_path).name != 'cgs_study' or not metrics.get('build_manifest_sha256'):
        raise RuntimeError('retained packet lacks binary path or build-manifest identity')
    build_receipt = study_path(STUDY / row.get('build_evidence', ''))
    if sha(build_receipt) != metrics.get('build_manifest_sha256'):
        raise RuntimeError('retained packet build-manifest hash differs from Scheduler receipt')
    manifest = read_json(build_receipt)
    study_build, qualifier_build = manifest.get('cgs_study', {}), manifest.get('qualification_binary', {})
    if (manifest.get('schema') != 'cgs.grouping-probe-build.v1' or
            manifest.get('archive_sha256') != data['source_archive_sha256'] or
            manifest.get('binary_sha256') != data['binary_sha256'] or
            manifest.get('binary_path') != binary_path or
            study_build.get('binary_sha256') != data['binary_sha256'] or
            study_build.get('original_receipt_sha256') != metrics.get('cgs_study_build_receipt_sha256') or
            qualifier_build.get('binary_sha256') != metrics.get('qualification_binary_sha256') or
            qualifier_build.get('original_receipt_sha256') != metrics.get('qualification_build_receipt_sha256')):
        raise RuntimeError('adapter manifest does not bind the exact source, binaries and original receipts')
    identity = read_json(path / 'runtime-wrapper-identity.json')
    snapshot_inputs = identity.get('snapshot_inputs', {})
    if (snapshot_inputs.get('.autobench/data/frozen-build.json') != study_build.get('original_receipt_sha256') or
            snapshot_inputs.get('.autobench/data/qualification-build.json') != qualifier_build.get('original_receipt_sha256') or
            snapshot_inputs.get('.autobench/data/frozen-binary') != data['binary_sha256'] or
            snapshot_inputs.get('.autobench/data/qualification-binary') != qualifier_build.get('binary_sha256')):
        raise RuntimeError('immutable runtime snapshot does not bind the two original receipts and binaries')
    if phase == 'qualification':
        qualifier_sha = metrics.get('qualification_binary_sha256')
        if not isinstance(qualifier_sha, str) or len(qualifier_sha) != 64:
            raise RuntimeError('qualification packet lacks its separate qualifier binary identity')
        result_obj = metrics.get('qualification_result', {})
        proof_name = result_obj.get('qualification_proof_path')
        proof_sha = result_obj.get('qualification_proof_sha256')
        if not isinstance(proof_name, str) or not isinstance(proof_sha, str):
            raise RuntimeError('qualification packet lacks retained proof path and hash')
        proof_path = _resolve_packet_file(packet, proof_name)
        if sha(proof_path) != proof_sha:
            raise RuntimeError('qualification proof hash differs from packet metrics')
        proof = read_json(proof_path)
        proof_checks = {
            'schema': 'cgs.grouping-probe-proof.v1', 'status': 'pass',
            'archive_sha256': data['source_archive_sha256'],
            'binary_sha256': data['binary_sha256'],
            'qualification_binary_sha256': qualifier_sha,
            'scheduler_job_id': row['job_id'], 'scheduler_attempt_id': expected_attempt,
            'device_uuid': expected_uuid,
        }
        for key, value in proof_checks.items():
            if value is None or proof.get(key) != value:
                raise RuntimeError(f'qualification proof does not bind expected {key}')
        if proof.get('cgs_study_build_receipt_sha256') != study_build.get('original_receipt_sha256'):
            raise RuntimeError('qualification proof does not bind the retained Cellerator build receipt')
        qualifier_receipt_sha = proof.get('qualification_build_receipt_sha256')
        if qualifier_receipt_sha != qualifier_build.get('original_receipt_sha256'):
            raise RuntimeError('qualification proof lacks its qualifier build-receipt hash')
        campaigns = proof.get('campaigns', {})
        if set(campaigns) != {item['case'] for item in CASE_INPUTS.values()}:
            raise RuntimeError('qualification proof does not cover both frozen probe workloads')
        for name, expected in CASE_INPUTS.items():
            observed = campaigns.get(expected['case'])
            if (not isinstance(observed, dict) or
                    observed.get('workload_sha256') != expected['workload_sha256'] or
                    observed.get('workload_content_sha256') != expected['content_sha256']):
                raise RuntimeError(f'qualification proof lacks exact {name} workload/content identity')
            arms = observed.get('arms', {})
            if set(arms) != {'masked-direct-two-tile', 'cohort-direct-two-tile'}:
                raise RuntimeError(f'qualification proof lacks both qualified arms for {name}')
            for arm_name, arm in arms.items():
                raw_path = _resolve_packet_file(packet, arm.get('raw_path', ''))
                if sha(raw_path) != arm.get('raw_sha256'):
                    raise RuntimeError('qualification raw campaign artifact hash is invalid')
                raw = read_json(raw_path)
                descriptor = ('exact-gate-planes-no-materialized-workset' if arm_name == 'masked-direct-two-tile'
                              else 'exact-work-membership-and-raw-group-identity')
                if (raw.get('schema') != 'cgs.campaign.v1' or raw.get('status') != 'ok' or
                        raw.get('accuracy_accepted') is not True or raw.get('descriptor_validation') != descriptor or
                        raw.get('validation_scope') != 'first-and-last-complete-steps' or
                        raw.get('case') != expected['case'] or raw.get('backend') != arm_name or
                        raw.get('device_uuid') != expected_uuid or
                        raw.get('workload_sha256') != expected['workload_sha256'] or
                        raw.get('workload_content_sha256') != expected['content_sha256'] or
                        raw.get('gate') != 'supplied' or raw.get('body') != 'product' or
                        raw.get('precision') != 'fp16' or raw.get('condition_update') != 'none' or
                        raw.get('input') != 'synthetic' or raw.get('tile') != 32 or raw.get('stages') != 1):
                    raise RuntimeError('qualification native output does not match its frozen contract')
        sanitizers = proof.get('sanitizers', {})
        if set(sanitizers) != {'memcheck', 'racecheck', 'initcheck', 'synccheck'} or any(
                item.get('status') != 'pass' or item.get('zero_error_summary') is not True
                for item in sanitizers.values()):
            raise RuntimeError('qualification proof lacks all four zero-error sanitizer results')
        checks_by_tool = {item.get('tool'): item for item in proof.get('qualification_checks', [])
                          if item.get('kind') == 'compute-sanitizer'}
        if set(checks_by_tool) != set(sanitizers):
            raise RuntimeError('qualification proof lacks the four sanitizer log records')
        for tool, summary in sanitizers.items():
            record = checks_by_tool[tool]
            for path_key, hash_key in (('stdout_path', 'stdout_sha256'), ('stderr_path', 'stderr_sha256')):
                log = _resolve_packet_file(packet, record.get(path_key, ''))
                if sha(log) != record.get(hash_key) or summary.get(hash_key) != record.get(hash_key):
                    raise RuntimeError(f'qualification {tool} log path/hash is invalid')
        return {'qualification_proof_sha256': proof_sha,
                'qualification_binary_sha256': qualifier_sha,
                'build_manifest_sha256': metrics['build_manifest_sha256'],
                'cgs_study_build_receipt_sha256': study_build['original_receipt_sha256'],
                'qualification_build_receipt_sha256': qualifier_build['original_receipt_sha256']}
    else:
        qualification = phase_gpu_job(data, 'qualification')
        if not qualification or qualification.get('state') != 'passed':
            raise RuntimeError('timing phase cannot be accepted before qualification passes')
        if (metrics.get('qualification_binary_sha256') != qualification.get('qualification_binary_sha256') or
                study_build.get('original_receipt_sha256') != qualification.get('cgs_study_build_receipt_sha256') or
                qualifier_build.get('original_receipt_sha256') != qualification.get('qualification_build_receipt_sha256')):
            raise RuntimeError('timing packet changed the qualified binaries or original build receipts')
        collection_name = metrics.get('collection_path')
        collection_sha = metrics.get('collection_sha256')
        if not isinstance(collection_name, str) or not isinstance(collection_sha, str):
            raise RuntimeError('timing packet lacks its retained collection path and hash')
        collection_path = _resolve_packet_file(packet, collection_name)
        if sha(collection_path) != collection_sha:
            raise RuntimeError('timing collection hash differs from packet metrics')
        if (metrics.get('qualification_proof_sha256') != qualification.get('qualification_proof_sha256') or
                metrics.get('serial_telemetry_clean') is not True):
            raise RuntimeError('timing packet differs from passed qualification proof or clean telemetry')
        proof_path = _resolve_packet_file(packet, metrics.get('qualification_proof_path', ''))
        if sha(proof_path) != qualification.get('qualification_proof_sha256'):
            raise RuntimeError('timing packet qualification-proof bytes differ from the accepted proof')
        timing = read_json(collection_path)
        expected = CASE_INPUTS[phase]
        checks = {
            'schema': 'cgs.grouping-probe-timing.v1', 'status': 'pass',
            'case': expected['case'], 'workload_sha256': expected['workload_sha256'],
            'workload_content_sha256': expected['content_sha256'],
            'qualification_proof_sha256': qualification['qualification_proof_sha256'],
            'device_uuid': expected_uuid,
        }
        for key, value in checks.items():
            if timing.get(key) != value:
                raise RuntimeError(f'retained timing data does not bind expected {key}')
        rounds = timing.get('rounds')
        orders = _expected_orders()
        if (not isinstance(rounds, list) or len(rounds) != 6 or
                timing.get('round_order_rule') != 'AB,BA,AB,BA,AB,BA'):
            raise RuntimeError('timing collection must contain exactly six paired rounds')
        expected_arms = {'masked-direct-two-tile', 'cohort-direct-two-tile'}
        ratios = []
        for index, (round_row, order) in enumerate(zip(rounds, orders)):
            if (round_row.get('round_id') != f'round-{index:02d}' or
                    round_row.get('order') != order or round_row.get('status') != 'pass' or
                    set(round_row.get('arms', {})) != expected_arms):
                raise RuntimeError(f'timing collection round {index:02d} is incomplete or out of order')
            for arm_name, arm in round_row['arms'].items():
                samples = arm.get('resident_complete_horizon_ms')
                recovery = arm.get('output_recovery_ms')
                cold = arm.get('cold_execution_preparation_ms')
                if (arm.get('status') != 'pass' or arm.get('device_uuid') != expected_uuid or
                        arm.get('workload_sha256') != expected['workload_sha256'] or
                        arm.get('workload_content_sha256') != expected['content_sha256'] or
                        not isinstance(samples, list) or len(samples) != 3 or
                        not all(not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in samples) or
                        arm.get('resident_median_ms') != sorted(samples)[1] or
                        not isinstance(recovery, list) or len(recovery) != 3 or
                        not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in recovery) or
                        not isinstance(cold, (int, float)) or not math.isfinite(cold) or cold < 0):
                    raise RuntimeError(f'timing arm {arm_name} round {index:02d} has invalid cost fields')
                for path_key, hash_key in (('raw_path', 'raw_sha256'),
                                           ('stdout_path', 'stdout_sha256'),
                                           ('stderr_path', 'stderr_sha256')):
                    raw_file = _resolve_packet_file(packet, arm.get(path_key, ''))
                    if sha(raw_file) != arm.get(hash_key):
                        raise RuntimeError(f'timing arm {arm_name} {path_key} hash is invalid')
                raw = read_json(_resolve_packet_file(packet, arm['raw_path']))
                descriptor = ('exact-gate-planes-no-materialized-workset' if arm_name == 'masked-direct-two-tile'
                              else 'exact-work-membership-and-raw-group-identity')
                if (raw.get('schema') != 'cgs.campaign.v1' or raw.get('status') != 'ok' or
                        raw.get('accuracy_accepted') is not True or
                        raw.get('validation_scope') != 'first-and-last-complete-steps' or
                        raw.get('descriptor_validation') != descriptor or
                        raw.get('gate') != 'supplied' or raw.get('body') != 'product' or
                        raw.get('precision') != 'fp16' or raw.get('condition_update') != 'none' or
                        raw.get('input') != 'synthetic' or raw.get('tile') != 32 or raw.get('stages') != 1 or
                        raw.get('case') != expected['case'] or raw.get('backend') != arm_name or
                        raw.get('device_uuid') != expected_uuid or raw.get('workload_sha256') != expected['workload_sha256'] or
                        raw.get('workload_content_sha256') != expected['content_sha256'] or
                        raw.get('resident_complete_horizon_ms') != samples or raw.get('output_recovery_ms') != recovery or
                        raw.get('cold_execution_preparation_ms') != cold):
                    raise RuntimeError('timing summary differs from its independently validated native output')
            ratio = (round_row['arms']['cohort-direct-two-tile']['resident_median_ms'] /
                     round_row['arms']['masked-direct-two-tile']['resident_median_ms'])
            if not math.isclose(round_row.get('paired_ratio_cohort_over_masked', -1), ratio, rel_tol=1e-12, abs_tol=1e-12):
                raise RuntimeError('paired round ratio differs from the arm medians')
            ratios.append(ratio)
        for key, expected_value in paired_summary(ratios).items():
            actual = timing.get(key)
            if isinstance(expected_value, list):
                if not isinstance(actual, list) or len(actual) != len(expected_value) or any(
                        not isinstance(a, (int, float)) or isinstance(a, bool) or
                        not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                        for a, b in zip(actual, expected_value)):
                    raise RuntimeError(f'paired summary differs from its recorded rounds: {key}')
            elif isinstance(expected_value, bool):
                if actual is not expected_value:
                    raise RuntimeError(f'paired decision differs from its recorded rounds: {key}')
            elif not isinstance(actual, (int, float)) or isinstance(actual, bool) or not math.isclose(actual, expected_value, rel_tol=1e-12, abs_tol=1e-12):
                raise RuntimeError(f'paired summary differs from its recorded rounds: {key}')
        return {'collection_sha256': collection_sha, 'build_manifest_sha256': metrics['build_manifest_sha256'],
                'qualification_proof_sha256': qualification.get('qualification_proof_sha256')}


def job_gpu_uuid(row: dict) -> str:
    resource = row.get('gpu_resource_id', row.get('expected_gpu_resource_id', ''))
    return resource.rsplit(':', 1)[-1]


def settle(args: argparse.Namespace) -> dict:
    path = study_path(args.state)
    with lock_context() as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = local_load(path)
        job = phase_gpu_job(data, args.phase)
        if job is None or job['state'] not in ('submitted', 'unknown') or not job.get('job_id'):
            raise RuntimeError('phase has no uniquely attributable submitted GPU job')
        rows = [row for row in global_rows() if row.get('id') == job.get('global_reservation_id')]
        if len(rows) != 1:
            raise RuntimeError('shared campaign reservation is missing or ambiguous')
        row = rows[0]
        if (row.get('job_id') != job['job_id'] or row.get('kind') != 'correctness' or
                row.get('lane_root') != data['lane_root'] or
                row.get('lane') != job['lane'] or row.get('reserved_seconds') != job['reserved_seconds'] or
                row.get('gpu_resource_id') != job.get('expected_gpu_resource_id')):
            raise RuntimeError('shared campaign reservation identity differs from local probe record')
        if row.get('state') != 'settled':
            raise RuntimeError('campaign controller has not verified terminal cleanup; reservation remains open')
        receipt_path = study_path(STUDY / row['receipt'])
        receipt = read_json(receipt_path)
        scheduler_job = receipt.get('job', receipt)
        state = scheduler_job.get('state')
        if (scheduler_job.get('id') != job['job_id'] or scheduler_job.get('cleanup_verified') is not True or
                state not in TERMINAL_STATES or row.get('terminal_state') != state):
            raise RuntimeError('Scheduler receipt does not prove this job terminal with verified cleanup')
        scheduler_attempt = scheduler_job.get('attempt')
        if not isinstance(scheduler_attempt, dict) or not scheduler_attempt.get('attempt_id'):
            raise RuntimeError('Scheduler receipt lacks the actual execution attempt identity')
        row = dict(row, scheduler_attempt_id=scheduler_attempt['attempt_id'])
        charged = float(row.get('charged_seconds', -1))
        if charged < 0 or charged > job['reserved_seconds']:
            job.update(state='unknown', charged_seconds=job['reserved_seconds'],
                       error='actual GPU charge exceeds local phase reservation')
            local_save(path, data)
            raise RuntimeError('GPU charge exceeds the phase budget; full reservation remains charged')
        packet_path = study_path(STUDY / row.get('packet_evidence', ''))
        if not packet_path.is_dir():
            raise RuntimeError('retained packet evidence is missing; reservation remains open')
        packet_result = None
        packet_error = None
        if state in ('succeeded', 'completed') and row.get('worker_status') == 'succeeded':
            try:
                packet_result = _validate_phase_result(packet_path, args.phase, data, row)
            except (RuntimeError, OSError, ValueError, TypeError, KeyError) as exc:
                packet_error = str(exc)
        passed = (state in ('succeeded', 'completed') and row.get('worker_status') == 'succeeded' and
                  packet_result is not None)
        job.update(state='passed' if passed else 'failed', charged_seconds=charged,
                   terminal_state=state, worker_status=row.get('worker_status'),
                   receipt=str(receipt_path.relative_to(STUDY)),
                   packet_evidence=row.get('packet_evidence'))
        if not passed:
            job['failure'] = packet_error or 'worker or Scheduler completion check failed; retries are disabled'
        else:
            job.update(packet_result)
        local_save(path, data)
        return job


def initialize(args: argparse.Namespace) -> dict:
    path = study_path(args.state)
    lane_root = study_path(args.lane_root)
    for name in ('source_sha256', 'binary_sha256'):
        value = getattr(args, name)
        if len(value) != 64 or any(char not in '0123456789abcdef' for char in value):
            raise RuntimeError(f'{name} must be a lowercase SHA-256')
    with lock_context() as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            raise RuntimeError('probe state already exists; no reinitialization is allowed')
        original = STUDY / 'runs/grouping-probe-v1/probe-budget.json'
        anchor = STUDY / 'runs/grouping-probe-allowance-anchor.json'
        if original.exists() or anchor.exists():
            raise RuntimeError('existing probe allowance must be continued through rebind, not reset')
        ledger = global_rows()
        if any(row.get('state') in ('reserving', 'unknown', 'submitted') for row in ledger):
            raise RuntimeError('shared study ledger has unresolved GPU reservations')
        value = {
            'schema': 'cgs.probe-budget.v1', 'created': time.time(),
            'budget_seconds': LOCAL_BUDGET, 'max_gpu_jobs': MAX_GPU_JOBS,
            'cleanup_margin_seconds': CLEANUP_MARGIN, 'phases': list(PHASES),
            'phase_limits_seconds': PHASE_LIMITS, 'phase_lanes': PHASE_LANES,
            'lane_root': str(lane_root), 'source_archive_sha256': args.source_sha256,
            'binary_sha256': args.binary_sha256, 'case_inputs': CASE_INPUTS,
            'expected_gpu_resource_id': None, 'jobs': [], 'preparations': [],
        }
        local_save(path, value)
        return value


def rebind(args: argparse.Namespace) -> dict:
    """Carry the existing untouched allowance into a new immutable binding."""
    parent = study_path(args.parent_state)
    path = study_path(args.state)
    lane_root = study_path(args.lane_root)
    if parent != (STUDY / 'runs/grouping-probe-v1/probe-budget.json').resolve():
        raise RuntimeError('rebind requires the original scoped probe allowance')
    if path != (STUDY / 'runs/grouping-probe-v2/probe-budget.json').resolve() or lane_root != (STUDY / 'runs/grouping-probe-v2/lanes').resolve():
        raise RuntimeError('rebind requires the designated fresh study-local binding')
    for name in ('source_sha256', 'binary_sha256'):
        value = getattr(args, name)
        if len(value) != 64 or any(char not in '0123456789abcdef' for char in value):
            raise RuntimeError(f'{name} must be a lowercase SHA-256')
    with lock_context() as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        original = local_load(parent)
        rows = global_rows()
        if path.exists() or parent.with_name(parent.name + '.rebound').exists():
            raise RuntimeError('the probe allowance already has a successor')
        if original['jobs'] or original['preparations'] or any(r.get('lane_root') == original['lane_root'] for r in rows):
            raise RuntimeError('original binding is not provably unused; no reset permitted')
        if any(r.get('state') in ('reserving', 'unknown', 'submitted') for r in rows):
            raise RuntimeError('shared study ledger has unresolved GPU reservations')
        value = dict(original, created=time.time(), lane_root=str(lane_root),
                     source_archive_sha256=args.source_sha256, binary_sha256=args.binary_sha256,
                     binding_parent={'path': str(parent), 'sha256': sha(parent), 'verified_jobs': 0,
                                     'verified_global_reservations': 0})
        marker = parent.with_name(parent.name + '.rebound')
        with marker.open('x') as stream:
            json.dump({'successor': str(path), 'parent_sha256': sha(parent)}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        local_save(path, value)
        return value


def status(args: argparse.Namespace) -> dict:
    data = local_load(study_path(args.state))
    return {
        'schema': data['schema'], 'budget_seconds': LOCAL_BUDGET,
        'used_or_reserved_seconds': used_seconds(data), 'gpu_job_count': gpu_job_count(data),
        'max_gpu_jobs': MAX_GPU_JOBS, 'jobs': data['jobs'],
        'preparations': data['preparations'], 'phases': data['phases'],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init')
    init.add_argument('--state', required=True)
    init.add_argument('--lane-root', required=True)
    init.add_argument('--source-sha256', dest='source_sha256', required=True)
    init.add_argument('--binary-sha256', dest='binary_sha256', required=True)
    reb = sub.add_parser('rebind')
    reb.add_argument('--parent-state', required=True)
    reb.add_argument('--state', required=True)
    reb.add_argument('--lane-root', required=True)
    reb.add_argument('--source-sha256', dest='source_sha256', required=True)
    reb.add_argument('--binary-sha256', dest='binary_sha256', required=True)
    rec = sub.add_parser('reconcile')
    rec.add_argument('--state', required=True)
    rec.add_argument('--phase', choices=PHASES, required=True)
    settle_p = sub.add_parser('settle')
    settle_p.add_argument('--state', required=True)
    settle_p.add_argument('--phase', choices=PHASES, required=True)
    stat_p = sub.add_parser('status')
    stat_p.add_argument('--state', required=True)
    args = parser.parse_args()
    if args.command == 'init':
        result = initialize(args)
    elif args.command == 'rebind':
        result = rebind(args)
    elif args.command == 'reconcile':
        result = reconcile(args)
    elif args.command == 'settle':
        result = settle(args)
    else:
        result = status(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
