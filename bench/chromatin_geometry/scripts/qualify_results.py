#!/usr/bin/env python3
"""Join retained KDA timing records to study admission, source, and raw campaigns.

This is an offline, fail-closed exporter. It never schedules work or modifies the
ledger. Each KDA round contributes one independent pair; samples within a
round are summarized together, not mistaken for independent trials.
"""
import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import math
import re
import statistics
import tarfile
import tomllib
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
CONTRACT = REPO / 'planning/chromatin-geometry-study-2026-11-02/contract-freeze-v2.json'
MATRIX = STUDY / 'docs/measurement-matrix-v2.json'
CONTROLS = {'scalar', 'bitmap', 'direct', 'library', 'library-batched'}
EXPERIMENTAL = {'cohort-direct', 'cohort-shared', 'cohort-direct-two-tile', 'masked-direct-two-tile'}
# Reviewed serialization guard plus input-retention wrapper in captured closure.
REVIEWED_WRAPPER_SHA256 = 'ac6a8d225c049abc9bc4703fe09f4e287b283a607d2fa71ce3431da02c58c6e7'
REVIEWED_DRIVER_SHA256 = 'f760a9f14bbf33069510bfda647ac3288adf705797a6606451ea9e1042d7544c'
REVIEWED_V3_WRAPPER_SHA256 = '3b2c7b14e2c960893c8465862d6657f97020ec10b6af14a8c083e81a9745399b'
REVIEWED_V3_DRIVER_SHA256 = '220e2b2340cead9b4ff857d361e02026604de0e297d1ca58d78837cf1afa8e7c'
REVIEWED_SANITIZER_ADAPTER_SHA256 = 'accf2f0a7a484329a3d936ef11e9ca18fe6645f14f1cb36ac22d9b9907bea40f'
REVIEWED_V3_SANITIZER_ADAPTER_SHA256 = '42ac54160093d6daf3a370f6230e804abc85b377cf2c89b908d2a143b68fcbb1'
REVIEWED_TRIAGE_ADAPTER_SHA256 = '6413a3bdbe95eeff881e2ef35bcb864d635168041f827eff647e635d52a7522a'
FIELDS = ('case', 'gate', 'body', 'precision', 'condition_update', 'input', 'seed',
          'reuse_horizon', 'parameter_update', 'nominal_input_change_percent',
          'applied_input_change_percent', 'tile', 'stages', 'workload_sha256', 'device_uuid')
BOUNDARY = 'cold_execution_preparation_ms + recurrent_complete_horizon_ms + output_recovery_ms; fixture, oracle, reset and warmup excluded'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stamp(value):
    return dt.datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()


def job_receipt(study, row, successful=False):
    job = read(study / row['receipt'])
    job = job.get('job', job)
    attempt = job['attempt']
    require(job.get('id') == row.get('job_id') and isinstance(attempt.get('attempt_id'), str) and bool(attempt['attempt_id']), 'Scheduler job/attempt mismatch')
    require(job.get('cleanup_verified') is True and attempt.get('cleanup_verified') is True, 'Scheduler cleanup unverified')
    require(job.get('state') in ('succeeded', 'completed', 'failed', 'cancelled', 'canceled', 'timed_out', 'abandoned'), 'Scheduler job not terminal')
    if successful:
        require(job.get('state') in ('succeeded', 'completed'), 'Scheduler job failed')
    start, end = stamp(attempt['created_at']), stamp(attempt['finished_at'])
    require(start <= end and end-start <= row['reserved_seconds'] + 1e-5, 'invalid attempt time or reservation')
    alloc = [a for a in attempt['execution_context']['allocations'] if a.get('resource_id', '').startswith('gpu:')]
    require(len(alloc) == 1 and alloc[0].get('mode') == 'exclusive' and alloc[0]['resource_id'] == row.get('expected_gpu_resource_id') == row.get('gpu_resource_id'), 'exclusive GPU allocation missing')
    return start, end, alloc[0]['resource_id'].split(':')[-1]


def unreserved_audit(study, row):
    """Read actual cancellation receipts without changing the budget ledger."""
    job_id = row.get('job_id')
    require(isinstance(job_id, str) and re.fullmatch(r'[0-9a-f-]{36}', job_id) is not None and row.get('packet_evidence') == 'runs/failures/' + job_id, 'invalid unreserved audit evidence location')
    receipt_path = study / row['packet_evidence'] / 'receipt.json'
    require(not receipt_path.is_symlink(), 'unreserved audit receipt must be a regular file')
    document = read(receipt_path)
    job = document.get('job', document)
    require(job.get('id') == job_id and job.get('job_id', job_id) == job_id and job.get('state') in ('cancelled', 'canceled') and job.get('cleanup_verified') is True and row.get('cleanup_verified') is True and row.get('terminal_state') in ('cancelled', 'canceled'), 'unreserved cancellation or cleanup unverified')
    spec_argv = job.get('spec', {}).get('argv', [])
    require(isinstance(spec_argv, list) and spec_argv and Path(spec_argv[-1]).name == str(row.get('attempt_id')) + '.json', 'unreserved Autobench attempt identity mismatch')
    evidence = {'job_id':job_id, 'classification':'unreserved auto-submission; cancelled and cleanup verified', 'receipt_path':str(receipt_path.relative_to(study)), 'receipt_sha256':sha(receipt_path), 'charged_seconds':row.get('charged_seconds')}
    attempt = job.get('attempt')
    if attempt is None:
        require(job.get('attempt_id') is None and job.get('lease_id') is None and job.get('lease_binding') is None and row.get('charged_seconds') == 0 and job.get('attempt_count', 0) in (None, 0), 'windowless cancellation has attempt, lease or charge')
        evidence['window_status'] = 'cancelled before Scheduler attempt; no GPU window inferred'
        return None, evidence
    require(isinstance(attempt, dict) and isinstance(attempt.get('attempt_id'), str) and job.get('attempt_id') == attempt['attempt_id'] and attempt.get('cleanup_verified') is True, 'unreserved Scheduler attempt mismatch or cleanup unverified')
    start, end = stamp(attempt['created_at']), stamp(attempt['finished_at'])
    charged = row.get('charged_seconds')
    require(isinstance(charged, (int,float)) and not isinstance(charged,bool) and math.isfinite(charged) and 0 <= end-start <= row['reserved_seconds'] + 1e-5 and abs(charged-(end-start)) <= 0.01, 'unreserved execution window or charge mismatch')
    alloc = [a for a in attempt['execution_context']['allocations'] if a.get('resource_id','').startswith('gpu:')]
    require(len(alloc) == 1 and alloc[0].get('mode') == 'exclusive', 'unreserved receipt does not establish one exclusive GPU')
    uuid = alloc[0]['resource_id'].split(':')[-1]
    require(uuid.startswith('GPU-'), 'unreserved receipt lacks physical GPU UUID')
    evidence.update(window_status='actual Scheduler attempt window', scheduler_attempt_id=attempt['attempt_id'], created_at=attempt['created_at'], finished_at=attempt['finished_at'], gpu_resource_id=alloc[0]['resource_id'])
    return (start,end,uuid), evidence


def reserved_windowless_cancellation(study, row):
    """Verify a submitted job canceled before any Scheduler attempt or lease."""
    receipt_path = study / row['receipt']
    require(not receipt_path.is_symlink(), 'cancellation receipt must be a regular file')
    document = read(receipt_path); job = document.get('job', document)
    require(job.get('id') == row.get('job_id') and job.get('job_id',row.get('job_id')) == row.get('job_id') and job.get('state') in ('cancelled','canceled') and job.get('cleanup_verified') is True and row.get('terminal_state') in ('cancelled','canceled') and row.get('worker_status') == 'not_started', 'unverified pre-attempt cancellation')
    require(job.get('attempt') is None and job.get('attempt_id') is None and job.get('lease_id') is None and job.get('lease_binding') is None and job.get('attempt_count', 0) in (None,0) and row.get('charged_seconds') == 0, 'cancellation has an attempt, lease or GPU charge')
    argv = job.get('spec',{}).get('argv',[])
    require(isinstance(argv,list) and argv and Path(argv[-1]).name == str(row.get('attempt_id')) + '.json', 'cancelled Autobench attempt identity mismatch')
    return {'job_id':row['job_id'],'classification':'reserved submission cancelled before Scheduler attempt; no GPU window inferred','receipt_path':row['receipt'],'receipt_sha256':sha(receipt_path),'charged_seconds':0}


def ledger_windows(study, ledger, return_audits=False):
    windows, audits, seen = {}, [], set()
    for row in ledger['reservations']:
        state = row.get('state')
        if state in ('no_submission', 'cpu_build'):
            continue
        require(state == 'settled', 'unsettled or unknown study GPU reservation')
        job = row.get('job_id')
        require(job and job not in seen, 'duplicate or missing study job')
        seen.add(job)
        if row.get('kind') == 'unreserved_auto_submission' and not row.get('receipt'):
            window, audit = unreserved_audit(study,row)
            audits.append(audit)
            if window is not None:
                windows[job] = window
            continue
        require(isinstance(row.get('receipt'), str) and bool(row['receipt']), 'settled study reservation lacks a ledger receipt path: ' + job)
        document = read(study/row['receipt']); scheduler_job = document.get('job',document)
        if scheduler_job.get('attempt') is None:
            audits.append(reserved_windowless_cancellation(study,row))
            continue
        windows[job] = job_receipt(study, row)
    return (windows, audits) if return_audits else windows


def telemetry(data, job, uuid):
    require(data.get('schema') == 'cgs.serial-telemetry.v1' and data.get('scheduler_job_id') == job and data.get('uuid') == uuid and data.get('clean') is True, 'telemetry identity or status mismatch')
    samples = data['samples']
    phases = [s.get('phase') for s in samples]
    require(len(samples) >= 3 and phases[0] == 'before' and phases[-1] == 'after' and 'during' in phases and phases == sorted(phases, key={'before': 0, 'during': 1, 'after': 2}.get), 'missing or unordered telemetry phases')
    inventory = None
    previous = 0
    for s in samples:
        require(s.get('issues') == [], 'telemetry recorded contamination or observation failure')
        ts = s.get('sampled_at_epoch_ns')
        require(isinstance(ts, int) and ts > previous, 'invalid telemetry timestamps')
        previous = ts
        devices = s.get('devices', [])
        ids = [d.get('uuid') for d in devices]
        require(uuid in ids and len(set(ids)) == len(ids), 'telemetry device inventory missing or duplicated')
        inventory = set(ids) if inventory is None else inventory
        require(inventory == set(ids), 'telemetry inventory changed')
        for d in devices:
            util = d.get('utilization_percent')
            require(type(util) is int and 0 <= util <= 100 and (d['uuid'] == uuid or util == 0), 'foreign device busy or invalid utilization')
            if (s is samples[0] or s is samples[-1]) and d['uuid'] == uuid:
                require(util == 0, 'target not quiescent')
        processes = s.get('processes')
        require(isinstance(processes, list) and all(p.get('uuid') == uuid and p.get('owned') is True for p in processes), 'foreign or unattributed compute process')
        if s['phase'] == 'during' and any(d['uuid'] == uuid and d['utilization_percent'] for d in devices):
            require(bool(processes), 'target activity not attributable')



def telemetry_v2(data, job, uuid, root, metrics, config, attempt_start, attempt_end, require_process_proof=False):
    """Recheck the bounded retrospective attribution against retained outputs."""
    require(data.get('schema') == ('cgs.serial-telemetry.v3' if require_process_proof else 'cgs.serial-telemetry.v2') and data.get('scheduler_job_id') == job and data.get('uuid') == uuid and data.get('clean') is True and data.get('retrospective_window_bound_ns') == 1000000000 and data.get('lifetime_issues') == [], 'telemetry v2 identity, bound or status mismatch')
    events_path = root / 'native-child-lifetimes.jsonl'
    require(sha(events_path) == data.get('lifetime_sha256'), 'native lifetime history hash mismatch')
    starts, ends = {}, {}
    for line in events_path.read_text().splitlines():
        event = json.loads(line)
        table = starts if event.get('event') == 'start' else ends if event.get('event') == 'end' else None
        require(table is not None and isinstance(event.get('invocation_id'), str) and event['invocation_id'] and event['invocation_id'] not in table, 'invalid or duplicate native lifetime event')
        if require_process_proof and event['event'] == 'end':
            require(event['invocation_id'] in starts, 'native child ended before its start event')
        table[event['invocation_id']] = event
    require(starts and set(starts) == set(ends), 'native lifetime history missing or unpaired')
    seen_pids = set()
    packet_command = read(root / 'registered-command.json')
    original_output = Path(packet_command['result_directory'])
    expected = config['cases'][0]
    for ident, child in starts.items():
        end = ends[ident]
        pid, argv = child.get('pid'), child.get('argv')
        require(type(pid) is int and pid > 0 and pid not in seen_pids and end.get('pid') == pid and child.get('device_uuid') == uuid and child.get('binary_sha256') == metrics['binary_sha256'] and end.get('returncode') == 0, 'native process PID, UUID, binary or exit mismatch')
        seen_pids.add(pid)
        if require_process_proof:
            require(type(child.get('proc_start_ticks')) is int and child['proc_start_ticks'] > 0 and end.get('proc_start_ticks') == child['proc_start_ticks'], 'native process start-tick identity mismatch')
        low, high, reap = child.get('spawn_lower_ns'), child.get('spawn_upper_ns'), end.get('reap_upper_ns')
        require(all(type(x) is int for x in (low, high, reap)) and attempt_start*1e9 <= low <= high <= reap <= attempt_end*1e9, 'invalid native process lifetime bounds')
        if require_process_proof:
            reap_lower = end.get('reap_lower_ns')
            require(type(reap_lower) is int and high <= reap_lower <= reap, 'invalid lower reap bound')
        require(isinstance(argv, list) and argv and all(isinstance(x, str) for x in argv) and Path(argv[0]).name == 'cgs_study', 'invalid native command')
        def flag(name):
            require(argv.count(name) == 1 and argv.index(name)+1 < len(argv), 'missing or duplicate native command flag: ' + name)
            return argv[argv.index(name)+1]
        require(flag('--mode') == 'campaign' and flag('--case') == expected['id'] and flag('--backend') == child['backend'] and flag('--output') == end.get('output_path'), 'native command or output identity mismatch')
        for arg, key in (('--gate','gate'), ('--body','body'), ('--precision','precision'), ('--tile','tile'), ('--stages','stages')):
            require(flag(arg) == str(expected['parameters'][key]), 'native command parameter mismatch: ' + arg)
        replay = expected['parameters'].get('replay')
        if replay:
            require(flag('--replay') == str(Path(metrics['binary_path']).parent / 'source/replay' / replay), 'native replay path differs from captured input')
        else:
            require('--replay' not in argv, 'unexpected native replay input')
        original = Path(end['output_path'])
        require(original.is_absolute() and '..' not in original.parts, 'invalid native output path')
        marker = '/runs/kda-experiments/attempts/'
        if marker in str(original):
            relative = str(original).split(marker, 1)[1]
            retained = root / 'kda/kda-workspace/runs/kda-experiments/attempts' / relative
            requests = [p for p in retained.parent.glob('*.json') if p.name not in ('campaign.json','reference-0.json','candidate-0.json') and not p.name.startswith(('reference-', 'candidate-'))]
            require(len(requests) == 1, 'native KDA request identity missing')
            request = read(requests[0])
            require(argv[0] == str(Path(request['candidate']['build']['directory']) / 'cgs_study'), 'native KDA binary command differs from retained request')
        else:
            require(original.is_relative_to(original_output), 'native output is outside retained packet')
            relative = original.relative_to(original_output)
            require(relative.parts[:2] in (('kda', 'supplemental'), ('kda', 'update-ablation')), 'native output is outside supplemental matrix')
            retained = root / relative
            require(argv[0] == metrics['binary_path'], 'supplemental native binary command differs from worker')
        if retained.is_relative_to(root / 'kda/update-ablation'):
            require(config.get('supplemental_updates') == ['rebuild', 'incremental'] and flag('--condition-update') in config['supplemental_updates'] and expected['parameters']['gate'] != 'ste', 'unapproved condition update mode')
        else:
            require(flag('--condition-update') == expected['parameters']['condition_update'], 'native condition update differs from frozen config')
        require(not retained.is_symlink() and sha(retained) == end.get('output_sha256'), 'native retained output hash mismatch')
        raw = read(retained)
        require(raw.get('condition_update') == flag('--condition-update'), 'native output condition update mismatch')
        require(raw.get('schema') == 'cgs.campaign.v1' and raw.get('status') == 'ok' and raw.get('accuracy_accepted') is True and raw.get('validation_scope') == 'first-and-last-complete-steps' and raw.get('device_uuid') == uuid and raw.get('backend') == child['backend'] and raw.get('case') == expected['id'] and raw.get('workload_sha256') == expected['input']['campaign_workload_sha256'], 'native output correctness or identity mismatch')
    samples = data.get('samples', [])
    phases = [s.get('phase') for s in samples]
    require(len(samples) >= 3 and phases[0] == 'before' and phases[-1] == 'after' and 'during' in phases and phases == sorted(phases, key={'before':0,'during':1,'after':2}.get), 'missing or unordered telemetry phases')
    inventory, previous = None, 0
    by_pid = {x['pid']: ident for ident, x in starts.items()}
    for sample in samples:
        require(sample.get('issues') == [], 'telemetry contamination or observation failure')
        a, b = sample.get('gpu_query_start_ns'), sample.get('gpu_query_end_ns')
        require(type(a) is int and type(b) is int and previous < a <= b and sample.get('sampled_at_epoch_ns') == a and attempt_start*1e9 <= a <= b <= attempt_end*1e9, 'invalid GPU query bounds')
        previous = a
        devices, processes = sample.get('devices', []), sample.get('processes')
        ids = [d.get('uuid') for d in devices]
        require(uuid in ids and len(ids) == len(set(ids)), 'telemetry inventory missing or duplicated')
        inventory = set(ids) if inventory is None else inventory
        require(set(ids) == inventory and isinstance(processes, list), 'telemetry inventory changed or process list missing')
        for d in devices:
            util = d.get('utilization_percent')
            require(type(util) is int and 0 <= util <= 100 and (d['uuid'] == uuid or util == 0), 'foreign GPU busy or invalid utilization')
            if sample is samples[0] or sample is samples[-1]:
                require(d['uuid'] != uuid or util == 0, 'target not quiescent at boundary')
        if require_process_proof:
            ps, pe = sample.get('process_query_start_ns'), sample.get('process_query_end_ns')
            require(type(ps) is int and type(pe) is int and b <= ps <= pe <= attempt_end*1e9, 'process-query time bounds missing or invalid')
            for process in processes:
                ident = process.get('invocation_id')
                require(process.get('uuid') == uuid and process.get('owned') is True and ident in starts and process.get('pid') == starts[ident]['pid'] == ends[ident]['pid'] and process.get('proc_start_ticks') == starts[ident]['proc_start_ticks'] == ends[ident]['proc_start_ticks'] and starts[ident]['spawn_upper_ns'] <= ps <= pe <= ends[ident]['reap_lower_ns'], 'foreign, reused, unverified or expired GPU process')
        else:
            require(all(p.get('uuid') == uuid and p.get('owned') is True and p.get('pid') in by_pid and ends[by_pid[p['pid']]]['reap_upper_ns'] >= b for p in processes), 'foreign, unknown or expired GPU process')
        busy = any(d['uuid'] == uuid and d['utilization_percent'] > 0 for d in devices)
        if busy and not processes:
            require(sample.get('requires_lifetime_attribution') is True, 'missing retrospective attribution request')
            claims = sample.get('attributed_invocation_ids')
            require(isinstance(claims, list) and claims and len(set(claims)) == len(claims), 'missing or duplicate retrospective owner IDs')
            for ident in claims:
                require(ident in starts and starts[ident]['spawn_upper_ns'] <= a and ends[ident]['reap_upper_ns'] >= a - 1000000000, 'retrospective owner does not overlap the bounded past window')
        else:
            require(not sample.get('requires_lifetime_attribution') and not sample.get('attributed_invocation_ids'), 'unexpected retrospective owner claim')
    return {'lifetime_sha256': sha(events_path), 'retrospective_window_bound_ns': 1000000000, 'process_query_identity_verified': require_process_proof}

def verify_archive(archive, config, frozen_path, matrix_path, wrapper_sha=REVIEWED_WRAPPER_SHA256, driver_sha=REVIEWED_DRIVER_SHA256):
    digest = config['source_archive_sha256']
    require(sha(archive) == digest, 'source archive hash mismatch')
    frozen = read(frozen_path)
    require(frozen.get('format') == 'cgs.contract-freeze.v1' and sha(frozen_path) == config['contract_freeze_sha256'], 'frozen contract mismatch')
    needed = {'Cellerator/' + item['path']: item['sha256'] for item in frozen['source_files'] if not item['path'].startswith('bench/chromatin_geometry/data/cache/')}
    needed['Cellerator/planning/chromatin-geometry-study-2026-11-02/' + frozen_path.name] = sha(frozen_path)
    needed['Cellerator/bench/chromatin_geometry/docs/' + matrix_path.name] = sha(matrix_path)
    needed['Cellerator/bench/chromatin_geometry/scripts/run_packet.py'] = wrapper_sha
    needed['Cellerator/bench/chromatin_geometry/scripts/kda_driver.py'] = driver_sha
    for case in config['cases']:
        if case['parameters'].get('replay'):
            needed['replay/' + case['parameters']['replay']] = case['input']['replay_sha256']
            public = frozen['public_replay']
            require(case['input']['replay_sha256'] == public['replay_sha256'], 'replay differs from frozen public input')
            needed.update({'replay/pbmc_unsorted_10k_filtered_feature_bc_matrix.h5': public['source_sha256'],
                           'replay/pbmc-atac-cgatac01/manifest.json': public['manifest_sha256'],
                           'replay/pbmc-atac-cgatac01/cells.tsv': public['cells_sha256'],
                           'replay/pbmc-atac-cgatac01/features.tsv': public['features_sha256']})
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        require(len(set(names)) == len(names) and all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members), 'unsafe or duplicate archive members')
        manifest = json.load(tar.extractfile('SOURCE-MANIFEST.json'))['files']
        require(set(names) == set(manifest) | {'SOURCE-MANIFEST.json'}, 'archive manifest membership mismatch')
        for name, expected in needed.items():
            require(name in manifest and manifest[name]['sha256'] == expected, 'frozen source, replay, matrix or runtime wrapper mismatch: ' + name)
        for name in sorted(needed, key=lambda n: tar.getmember(n).offset_data):
            expected = needed[name]
            h = hashlib.sha256()
            with tar.extractfile(name) as f:
                for block in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(block)
            require(h.hexdigest() == expected, 'archive member bytes mismatch: ' + name)
    return {'archive_sha256': digest, 'contract_sha256': sha(frozen_path), 'matrix_sha256': sha(matrix_path), 'wrapper_sha256': needed['Cellerator/bench/chromatin_geometry/scripts/run_packet.py'], 'driver_sha256': needed['Cellerator/bench/chromatin_geometry/scripts/kda_driver.py']}


def campaign(raw, response, row, expected, uuid):
    require(raw.get('schema') == 'cgs.campaign.v1' and raw.get('status') == 'ok' and raw.get('accuracy_accepted') is True and raw.get('validation_scope') == 'first-and-last-complete-steps', 'raw campaign correctness missing')
    require(raw.get('backend') == row['candidate_id'] and raw.get('case') == row['case_id'] and raw.get('device_uuid') == uuid, 'raw campaign arm identity mismatch')
    require(raw.get('workload_sha256') == expected['input']['campaign_workload_sha256'] == response.get('observed_workload_sha256'), 'raw workload hash mismatch')
    for key in ('gate', 'body', 'precision', 'condition_update', 'tile', 'stages'):
        require(raw.get(key) == expected['parameters'].get(key), 'campaign parameter mismatch: ' + key)
    require(raw.get('cold_device_preparation_includes_host_condition_maps') is True, 'missing condition-map preparation')
    cold = raw['cold_execution_preparation_ms']
    parts = [raw[k] for k in ('replay_input_preparation_ms', 'host_layout_preparation_ms', 'cold_device_preparation_ms')]
    require(all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0 for v in [cold, *parts]), 'invalid cold component')
    component = sum(parts)
    require(math.isclose(cold, component, rel_tol=1e-9, abs_tol=1e-8), 'cold preparation accounting mismatch')
    rec, out = raw['recurrent_complete_horizon_ms'], raw['output_recovery_ms']
    require(len(rec) > 0 and len(rec) == len(out) and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0 for v in [*rec, *out]), 'missing or invalid component timings')
    values = [1000 * (cold + r + d) for r, d in zip(rec, out)]
    require(all(math.isfinite(v) and v > 0 for v in values), 'invalid component timings')
    samples = json.loads(row['samples_json'])
    require(response.get('samples') == samples and len(values) == len(samples) and all(math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-6) for a,b in zip(values,samples)), 'KDA samples disagree with raw component accounting')
    require(response.get('status') == 'ok' and response.get('case_id') == row['case_id'] and response.get('request_id'), 'KDA response mismatch')
    return statistics.median(values)



def frozen_workload(raw, frozen, matrix):
    matches = [w for w in frozen['workloads'] if w['id'] == raw.get('case')]
    require(len(matches) == 1, 'case absent or duplicated in frozen contract')
    w = matches[0]
    require(raw.get('seed') == w['seed'] and raw.get('reuse_horizon') == w['reuse_horizon'] and raw.get('parameter_update') is (w['update_policy'] == 'update_every_step') and raw.get('nominal_input_change_percent') == w['requested_leaf_change_percent'] and raw.get('applied_input_change_percent') == (0 if raw['condition_update'] == 'none' else w['requested_leaf_change_percent']), 'workload seed, reuse or update differs from frozen contract')
    matching = [m for m in matrix['scenarios'] if matrix['workloads'][m['workload']] == raw['case'] and m['input'] == raw['input'] and m['gate'] == raw['gate'] and m['body'] == raw['body'] and m['precision'] == raw['precision'] and raw['condition_update'] in m['updates']]
    require(len(matching) == 1, 'workload is outside frozen measurement matrix')


def runtime_wrapper(root, row, config, expected_hash):
    identity = read(root / 'runtime-wrapper-identity.json')
    command_path, manifest_path = root / 'registered-command.json', root / 'snapshot-manifest.json'
    command, manifest_record = read(command_path), read(manifest_path)
    require(manifest_record.get('schema') == 'cgs.reconstructed-snapshot.v1' and manifest_record.get('snapshot_id') == identity.get('snapshot_id') and isinstance(manifest_record.get('files'), dict), 'snapshot reconstruction evidence missing')
    manifest = manifest_record['files']
    fingerprint = hashlib.sha256('\n'.join(f'{key}\0{manifest[key]}' for key in sorted(manifest)).encode()).hexdigest()
    require(fingerprint == identity.get('snapshot_id'), 'reconstructed snapshot fingerprint mismatch')
    wrapper_hash = sha(root / 'runtime-wrapper.py')
    require(identity.get('schema') == 'cgs.runtime-wrapper.v1' and identity.get('job_id') == row['job_id'] and identity.get('attempt_id') == row['attempt_id'] and identity.get('wrapper_sha256') == wrapper_hash == expected_hash and identity.get('packet_manifest_sha256') == sha(command_path) and identity.get('snapshot_manifest_sha256') == sha(manifest_path), 'retained runtime wrapper identity mismatch')
    wd = Path(command['working_directory'])
    argv = command['argv']
    require(wd.name == identity.get('snapshot_id') and len(argv) > 2 and argv[0] == 'python3' and argv[1] == str(wd / 'scripts/run_packet.py') and argv[2] == 'ste-test', 'registered command is not bound to immutable wrapper snapshot')
    require(Path(command['result_directory']).name == row['attempt_id'] and command['result_directory'] == argv[argv.index('--output')+1], 'registered command scratch attempt mismatch')
    require(argv[argv.index('--archive-sha256')+1] == config['source_archive_sha256'] and argv[argv.index('--suite')+1] == 'kda' and argv[argv.index('--uuid')+1] == row['gpu_resource_id'].split(':')[-1] and argv[argv.index('--config')+1] == str(wd / 'experiment.json'), 'registered command source, suite, device or configuration mismatch')
    require(manifest.get('scripts/run_packet.py') == wrapper_hash and manifest.get('.autobench/data/closure.tar.gz') == config['source_archive_sha256'] and manifest.get('experiment.json') == sha(root / 'experiment.json') and manifest.get('contract-freeze.json') == sha(root / 'contract-freeze.json') and manifest.get('interference-evidence.json') == sha(root / 'interference-evidence.json') and manifest.get('.autobench/data/frozen-binary') == config['binary_sha256'] and identity.get('snapshot_inputs') == manifest, 'snapshot manifest inputs mismatch')
    with (root / 'registration.toml').open('rb') as stream:
        registration = tomllib.load(stream)
    packets = registration.get('packet', [])
    require(len(packets) == 1 and packets[0].get('kill_seconds', 901) <= 900 and packets[0].get('argv', [None, None])[1] == '{snapshot}/scripts/run_packet.py', 'registration does not bind a bounded wrapper packet')
    gpu = [x for x in packets[0].get('resources', {}).get('scheduler', []) if x.get('kind') == 'gpu']
    require(len(gpu) == 1 and gpu[0].get('mode') == 'exclusive' and gpu[0].get('selectors', {}).get('resource_id') == row['gpu_resource_id'], 'registration lacks the expected exclusive GPU')
    return {'runtime_wrapper_sha256': wrapper_hash, 'runtime_wrapper_identity_sha256': sha(root / 'runtime-wrapper-identity.json'), 'registered_command_sha256': sha(command_path), 'snapshot_manifest_sha256': sha(manifest_path), 'registration_sha256': sha(root / 'registration.toml'), 'snapshot_id': identity['snapshot_id'], 'snapshot_manifest_origin': 'independently reconstructed and fingerprint verified; original SDK manifest unavailable'}


def verify_triage_adapter(study, row, metrics, summary, config):
    """Bind exact-binary triage to its actual snapshot, receipt and raw arms."""
    root = study / row['packet_evidence']
    identity = read(root / 'runtime-wrapper-identity.json')
    command_path, manifest_path = root / 'registered-command.json', root / 'snapshot-manifest.json'
    command, manifest_record = read(command_path), read(manifest_path)
    files = manifest_record['files']
    fingerprint = hashlib.sha256('\n'.join(f'{k}\0{files[k]}' for k in sorted(files)).encode()).hexdigest()
    require(manifest_record.get('schema') == 'cgs.reconstructed-snapshot.v1' and manifest_record.get('snapshot_id') == identity.get('snapshot_id') == fingerprint and identity.get('snapshot_inputs') == files and identity.get('schema') == 'cgs.runtime-wrapper.v1' and identity.get('job_id') == row['job_id'] and identity.get('attempt_id') == row['attempt_id'] and identity.get('packet_manifest_sha256') == sha(command_path) and identity.get('snapshot_manifest_sha256') == sha(manifest_path), 'triage snapshot identity mismatch')
    require(identity.get('wrapper_sha256') == files.get('scripts/triage_binary.py') == sha(root/'runtime-wrapper.py') == REVIEWED_TRIAGE_ADAPTER_SHA256, 'triage runtime adapter mismatch')
    build = read(study/row['build_evidence'])
    require(build.get('schema') == 'cgs.build.v1' and build.get('suite') == 'triage-binary' and build.get('target') == 'cgs_study' and build.get('archive_sha256') == metrics.get('archive_sha256') and build.get('binary_sha256') == metrics.get('binary_sha256'), 'triage adapter build identity mismatch')
    original, original_sha = original_build(study,row)
    require(files.get('.autobench/data/closure.tar.gz') == metrics['archive_sha256'] and files.get('.autobench/data/frozen-binary') == metrics['binary_sha256'] and files.get('.autobench/data/frozen-build.json') == original_sha and files.get('experiment.json') == sha(root/'experiment.json') and original.get('target') == 'cgs_study', 'triage frozen inputs differ from exact original build')
    wd=Path(command['working_directory']); argv=command['argv']
    require(wd.name == fingerprint and argv[:3] == ['python3',str(wd/'scripts/triage_binary.py'),'run'] and Path(command['result_directory']).name == row['attempt_id'], 'triage registered command or scratch identity mismatch')
    for flag, value in (('--config',str(wd/'experiment.json')),('--uuid',metrics['uuid']),('--output',str(Path(command['result_directory'])/'triage-packet'))):
        require(argv.count(flag) == 1 and argv[argv.index(flag)+1] == value, 'triage command mismatch: ' + flag)
    registration = tomllib.loads((root/'registration.toml').read_text()); packets=registration.get('packet',[])
    require(len(packets) == 1 and packets[0].get('kill_seconds',901) <= 900 and packets[0].get('argv',[None,None])[1] == '{snapshot}/scripts/triage_binary.py', 'triage registration mismatch')
    gpu = [x for x in packets[0].get('resources',{}).get('scheduler',[]) if x.get('kind') == 'gpu']
    require(len(gpu) == 1 and gpu[0].get('mode') == 'exclusive' and gpu[0].get('selectors',{}).get('resource_id') == row['gpu_resource_id'], 'triage registration lacks exclusive GPU')
    receipt = read(study/row['receipt']); attempt = receipt.get('job',receipt)['attempt']['attempt_id']; worker = read(root/'result.json')
    require(metrics.get('schema') == 'cgs.packet.v1' and metrics.get('suite') == 'triage' and metrics.get('scheduler_job_id') == row['job_id'] and metrics.get('scheduler_attempt_id') == attempt and metrics.get('returncode') == 0 and worker.get('scheduler_job_id') == row['job_id'] and worker.get('scheduler_attempt_id') == attempt and worker.get('status') == 'succeeded' and worker.get('returncode') == 0, 'triage worker or Scheduler attempt mismatch')
    nested = root/'triage-packet'
    aliases_path = root/'triage-aliases.json'
    alias_manifest = read(aliases_path)
    require(alias_manifest.get('schema') == 'cgs.triage-aliases.v1' and isinstance(alias_manifest.get('files'), list), 'triage alias manifest missing')
    aliases = {entry.get('alias'):entry for entry in alias_manifest['files']}
    require(len(aliases) == len(alias_manifest['files']), 'duplicate triage aliases')
    def checked_alias(relative):
        entry = aliases.get(relative, {})
        require(entry.get('original') == 'triage-packet/' + relative and entry.get('sha256') == sha(root/relative) == sha(nested/relative), 'triage alias differs from its retained nested original: ' + relative)
    for relative in ('metrics.json','experiment.json','triage/summary.json'):
        checked_alias(relative)
    original_config = read(root/'experiment.json'); cases = original_config.get('runs',[]); records = summary.get('records',[])
    require(original_config.get('schema') == 'cgs.triage.v1' and summary.get('schema') == 'cgs.triage-result.v1' and summary.get('status') == 'pass' and summary.get('device_uuid') == metrics['uuid'] and len(cases) == len(records) and bool(cases), 'triage case summary incomplete')
    expected = config['cases'][0]; expected_arms = {config['incumbent'], *config['candidates']}; seen = set(); evidence=[]
    for i,(case,record) in enumerate(zip(cases,records)):
        require(record.get('run') == i and record.get('validation') == 'pass' and record.get('backend') == case.get('backend') and record.get('match_group') == case.get('match_group'), 'triage record identity mismatch')
        raw_path = root/'triage'/f'run-{i:02d}.json'; raw=read(raw_path)
        checked_alias(f'triage/run-{i:02d}.json')
        require(raw.get('schema') == 'cgs.campaign.v1' and raw.get('status') == 'ok' and raw.get('accuracy_accepted') is True and raw.get('validation_scope') == 'first-and-last-complete-steps' and raw.get('device_uuid') == metrics['uuid'] and raw.get('workload_sha256') == record.get('workload_sha256') and all(raw.get(k) == case.get(k) for k in ('case','backend','gate','body','precision','condition_update','tile','stages')), 'triage raw complete-chain identity mismatch')
        if case.get('backend') in expected_arms and case.get('case') == expected['id'] and all(case.get(k) == expected['parameters'].get(k) for k in ('gate','body','precision','condition_update','tile','stages')) and raw.get('workload_sha256') == expected['input']['campaign_workload_sha256'] and case.get('replay_sha256') == expected['input'].get('replay_sha256'):
            require(case['backend'] not in seen, 'duplicate matched triage arm')
            seen.add(case['backend']); evidence.append({'backend':case['backend'],'path':str(raw_path.relative_to(study)),'original_path':str((nested/'triage'/f'run-{i:02d}.json').relative_to(study)),'sha256':sha(raw_path)})
    require(seen == expected_arms, 'triage omits a matched required arm')
    return {'triage_adapter_sha256':sha(root/'runtime-wrapper.py'),'triage_snapshot_id':fingerprint,'triage_snapshot_sha256':sha(manifest_path),'triage_command_sha256':sha(command_path),'triage_original_build_receipt_sha256':original_sha,'triage_alias_manifest_sha256':sha(aliases_path),'triage_config_sha256':sha(root/'experiment.json'),'triage_summary_sha256':sha(root/'triage/summary.json'),'triage_arm_records':evidence}

def packet(study, row, windows, frozen_path, matrix_path, archive_cache, wrapper_sha=REVIEWED_WRAPPER_SHA256, driver_sha=REVIEWED_DRIVER_SHA256):
    job = row['job_id']
    require(row.get('state') == 'settled' and row.get('serialized_confirmation') is True, 'reservation is not settled serialized confirmation')
    require(row.get('worker_status') == 'succeeded', 'GPU worker did not succeed: ' + str(row.get('worker_status')))
    start, end, uuid = job_receipt(study, row, True)
    require(all(other == job or end <= w[0] or start >= w[1] for other, w in windows.items()), 'overlapping study GPU job')
    root = study / row['packet_evidence']
    metrics, worker = read(root / 'metrics.json'), read(root / 'result.json')
    receipt_data = read(study / row['receipt'])
    scheduler_attempt = receipt_data.get('job', receipt_data)['attempt']['attempt_id']
    for obj in (metrics, worker):
        require(obj.get('scheduler_job_id') == job and obj.get('scheduler_attempt_id') == scheduler_attempt, 'worker Scheduler job/attempt mismatch')
    require(metrics.get('schema') == 'cgs.packet.v1' and metrics.get('suite') == 'kda' and metrics.get('uuid') == uuid and metrics.get('success') is True and metrics.get('serial_telemetry_clean') is True and metrics.get('returncode') == 0 and worker.get('status') == 'succeeded' and worker.get('returncode') == 0, 'worker packet failed')
    guard = read(root / 'serial-telemetry.json')
    lifetime_provenance = telemetry_v2(guard, job, uuid, root, metrics, read(root / 'experiment.json'), start, end, require_process_proof=frozen_path.name == 'contract-freeze-v3.json')
    config = read(root / 'experiment.json')
    require(sha(root / 'contract-freeze.json') == sha(frozen_path), 'retained contract bytes mismatch')
    require(config.get('schema') == 'cgs.kda-config.v1' and len(config.get('cases', [])) == 1, 'invalid KDA config')
    require(config.get('source_archive_sha256') == metrics.get('archive_sha256') and config.get('binary_sha256') == metrics.get('binary_sha256'), 'config source or binary mismatch')
    require(read(root / 'kda/experiment.json') == config, 'worker and engine config differ')
    engine = read(root / 'kda/metrics.json')
    require(engine.get('schema') == 'cgs.kda-packet.v1' and engine.get('scheduler_job_id') == job and engine.get('uuid') == uuid and engine.get('exit_code') == 0 and engine.get('engine_result', {}).get('status') in ('complete', 'ok'), 'KDA packet failed')
    fallback_path = root / 'interference-evidence.json'
    fallback = read(fallback_path)
    require(sha(fallback_path) == config.get('interference_attestation', {}).get('sha256'), 'serialized guard attestation hash mismatch')
    require(fallback.get('schema') == 'cgs.serialized-fallback.v1' and fallback.get('device_uuid') == uuid and fallback.get('source_archive_sha256') == metrics['archive_sha256'] and fallback.get('binary_sha256') == metrics['binary_sha256'] and fallback.get('workload_sha256') == config['cases'][0]['input']['campaign_workload_sha256'] and fallback.get('triage_job_id') == config.get('triage', {}).get('job_id') and fallback.get('controller_serialization_required') is True and fallback.get('all_device_telemetry_required') is True, 'serialized fallback does not bind measurement')
    build_path = study / row['build_evidence']
    build = read(build_path)
    binary = study / 'runs/builds' / metrics['binary_sha256'] / 'binary'
    require(build.get('schema') == 'cgs.build.v1' and build.get('archive_sha256') == metrics['archive_sha256'] and build.get('binary_sha256') == metrics['binary_sha256'] and sha(binary) == metrics['binary_sha256'], 'build or binary identity mismatch')
    archive = study / 'runs/closures' / (metrics['archive_sha256'] + '.tar.gz')
    cache_key = (str(archive), sha(frozen_path), sha(matrix_path), wrapper_sha, driver_sha, json.dumps(config['cases'], sort_keys=True))
    if cache_key not in archive_cache:
        archive_cache[cache_key] = verify_archive(archive, config, frozen_path, matrix_path, wrapper_sha, driver_sha)
    wrapper = runtime_wrapper(root, row, config, archive_cache[cache_key]['wrapper_sha256'])
    provenance = dict(archive_cache[cache_key], **wrapper, **lifetime_provenance, binary_sha256=sha(binary), build_receipt_sha256=sha(build_path), scheduler_receipt_sha256=sha(study / row['receipt']), packet_metrics_sha256=sha(root / 'metrics.json'), worker_result_sha256=sha(root / 'result.json'), telemetry_sha256=sha(root / 'serial-telemetry.json'), fallback_sha256=sha(fallback_path), config_sha256=sha(root / 'experiment.json'))
    triage = config['triage']
    troot = study / 'runs/packets' / triage['job_id']
    require(sha(study / 'runs/receipts' / (triage['job_id'] + '.json')) == triage['receipt_sha256'] and sha(troot / 'metrics.json') == triage['packet_metrics_sha256'], 'triage evidence hash mismatch')
    tm, ts = read(troot / 'metrics.json'), read(troot / 'triage/summary.json')
    require(tm.get('success') is True and tm.get('archive_sha256') == metrics['archive_sha256'] and tm.get('binary_sha256') == metrics['binary_sha256'] and tm.get('uuid') == uuid and ts.get('status') == 'pass', 'triage source, binary, device or correctness mismatch')
    trow = next((r for r in read(study / 'runs/campaign-budget.json')['reservations'] if r.get('job_id') == triage['job_id'] and r.get('state') == 'settled'), None)
    require(trow is not None, 'triage reservation missing')
    job_receipt(study, trow, True)
    if frozen_path.name == 'contract-freeze-v3.json':
        provenance.update(verify_triage_adapter(study, trow, tm, ts, config))
    snapshot_files = read(root / 'snapshot-manifest.json')['files']
    require(snapshot_files.get('.autobench/data/frozen-build.json') == sha(study / trow['build_evidence']), 'snapshot frozen build receipt differs from triage')
    expected = config['cases'][0]
    workspace = root / 'kda/kda-workspace'
    with (workspace / 'benchmark.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    require(bool(rows), 'no retained KDA measurements')
    require(all(r.get('metric') == 'complete_operational_horizon' and r.get('unit') == 'us' and r.get('direction') == 'minimize' and r.get('case_id') == expected['id'] for r in rows), 'KDA timing scope mismatch')
    phases = {}
    for index, r in enumerate(rows):
        require(r['candidate_id'] in CONTROLS | EXPERIMENTAL, 'unknown measurement arm')
        run, phase, rnd = r['run_id'], r['phase'], int(r['round'])
        require(phase in ('exploration', 'confirmation'), 'unexpected KDA timing phase')
        key = (run, phase, rnd, r['candidate_id'])
        require(key not in phases, 'duplicate KDA arm/round')
        results = list((workspace / 'runs/kda-experiments/attempts' / run / 'stages' / r['candidate_id'] / phase / f'round-{rnd:03d}').glob('*/responses/*.json'))
        require(len(results) == 1 and sha(results[0]) == r['result_sha256'], 'missing or mismatched KDA measurement response')
        response = read(results[0])
        stage = results[0].parent.parent
        request = read(stage / 'requests' / (response['request_id'] + '.json'))
        raw_path = stage / 'requests/campaign.json'
        raw = read(raw_path)
        frozen_workload(raw, read(frozen_path), read(matrix_path))
        require(request.get('request_id') == response['request_id'] and request.get('round') == rnd and request.get('phase') == phase and request.get('candidate', {}).get('id') == r['candidate_id'], 'KDA request identity mismatch')
        case_request = request.get('case', {})
        case_descriptor = {k:v for k,v in case_request.items() if k != 'workload_sha256'}
        require(case_request.get('id') == expected['id'] and request.get('environment', {}).get('device_uuid') == uuid and digest_json(case_descriptor) == case_request.get('workload_sha256') == r['workload_sha256'] == response.get('workload_sha256') and digest_json(request['environment']) == r['environment_fingerprint'], 'KDA request workload, environment or digest mismatch')
        require(case_request.get('input') == expected['input'] and all(case_request.get('parameters', {}).get(k) == expected['parameters'].get(k) for k in ('gate', 'body', 'precision', 'condition_update', 'tile', 'stages')), 'KDA input or operation identity mismatch')
        value = campaign(raw, response, r, expected, uuid)
        # The independent per-arm validation stages are retained alongside timing.
        validation = workspace / 'runs/kda-experiments/attempts' / run / 'stages' / r['candidate_id'] / 'validation'
        responses = list((validation / 'responses').glob('*.json'))
        require(len(responses) == 1, 'missing per-arm validation response')
        valid = read(responses[0])
        require(valid.get('status') == 'ok' and len(valid.get('cases', [])) == 1 and valid['cases'][0].get('status') == 'pass' and valid['cases'][0].get('observed_workload_sha256') == raw['workload_sha256'], 'per-arm validation failed')
        # The raw reference and candidate independently validate actual outputs.
        for name, arm in [('reference-0.json', 'scalar'), ('candidate-0.json', r['candidate_id'])]:
            vraw = read(validation / 'requests' / name)
            require(vraw.get('status') == 'ok' and vraw.get('accuracy_accepted') is True and vraw.get('validation_scope') == 'first-and-last-complete-steps' and vraw.get('backend') == arm and all(vraw.get(k) == raw.get(k) for k in FIELDS), 'raw per-arm validation mismatch')
        phases[key] = {'index': index, 'us': value, 'raw': raw, 'raw_sha256': sha(raw_path), 'response_sha256': sha(results[0])}
    # KDA's two-arm confirmation is exploratory. The separately retained
    # complete-control matrix runs inside this same monitored worker packet.
    supplement_root = root / 'kda/supplemental'
    supplement_path = supplement_root / 'summary.json'
    supplement = read(supplement_path)
    declared = [config['incumbent'], *config['candidates']]
    allowed_arms = read(matrix_path).get('execution_arms')
    if allowed_arms is not None:
        require(set(declared) <= set(allowed_arms), 'configured arm outside selected frozen matrix')
    require(supplement.get('schema') == 'cgs.supplemental.v1' and supplement.get('scheduler_job_id') == job and supplement.get('device_uuid') == uuid and supplement.get('source_archive_sha256') == metrics['archive_sha256'] and supplement.get('binary_sha256') == metrics['binary_sha256'], 'supplemental matrix identity mismatch')
    require(supplement.get('status') == 'pass' and len(declared) == len(set(declared)) and set(supplement.get('arms', [])) == set(declared), 'supplemental matrix failed or arms differ from config')
    require(len(supplement.get('rounds', [])) == 6, 'supplemental matrix requires six independent rounds')
    run_ids = {k[0] for k in phases}
    require(len(run_ids) == 1, 'KDA packet contains mixed engine runs')
    run = next(iter(run_ids))
    orders = []
    for n, round_data in enumerate(supplement['rounds']):
        require(round_data.get('round_id') == f'round-{n:02d}', 'duplicate or unordered supplemental round')
        order = round_data['order']
        records = round_data['records']
        require(len(order) == len(declared) and set(order) == set(declared) and len(records) == len(declared) and [r.get('backend') for r in records] == order, 'supplemental round omits or duplicates arms')
        orders.append(order)
        for position, record in enumerate(records):
            arm = record['backend']
            require(record.get('position') == position and record.get('status') == 'pass', 'supplemental arm failed or order mismatched')
            relative = Path(record['path'])
            require(not relative.is_absolute() and '..' not in relative.parts, 'unsafe supplemental evidence path')
            path = supplement_root / relative
            require(not path.is_symlink() and sha(path) == record.get('raw_sha256'), 'supplemental raw hash mismatch')
            raw = read(path)
            frozen_workload(raw, read(frozen_path), read(matrix_path))
            require(record.get('workload_sha256') == raw.get('workload_sha256'), 'supplemental workload mismatch')
            row_data = {'candidate_id': arm, 'case_id': expected['id'], 'samples_json': json.dumps([record['complete_us']])}
            response = {'status': 'ok', 'request_id': 'supplemental', 'case_id': expected['id'], 'observed_workload_sha256': record['workload_sha256'], 'samples': [record['complete_us']]}
            value = campaign(raw, response, row_data, expected, uuid)
            reference = [v['raw'] for k,v in phases.items() if k[0] == run and k[3] == arm]
            require(reference and all(raw.get(f) == reference[0].get(f) for f in FIELDS), 'supplemental arm lacks matching KDA validation and workload')
            phases[(run, 'supplemental', n, arm)] = {'index': n*len(order)+position, 'us': value, 'raw': raw, 'raw_sha256': sha(path), 'raw_path': str(path.relative_to(study))}
    require(all(orders[i+1] == list(reversed(orders[i])) for i in (0,2,4)), 'supplemental paired order does not rotate and reverse')
    updates = update_ablation_records(study, root, config, metrics, expected, uuid, frozen_path, matrix_path, declared, phases, run)
    return {'job': job, 'uuid': uuid, 'provenance': provenance, 'config': config, 'phases': phases, 'benchmark_sha256': sha(workspace / 'benchmark.csv'), 'supplemental_sha256': sha(supplement_path), 'updates': updates, 'matrix_schema':read(matrix_path).get('schema')}



def update_ablation_records(study, root, config, metrics, expected, uuid, frozen_path, matrix_path, declared, phases, run):
    modes = config.get('supplemental_updates')
    update_root = root / 'kda/update-ablation'
    if modes is None:
        require(not update_root.exists(), 'undeclared update ablation')
        return None
    require(modes == ['rebuild', 'incremental'] and expected['parameters']['gate'] != 'ste' and expected['parameters']['condition_update'] in modes, 'update pairing is not declared for an applicable scenario')
    summary_path = update_root / 'summary.json'
    summary = read(summary_path)
    arms = [f'{arm}@{mode}' for arm in declared for mode in modes]
    require(summary.get('schema') == 'cgs.update-ablation.v1' and summary.get('scheduler_job_id') == metrics['scheduler_job_id'] and summary.get('device_uuid') == uuid and summary.get('source_archive_sha256') == metrics['archive_sha256'] and summary.get('binary_sha256') == metrics['binary_sha256'] and summary.get('update_modes') == modes and summary.get('status') == 'pass' and len(summary.get('arms',[])) == len(arms) and set(summary['arms']) == set(arms), 'update ablation identity, mode or status mismatch')
    require(len(summary.get('rounds',[])) == 6, 'update ablation requires six paired rounds')
    records, orders, workload_fields = {}, [], tuple(f for f in FIELDS if f != 'condition_update')
    for n, rd in enumerate(summary['rounds']):
        order, rows = rd['order'], rd['records']
        require(rd.get('round_id') == f'round-{n:02d}' and len(order) == len(arms) and set(order) == set(arms) and len(rows) == len(arms) and [r.get('arm_id') for r in rows] == order, 'update ablation round is incomplete or unordered')
        orders.append(order)
        for position, record in enumerate(rows):
            arm, mode = record.get('backend'), record.get('update_mode')
            require(arm in declared and mode in modes and record.get('condition_update') == mode and record.get('arm_id') == f'{arm}@{mode}' and record.get('position') == position and record.get('status') == 'pass', 'update ablation arm identity or status mismatch')
            relative = Path(record['path'])
            require(not relative.is_absolute() and '..' not in relative.parts, 'unsafe update ablation path')
            path = update_root / relative
            require(not path.is_symlink() and sha(path) == record.get('raw_sha256'), 'update ablation raw hash mismatch')
            raw = read(path)
            frozen_workload(raw, read(frozen_path), read(matrix_path))
            require(raw.get('applied_input_change_percent', 0) > 0 and record.get('workload_sha256') == raw.get('workload_sha256'), 'update ablation does not change the same input workload')
            matched = json.loads(json.dumps(expected)); matched['parameters']['condition_update'] = mode
            row_data = {'candidate_id':arm, 'case_id':expected['id'], 'samples_json':json.dumps([record['complete_us']])}
            response = {'status':'ok','request_id':'update-ablation','case_id':expected['id'],'observed_workload_sha256':record['workload_sha256'],'samples':[record['complete_us']]}
            value = campaign(raw, response, row_data, matched, uuid)
            reference = [v['raw'] for k,v in phases.items() if k[0] == run and k[3] == arm]
            require(reference and all(raw.get(f) == reference[0].get(f) for f in workload_fields) and raw.get('numerical_contract') == reference[0].get('numerical_contract'), 'update ablation differs from validated workload or numerical contract')
            records[(n,arm,mode)] = {'index':n*len(order)+position,'us':value,'raw':raw,'raw_path':str(path.relative_to(study)),'raw_sha256':sha(path)}
    require(all(orders[i+1] == list(reversed(orders[i])) for i in (0,2,4)), 'update ablation order is not reversed')
    return {'summary_sha256':sha(summary_path),'records':records}


def numerical_archive_manifest(archive, expected, contract_name='contract-freeze-v2.json'):
    """Hash all numerical source/header bytes and verify the archive manifest."""
    require(sha(archive) == expected, 'sanitizer source archive hash mismatch')
    def selected(name):
        return (name.startswith(('Cellerator/include/', 'Cellerator/src/', 'Cellerator/cmake/', 'Cellerator/components/CellShard/include/')) or name in ('Cellerator/CMakeLists.txt', 'Cellerator/bench/chromatin_geometry/CMakeLists.txt', 'Cellerator/planning/chromatin-geometry-study-2026-11-02/' + contract_name) or name.startswith('Cellerator/bench/chromatin_geometry/') and name.endswith(('.c', '.cc', '.cu', '.h', '.hh', '.cuh')))
    actual, manifest = {}, None
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            require(member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts, 'unsafe numerical source archive')
            if member.name == 'SOURCE-MANIFEST.json':
                require(manifest is None, 'duplicate source manifest')
                manifest = json.load(tar.extractfile(member))['files']
            elif selected(member.name):
                require(member.name not in actual, 'duplicate numerical source archive member')
                h = hashlib.sha256()
                with tar.extractfile(member) as stream:
                    for block in iter(lambda: stream.read(1024*1024), b''):
                        h.update(block)
                actual[member.name] = h.hexdigest()
    require(manifest is not None and actual and actual == {p: info['sha256'] for p,info in manifest.items() if selected(p)}, 'numerical archive files differ from manifest')
    return actual


def sanitizer_snapshot(root, row, metric, original_build_sha, adapter_sha=REVIEWED_SANITIZER_ADAPTER_SHA256, backend='cohort-shared'):
    identity = read(root / 'runtime-wrapper-identity.json')
    command_path, manifest_path = root / 'registered-command.json', root / 'snapshot-manifest.json'
    command, reconstructed = read(command_path), read(manifest_path)
    require(reconstructed.get('schema') == 'cgs.reconstructed-snapshot.v1' and reconstructed.get('snapshot_id') == identity.get('snapshot_id'), 'sanitizer snapshot reconstruction missing')
    manifest = reconstructed['files']
    digest = hashlib.sha256('\n'.join(f'{k}\0{manifest[k]}' for k in sorted(manifest)).encode()).hexdigest()
    wrapper_hash = sha(root / 'runtime-wrapper.py')
    require(digest == identity.get('snapshot_id') and identity.get('snapshot_inputs') == manifest and identity.get('schema') == 'cgs.runtime-wrapper.v1' and identity.get('job_id') == row['job_id'] and identity.get('attempt_id') == row['attempt_id'] and identity.get('wrapper_sha256') == wrapper_hash == manifest.get('scripts/sanitize_chain.py') == adapter_sha and identity.get('packet_manifest_sha256') == sha(command_path) and identity.get('snapshot_manifest_sha256') == sha(manifest_path), 'sanitizer snapshot identity mismatch')
    require(manifest.get('.autobench/data/closure.tar.gz') == metric['archive_sha256'] and manifest.get('.autobench/data/frozen-binary') == metric['binary_sha256'] and manifest.get('.autobench/data/frozen-build.json') == original_build_sha, 'sanitizer snapshot source, binary or build mismatch')
    wd = Path(command['working_directory']); argv = command['argv']
    require(wd.name == digest and argv[:3] == ['python3',str(wd/'scripts/sanitize_chain.py'),'run'] and Path(command['result_directory']).name == row['attempt_id'] and argv[argv.index('--sanitizer')+1] == metric['sanitizer'] and argv[argv.index('--uuid')+1] == metric['uuid'] and (backend == 'cohort-shared' and ('--backend' not in argv or argv[argv.index('--backend')+1] == backend) or backend != 'cohort-shared' and '--backend' in argv and argv[argv.index('--backend')+1] == backend), 'sanitizer registered command mismatch')
    return {'snapshot_id':digest,'reconstructed_snapshot_sha256':sha(manifest_path),'adapter_sha256':wrapper_hash,'registered_command_sha256':sha(command_path)}


def original_build(study, row):
    build = read(study / row['build_evidence'])
    original_sha = build.get('original_build_receipt_sha256')
    if original_sha:
        path = study / 'runs/builds' / build['binary_sha256'] / (original_sha + '.json')
        require(sha(path) == original_sha, 'original binary build receipt hash mismatch')
        original = read(path)
        require(original.get('archive_sha256') == build.get('archive_sha256') and original.get('binary_sha256') == build.get('binary_sha256'), 'copied binary and original build receipt differ')
        return original, original_sha
    return build, sha(study / row['build_evidence'])


def compiler_signature(build):
    command = build.get('command')
    require(isinstance(command,list) and command and command[0] == 'cmake', 'original CMake build command unavailable')
    flags = sorted(x for x in command if x.startswith('-D'))
    require(flags and any(x == '-DCMAKE_CUDA_ARCHITECTURES=70' for x in flags), 'original CUDA compiler flags missing')
    toolchain = build.get('toolchain')
    require(isinstance(toolchain,dict) and toolchain.get('tools') and toolchain.get('linked_file_sha256'), 'original compiler or library identity missing')
    return {'flags': flags, 'tools':toolchain['tools'], 'linked_file_sha256':toolchain['linked_file_sha256'], 'host_pkg_config_versions':toolchain.get('host_pkg_config_versions')}


def sanitizer_coverage(study, ledger, measured, frozen_path=CONTRACT):
    needed = {'memcheck','racecheck','initcheck','synccheck'}
    is_v3 = frozen_path.name == 'contract-freeze-v3.json'
    expected_backend = 'cohort-direct-two-tile' if is_v3 else 'cohort-shared'
    expected_stages = 1 if is_v3 else 2
    adapter_sha = REVIEWED_V3_SANITIZER_ADAPTER_SHA256 if is_v3 else REVIEWED_SANITIZER_ADAPTER_SHA256
    triage_id = measured['config']['triage']['job_id']
    triage = next(r for r in ledger['reservations'] if r.get('job_id') == triage_id and r.get('state') == 'settled')
    measured_build, measured_build_sha = original_build(study, triage)
    measured_signature = compiler_signature(measured_build)
    measured_archive = measured['provenance']['archive_sha256']
    require(measured_build.get('archive_sha256') == measured_archive and measured_build.get('binary_sha256') == measured['provenance']['binary_sha256'], 'measured triage build source or binary mismatch')
    cache = {}
    def sources(digest):
        if digest not in cache:
            cache[digest] = numerical_archive_manifest(study/'runs/closures'/(digest+'.tar.gz'),digest,frozen_path.name)
        return cache[digest]
    reference = sources(measured_archive)
    records, failures, excluded = [], [], []
    expected_cases = {(gate,body) for gate in ('supplied','ste','teacher') for body in ('product','linear')}
    for row in ledger['reservations']:
        if row.get('state') != 'settled' or not row.get('packet_evidence'):
            continue
        root = study / row['packet_evidence']; path = root/'chain/metrics.json'
        if not path.is_file():
            continue
        try:
            metric = read(path)
            if metric.get('suite') != 'sanitize-chain':
                continue
            if is_v3 and (metric.get('archive_sha256') != measured_archive or metric.get('binary_sha256') != measured['provenance']['binary_sha256']):
                excluded.append({'job_id':row.get('job_id'),'tool':metric.get('sanitizer'),'archive_sha256':metric.get('archive_sha256'),'binary_sha256':metric.get('binary_sha256'),'classification':'outside exact v3 measured source and binary'})
                continue
            require(metric.get('sanitizer') in needed and metric.get('success') is True and row.get('worker_status') == 'succeeded', 'sanitizer worker failed')
            _, _, uuid = job_receipt(study,row,True)
            receipt = read(study/row['receipt']); scheduler_attempt = receipt.get('job',receipt)['attempt']['attempt_id']
            worker = read(root/'result.json')
            require(metric.get('schema') == 'cgs.packet.v1' and metric.get('scheduler_job_id') == row['job_id'] and metric.get('scheduler_attempt_id') == scheduler_attempt and metric.get('uuid') == uuid and worker.get('status') == 'succeeded' and worker.get('returncode') == 0 and worker.get('scheduler_attempt_id') == scheduler_attempt and worker.get('scheduler_job_id') == row['job_id'], 'sanitizer Scheduler/worker identity mismatch')
            build, build_sha = original_build(study,row)
            require(build.get('archive_sha256') == metric['archive_sha256'] and build.get('binary_sha256') == metric['binary_sha256'] and sha(study/'runs/builds'/metric['binary_sha256']/'binary') == metric['binary_sha256'], 'sanitizer binary or build mismatch')
            require(compiler_signature(build) == measured_signature, 'sanitizer and measured compiler flags or toolchain differ')
            require(sources(metric['archive_sha256']) == reference, 'sanitizer numerical source/header bytes differ')
            snapshot = sanitizer_snapshot(root,row,metric,build_sha,adapter_sha,expected_backend)
            cases = metric.get('cases', [])
            require(len(cases) == 6 and {(c.get('gate'),c.get('body')) for c in cases} == expected_cases, 'sanitizer gate/body coverage incomplete')
            for case in cases:
                require(case.get('success') is True and case.get('status') == 'pass' and case.get('returncode') == 0 and case.get('errors') == [] and case.get('sanitizer_zero_error_summary') is True and case.get('backend') == expected_backend and case.get('case') == 'dag2304-64root-complete-consumer-diverse-dense-low-update-r1-change100' and case.get('tile') == 64 and case.get('stages') == expected_stages and case.get('precision') == 'fp16' and case.get('condition_update') == ('none' if case.get('gate') == 'ste' else 'incremental'), 'sanitizer case failed or is outside the representative scope')
                argv = case['argv']
                require('--tool' in argv and argv[argv.index('--tool')+1] == metric['sanitizer'] and '--error-exitcode' in argv and argv[argv.index('--error-exitcode')+1] == '99' and metric['binary_path'] in argv and '--target-processes' in argv and argv[argv.index('--target-processes')+1] == 'all', 'sanitizer invocation mismatch')
                for flag, key in (('--case','case'),('--backend','backend'),('--gate','gate'),('--body','body'),('--precision','precision'),('--condition-update','condition_update'),('--tile','tile'),('--stages','stages'),('--warmups','warmups'),('--repeats','repeats')):
                    require(argv.count(flag) == 1 and argv[argv.index(flag)+1] == str(case[key]), 'sanitizer case command mismatch: ' + flag)
                for label in ('campaign','sanitizer_stdout','sanitizer_stderr'):
                    rel = Path(case[label+'_path'])
                    require(not rel.is_absolute() and '..' not in rel.parts and sha(root/'chain'/rel) == case[label+'_sha256'], 'sanitizer raw evidence mismatch')
                output = (root/'chain'/case['sanitizer_stdout_path']).read_text() + (root/'chain'/case['sanitizer_stderr_path']).read_text()
                require(re.search(r'ERROR SUMMARY:\s*0 errors\b|0 hazards displayed \(0 errors\b',output) is not None, 'sanitizer zero-error summary missing')
                raw = read(root/'chain'/case['campaign_path'])
                require(raw == case['campaign'] and raw.get('accuracy_accepted') is True and raw.get('validation_scope') == 'first-and-last-complete-steps' and raw.get('device_uuid') == uuid and raw.get('gate') == case['gate'] and raw.get('body') == case['body'] and raw.get('backend') == case['backend'] and raw.get('tile') == case['tile'] and raw.get('stages') == case['stages'] and raw.get('precision') == case['precision'] and raw.get('condition_update') == case['condition_update'] and raw.get('input') == 'synthetic' and raw.get('descriptor_validation') == 'exact-work-membership-and-raw-group-identity' and raw.get('case') == case['case'] and isinstance(raw.get('workload_sha256'), str) and re.fullmatch(r'[0-9a-f]{64}', raw['workload_sha256']) is not None, 'sanitizer raw complete-chain validation mismatch')
            records.append({'job_id':row['job_id'],'tool':metric['sanitizer'],'uuid':uuid,'sanitized_archive_sha256':metric['archive_sha256'],'sanitized_binary_sha256':metric['binary_sha256'],'sanitized_build_receipt_sha256':build_sha,'measured_archive_sha256':measured_archive,'measured_binary_sha256':measured['provenance']['binary_sha256'],'measured_build_receipt_sha256':measured_build_sha,'numerical_source_and_contract_files':len(reference),'compiler_signature_sha256':digest_json(measured_signature),'scheduler_receipt_sha256':sha(study/row['receipt']),'metrics_sha256':sha(path),**snapshot})
        except (ValueError, KeyError, IndexError, TypeError, OSError, json.JSONDecodeError, tarfile.TarError) as exc:
            failures.append({'job_id':row.get('job_id'),'reason':str(exc)})
    missing = sorted(needed - {r['tool'] for r in records})
    primary = []
    for tool in sorted(needed - set(missing)):
        matching = [r for r in records if r['tool'] == tool]
        matching.sort(key=lambda r: (r['sanitized_binary_sha256'] != r['measured_binary_sha256'] or r['sanitized_archive_sha256'] != r['measured_archive_sha256'], r['job_id']))
        primary.append(matching[0])
    exact = not missing and all(r['sanitized_binary_sha256'] == r['measured_binary_sha256'] and r['sanitized_archive_sha256'] == r['measured_archive_sha256'] for r in primary)
    return {'status': 'representative sanitizer tools passed' if not missing else 'incomplete', 'coverage_basis': 'exact measured binary and archive' if exact else 'source-identical builds' if not missing else 'incomplete', 'missing_tools':missing, 'primary_records':primary, 'records':records,'failures':failures,'other_source_records':excluded,'scope':f'Six complete-chain gate/body cases on {expected_backend}, tile 64, {expected_stages} staging buffer(s), FP16 synthetic workload; sanitized and measured binaries, source archives and devices remain explicit. This is representative, not exhaustive.'}

def comparisons(packets):
    spec = importlib.util.spec_from_file_location('cgs_paired_calculator', Path(__file__).with_name('analyze_pairs.py'))
    calculator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(calculator)
    reports = []
    for p in packets:
        rows = p['phases']
        for run, phase in sorted({(k[0], k[1]) for k in rows if k[1] == 'supplemental'}):
            arms = {k[3] for k in rows if k[:2] == (run, phase)}
            candidates = set(p['config']['candidates']) & EXPERIMENTAL
            for candidate in sorted(candidates):
                rounds = sorted({k[2] for k in rows if k[:2] == (run, phase)})
                candidate_rows = [v for k,v in rows.items() if k[:2] == (run, phase) and k[3] == candidate]
                if not candidate_rows:
                    reports.append({'job_id': p['job'], 'run_id': run, 'phase': phase, 'candidate': candidate, 'decision': 'inconclusive: no retained measurements for configured arm'})
                    continue
                sample = candidate_rows[0]['raw']
                controls = {'scalar', 'bitmap', 'direct'} | ({'library', 'library-batched'} if sample['body'] == 'linear' else set())
                if p.get('matrix_schema') == 'cgs.measurement-matrix.v4' and candidate != 'masked-direct-two-tile':
                    controls.add('masked-direct-two-tile')
                report = {'job_id': p['job'], 'run_id': run, 'phase': phase, 'case': sample['case'], 'candidate': candidate, 'required_controls': sorted(controls), 'device_uuid':p['uuid'], 'gate':sample['gate'], 'body':sample['body'], 'precision':sample['precision'], 'input':sample['input'], 'condition_update':sample['condition_update'], 'tile':sample['tile'], 'stages':sample['stages'], 'decision': 'inconclusive: incomplete matched controls or rounds'}
                if not rounds or any((run, phase, n, arm) not in rows for n in rounds for arm in controls | {candidate}):
                    reports.append(report)
                    continue
                all_rows = [rows[(run, phase, n, arm)] for n in rounds for arm in controls | {candidate}]
                if any(any(v['raw'].get(f) != sample.get(f) for f in FIELDS) for v in all_rows):
                    report['decision'] = 'rejected: unmatched semantic identity'
                    reports.append(report)
                    continue
                fastest = min(controls, key=lambda arm: statistics.median(rows[(run, phase, n, arm)]['us'] for n in rounds))
                pairs = []
                for n in rounds:
                    control, cand = rows[(run, phase, n, fastest)], rows[(run, phase, n, candidate)]
                    pairs.append({'case': sample['case'], 'repeat': n, 'device_uuid': p['uuid'], 'order': 'control-first' if control['index'] < cand['index'] else 'candidate-first', 'control_us': control['us'], 'candidate_us': cand['us']})
                calculation = calculator.analyze({'schema': 'cgs.paired-trials.v1', 'validation_passed': True, 'interference_check_passed': True, 'measurement_boundary': BOUNDARY, 'strongest_matched_control': fastest, 'pairs': pairs})['cases'][0]
                decision = calculation['decision']
                if decision == 'calculator-threshold-met':
                    decision = 'paired threshold met; study-wide recommendation pending sanitizer coverage'
                elif decision == 'not established':
                    if calculation['median_speedup'] <= 1.0:
                        decision = 'rejected: no median improvement'
                    elif calculation['median_latency_reduction'] <= 0.05 + 1e-12:
                        decision = 'practically inconclusive: latency reduction does not exceed 5%'
                    else:
                        decision = 'statistically inconclusive: paired interval includes no improvement'
                report.update(strongest_control=fastest, control_median_us={a: statistics.median(rows[(run,phase,n,a)]['us'] for n in rounds) for a in sorted(controls)}, pairs=pairs, calculation=calculation, decision=decision)
                reports.append(report)
    return reports



def staging_ablations(packets):
    """Pair shared staging against the equal-ownership cached two-tile arm."""
    spec = importlib.util.spec_from_file_location('cgs_staging_calculator', Path(__file__).with_name('analyze_pairs.py'))
    calculator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(calculator)
    reports = []
    control, candidate = 'cohort-direct-two-tile', 'cohort-shared'
    for p in packets:
        configured = set(p['config']['candidates']) | {p['config'].get('incumbent')}
        if control not in configured:
            continue
        rows = p['phases']
        for run, phase in sorted({(k[0], k[1]) for k in rows if k[1] == 'supplemental'}):
            rounds = sorted({k[2] for k in rows if k[:2] == (run, phase)})
            report = {'job_id':p['job'], 'run_id':run, 'phase':phase, 'comparison':'matched cached two-tile versus shared staging', 'control':control, 'candidate':candidate, 'decision':'inconclusive: incomplete matched arms or rounds'}
            if not rounds or any((run,phase,n,arm) not in rows for n in rounds for arm in (control,candidate)):
                reports.append(report)
                continue
            exemplar = rows[(run,phase,rounds[0],control)]['raw']
            if any(any(rows[(run,phase,n,arm)]['raw'].get(f) != exemplar.get(f) for f in FIELDS) for n in rounds for arm in (control,candidate)):
                report['decision'] = 'rejected: unmatched semantic identity'
                reports.append(report)
                continue
            pairs=[]
            for n in rounds:
                a,b = rows[(run,phase,n,control)], rows[(run,phase,n,candidate)]
                pairs.append({'case':exemplar['case'],'repeat':n,'device_uuid':p['uuid'], 'order':'control-first' if a['index'] < b['index'] else 'candidate-first', 'control_us':a['us'],'candidate_us':b['us']})
            calculation = calculator.analyze({'schema':'cgs.paired-trials.v1','validation_passed':True,'interference_check_passed':True,'measurement_boundary':BOUNDARY,'strongest_matched_control':control,'pairs':pairs})['cases'][0]
            report.update(case=exemplar['case'], device_uuid=p['uuid'], gate=exemplar['gate'], body=exemplar['body'], precision=exemplar['precision'], input=exemplar['input'], condition_update=exemplar['condition_update'], tile=exemplar['tile'], stages=exemplar['stages'], pairs=pairs, calculation=calculation, decision=calculation['decision'], interpretation='Staging ablation only; does not replace the fastest applicable conventional control for specialization.')
            reports.append(report)
    return reports


def masked_ownership_ablations(packets):
    """Compare two-tile cached cohorts with the matched unsorted masked arm."""
    spec=importlib.util.spec_from_file_location('cgs_masked_calculator', Path(__file__).with_name('analyze_pairs.py'))
    calculator=importlib.util.module_from_spec(spec);spec.loader.exec_module(calculator)
    reports=[]; control='masked-direct-two-tile'; candidate='cohort-direct-two-tile'
    for p in packets:
        if p.get('matrix_schema') != 'cgs.measurement-matrix.v4':
            continue
        rows=p['phases']
        for run,phase in sorted({(k[0],k[1]) for k in rows if k[1] == 'supplemental'}):
            rounds=sorted({k[2] for k in rows if k[:2] == (run,phase)})
            report={'job_id':p['job'],'run_id':run,'phase':phase,'comparison':'unsorted masked versus cached two-tile','control':control,'candidate':candidate,'decision':'inconclusive: incomplete matched arms or rounds'}
            if not rounds or any((run,phase,n,arm) not in rows for n in rounds for arm in (control,candidate)):
                reports.append(report);continue
            exemplar=rows[(run,phase,rounds[0],control)]['raw']
            if any(any(rows[(run,phase,n,arm)]['raw'].get(f) != exemplar.get(f) for f in FIELDS) for n in rounds for arm in (control,candidate)):
                report['decision']='rejected: unmatched semantic identity';reports.append(report);continue
            pairs=[]
            for n in rounds:
                a,b=rows[(run,phase,n,control)],rows[(run,phase,n,candidate)]
                pairs.append({'case':exemplar['case'],'repeat':n,'device_uuid':p['uuid'],'order':'control-first' if a['index'] < b['index'] else 'candidate-first','control_us':a['us'],'candidate_us':b['us']})
            calculation=calculator.analyze({'schema':'cgs.paired-trials.v1','validation_passed':True,'interference_check_passed':True,'measurement_boundary':BOUNDARY,'strongest_matched_control':control,'pairs':pairs})['cases'][0]
            report.update(case=exemplar['case'],device_uuid=p['uuid'],gate=exemplar['gate'],body=exemplar['body'],precision=exemplar['precision'],input=exemplar['input'],condition_update=exemplar['condition_update'],tile=exemplar['tile'],stages=exemplar['stages'],pairs=pairs,calculation=calculation,decision=calculation['decision'],interpretation='Ownership ablation only; specialization selection uses all applicable controls.')
            reports.append(report)
    return reports


def update_mode_packets(packets):
    """Expose each mode's own balanced rounds without pooling across matrices."""
    out=[]
    for p in packets:
        if not p.get('updates'):
            continue
        for mode in ('rebuild','incremental'):
            phases={('update-'+mode, 'supplemental', n, arm): value for (n,arm,m),value in p['updates']['records'].items() if m == mode}
            out.append({'job':p['job'], 'uuid':p['uuid'], 'config':p['config'], 'phases':phases, 'matrix_schema':p.get('matrix_schema')})
    return out


def update_ablations(packets):
    spec = importlib.util.spec_from_file_location('cgs_update_calculator', Path(__file__).with_name('analyze_pairs.py'))
    calculator = importlib.util.module_from_spec(spec); spec.loader.exec_module(calculator)
    reports=[]
    for p in packets:
        if not p.get('updates'):
            continue
        records=p['updates']['records']
        arms=sorted({k[1] for k in records})
        for arm in arms:
            rounds=sorted({k[0] for k in records})
            report={'job_id':p['job'],'device_uuid':p['uuid'],'backend':arm,'control':'rebuild','candidate':'incremental','decision':'inconclusive: incomplete matched update rounds'}
            if not rounds or any((n,arm,mode) not in records for n in rounds for mode in ('rebuild','incremental')):
                reports.append(report); continue
            exemplar=records[(rounds[0],arm,'rebuild')]['raw']
            fields=tuple(f for f in FIELDS if f != 'condition_update')
            if any(any(records[(n,arm,mode)]['raw'].get(f) != exemplar.get(f) for f in fields) or records[(n,arm,mode)]['raw'].get('numerical_contract') != exemplar.get('numerical_contract') for n in rounds for mode in ('rebuild','incremental')):
                report['decision']='rejected: unmatched semantic identity'; reports.append(report); continue
            pairs=[]
            for n in rounds:
                a,b=records[(n,arm,'rebuild')],records[(n,arm,'incremental')]
                pairs.append({'case':exemplar['case'],'repeat':n,'device_uuid':p['uuid'],'order':'control-first' if a['index'] < b['index'] else 'candidate-first','control_us':a['us'],'candidate_us':b['us']})
            calculation=calculator.analyze({'schema':'cgs.paired-trials.v1','validation_passed':True,'interference_check_passed':True,'measurement_boundary':BOUNDARY,'strongest_matched_control':'rebuild','pairs':pairs})['cases'][0]
            report.update(case=exemplar['case'],gate=exemplar['gate'],body=exemplar['body'],precision=exemplar['precision'],input=exemplar['input'],tile=exemplar['tile'],stages=exemplar['stages'],pairs=pairs,calculation=calculation,decision=calculation['decision'],interpretation='Same-packet, same-backend update-strategy ablation; separate from specialization selection.')
            reports.append(report)
    return reports


FROZEN_SELECTIONS = {
    ('contract-freeze-v2.json', 'measurement-matrix-v2.json'): ('237da579f5068fc26469f52c79b7ae33beb1ab0af057129a27c186193c5c8147', 'c2d40e206fa21692a704a65e029a159f38cb023183e4cc7a28481b4f35749241'),
    ('contract-freeze-v3.json', 'measurement-matrix-v3.json'): ('4033eea4a8168f5e898212873252c8782943d12f5325e5a31ca45487f9751f06', '0d814998d7a7927f82ad49e10cb95d5d7d09af3e98ff391e0b0ccd6491cebd26'),
}


def validate_selection(frozen_path, matrix_path):
    identities = FROZEN_SELECTIONS.get((frozen_path.name, matrix_path.name))
    require(identities is not None and (sha(frozen_path), sha(matrix_path)) == identities, 'unreviewed, mixed or altered contract/matrix selection')
    matrix = read(matrix_path)
    require(matrix.get('frozen_before_timing') is True and matrix.get('schema') == 'cgs.measurement-matrix.' + frozen_path.stem.rsplit('-',1)[-1], 'measurement matrix is not the selected frozen version')
    return identities

def qualify(study, frozen_path, matrix_path, wrapper_sha=None, driver_sha=None):
    is_v3 = frozen_path.name == 'contract-freeze-v3.json'
    wrapper_sha = wrapper_sha or (REVIEWED_V3_WRAPPER_SHA256 if is_v3 else REVIEWED_WRAPPER_SHA256)
    driver_sha = driver_sha or (REVIEWED_V3_DRIVER_SHA256 if is_v3 else REVIEWED_DRIVER_SHA256)
    require(all(re.fullmatch(r'[0-9a-f]{64}', x or '') for x in (wrapper_sha, driver_sha)), 'reviewed wrapper and driver SHA-256 values are required')
    validate_selection(frozen_path, matrix_path)
    ledger_path = study / 'runs/campaign-budget.json'
    ledger = read(ledger_path)
    accepted, failures, excluded, cache = [], [], [], {}
    try:
        windows, audits = ledger_windows(study, ledger, return_audits=True)
        ledger_error = None
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        windows, audits, ledger_error = {}, [], str(exc)
        failures.append({'classification': 'unqualified ledger evidence', 'reason': ledger_error})
    for row in ledger['reservations']:
        if not row.get('serialized_confirmation'):
            continue
        if row.get('state') == 'cpu_build':
            continue
        if row.get('state') == 'no_submission' and not row.get('job_id'):
            failures.append({'job_id': None, 'reservation_id': row.get('id'), 'classification': 'no submission', 'reason': 'reservation settled without a GPU job'})
            continue
        try:
            config_path = study / row['packet_evidence'] / 'experiment.json'
            if config_path.is_file():
                recorded = read(config_path).get('contract_freeze_sha256')
                if recorded != sha(frozen_path):
                    excluded.append({'job_id': row.get('job_id'), 'contract_sha256': recorded, 'classification': 'outside selected frozen contract', 'worker_status': row.get('worker_status')})
                    continue
            if row.get('state') == 'settled' and row.get('worker_status') != 'succeeded':
                raise ValueError('GPU worker did not succeed: ' + str(row.get('worker_status')))
            require(ledger_error is None, 'cannot establish study serialization: ' + str(ledger_error))
            accepted.append(packet(study, row, windows, frozen_path, matrix_path, cache, wrapper_sha, driver_sha))
        except (ValueError, KeyError, IndexError, TypeError, OSError, json.JSONDecodeError, tarfile.TarError, tomllib.TOMLDecodeError) as e:
            failure = {'job_id': row.get('job_id'), 'reservation_id': row.get('id'), 'classification': 'unqualified or incomplete evidence', 'reason': str(e)}
            try:
                worker_path = study / row['packet_evidence'] / 'result.json'
                worker_result = read(worker_path)
                failure['worker_status'] = worker_result.get('status')
                failure['worker_returncode'] = worker_result.get('returncode')
                failure['worker_storage'] = worker_result.get('storage')
                failure['worker_result_sha256'] = sha(worker_path)
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                pass
            try:
                packet_root = study / row['packet_evidence']
                config = read(packet_root / 'experiment.json')
                failure['source_archive_sha256'] = config.get('source_archive_sha256')
                failure['binary_sha256'] = config.get('binary_sha256')
                failure['device_uuid'] = config.get('interference_attestation', {}).get('device_uuid')
                failure['config_sha256'] = sha(packet_root / 'experiment.json')
                failure['scheduler_receipt_sha256'] = sha(study / row['receipt'])
                if (packet_root / 'serial-telemetry.json').is_file():
                    guard = read(packet_root / 'serial-telemetry.json')
                    issues = {}
                    for sample in guard.get('samples', []):
                        for issue in sample.get('issues', []):
                            issues[issue] = issues.get(issue, 0) + 1
                    failure['telemetry'] = {'sha256': sha(packet_root / 'serial-telemetry.json'), 'clean': guard.get('clean'), 'issues': issues}
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                pass
            try:
                summary_path = study / row['packet_evidence'] / 'kda/supplemental/summary.json'
                summary = read(summary_path)
                failure['supplemental_summary_sha256'] = sha(summary_path)
                failure['arm_records'] = [{'round': r.get('round_id'), 'backend': a.get('backend'), 'status': a.get('status'), 'reason': a.get('reason'), 'path': a.get('path'), 'sha256': a.get('raw_sha256'), 'qualification': 'unqualified'} for r in summary.get('rounds', []) for a in r.get('records', [])]
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                try:
                    partial_root = study / row['packet_evidence'] / 'kda/supplemental'
                    partial = []
                    for path in sorted(partial_root.glob('round-*.json')):
                        record = read(path)
                        partial.append({'path': str(path.relative_to(study)), 'sha256': sha(path), 'backend': record.get('backend'), 'case': record.get('case'), 'status': 'unqualified partial record'})
                    if partial:
                        failure['partial_arm_records'] = partial
                except (OSError, KeyError, TypeError, json.JSONDecodeError):
                    pass
            try:
                update_path = study / row['packet_evidence'] / 'kda/update-ablation/summary.json'
                update_summary = read(update_path)
                failure['update_ablation_summary_sha256'] = sha(update_path)
                failure['update_arm_records'] = [{'round': r.get('round_id'), 'backend': a.get('backend'), 'update_mode': a.get('update_mode'), 'status': a.get('status'), 'path': a.get('path'), 'sha256': a.get('raw_sha256'), 'qualification': 'unqualified'} for r in update_summary.get('rounds', []) for a in r.get('records', [])]
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                pass
            failures.append(failure)
    reports = comparisons(accepted)
    coverage = {}
    for p in accepted:
        try:
            coverage[p['job']] = sanitizer_coverage(study, ledger, p, frozen_path)
        except (ValueError, KeyError, IndexError, TypeError, OSError, json.JSONDecodeError, tarfile.TarError, StopIteration) as exc:
            coverage[p['job']] = {'status': 'incomplete', 'reason': str(exc)}
    mode_reports = comparisons(update_mode_packets(accepted))
    mode_staging = staging_ablations(update_mode_packets(accepted))
    for report in mode_reports + mode_staging:
        report['analysis_role'] = 'supplemental same-mode pairs from update ablation; separate from primary base matrix, not pooled or selected post hoc'
    for report in reports + mode_reports:
        if report['decision'] == 'paired threshold met; study-wide recommendation pending sanitizer coverage' and coverage.get(report['job_id'], {}).get('status') == 'representative sanitizer tools passed':
            report['decision'] = 'paired threshold met; representative sanitizer coverage passed'
    return {'schema': 'cgs.qualification.v1', 'exporter_sha256': sha(Path(__file__)), 'calculator_sha256': sha(Path(__file__).with_name('analyze_pairs.py')), 'ledger_sha256': sha(ledger_path), 'contract_sha256': sha(frozen_path), 'matrix_sha256': sha(matrix_path), 'reviewed_wrapper_sha256': wrapper_sha, 'reviewed_driver_sha256': driver_sha, 'measurement_boundary': BOUNDARY, 'method': 'Serialized study jobs with sampled all-device pre/during/post telemetry and verified native-child lifetimes for positive target readings without a current process, bounded by one second. Ownership verification relies on NVML current compute-process query semantics; unreaped PID and start-tick identity prevent instance reuse across the recorded query. Sampling cannot exclude activity between observations; no host-wide exclusivity is claimed. KDA internal selection is not promotion evidence. Sanitizer and remaining study-wide coverage are reported separately; threshold results do not establish unfinished coverage.', 'packets': [{'job_id': p['job'], 'device_uuid': p['uuid'], 'provenance': p['provenance'], 'benchmark_sha256': p['benchmark_sha256'], 'supplemental_sha256': p['supplemental_sha256'], 'update_ablation_sha256': p['updates']['summary_sha256'] if p.get('updates') else None, 'update_raw_records': [{'round':k[0], 'backend':k[1], 'update_mode':k[2], 'path':v['raw_path'], 'sha256':v['raw_sha256']} for k,v in p['updates']['records'].items()] if p.get('updates') else [], 'raw_records': [{'round': k[2], 'backend': k[3], 'path': v['raw_path'], 'sha256': v['raw_sha256']} for k,v in p['phases'].items() if k[1] == 'supplemental']} for p in accepted], 'serialization_audit': audits, 'comparisons': reports, 'staging_ablations': staging_ablations(accepted), 'masked_ownership_ablations': masked_ownership_ablations(accepted), 'update_mode_comparisons': mode_reports, 'update_mode_staging_ablations': mode_staging, 'update_ablations': update_ablations(accepted), 'cross_strategy_pairing': 'only separately retained same-packet paired update ablations qualify; independent scenario packets are not pooled', 'sanitizer_coverage': coverage, 'failures': failures, 'other_contract_packets': excluded, 'capacity_unsupported': [{'arm': 'cohort-shared', 'tile': 128, 'stages': 2, 'basis': read(matrix_path)['capacity_rejection'], 'basis_type': 'frozen matrix declaration; no submitted measurement'}], 'coverage': 'Only listed comparisons are evaluated; all other frozen matrix cases and unsupported arms remain incomplete unless independently evidenced.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, default=STUDY)
    p.add_argument('--contract', type=Path, default=CONTRACT)
    p.add_argument('--matrix', type=Path, default=MATRIX)
    p.add_argument('--reviewed-wrapper-sha256', help='explicit reviewed runtime wrapper identity; otherwise use selected contract review')
    p.add_argument('--reviewed-driver-sha256', help='explicit reviewed native driver identity; otherwise use selected contract review')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = qualify(a.study, a.contract, a.matrix, a.reviewed_wrapper_sha256, a.reviewed_driver_sha256)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'output': str(a.output), 'accepted_packets': len(result['packets']), 'comparisons': len(result['comparisons']), 'failures': len(result['failures'])}))


if __name__ == '__main__':
    main()
