#!/usr/bin/env python3
"""Run a prebuilt KDA evaluation inside a Scheduler-admitted Autobench packet."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path


def supplemental_matrix(binary, task, config, output, updates=None):
    """Collect all controls and candidates in balanced independent rounds."""
    driver_path = Path(__file__).with_name('kda_driver.py')
    spec = importlib.util.spec_from_file_location('cgs_supplemental_driver', driver_path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    backends = [config['incumbent'], *config['candidates']]
    modes = updates or [task['cases'][0]['parameters']['condition_update']]
    if (len(set(backends)) != len(backends) or len(task['cases']) != 1 or
            not modes or len(set(modes)) != len(modes) or not set(modes) <= {'none', 'incremental', 'rebuild'}):
        raise RuntimeError('supplemental matrix requires unique arms and one matched case')
    if len(modes) > 1 and set(modes) != {'incremental', 'rebuild'}:
        raise RuntimeError('the paired update ablation requires incremental and rebuild')
    arms = [f'{backend}@{mode}' for backend in backends for mode in modes] if len(modes) > 1 else backends
    output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(20261102)
    rounds = []
    for pair in range(3):
        forward = list(arms)
        rng.shuffle(forward)
        for order in (forward, list(reversed(forward))):
            index = len(rounds)
            records = []
            for position, arm in enumerate(order):
                backend, mode = arm.rsplit('@', 1) if len(modes) > 1 else (arm, modes[0])
                raw = output / f'round-{index:02d}-{arm}.json'
                record = {'arm_id': arm, 'backend': backend, 'condition_update': mode, 'update_mode': mode,
                          'position': position, 'path': raw.name}
                try:
                    case = copy.deepcopy(task['cases'][0])
                    case['parameters']['condition_update'] = mode
                    data = driver.run_one(binary, backend, case, 1, 1, raw)
                    cold = data['cold_execution_preparation_ms']
                    components = (data['replay_input_preparation_ms'] + data['host_layout_preparation_ms'] +
                                  data['cold_device_preparation_ms'])
                    if abs(cold - components) > max(1e-8, abs(cold) * 1e-9):
                        raise RuntimeError('cold cost components do not sum to the operational boundary')
                    record.update(status='pass', raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
                                  complete_us=1000 * (cold + data['recurrent_complete_horizon_ms'][0] +
                                                      data['output_recovery_ms'][0]),
                                  workload_sha256=data['workload_sha256'])
                except Exception as exc:
                    record.update(status='failed', reason=str(exc))
                records.append(record)
            rounds.append({'round_id': f'round-{index:02d}', 'order': order, 'records': records})
    result = {'schema': 'cgs.update-ablation.v1' if len(modes) > 1 else 'cgs.supplemental.v1',
              'update_modes': modes, 'scheduler_job_id': os.environ['AUTOBENCH_SCHEDULER_JOB_ID'],
              'device_uuid': os.environ['CUDA_VISIBLE_DEVICES'], 'source_archive_sha256': config['source_archive_sha256'],
              'binary_sha256': config['binary_sha256'], 'arms': arms, 'rounds': rounds,
              'scope': 'independent complete operational horizons; paired reverse-order rounds',
              'status': 'pass' if all(r['status'] == 'pass' for x in rounds for r in x['records']) else 'failed'}
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kda-root', type=Path, required=True)
    p.add_argument('--task', type=Path, required=True)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get('AUTOBENCH_SCHEDULER_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-'):
        raise RuntimeError('KDA GPU evaluation requires the assigned Scheduler UUID')
    config = json.loads(args.config.read_text())
    attestation = config.get('interference_attestation', {})
    evidence_path = args.config.parent / 'interference-evidence.json'
    if not evidence_path.is_file():
        raise RuntimeError('confirmation requires retained isolated-versus-concurrent interference evidence')
    evidence_bytes = evidence_path.read_bytes()
    if hashlib.sha256(evidence_bytes).hexdigest() != attestation.get('sha256'):
        raise RuntimeError('interference evidence identity differs from frozen configuration')
    evidence = json.loads(evidence_bytes)
    if evidence.get("schema") == "cgs.serialized-fallback.v1":
        cases = config.get("cases", [])
        expected = {"device_uuid": os.environ["CUDA_VISIBLE_DEVICES"],
                    "source_archive_sha256": config.get("source_archive_sha256"),
                    "binary_sha256": config.get("binary_sha256"),
                    "controller_serialization_required": True, "all_device_telemetry_required": True}
        if len(cases) != 1 or any(evidence.get(k) != v for k, v in expected.items()) or evidence.get("workload_sha256") != cases[0].get("input", {}).get("campaign_workload_sha256"):
            raise RuntimeError("serialized fallback does not bind this exact device and workload")
    else:
        if (evidence.get('schema') != 'cgs.interference.v1' or evidence.get('status') not in ('passed', 'serialized') or
                evidence.get('device_uuid') != os.environ['CUDA_VISIBLE_DEVICES'] or
                evidence.get('source_archive_sha256') != config.get('source_archive_sha256') or
                evidence.get('binary_sha256') != config.get('binary_sha256') or
                not evidence.get('isolated_receipts') or not evidence.get('concurrent_receipts')):
            raise RuntimeError('interference evidence does not qualify this device and timing mode')
        cases = config.get('cases', [])
        if len(cases) != 1 or evidence.get('workload_sha256') != cases[0].get('input', {}).get('campaign_workload_sha256'):
            raise RuntimeError('interference evidence does not bind the exact campaign workload')
        identity = evidence.get('case_identity', {})
        parameters = cases[0].get('parameters', {})
        if identity.get('case') != cases[0].get('id') or any(identity.get(k) != parameters.get(k) for k in ('gate', 'body', 'precision', 'condition_update', 'tile', 'stages')):
            raise RuntimeError('interference evidence does not bind the same operation and cost scope')
        ids = evidence['isolated_receipts'] + evidence['concurrent_receipts']
        if len(set(ids)) != len(ids) or len(evidence['isolated_receipts']) < 2 or len(evidence['concurrent_receipts']) < 2:
            raise RuntimeError('interference evidence needs distinct repeated isolated and concurrent receipts')
    candidates = config['candidates']
    command = [sys.executable, '-m', 'kda_experiments', '--task', str(args.task.resolve(strict=True)),
               '--workspace', str(args.workspace.resolve(strict=True)), 'evaluate',
               '--candidates', *candidates, '--incumbent', config['incumbent']]
    env = dict(os.environ, PYTHONPATH=str(args.kda_root.resolve(strict=True)))
    env.pop('CGS_KDA_CPU_PREPARE', None)
    process = subprocess.run(command, text=True, capture_output=True, env=env)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'kda.stdout').write_text(process.stdout)
    (args.output / 'kda.stderr').write_text(process.stderr)
    shutil.copy2(args.task, args.output / 'task.json')
    shutil.copy2(args.config, args.output / 'experiment.json')
    result = None
    for line in reversed(process.stdout.splitlines()):
        try:
            result = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    # Copy complete KDA records before Autobench scratch cleanup, including
    # rejected and inconclusive alternatives. Do not alter engine workspace.
    shutil.copytree(args.workspace, args.output / 'kda-workspace')
    supplemental = None
    update_ablation = None
    if process.returncode == 0 and isinstance(result, dict) and result.get('status') in ('complete', 'ok'):
        supplemental = supplemental_matrix(args.binary, json.loads(args.task.read_text()), config, args.output / 'supplemental')
        if config.get('supplemental_updates'):
            update_ablation = supplemental_matrix(args.binary, json.loads(args.task.read_text()), config,
                                                  args.output / 'update-ablation', config['supplemental_updates'])
    summary = {'schema': 'cgs.kda-packet.v1', 'scheduler_job_id': os.environ['AUTOBENCH_SCHEDULER_JOB_ID'],
               'uuid': os.environ['CUDA_VISIBLE_DEVICES'], 'exit_code': process.returncode,
               'engine_result': result, 'supplemental_status': supplemental and supplemental['status'],
               'update_ablation_status': update_ablation and update_ablation['status'],
               'selection_status': 'preliminary; 5-percent latency and independent receipt gates remain'}
    (args.output / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, sort_keys=True))
    if (process.returncode or not isinstance(result, dict) or result.get('status') not in ('complete', 'ok') or
            not supplemental or supplemental['status'] != 'pass' or (update_ablation and update_ablation['status'] != 'pass')):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
