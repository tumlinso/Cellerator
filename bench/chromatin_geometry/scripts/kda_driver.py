#!/usr/bin/env python3
"""Trusted KDA command adapter for the complete native campaign runner.

All GPU commands here require the enclosing Autobench/Scheduler job. Candidate
backend configurations are separate from this validator and reference binary.
"""
import hashlib
import ctypes
import fcntl
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, sort_keys=True, allow_nan=False) + '\n')


def lifetime_event(path, event):
    # One append write per event keeps separate driver processes from mixing
    # partial JSON records. Missing/unpaired events are rejected by the monitor.
    data = (json.dumps(event, sort_keys=True) + '\n').encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        if os.write(fd, data) != len(data):
            raise RuntimeError('short native-child lifetime write')
    finally:
        os.close(fd)


def proc_start_ticks(pid):
    # /proc/<pid>/stat's comm field may contain spaces and parentheses.
    stat = Path(f'/proc/{pid}/stat').read_text()
    return int(stat[stat.rindex(')') + 2:].split()[19])


def run_tracked(argv, events, lock_path, metadata, output):
    """Keep the native PID allocated throughout every locked process query.

    waitid observes completion without reaping; only the locked wait releases
    the PID. Regular temporary files avoid deadlock if a child emits much text.
    """
    invocation = str(uuid.uuid4())
    with open(lock_path, 'r+b') as lock, tempfile.TemporaryFile(mode='w+t') as out, \
            tempfile.TemporaryFile(mode='w+t') as err:
        fcntl.flock(lock, fcntl.LOCK_EX)
        child = None
        try:
            lower = time.time_ns()
            child = subprocess.Popen(argv, text=True, stdout=out, stderr=err)
            upper = time.time_ns()
            ticks = proc_start_ticks(child.pid)
            lifetime_event(events, {'event': 'start', 'invocation_id': invocation, 'pid': child.pid,
                                    'proc_start_ticks': ticks, 'spawn_lower_ns': lower,
                                    'spawn_upper_ns': upper, 'argv': argv, **metadata})
        except BaseException:
            if child is not None:
                child.kill()
                child.wait()
            raise
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
        try:
            os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT)
        except BaseException:
            # A terminated launcher must not leave its child running; the same
            # lock also serializes exceptional reaping with observation.
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                child.kill()
                child.wait()
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
            raise
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            reap_lower = time.time_ns()
            child.wait()
            reap_upper = time.time_ns()
            lifetime_event(events, {'event': 'end', 'invocation_id': invocation, 'pid': child.pid,
                                    'proc_start_ticks': ticks, 'reap_lower_ns': reap_lower,
                                    'reap_upper_ns': reap_upper, 'returncode': child.returncode,
                                    'output_path': str(output),
                                    'output_sha256': sha(output) if Path(output).is_file() else None})
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
        out.seek(0)
        err.seek(0)
        return subprocess.CompletedProcess(argv, child.returncode, out.read(), err.read())


def candidate(request):
    root = Path(request['candidate']['source_dir'])
    return json.loads((root / 'backend.json').read_text())['backend']


def build(binary, trusted_sources, request, result):
    if os.environ.get('CGS_KDA_CPU_PREPARE') != '1' or os.environ.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-'):
        raise RuntimeError('candidate preparation must finish before GPU admission')
    root = Path(request['candidate']['source_dir'])
    for name in ('consumers.cu', 'conditions.cu', 'grouped_linear_library.cu'):
        if sha(root / name) != sha(Path(trusted_sources) / name):
            raise RuntimeError(f'candidate source does not match the compiled trusted closure: {name}')
    output = Path(request['build']['directory'])
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / 'cgs_study'
    shutil.copy2(binary, artifact)
    write(result, {'schema_version': 1, 'request_id': request['request_id'], 'status': 'ok',
                   'artifacts': [{'path': 'cgs_study', 'sha256': sha(artifact)}]})


def probe(request, result):
    uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not os.environ.get('AUTOBENCH_SCHEDULER_JOB_ID') or not uuid.startswith('GPU-') or ',' in uuid:
        raise RuntimeError('KDA probe requires an admitted single-UUID Autobench GPU packet')
    gpu = subprocess.check_output(['nvidia-smi', '-i', uuid, '--query-gpu=uuid,name,driver_version,compute_cap',
                                   '--format=csv,noheader'], text=True).strip()
    fields = [x.strip() for x in gpu.split(',')]
    if len(fields) != 4 or fields[0] != uuid:
        raise RuntimeError('GPU probe returned a different UUID')
    runtime = ctypes.CDLL('/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/lib64/libcudart.so')
    runtime_version = ctypes.c_int()
    if runtime.cudaRuntimeGetVersion(ctypes.byref(runtime_version)) != 0:
        raise RuntimeError('CUDA runtime version probe failed')
    compiler = subprocess.check_output(['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc', '--version'], text=True)
    write(result, {'schema_version': 1, 'request_id': request['request_id'], 'status': 'ok',
                   'environment': {'kind': 'cuda', 'architecture': 'sm_' + fields[3].replace('.', ''),
                                   'device_uuid': uuid, 'driver_version': fields[2],
                                   'runtime_version': str(runtime_version.value),
                                   'tools': {'nvcc_version': compiler.strip()}, 'nvidia_smi': gpu}})


def run_one(binary, backend, case, warmups, repeats, output):
    if not os.environ.get('AUTOBENCH_SCHEDULER_JOB_ID'):
        raise RuntimeError('CUDA evaluation requires Autobench Scheduler admission')
    p = case['parameters']
    argv = [str(binary), '--mode', 'campaign', '--case', case['id'], '--backend', backend,
            '--gate', p['gate'], '--body', p['body'], '--precision', p['precision'],
            '--condition-update', p['condition_update'], '--tile', str(p['tile']),
            '--stages', str(p['stages']), '--warmups', str(warmups), '--repeats', str(repeats),
            '--output', str(output)]
    if p.get('replay'):
        argv.extend(['--replay', p['replay']])
    events = os.environ.get('CGS_LIFETIME_EVENTS')
    lock_path = os.environ.get('CGS_PROCESS_LOCK')
    if bool(events) != bool(lock_path):
        raise RuntimeError('native process tracking requires both event and lock paths')
    if events:
        proc = run_tracked(argv, events, lock_path,
                           {'device_uuid': os.environ.get('CUDA_VISIBLE_DEVICES'),
                            'binary_sha256': sha(binary), 'backend': backend}, output)
    else:
        proc = subprocess.run(argv, text=True, capture_output=True)
    if proc.returncode:
        raise RuntimeError(f'campaign {backend} failed ({proc.returncode}): {proc.stderr[-4000:]}')
    data = json.loads(Path(output).read_text())
    if (data.get('schema') != 'cgs.campaign.v1' or data.get('status') != 'ok' or
            data.get('accuracy_accepted') is not True or
            data.get('validation_scope') != 'first-and-last-complete-steps' or
            data.get('case') != case['id'] or data.get('backend') != backend or
            data.get('device_uuid') != os.environ.get('CUDA_VISIBLE_DEVICES')):
        raise RuntimeError('campaign response failed identity or independent-validation checks')
    for name in ('gate', 'body', 'precision', 'condition_update'):
        if data.get(name) != p[name]:
            raise RuntimeError(f'campaign response mismatches case {name}')
    if data.get('tile') != p['tile'] or data.get('stages') != p['stages']:
        raise RuntimeError('campaign response mismatches tile/staging')
    digest = data.get('workload_sha256')
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError('campaign workload digest missing')
    expected = case.get('input', {}).get('campaign_workload_sha256')
    if not isinstance(expected, str) or len(expected) != 64 or digest != expected:
        raise RuntimeError('campaign input does not match its independently frozen workload digest')
    return data


def validate(request_path, request, result):
    binary = Path(request['candidate']['build']['directory']) / 'cgs_study'
    backend = candidate(request)
    rows = []
    for i, case in enumerate(request['cases']):
        case_result = {'case_id': case['id'], 'workload_sha256': case['workload_sha256']}
        try:
            base = Path(request_path).parent
            scalar = run_one(binary, 'scalar', case, 0, 1, base / f'reference-{i}.json')
            actual = run_one(binary, backend, case, 0, 1, base / f'candidate-{i}.json')
            if scalar['workload_sha256'] != actual['workload_sha256']:
                raise RuntimeError('candidate and matched scalar use different workload bytes')
            case_result.update(status='pass', observed_workload_sha256=actual['workload_sha256'])
        except Exception as exc:
            case_result.update(status='fail', reason=str(exc))
        rows.append(case_result)
    write(result, {'schema_version': 1, 'request_id': request['request_id'], 'status': 'ok', 'cases': rows})


def benchmark(request_path, request, result):
    binary = Path(request['candidate']['build']['directory']) / 'cgs_study'
    case, timing = request['case'], request['timing']
    raw = Path(request_path).parent / 'campaign.json'
    data = run_one(binary, candidate(request), case, timing['warmup'], timing['repetitions'], raw)
    recurrent, recovery = data['recurrent_complete_horizon_ms'], data['output_recovery_ms']
    if len(recurrent) != timing['repetitions'] or len(recovery) != len(recurrent):
        raise RuntimeError('campaign omitted required independent recurrent/recovery samples')
    cold = data['cold_execution_preparation_ms']
    components = (data['replay_input_preparation_ms'] + data['host_layout_preparation_ms'] +
                  data['cold_device_preparation_ms'])
    if (not math.isclose(cold, components, rel_tol=1e-9, abs_tol=1e-8) or
            data.get('cold_device_preparation_includes_host_condition_maps') is not True):
        raise RuntimeError('operational cold preparation omits layout or condition-map preparation')
    samples = [1000.0 * (cold + r + d) for r, d in zip(recurrent, recovery)]
    if any(not math.isfinite(x) or x <= 0 for x in samples):
        raise RuntimeError('invalid complete-horizon time')
    write(result, {'schema_version': 1, 'request_id': request['request_id'], 'status': 'ok',
                   'case_id': case['id'], 'workload_sha256': case['workload_sha256'],
                   'observed_workload_sha256': data['workload_sha256'],
                   'timing_scope': timing['scope'], 'metric': timing['metric'], 'unit': timing['unit'],
                   'warmup': timing['warmup'], 'repetitions': timing['repetitions'], 'samples': samples})


def main(argv):
    if len(argv) < 4:
        raise SystemExit('usage: kda_driver.py MODE [BUILD_BINARY TRUSTED_BACKENDS] REQUEST RESULT')
    mode, request_path, result = argv[1], argv[-2], argv[-1]
    request = json.loads(Path(request_path).read_text())
    if mode == 'build' and len(argv) == 6:
        build(argv[2], argv[3], request, result)
    elif mode == 'probe':
        probe(request, result)
    elif mode == 'validate':
        validate(request_path, request, result)
    elif mode == 'benchmark':
        benchmark(request_path, request, result)
    else:
        raise RuntimeError('invalid KDA driver invocation')


if __name__ == '__main__':
    main(sys.argv)
