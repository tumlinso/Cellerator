#!/usr/bin/env python3
"""CPU build and admitted GPU study packet, both bound to a closure."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path

NVCC = '/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc'
SANITIZER = '/home/tumlinson/Software/cuda-sanitizer-12.9/usr/local/cuda-12.9/bin/compute-sanitizer'


def is_descendant(pid, root):
    """Recognize only a live child of the KDA process; uncertainty fails closed."""
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == root:
            return True
        seen.add(pid)
        try:
            lines = Path(f'/proc/{pid}/status').read_text().splitlines()
            pid = int(next(x.split()[1] for x in lines if x.startswith('PPid:')))
        except (OSError, StopIteration, ValueError):
            return False
    return False


def proc_start_ticks(pid):
    stat = Path(f'/proc/{pid}/stat').read_text()
    return int(stat[stat.rindex(')') + 2:].split()[19])


def active_native_starts(path):
    starts, ends = {}, set()
    if not path.is_file():
        return starts
    for line in path.read_text().splitlines():
        event = json.loads(line)
        ident = event['invocation_id']
        if not isinstance(ident, str) or not ident:
            raise ValueError('invalid native invocation ID')
        if event.get('event') == 'start' and ident not in starts and ident not in ends:
            starts[ident] = event
        elif event.get('event') == 'end' and ident in starts and ident not in ends:
            ends.add(ident)
        else:
            raise ValueError('invalid native lifetime sequence')
    return {ident: event for ident, event in starts.items() if ident not in ends}


def serial_sample(uuid, root_pid, lock_path=None, lifetime_path=None):
    sampled = time.time_ns()
    try:
        g = subprocess.run(['nvidia-smi', '--query-gpu=uuid,utilization.gpu',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True)
        query_end = time.time_ns()
        if bool(lock_path) != bool(lifetime_path):
            raise ValueError('process observation requires both lock and lifetime paths')
        # The lock spans the process query AND capture of process instance
        # identity. Launchers cannot spawn or reap a tracked child during it.
        context = open(lock_path, 'r+b') if lock_path else nullcontext(None)
        with context as lock:
            if lock is not None:
                fcntl.flock(lock, fcntl.LOCK_EX)
            process_query_start = time.time_ns()
            p = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
                                '--format=csv,noheader'], capture_output=True, text=True)
            process_query_end = time.time_ns()
            active = active_native_starts(Path(lifetime_path)) if lock is not None else {}
            processes, process_issues = [], []
            for row in csv.reader(p.stdout.splitlines(), skipinitialspace=True):
                if not row or row[0].strip().lower() in ('[no running processes found]', 'no running processes found'):
                    continue
                try:
                    device, pid = row[0].strip(), int(row[1].strip())
                    matches = [(ident, start) for ident, start in active.items()
                               if start.get('pid') == pid and start.get('device_uuid') == device]
                    ticks = proc_start_ticks(pid) if len(matches) == 1 else None
                    ident, start = matches[0] if len(matches) == 1 else (None, {})
                    owned = (device == uuid and isinstance(ticks, int) and ticks > 0 and
                             start.get('proc_start_ticks') == ticks and is_descendant(pid, root_pid))
                    processes.append({'uuid': device, 'pid': pid, 'owned': owned,
                                      'invocation_id': ident, 'proc_start_ticks': ticks})
                    if not owned:
                        process_issues.append('foreign or unverified GPU compute process observed')
                except (IndexError, ValueError, OSError):
                    process_issues.append('invalid or unreadable compute-process identity')
    except OSError as exc:
        return {'sampled_at_epoch_ns': sampled, 'devices': [], 'processes': [],
                'issues': ['nvidia-smi unavailable: ' + str(exc)]}
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {'sampled_at_epoch_ns': sampled, 'devices': [], 'processes': [],
                'issues': ['invalid process observation metadata: ' + str(exc)]}
    issues, devices = process_issues, []
    if g.returncode or p.returncode:
        issues.append('nvidia-smi telemetry command failed')
    for row in csv.reader(g.stdout.splitlines(), skipinitialspace=True):
        try:
            device, utilization = row[0].strip(), int(row[1].strip())
            devices.append({'uuid': device, 'utilization_percent': utilization})
            if not device.startswith('GPU-') or not 0 <= utilization <= 100:
                issues.append('invalid device identity or utilization')
            if device != uuid and utilization != 0:
                issues.append('non-target device utilization observed')
        except (IndexError, ValueError):
            issues.append('invalid device telemetry')
    if not devices or uuid not in {d['uuid'] for d in devices} or len({d['uuid'] for d in devices}) != len(devices):
        issues.append('incomplete or duplicate device inventory')
    return {'sampled_at_epoch_ns': sampled, 'gpu_query_start_ns': sampled, 'gpu_query_end_ns': query_end,
            'process_query_start_ns': process_query_start, 'process_query_end_ns': process_query_end,
            'devices': devices, 'processes': processes,
            'issues': sorted(set(issues))}


def run_serial_monitored(command, env, uuid, output, binary_sha256):
    before = serial_sample(uuid, 0)
    before['phase'] = 'before'
    if any(d['uuid'] == uuid and d['utilization_percent'] != 0 for d in before['devices']):
        before['issues'].append('target was not quiescent before confirmation')
    if before['issues']:
        evidence = {'schema': 'cgs.serial-telemetry.v3', 'scheduler_job_id': os.environ['AUTOBENCH_SCHEDULER_JOB_ID'],
                    'uuid': uuid, 'samples': [before], 'clean': False,
                    'limitation': 'Confirmation was not started because initial sampled quiescence failed.'}
        (output / 'serial-telemetry.json').write_text(json.dumps(evidence, indent=2) + '\n')
        return subprocess.CompletedProcess(command, 1, '', 'initial GPU telemetry was not clean'), False
    lifetime_path = output / 'native-child-lifetimes.jsonl'
    lock_path = output / 'native-process-observation.lock'
    if lifetime_path.exists() or lock_path.exists():
        raise RuntimeError('native-child lifetime and lock paths must be fresh')
    lock_path.touch(exist_ok=False)
    env = dict(env, CGS_LIFETIME_EVENTS=str(lifetime_path), CGS_PROCESS_LOCK=str(lock_path))
    with tempfile.TemporaryFile(mode='w+t') as out, tempfile.TemporaryFile(mode='w+t') as err:
        process = subprocess.Popen(command, env=env, text=True, stdout=out, stderr=err)
        samples = [before]
        while process.poll() is None:
            sample = serial_sample(uuid, process.pid, lock_path, lifetime_path)
            sample['phase'] = 'during'
            target_busy = any(d['uuid'] == uuid and d['utilization_percent'] > 0 for d in sample['devices'])
            owned = any(p['uuid'] == uuid and p['owned'] for p in sample['processes'])
            if target_busy and not owned:
                sample['requires_lifetime_attribution'] = True
            samples.append(sample)
            if process.poll() is None:
                time.sleep(0.5)
        process.communicate()
        out.seek(0)
        err.seek(0)
        stdout, stderr = out.read(), err.read()
    deadline = time.monotonic() + 10
    while True:
        after = serial_sample(uuid, 0)
        after['phase'] = 'after'
        if any(d['uuid'] == uuid and d['utilization_percent'] > 0 for d in after['devices']):
            after['requires_lifetime_attribution'] = True
        samples.append(after)
        if any(d['uuid'] == uuid and d['utilization_percent'] == 0 for d in after['devices']):
            break
        if time.monotonic() >= deadline:
            after['issues'].append('target did not return to quiescence')
            break
        time.sleep(0.5)
    inventory = {d['uuid'] for d in before['devices']}
    for sample in samples:
        if {d['uuid'] for d in sample['devices']} != inventory:
            sample['issues'].append('device inventory changed during observation')
    starts, ends, lifetime_issues = {}, {}, []
    try:
        for line in lifetime_path.read_text().splitlines():
            event = json.loads(line)
            target = starts if event.get('event') == 'start' else ends if event.get('event') == 'end' else None
            if target is None or event.get('invocation_id') in target:
                raise ValueError('invalid or duplicate lifetime event')
            target[event['invocation_id']] = event
        if not starts or set(starts) != set(ends):
            raise ValueError('missing or unpaired native-child lifetime events')
        pids = set()
        for ident, start in starts.items():
            end, argv = ends[ident], start.get('argv', [])
            output_path = Path(end['output_path'])
            if (start.get('device_uuid') != uuid or start.get('binary_sha256') != binary_sha256 or
                    not isinstance(start.get('pid'), int) or start['pid'] in pids or end.get('pid') != start['pid'] or
                    not isinstance(start.get('proc_start_ticks'), int) or start['proc_start_ticks'] <= 0 or
                    end.get('proc_start_ticks') != start['proc_start_ticks'] or
                    not argv or Path(argv[0]).name != 'cgs_study' or '--mode' not in argv or 'campaign' not in argv or
                    '--backend' not in argv or argv[argv.index('--backend') + 1] != start.get('backend') or
                    '--output' not in argv or argv[argv.index('--output') + 1] != str(output_path) or
                    end.get('returncode') != 0 or not output_path.is_file() or sha(output_path) != end.get('output_sha256') or
                    not start['spawn_lower_ns'] <= start['spawn_upper_ns'] <= end['reap_lower_ns'] <= end['reap_upper_ns']):
                raise ValueError('native-child identity, output, or lifetime is invalid')
            pids.add(start['pid'])
    except (OSError, ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        lifetime_issues.append(str(exc))
    for sample in samples:
        for process_entry in sample['processes']:
            if not process_entry.get('owned'):
                continue
            ident = process_entry.get('invocation_id')
            start, end = starts.get(ident, {}), ends.get(ident, {})
            try:
                valid = (process_entry.get('uuid') == uuid and process_entry.get('pid') == start['pid'] and
                         process_entry.get('proc_start_ticks') == start['proc_start_ticks'] == end['proc_start_ticks'] and
                         start['spawn_upper_ns'] <= sample['process_query_start_ns'] <=
                         sample['process_query_end_ns'] <= end['reap_lower_ns'])
            except (KeyError, TypeError):
                valid = False
            if not valid or lifetime_issues:
                sample['issues'].append('owned process instance lacks synchronized lifetime proof')
        if sample.get('requires_lifetime_attribution'):
            matches = [ident for ident, start in starts.items() if ident in ends and
                       start.get('spawn_upper_ns', 0) <= sample['gpu_query_start_ns'] and
                       ends[ident].get('reap_upper_ns', 0) >= sample['gpu_query_start_ns'] - 1000000000]
            sample['attributed_invocation_ids'] = sorted(matches)
            if not matches or lifetime_issues:
                sample['issues'].append('target utilization has no attributable bounded native-child lifetime')
    evidence = {'schema': 'cgs.serial-telemetry.v3', 'scheduler_job_id': os.environ['AUTOBENCH_SCHEDULER_JOB_ID'],
                'uuid': uuid, 'samples': samples,
                'retrospective_window_bound_ns': 1000000000, 'lifetime_issues': lifetime_issues,
                'lifetime_sha256': sha(lifetime_path) if lifetime_path.is_file() else None,
                'clean': not lifetime_issues and any(s['phase'] == 'during' for s in samples) and all(not s['issues'] for s in samples),
                'limitation': 'A bounded native-child overlap explains retrospective target utilization; sampled telemetry cannot exclude foreign activity between observations.'}
    (output / 'serial-telemetry.json').write_text(json.dumps(evidence, indent=2) + '\n')
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr), evidence['clean']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def executable_version(argv):
    try:
        result = subprocess.run(argv, check=True, text=True, capture_output=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def toolchain(binary):
    ldd = subprocess.check_output(['ldd', str(binary)], text=True)
    dependencies = {}
    for line in ldd.splitlines():
        fields = line.strip().split()
        paths = [x for x in fields if x.startswith('/')]
        for name in paths:
            path = Path(name)
            if path.is_file():
                dependencies[str(path)] = sha(path)
    tools = {}
    for name, path, version in [('nvcc', NVCC, [NVCC, '--version']),
                                ('host_cxx', '/usr/bin/g++-12', ['/usr/bin/g++-12', '--version']),
                                ('cmake', '/usr/bin/cmake', ['/usr/bin/cmake', '--version']),
                                ('sanitizer', SANITIZER, [SANITIZER, '--version'])]:
        resolved = Path(path).resolve(strict=True)
        tools[name] = {'path': str(resolved), 'sha256': sha(resolved),
                       'version': executable_version(version)}
    return {'tools': tools, 'ldd': ldd, 'linked_file_sha256': dependencies,
            'host_pkg_config_versions': {name: executable_version(['pkg-config', '--modversion', name])
                                         for name in ('hdf5', 'openssl')}}


def extract(archive, target, expected):
    if sha(archive) != expected:
        raise RuntimeError('source archive SHA-256 mismatch')
    # Autobench owns the build directory; never merge with previous bytes.
    target.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        names = [x.name for x in members]
        if len(set(names)) != len(names) or 'SOURCE-MANIFEST.json' not in names:
            raise RuntimeError('invalid source archive membership')
        for item in members:
            name = Path(item.name)
            if not item.isfile() or name.is_absolute() or '..' in name.parts:
                raise RuntimeError('unsafe source archive member')
            dst = target / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(item) as src, dst.open('wb') as out:
                while data := src.read(1024 * 1024):
                    out.write(data)
            os.chmod(dst, item.mode & 0o777)
    manifest = json.loads((target / 'SOURCE-MANIFEST.json').read_text())
    if set(names) != set(manifest['files']) | {'SOURCE-MANIFEST.json'}:
        raise RuntimeError('archive membership differs from manifest')
    for name, info in manifest['files'].items():
        path = target / name
        if path.stat().st_size != info['size'] or sha(path) != info['sha256']:
            raise RuntimeError(f'archive content mismatch: {name}')


def build(args):
    build_dir = Path(args.build).resolve()
    source = build_dir / 'source'
    extract(Path(args.archive), source, args.archive_sha256)
    root = source / 'Cellerator'
    study = root / 'bench/chromatin_geometry'
    if args.suite in ('interference', 'kda', 'profile'):
        if not args.frozen_binary or not args.frozen_build_receipt:
            raise RuntimeError('interference needs a snapshotted qualified binary and build receipt')
        original = json.loads(Path(args.frozen_build_receipt).read_text())
        if original['archive_sha256'] != args.archive_sha256 or sha(Path(args.frozen_binary)) != original['binary_sha256']:
            raise RuntimeError('qualified interference binary does not match its source archive')
        binary = build_dir / 'cgs_study'
        shutil.copy2(args.frozen_binary, binary)
        # Retained artifacts are data files; Autobench snapshots also discard
        # executable mode. Restore it only on the verified local build copy.
        binary.chmod(binary.stat().st_mode | 0o100)
        shutil.copy2(args.frozen_build_receipt, build_dir / 'original-build.json')
        command = ['copy-verified-qualified-binary', args.frozen_binary]
    elif args.suite == 'ste':
        binary = build_dir / 'cgs-ste-gpu-tests'
        command = [NVCC, '-std=c++17', '-arch=sm_70', '-lineinfo', '-I' + str(study),
                   '-I' + str(root / 'include'), str(study / 'learning/ste.cu'), str(study / 'learning/ste_chain.cu'),
                   str(study / 'learning/ste_tests.cu'), str(study / 'learning/ste_gpu_test_main.cc'),
                   '-o', str(binary)]
        subprocess.run(command, check=True)
    else:
        cmake = build_dir / 'cmake'
        command = ['cmake', '-S', str(study), '-B', str(cmake), '-DCMAKE_BUILD_TYPE=Release',
                   '-DCMAKE_CUDA_ARCHITECTURES=70', '-DCMAKE_CUDA_COMPILER=' + NVCC,
                   '-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12', '-DCMAKE_CXX_COMPILER=/usr/bin/g++-12',
                   '-DCMAKE_CUDA_FLAGS=-lineinfo', '-DBUILD_TESTING=OFF']
        subprocess.run(command, check=True)
        subprocess.run(['cmake', '--build', str(cmake), '--target', args.target, '--parallel', '2'], check=True)
        binary = cmake / args.target
    receipt = {'schema': 'cgs.build.v1', 'archive_sha256': args.archive_sha256,
               'binary_sha256': sha(binary), 'binary_relative_path': str(binary.relative_to(build_dir)),
               'suite': args.suite, 'target': args.target, 'command': command,
               'toolchain': toolchain(binary)}
    (build_dir / 'build.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if args.suite == 'kda':
        if not args.kda_workspace or not args.config:
            raise RuntimeError('KDA build requires its workspace and frozen configuration')
        subprocess.run([sys.executable, str(study / 'scripts/kda_setup.py'),
                        '--source', str(root), '--kda-root', str(source / 'KDA'),
                        '--binary', str(binary), '--workspace', args.kda_workspace,
                        '--config', args.config], check=True)
        setup = {'workspace': args.kda_workspace, 'task_sha256': sha(Path(args.kda_workspace) / 'task.json'),
                 'config_sha256': sha(Path(args.config)), 'archive_sha256': args.archive_sha256,
                 'binary_sha256': sha(binary)}
        (build_dir / 'kda-setup.json').write_text(json.dumps(setup, indent=2) + '\n')
    print(json.dumps(receipt, sort_keys=True))


def quiescence(uuid):
    observations = []
    consecutive = 0
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        raw = subprocess.check_output(['nvidia-smi', '-i', uuid, '--query-gpu=uuid,utilization.gpu',
                                       '--format=csv,noheader,nounits'], text=True).strip()
        fields = [x.strip() for x in raw.split(',')]
        if len(fields) != 2 or fields[0] != uuid:
            raise RuntimeError('UUID-scoped quiescence observation unavailable')
        utilization = int(fields[1])
        observations.append({'uuid': fields[0], 'utilization_percent': utilization, 'time': time.time()})
        consecutive = consecutive + 1 if utilization == 0 else 0
        if consecutive == 3:
            return observations
        time.sleep(1)
    raise RuntimeError('GPU did not satisfy three consecutive idle samples')


def run(args):
    if not os.environ.get('AUTOBENCH_SCHEDULER_JOB_ID') or not os.environ.get('AUTOBENCH_SCHEDULER_ATTEMPT_ID'):
        raise RuntimeError('GPU execution requires an admitted Autobench Scheduler worker')
    current = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if current.startswith('GPU-') and current != args.uuid:
        raise RuntimeError('visible CUDA UUID conflicts with requested Scheduler UUID')
    receipt = json.loads((Path(args.build) / 'build.json').read_text())
    binary = Path(args.build) / receipt['binary_relative_path']
    if receipt['suite'] != args.suite or receipt['target'] != args.target:
        raise RuntimeError('requested suite or target differs from build receipt')
    if sha(binary) != receipt['binary_sha256'] or receipt['archive_sha256'] != args.archive_sha256:
        raise RuntimeError('binary or source identity changed after build')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.config:
        shutil.copy2(args.config, output / 'experiment.json')
    if args.suite == 'kda':
        snapshot_dir = Path(args.config).parent
        for name in ('interference-evidence.json', 'contract-freeze.json'):
            source = snapshot_dir / name
            if not source.is_file():
                raise RuntimeError(f'KDA evidence input is missing: {name}')
            shutil.copy2(source, output / name)
    observations = quiescence(args.uuid)
    device_information = subprocess.check_output(
        ['nvidia-smi', '-i', args.uuid, '--query-gpu=uuid,name,driver_version',
         '--format=csv,noheader'], text=True).strip()
    if args.suite == 'kda':
        if not args.kda_workspace or not args.config or args.sanitizer != 'none':
            raise RuntimeError('KDA evaluation requires its prepared workspace and no sanitizer wrapper')
        setup = json.loads((Path(args.build) / 'kda-setup.json').read_text())
        task = Path(args.kda_workspace) / 'task.json'
        if (setup['workspace'] != args.kda_workspace or setup['task_sha256'] != sha(task) or
                setup['config_sha256'] != sha(Path(args.config)) or setup['binary_sha256'] != sha(binary)):
            raise RuntimeError('prepared KDA experiment identity changed')
        evaluator = Path(args.build) / 'source/Cellerator/bench/chromatin_geometry/scripts/kda_evaluate.py'
        command = [sys.executable, str(evaluator), '--kda-root', str(Path(args.build) / 'source/KDA'),
                   '--task', str(task), '--workspace', str(Path(args.kda_workspace) / 'engine'),
                   '--config', args.config, '--binary', str(binary), '--output', str(output / 'kda')]
    elif args.suite == 'triage':
        if not args.config or args.sanitizer != 'none':
            raise RuntimeError('triage needs its frozen configuration and no sanitizer wrapper')
        triage = Path(args.build) / 'source/Cellerator/bench/chromatin_geometry/scripts/triage.py'
        command = [sys.executable, str(triage), '--binary', str(binary), '--config', args.config,
                   '--output', str(output / 'triage'), '--replay-root', str(Path(args.build) / 'source/replay')]
    elif args.suite == 'profile':
        if not args.config or args.sanitizer != 'none':
            raise RuntimeError('profile needs frozen configuration and no sanitizer wrapper')
        profile = Path(args.build) / 'source/Cellerator/bench/chromatin_geometry/scripts/profile.py'
        command = [sys.executable, str(profile), '--binary', str(binary), '--config', args.config,
                   '--output', str(output / 'profile')]
    elif args.suite == 'interference':
        if not args.config or args.sanitizer != 'none' or not args.interference_mode:
            raise RuntimeError('interference needs a frozen config, mode and no sanitizer')
        probe = Path(args.build) / 'source/Cellerator/bench/chromatin_geometry/scripts/interference.py'
        command = [sys.executable, str(probe), 'run', '--mode', args.interference_mode,
                   '--config', args.config, '--binary', str(binary), '--output-dir', str(output / 'interference'),
                   '--repetition', str(args.repetition), '--source-root', str(Path(args.build) / 'source')]
        if args.interference_mode == 'concurrent':
            if not args.shared_study_root or not args.barrier_id:
                raise RuntimeError('concurrent interference needs study-local rendezvous')
            command += ['--shared-study-root', args.shared_study_root, '--barrier-id', args.barrier_id]
    else:
        command = [str(binary)] + (['teacher'] if args.target == 'cgs_learning_tests' else [])
    if args.sanitizer != 'none':
        command = [SANITIZER, '--tool', args.sanitizer, '--error-exitcode', '99'] + command
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.uuid)
    if args.suite == 'kda':
        proc, serial_telemetry_clean = run_serial_monitored(command, env, args.uuid, output, receipt['binary_sha256'])
    else:
        proc = subprocess.run(command, env=env, text=True, capture_output=True)
        serial_telemetry_clean = None
    (output / 'test.stdout').write_text(proc.stdout)
    (output / 'test.stderr').write_text(proc.stderr)
    result = {'schema': 'cgs.packet.v1', 'scheduler_job_id': os.environ['AUTOBENCH_SCHEDULER_JOB_ID'],
              'scheduler_attempt_id': os.environ['AUTOBENCH_SCHEDULER_ATTEMPT_ID'],
              'uuid': args.uuid, 'archive_sha256': args.archive_sha256,
              'binary_sha256': receipt['binary_sha256'], 'binary_path': str(binary), 'sanitizer': args.sanitizer,
              'device_information': device_information,
              'quiescence': observations, 'argv': command, 'returncode': proc.returncode,
              'suite': args.suite, 'target': args.target,
              'serial_telemetry_clean': serial_telemetry_clean,
              'success': proc.returncode == 0 and (args.suite != 'kda' or serial_telemetry_clean is True) and
                         (args.suite != 'ste' or 'STE CUDA validation passed' in proc.stdout) and
                         (args.target != 'cgs_learning_tests' or
                          ('teacher CUDA tests: PASS' in proc.stdout and 'teacher fit-to-circuit chain CUDA tests: PASS' in proc.stdout))}
    (output / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, sort_keys=True))
    if not result['success']:
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['build', 'ste-test'])
    p.add_argument('--archive', required=True)
    p.add_argument('--archive-sha256', required=True)
    p.add_argument('--build', required=True)
    p.add_argument('--suite', choices=['ste', 'combined', 'kda', 'triage', 'profile', 'interference'], default='ste')
    p.add_argument('--target', choices=['cgs-ste-gpu-tests', 'cgs_learning_tests', 'cgs_condition_tests',
                                        'cgs_consumer_tests', 'cgs_linear_library_tests', 'cgs_study'], default='cgs-ste-gpu-tests')
    p.add_argument('--kda-workspace')
    p.add_argument('--config')
    p.add_argument('--frozen-binary')
    p.add_argument('--frozen-build-receipt')
    p.add_argument('--interference-mode', choices=['isolated', 'concurrent'])
    p.add_argument('--repetition', type=int, default=0)
    p.add_argument('--shared-study-root')
    p.add_argument('--barrier-id')
    p.add_argument('--output')
    p.add_argument('--uuid')
    p.add_argument('--sanitizer', choices=['none', 'memcheck', 'racecheck', 'initcheck', 'synccheck'], default='none')
    args = p.parse_args()
    if args.mode == 'build':
        build(args)
    else:
        if not args.output or not args.uuid:
            p.error('ste-test requires --output and --uuid')
        run(args)


if __name__ == '__main__':
    main()
