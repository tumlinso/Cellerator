#!/usr/bin/env python3
"""Bind an actual native controller lease to a sealed NF1 test gate."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile

SHARED_LOCK = '/tmp/nf1-20260908-v1-device-evidence.lock'

def require(value, message):
    if not value:
        raise ValueError(message)


def validate_device_binding(receipt, expected, visible, inventory):
    require(receipt.get('format') == 'CUDA-FOREGROUND-LEASE/1', 'wrong lease format')
    require(receipt.get('state') == 'active', 'inactive lease receipt')
    reserved = {r[len('accelerator:'):] for r in receipt.get('resource_ids', []) if r.startswith('accelerator:')}
    require(expected and len(expected) == len(set(expected)), 'expected GPU set absent or duplicate')
    require(set(expected) == reserved, 'reserved device set differs from requested devices')
    require(visible, 'CUDA_VISIBLE_DEVICES must be supplied by controller')
    mapped = [inventory.get(token.strip(), token.strip()) for token in visible.split(',')]
    require(len(mapped) == len(set(mapped)) and set(mapped) == set(expected), 'visible devices differ from lease')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate-script', type=Path, required=True)
    parser.add_argument('--group', required=True)
    parser.add_argument('--verifier', type=Path, required=True)
    parser.add_argument('--gpu-uuid', action='append', required=True)
    args = parser.parse_args()
    lease_name = os.environ.get('TODO_GPU_LEASE_RECEIPT')
    require(lease_name, 'native TODO_GPU_LEASE_RECEIPT is required')
    lease_path = Path(lease_name).resolve()
    receipt_bytes = lease_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    script = args.gate_script.resolve()
    verifier = args.verifier.resolve()
    require(script.is_file() and verifier.is_file(), 'gate script or verifier absent')
    require(Path(receipt['project_root']).resolve() == Path.cwd().resolve(), 'lease reserves another worktree')
    probe = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'],
                           check=True, text=True, capture_output=True, timeout=30)
    inventory = {}
    for line in probe.stdout.splitlines():
        index, uuid = (part.strip() for part in line.split(',', 1))
        inventory[index] = uuid
    validate_device_binding(receipt, args.gpu_uuid, os.environ.get('CUDA_VISIBLE_DEVICES'), inventory)
    verify_argv = [sys.executable, '-B', str(verifier), '--lease-receipt', str(lease_path),
                   '--project-root', str(Path.cwd().resolve())]
    for gpu in args.gpu_uuid:
        verify_argv.extend(['--gpu-uuid', gpu])
    # Receipt text alone is never sufficient: inspect the actual native owner.
    verified = subprocess.run(verify_argv, check=True, text=True, capture_output=True, timeout=30)
    require(receipt_bytes == lease_path.read_bytes(), 'lease receipt changed during validation')
    base_path = Path(os.environ['NF1_EXECUTION_BINDINGS']).resolve()
    base_bytes = base_path.read_bytes()
    bindings = json.loads(base_bytes)
    evidence = Path(bindings['evidence_dir']).resolve()
    require(evidence.is_dir() and not evidence.is_relative_to(Path.cwd().resolve()), 'external evidence directory required')
    bindings['gpu_lease_receipt'] = str(lease_path)
    bindings['gpu_lease_verifier_argv'] = [word if word != str(lease_path) else '{lease_receipt}' for word in verify_argv]
    bindings['shared_gpu_lock_file'] = SHARED_LOCK
    bindings['adapter_evidence'] = {
        'lease_sha256': hashlib.sha256(receipt_bytes).hexdigest(),
        'base_bindings_sha256': hashlib.sha256(base_bytes).hexdigest(),
        'visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
        'verified_native_owner': json.loads(verified.stdout),
        'gate_script_sha256': hashlib.sha256(script.read_bytes()).hexdigest(),
        'verifier_sha256': hashlib.sha256(verifier.read_bytes()).hexdigest(),
    }
    fd, derived = tempfile.mkstemp(prefix='leased-bindings-', suffix='.json', dir=evidence)
    with os.fdopen(fd, 'w') as output:
        json.dump(bindings, output, indent=2)
        output.write('\n')
    os.chmod(derived, 0o444)
    os.environ['NF1_LEASED_BINDINGS'] = derived
    matrix = json.loads((script.parent.parent / 'machine/acceptance_matrix.json').read_text())
    group = matrix['gate_groups'][args.group]
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    test_receipt = evidence / ('leased-test-' + args.group + '-' + stamp + '.json')
    command = [sys.executable, '-B', str(script), '--group', args.group, '--bindings', derived, '--receipt', str(test_receipt)]
    def execute():
        live = subprocess.run(verify_argv, check=True, text=True, capture_output=True, timeout=30)
        require(receipt_bytes == lease_path.read_bytes(), 'lease changed while waiting for lock')
        result = subprocess.run(command, check=False)
        supplemental = []
        for name in bindings.get('supplemental_ctest_names', []):
            require(not group.get('requires_gpu'), 'supplemental GPU tests require adapter-owned lock')
            require(isinstance(name, str) and re.fullmatch(r'[a-zA-Z0-9_]+', name), 'invalid supplemental test name')
            argv = ['ctest', '--test-dir', bindings['build_dir'], '--no-tests=error',
                    '-R', '^' + name + '$', '--verbose']
            check = subprocess.run(argv, text=True, capture_output=True, timeout=150)
            supplemental.append(dict(argv=argv, returncode=check.returncode, stdout=check.stdout, stderr=check.stderr))
        final_returncode = result.returncode or (1 if any(x['returncode'] for x in supplemental) else 0)
        sidecar = dict(bindings['adapter_evidence'], live_verification=json.loads(live.stdout),
                       supplemental_tests=supplemental, bindings_path=derived, bindings_sha256=hashlib.sha256(Path(derived).read_bytes()).hexdigest(),
                       test_receipt=str(test_receipt), returncode=final_returncode, passed=False, required_group_passed=False,
                       test_receipt_sha256=hashlib.sha256(test_receipt.read_bytes()).hexdigest() if test_receipt.exists() else None)
        if test_receipt.exists():
            completed = json.loads(test_receipt.read_bytes())
            sidecar['source_commit'] = completed.get('source_commit')
            sidecar['source_fingerprint'] = completed.get('source_fingerprint')
            sidecar['executables'] = completed.get('executables')
            sidecar['cmake_cache_sha256'] = completed.get('cmake_cache_sha256')
            sidecar['required_group_passed'] = completed.get('passed') is True
            sidecar['passed'] = sidecar['required_group_passed'] and final_returncode == 0
        children = []
        for child in evidence.glob('leased-executable-*.json'):
            child_bytes = child.read_bytes()
            child_record = json.loads(child_bytes)
            if child_record.get('bindings_sha256') == sidecar['bindings_sha256']:
                children.append(dict(path=str(child), sha256=hashlib.sha256(child_bytes).hexdigest(),
                                     executable_sha256=child_record.get('executable_sha256'),
                                     sanitizer_sha256=child_record.get('sanitizer_sha256'), passed=child_record.get('passed')))
        sidecar['child_executable_evidence'] = children
        sidepath = evidence / ('lease-evidence-' + stamp + '.json')
        with sidepath.open('x') as output:
            json.dump(sidecar, output, indent=2)
            output.write('\n')
        sidepath.chmod(0o444)
        return final_returncode
    # Exactly one owner for the NF1 lock: GPU-labelled sealed runner owns it;
    # stronger device qualification of a host-labelled group requires this adapter.
    if group.get('requires_gpu'):
        return execute()
    with open(SHARED_LOCK, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return execute()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        raise SystemExit('run_gpu_gate: ' + str(error))
