#!/usr/bin/env python3
"""Execute CUDA qualification only inside the sealed gate's live lease and lock."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def require(value, why):
    if not value:
        raise ValueError(why)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--sanitizer', type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get('NF1_LEASED_BINDINGS'), 'leased parent bindings required')
    binding_path = Path(os.environ['NF1_LEASED_BINDINGS'])
    bindings = json.loads(binding_path.read_bytes())
    receipt = Path(os.environ['TODO_GPU_LEASE_RECEIPT']).resolve()
    require(receipt == Path(bindings['gpu_lease_receipt']).resolve(), 'lease path mismatch')
    # Contention alone does not identify the holder. The trusted sealed launch
    # supplies parent context; the native verifier below independently checks
    # lease ownership. Reject an uncontended lock and never wait on a second one.
    with open(bindings['shared_gpu_lock_file'], 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(lock, fcntl.LOCK_UN)
            raise ValueError('shared GPU lock is uncontended despite trusted sealed-parent launch')
    verifier = [word.replace('{lease_receipt}', str(receipt)) for word in bindings['gpu_lease_verifier_argv']]
    verified = subprocess.run(verifier, text=True, capture_output=True, check=True, timeout=30)
    before = digest(args.executable)
    record = dict(kind='nf1-leased-executable-v1', executable=str(args.executable.resolve()),
                  executable_sha256=before, sanitizer_sha256=digest(args.sanitizer),
                  bindings_sha256=digest(binding_path), lease_sha256=digest(receipt),
                  live_verification=json.loads(verified.stdout), passed=False, commands=[])
    fd, path = tempfile.mkstemp(prefix='leased-executable-', suffix='.json', dir=bindings['evidence_dir'])
    with os.fdopen(fd, 'w') as output:
        json.dump(record, output);output.flush();os.fsync(output.fileno())
        try:
            for command in [[str(args.executable)], [str(args.sanitizer), '--tool', 'memcheck', '--error-exitcode', '99', str(args.executable)]]:
                result = subprocess.run(command, text=True, capture_output=True, timeout=120)
                record['commands'].append(dict(argv=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
                print(result.stdout, end='');print(result.stderr, end='')
                require(result.returncode == 0, 'CUDA executable or Compute Sanitizer failed')
            require(before == digest(args.executable), 'executable changed during qualification')
            record['passed'] = True
        finally:
            output.seek(0);output.truncate();json.dump(record, output, indent=2);output.write('\n');output.flush();os.fsync(output.fileno())
    Path(path).chmod(0o444)
    print('Leased executable evidence: ' + path)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        raise SystemExit('run_leased_test: ' + str(error))
