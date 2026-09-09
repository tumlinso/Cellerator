#!/usr/bin/env python3
"""Execute CUDA qualification only inside the sealed gate's live lease and lock."""
import argparse
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET


def require(value, why):
    if not value:
        raise ValueError(why)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_regression_junit(path, expected):
    root = ET.parse(path).getroot()
    cases = list(root.iter('testcase'))
    require(len(cases) == len(expected) and {case.get('name') for case in cases} == set(expected),
            'regression execution names differ from required inventory')
    require(all(case.get('status') == 'run' for case in cases), 'regression test was not executed')
    for tag in ('skipped', 'failure', 'error'):
        require(not list(root.iter(tag)), 'regression report contains ' + tag)
    for suite in root.iter('testsuite'):
        require(all(int(suite.get(key, '0')) == 0 for key in ('failures', 'errors', 'disabled', 'skipped')),
                'regression suite contains failed or skipped tests')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--sanitizer', type=Path, required=True)
    parser.add_argument('--regression-test', action='append', default=[])
    parser.add_argument('--require-regressions', action='store_true')
    args = parser.parse_args()
    require(not args.require_regressions or args.regression_test, 'retained RU1 regressions must be configured')
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
    regression_files = {}
    regression_command = None
    if args.regression_test:
        require(len(set(args.regression_test)) == len(args.regression_test), 'duplicate regression test')
        require(all(re.fullmatch(r'ru1_[a-z0-9_]+', name) for name in args.regression_test), 'invalid regression test name')
        expression = '^(' + '|'.join(args.regression_test) + ')$'
        regression_command = ['ctest', '--test-dir', bindings['build_dir'], '--no-tests=error', '-R', expression, '--verbose']
        inventory = subprocess.run(regression_command + ['--show-only=json-v1'], text=True, capture_output=True, check=True, timeout=30)
        tests = json.loads(inventory.stdout)['tests']
        require({test['name'] for test in tests} == set(args.regression_test), 'regression inventory incomplete')
        for test in tests:
            require(not any(prop.get('name') == 'DISABLED' and prop.get('value')
                            for prop in test.get('properties', [])), 'disabled regression test')
            require(test.get('command') and Path(test['command'][0]).is_file(), 'regression executable absent')
            for word in test['command']:
                if Path(word).is_absolute() and Path(word).is_file():
                    regression_files[word] = digest(word)
        record['regression_inventory'] = tests
        record['regression_file_sha256'] = regression_files
    fd, path = tempfile.mkstemp(prefix='leased-executable-', suffix='.json', dir=bindings['evidence_dir'])
    with os.fdopen(fd, 'w') as output:
        json.dump(record, output);output.flush();os.fsync(output.fileno())
        try:
            for command in [[str(args.executable)], [str(args.sanitizer), '--tool', 'memcheck', '--error-exitcode', '99', str(args.executable)]]:
                result = subprocess.run(command, text=True, capture_output=True, timeout=120)
                record['commands'].append(dict(argv=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
                print(result.stdout, end='');print(result.stderr, end='')
                require(result.returncode == 0, 'CUDA executable or Compute Sanitizer failed')
            if regression_command:
                junit = Path(path).with_suffix('.xml')
                regression_command.extend(['--output-junit', str(junit)])
                result = subprocess.run(regression_command, text=True, capture_output=True, timeout=180)
                record['commands'].append(dict(argv=regression_command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
                print(result.stdout, end='');print(result.stderr, end='')
                if junit.is_file():
                    record['regression_junit'] = dict(path=str(junit), sha256=digest(junit))
                    junit.chmod(0o444)
                require(result.returncode == 0, 'retained RU1 regression failed')
                validate_regression_junit(junit, args.regression_test)
                require(all(digest(path) == value for path, value in regression_files.items()), 'regression source or executable changed')
            require(before == digest(args.executable), 'executable changed during qualification')
            record['passed'] = True
        finally:
            output.seek(0);output.truncate();json.dump(record, output, indent=2);output.write('\n');output.flush();os.fsync(output.fileno())
            os.fchmod(output.fileno(), 0o444)
    Path(path).chmod(0o444)
    print('Leased executable evidence: ' + path)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        raise SystemExit('run_leased_test: ' + str(error))
