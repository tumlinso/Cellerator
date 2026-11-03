#!/usr/bin/env python3
"""Stage one immutable runner into Autobench scratch and execute a driver.

Autobench snapshots copy file bytes without executable mode. The runner is
copied to its managed, size-limited attempt output, made executable, and
byte-checked. The literal @RUNNER@ in the driver's argv is replaced with the
staged path. Device Scheduler and Autobench retain the authoritative receipts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runner', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('driver', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.driver
    if command and command[0] == '--':
        command = command[1:]
    if not command or '@RUNNER@' not in command:
        parser.error('driver argv must include the literal @RUNNER@')
    job = os.environ.get('AUTOBENCH_SCHEDULER_JOB_ID')
    attempt = os.environ.get('AUTOBENCH_SCHEDULER_ATTEMPT_ID')
    if not job or not attempt:
        raise RuntimeError('a Scheduler-admitted Autobench worker is required')
    args.output.mkdir(parents=True, exist_ok=True)
    staged = args.output / 'staged-runner'
    shutil.copyfile(args.runner, staged)
    staged.chmod(0o700)
    original_hash = sha256(args.runner)
    if sha256(staged) != original_hash:
        raise RuntimeError('runner copy differs from frozen snapshot')
    command = [str(staged) if item == '@RUNNER@' else item for item in command]
    metadata = {'scheduler_job_id': job, 'scheduler_attempt_id': attempt,
                'snapshot_runner_sha256': original_hash, 'argv': command}
    (args.output / 'operator-stage.json').write_text(json.dumps(metadata, indent=2) + '\n')
    result = subprocess.run(command, check=False)
    if sha256(staged) != original_hash:
        raise RuntimeError('runner changed during execution')
    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
