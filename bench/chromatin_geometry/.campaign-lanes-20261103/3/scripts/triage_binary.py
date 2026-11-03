#!/usr/bin/env python3
"""Run the frozen complete-chain triage with a retained qualified binary."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tarfile
import time
from pathlib import Path


class AdapterError(RuntimeError):
    """A bound source, binary, or Scheduler execution contract failed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterError(f'cannot read JSON object {path}: {exc}') from exc
    if not isinstance(value, dict):
        raise AdapterError(f'JSON root is not an object: {path}')
    return value


def extract_archive(archive: Path, target: Path, expected_sha256: str) -> dict:
    if target.exists():
        if not target.is_dir() or any(target.iterdir()):
            raise AdapterError('source extraction directory must be new or empty')
    else:
        target.mkdir(parents=True)
    target.mkdir(parents=True, exist_ok=True)
    if sha256(archive) != expected_sha256:
        raise AdapterError('source archive digest differs from the registered identity')
    try:
        with tarfile.open(archive, 'r:gz') as tar:
            members = tar.getmembers()
            names = [member.name for member in members]
            if len(set(names)) != len(names) or 'SOURCE-MANIFEST.json' not in names:
                raise AdapterError('source archive has duplicate members or no manifest')
            for member in members:
                path = Path(member.name)
                if (not member.isfile() or path.is_absolute() or '..' in path.parts or
                        not path.parts or member.issym() or member.islnk()):
                    raise AdapterError(f'unsafe source archive member: {member.name}')
                destination = target / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    raise AdapterError(f'cannot read archive member: {member.name}')
                with source, destination.open('xb') as output:
                    shutil.copyfileobj(source, output)
                destination.chmod(member.mode & 0o777)
    except (OSError, tarfile.TarError) as exc:
        raise AdapterError(f'source archive extraction failed: {exc}') from exc

    manifest = read_object(target / 'SOURCE-MANIFEST.json')
    files = manifest.get('files')
    if not isinstance(files, dict) or set(names) != set(files) | {'SOURCE-MANIFEST.json'}:
        raise AdapterError('archive membership differs from its source manifest')
    for name, entry in files.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or not isinstance(entry, dict):
            raise AdapterError(f'invalid source manifest path: {name}')
        file = target / path
        if (not file.is_file() or file.stat().st_size != entry.get('size') or
                sha256(file) != entry.get('sha256')):
            raise AdapterError(f'archive content differs from its source manifest: {name}')
    return manifest


def build(args: argparse.Namespace) -> dict:
    if any(path.is_symlink() for path in (args.archive, args.binary, args.receipt)):
        raise AdapterError('archive, binary and build receipt must not be symlinks')
    archive = args.archive.resolve(strict=True)
    source_binary = args.binary.resolve(strict=True)
    original_receipt = args.receipt.resolve(strict=True)
    if not all(path.is_file() for path in (archive, source_binary, original_receipt)):
        raise AdapterError('archive, binary and build receipt must be regular files')
    if sha256(archive) != args.archive_sha256:
        raise AdapterError('archive bytes differ from requested archive SHA-256')
    original = read_object(original_receipt)
    if (original.get('schema') != 'cgs.build.v1' or original.get('target') != 'cgs_study' or
            original.get('archive_sha256') != args.archive_sha256 or
            sha256(source_binary) != original.get('binary_sha256')):
        raise AdapterError('qualified source, binary, and original build receipt do not agree')

    build_dir = args.build.resolve()
    if build_dir.exists():
        if not build_dir.is_dir() or any(build_dir.iterdir()):
            raise AdapterError('adapter build directory must be new or empty')
    else:
        build_dir.mkdir(parents=True)
    source_root = build_dir / 'source'
    extract_archive(archive, source_root, args.archive_sha256)
    binary = build_dir / 'cgs_study'
    shutil.copy2(source_binary, binary)
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    if sha256(binary) != original['binary_sha256']:
        raise AdapterError('copied qualified binary failed its digest check')
    shutil.copy2(original_receipt, build_dir / 'original-build.json')
    receipt = {
        'schema': 'cgs.build.v1', 'suite': 'triage-binary', 'target': 'cgs_study',
        'archive_sha256': args.archive_sha256, 'binary_sha256': original['binary_sha256'],
        'binary_relative_path': 'cgs_study', 'qualified_binary_source_path': str(source_binary),
        'original_build_receipt_sha256': sha256(original_receipt),
        'original_build_receipt_path': str(original_receipt),
        'source_archive_path': str(archive), 'command': ['copy-verified-qualified-binary'],
    }
    (build_dir / 'build.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    return receipt


def validate_build(build_dir: Path) -> tuple[dict, Path, Path]:
    build_dir = build_dir.resolve(strict=True)
    receipt = read_object(build_dir / 'build.json')
    if (receipt.get('schema') != 'cgs.build.v1' or receipt.get('suite') != 'triage-binary' or
            receipt.get('target') != 'cgs_study' or receipt.get('binary_relative_path') != 'cgs_study'):
        raise AdapterError('adapter build receipt does not identify retained cgs_study binary')
    binary = build_dir / 'cgs_study'
    source = build_dir / 'source'
    if (binary.is_symlink() or not binary.is_file() or
            not binary.stat().st_mode & stat.S_IXUSR or sha256(binary) != receipt.get('binary_sha256')):
        raise AdapterError('copied cgs_study binary is absent or changed')
    if (not source.is_dir() or sha256(receipt_path := build_dir / 'original-build.json') !=
            receipt.get('original_build_receipt_sha256')):
        raise AdapterError('original build receipt or verified source tree is absent')
    original = read_object(receipt_path)
    if (original.get('archive_sha256') != receipt.get('archive_sha256') or
            original.get('binary_sha256') != receipt.get('binary_sha256')):
        raise AdapterError('adapter build receipt differs from the original qualified build')
    return receipt, binary, source


def quiescence(uuid: str) -> list[dict]:
    observations = []
    end = time.monotonic() + 60
    consecutive = 0
    while time.monotonic() < end:
        result = subprocess.run(['nvidia-smi', '-i', uuid, '--query-gpu=uuid,utilization.gpu',
                                 '--format=csv,noheader,nounits'], text=True, capture_output=True)
        if result.returncode:
            raise AdapterError('Scheduler GPU quiescence telemetry failed')
        fields = [value.strip() for value in result.stdout.strip().split(',')]
        if len(fields) != 2 or fields[0] != uuid:
            raise AdapterError('UUID scoped GPU observation was incomplete')
        utilization = int(fields[1])
        observations.append({'uuid': uuid, 'utilization_percent': utilization, 'sampled_at': time.time()})
        consecutive = consecutive + 1 if utilization == 0 else 0
        if consecutive >= 3:
            return observations
        time.sleep(1)
    raise AdapterError('GPU did not provide three consecutive quiescent observations')


def run(args: argparse.Namespace) -> dict:
    job_id = os.environ.get('AUTOBENCH_SCHEDULER_JOB_ID', '').strip()
    attempt_id = os.environ.get('AUTOBENCH_SCHEDULER_ATTEMPT_ID', '').strip()
    if not job_id or not attempt_id:
        raise AdapterError('triage requires an admitted Autobench Scheduler worker')
    if not args.uuid.startswith('GPU-') or ',' in args.uuid:
        raise AdapterError('exact Scheduler GPU UUID is required')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if visible and visible != args.uuid:
        raise AdapterError('visible GPU differs from the requested Scheduler UUID')

    receipt, binary, source = validate_build(args.build)
    if args.config.is_symlink():
        raise AdapterError('triage config must be an immutable regular file')
    config = args.config.resolve(strict=True)
    if not config.is_file():
        raise AdapterError('triage config must be a regular file')
    output = args.output.resolve()
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise AdapterError('triage output directory must be new or empty')
    else:
        output.mkdir(parents=True)
    shutil.copy2(config, output / 'experiment.json')
    observations = quiescence(args.uuid)
    device = subprocess.check_output(['nvidia-smi', '-i', args.uuid, '--query-gpu=uuid,name,driver_version',
                                      '--format=csv,noheader'], text=True).strip()
    triage_script = source / 'Cellerator/bench/chromatin_geometry/scripts/triage.py'
    replay_root = source / 'replay'
    argv = ['python3', str(triage_script), '--binary', str(binary), '--config', str(config),
            '--output', str(output / 'triage'), '--replay-root', str(replay_root)]
    result = subprocess.run(argv, text=True, capture_output=True,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.uuid))
    (output / 'test.stdout').write_text(result.stdout)
    (output / 'test.stderr').write_text(result.stderr)
    summary_path = output / 'triage/summary.json'
    success = result.returncode == 0 and summary_path.is_file()
    if success:
        summary = read_object(summary_path)
        config_value = read_object(config)
        success = (summary.get('schema') == 'cgs.triage-result.v1' and summary.get('status') == 'pass' and
                   summary.get('device_uuid') == args.uuid and
                   len(summary.get('records', [])) == len(config_value.get('runs', [])))
    metrics = {
        'schema': 'cgs.packet.v1', 'scheduler_job_id': job_id, 'scheduler_attempt_id': attempt_id,
        'uuid': args.uuid, 'archive_sha256': receipt['archive_sha256'],
        'binary_sha256': receipt['binary_sha256'], 'binary_path': str(binary), 'sanitizer': 'none',
        'device_information': device, 'quiescence': observations, 'argv': argv,
        'returncode': result.returncode, 'suite': 'triage', 'target': 'cgs_study',
        'serial_telemetry_clean': None, 'success': bool(success),
    }
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2, sort_keys=True) + '\n')
    print(json.dumps(metrics, sort_keys=True))
    if not success:
        raise SystemExit(1)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)
    build_parser = sub.add_parser('build')
    build_parser.add_argument('--archive', type=Path, required=True)
    build_parser.add_argument('--archive-sha256', required=True)
    build_parser.add_argument('--binary', type=Path, required=True)
    build_parser.add_argument('--receipt', type=Path, required=True)
    build_parser.add_argument('--build', type=Path, required=True)
    run_parser = sub.add_parser('run')
    run_parser.add_argument('--build', type=Path, required=True)
    run_parser.add_argument('--config', type=Path, required=True)
    run_parser.add_argument('--uuid', required=True)
    run_parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'build':
        print(json.dumps(build(args), sort_keys=True))
    else:
        run(args)


if __name__ == '__main__':
    main()
