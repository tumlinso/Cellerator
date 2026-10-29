#!/usr/bin/env python3
"""Preserve generated payloads locally; leave compact evidence suitable for Git."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
PAYLOAD_SUFFIXES = {'.npz', '.pt', '.zip', '.so', '.o', '.fatbin', '.pyc'}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    destination = args.archive.resolve()
    if not destination.is_relative_to(Path('/mnt/block')):
        parser.error('the user requested local storage under /mnt/block')
    if destination.exists():
        parser.error('refusing to overwrite an existing archive')
    payloads = []
    for path in sorted(HERE.rglob('*')):
        if not path.is_file():
            continue
        with path.open('rb') as stream:
            is_elf = stream.read(4) == b'\x7fELF'
        if path.suffix in PAYLOAD_SUFFIXES or is_elf:
            payloads.append(path)
    entries = [{'path': str(path.relative_to(HERE)),
                'size': path.stat().st_size, 'sha256': digest(path)}
               for path in payloads]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as output:
        compressor = subprocess.Popen(['zstd', '-T2', '-3', '-c'],
                                      stdin=subprocess.PIPE, stdout=output)
        try:
            with tarfile.open(fileobj=compressor.stdin, mode='w|') as archive:
                for path in payloads:
                    archive.add(path, arcname=str(path.relative_to(HERE)), recursive=False)
        finally:
            compressor.stdin.close()
        if compressor.wait() != 0:
            raise RuntimeError('archive compression failed')
    # Verify every archived byte against the source hashes, without extracting
    # or modifying the original payloads.
    decompressor = subprocess.Popen(['zstd', '-d', '-c', str(destination)],
                                   stdout=subprocess.PIPE)
    expected = {row['path']: row for row in entries}
    seen = set()
    with tarfile.open(fileobj=decompressor.stdout, mode='r|') as archive:
        for member in archive:
            row = expected[member.name]
            result = hashlib.sha256()
            with archive.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1 << 20), b''):
                    result.update(block)
            if member.size != row['size'] or result.hexdigest() != row['sha256']:
                raise RuntimeError(f'archive verification failed: {member.name}')
            seen.add(member.name)
    decompressor.stdout.close()
    if decompressor.wait() != 0 or seen != set(expected):
        raise RuntimeError('archive inventory verification failed')
    manifest = {
        'schema': 'cellerator.adaptive_fields.local_artifact_archive.v1',
        'archive': str(destination), 'archive_sha256': digest(destination),
        'archive_bytes': destination.stat().st_size,
        'payload_bytes': sum(row['size'] for row in entries),
        'entry_count': len(entries), 'entries': entries,
        'verification': 'Every tar member size and SHA-256 matched its source inventory',
        'original_payloads_preserved': True,
        'restore': 'zstd -d -c ARCHIVE | tar -xf - -C EXPERIMENT_ROOT',
        'remote_payload_upload': False,
    }
    (HERE / 'LOCAL_ARTIFACTS.json').write_text(json.dumps(manifest, indent=2) + '\n')
    # These are experiment-local ignores, not changes to repository-wide rules.
    ignores = ['# Generated payloads are preserved in the verified /mnt/block archive.',
               '# See LOCAL_ARTIFACTS.json for exact paths, hashes, and restoration.',
               '__pycache__/', '*.npz', '*.pt', '*.zip']
    for row in entries:
        if Path(row['path']).suffix not in {'.npz', '.pt', '.zip', '.pyc'}:
            ignores.append('/' + row['path'])
    (HERE / '.gitignore').write_text('\n'.join(ignores) + '\n')
    print(json.dumps({key: manifest[key] for key in
                      ('archive', 'archive_sha256', 'archive_bytes',
                       'payload_bytes', 'entry_count', 'verification')}))


if __name__ == '__main__':
    main()
