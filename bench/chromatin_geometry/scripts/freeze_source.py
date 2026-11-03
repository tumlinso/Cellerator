#!/usr/bin/env python3
"""Create an immutable, self-contained study source closure, without symlinks."""
import argparse
import gzip
import hashlib
import io
import json
import os
import stat
import tarfile
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
CELL = REPO / 'components/CellShard'
KDA = Path('/home/tumlinson/kda')
SKIP = {'.git', '.autobench', '.project-control-as1', '.campaign-lanes', '.campaign-lanes-20261103',
        'build', 'runs', 'artifacts', '__pycache__', '.pytest_cache', 'cache'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def collect(root, rel, prefix, files):
    base = root / rel
    if not base.exists():
        raise RuntimeError(f'missing closure input: {base}')
    paths = [base] if base.is_file() else sorted(base.rglob('*'))
    for path in paths:
        relative = path.relative_to(root)
        if set(relative.parts) & SKIP:
            continue
        # Analysis is an output, not a build dependency. It can be written by
        # concurrent measurement owners while the immutable source is captured.
        if root == REPO and relative.parts[:3] == ('bench', 'chromatin_geometry', 'analysis'):
            continue
        if path.is_symlink():
            raise RuntimeError(f'symlink forbidden in source closure: {path}')
        if path.is_file():
            name = (Path(prefix) / relative).as_posix()
            if name in files:
                raise RuntimeError(f'duplicate archive member {name}')
            files[name] = path


def validate_contract(contract, entries):
    """Require every frozen input to occur with its exact hash in the closure."""
    value = json.loads(contract.read_text())
    if value.get('format') != 'cgs.contract-freeze.v1' or not isinstance(value.get('source_files'), list):
        raise RuntimeError('invalid frozen contract source inventory')
    cache_prefix = 'bench/chromatin_geometry/data/cache/'
    for item in value['source_files']:
        path, expected = item.get('path'), item.get('sha256')
        if not isinstance(path, str) or not isinstance(expected, str):
            raise RuntimeError('invalid frozen contract source identity')
        member = ('replay/' + path[len(cache_prefix):] if path.startswith(cache_prefix)
                  else 'Cellerator/' + path)
        if entries.get(member, {}).get('sha256') != expected:
            raise RuntimeError(f'frozen contract source absent or hash mismatch: {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--replay', type=Path, action='append', default=[])
    parser.add_argument('--ste-only', action='store_true', help='capture the complete standalone STE check closure')
    parser.add_argument('--contract-freeze', type=Path, help='additional approved frozen contract under Cellerator')
    args = parser.parse_args()
    files = {}
    if args.ste_only:
        sources = ('include/Cellerator/ops/mechanism/gated_product.hh',
                   'bench/chromatin_geometry/contract.hh',
                   'bench/chromatin_geometry/learning/ste.hh',
                   'bench/chromatin_geometry/learning/ste.cu',
                   'bench/chromatin_geometry/learning/ste_chain.hh',
                   'bench/chromatin_geometry/learning/ste_chain.cu',
                   'bench/chromatin_geometry/learning/ste_tests.cu',
                   'bench/chromatin_geometry/learning/ste_gpu_test_main.cc',
                   'bench/chromatin_geometry/docs/draft.md',
                   'bench/chromatin_geometry/scripts')
    else:
        sources = ('CMakeLists.txt', 'include', 'src', 'cmake', 'bench/chromatin_geometry',
                   'tests/native_data/gated_product_host_test.cc',
                   'tests/framework/cuda/grouped_linear_smoke.cu',
                   'planning/chromatin-geometry-study-2026-11-02/design-sources.json',
                   'planning/chromatin-geometry-study-2026-11-02/contract-freeze-v1.json',
                   'planning/chromatin-geometry-study-2026-11-02/protocol.md',
                   'planning/chromatin-geometry-study-2026-11-02/archive/freeze-contract-v1.cc',
                   'planning/chromatin-geometry-study-2026-11-02/freeze-contract.cc')
    if args.contract_freeze:
        contract = args.contract_freeze.resolve(strict=True)
        if not contract.is_relative_to(REPO) or args.contract_freeze.is_symlink():
            raise RuntimeError('frozen contract must be a regular file under Cellerator')
        relative = contract.relative_to(REPO).as_posix()
        if relative not in sources:
            sources = (*sources, relative)
        if contract.name.startswith('contract-freeze-grouping-probe-v'):
            sources = (*sources, 'planning/chromatin-geometry-study-2026-11-02/bounded-grouping-probe-v1.json',
                       'planning/chromatin-geometry-study-2026-11-02/contract-freeze-v3.json',
                       'planning/chromatin-geometry-study-2026-11-02/bounded-grouping-resume-2026-11-03.json')
    for name in sources:
        collect(REPO, name, 'Cellerator', files)
    collect(CELL, 'include', 'Cellerator/components/CellShard', files)
    collect(KDA, 'kda_experiments', 'KDA', files)
    for path in args.replay:
        resolved = path.resolve(strict=True)
        cache = (STUDY / 'data/cache').resolve(strict=True)
        if not resolved.is_relative_to(cache) or path.is_symlink() or not resolved.is_file():
            raise RuntimeError('replay inputs must be regular files inside study data/cache')
        name = 'replay/' + resolved.relative_to(cache).as_posix()
        if name in files:
            raise RuntimeError(f'duplicate replay member {name}')
        files[name] = resolved
    entries = {}
    for name, path in sorted(files.items()):
        st = path.stat()
        entries[name] = {'sha256': digest(path), 'size': st.st_size,
                         'mode': stat.S_IMODE(st.st_mode)}
    if args.contract_freeze:
        validate_contract(args.contract_freeze, entries)
    manifest = {'schema': 'cgs.source-closure.v1', 'files': entries}
    manifest_bytes = (json.dumps(manifest, sort_keys=True, separators=(',', ':')) + '\n').encode()
    fingerprint = hashlib.sha256(manifest_bytes).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f'{fingerprint}.tar.gz'
    tmp = out.with_suffix('.tmp')
    if out.exists():
        raise RuntimeError(f'refusing to overwrite existing closure {out}')
    try:
        with tmp.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=1, mtime=0, filename='') as gz, \
             tarfile.open(fileobj=gz, mode='w', format=tarfile.PAX_FORMAT) as tar:
            for name, path in sorted(files.items()):
                expected = entries[name]
                if digest(path) != expected['sha256']:
                    raise RuntimeError(f'input changed during capture: {path}')
                entry = tarfile.TarInfo(name)
                entry.size, entry.mode, entry.mtime = expected['size'], expected['mode'], 0
                entry.uid = entry.gid = 0
                with path.open('rb') as stream:
                    tar.addfile(entry, stream)
            entry = tarfile.TarInfo('SOURCE-MANIFEST.json')
            entry.size, entry.mode, entry.mtime = len(manifest_bytes), 0o644, 0
            tar.addfile(entry, io.BytesIO(manifest_bytes))
        for name, path in files.items():
            if digest(path) != entries[name]['sha256']:
                raise RuntimeError(f'input changed while capture was closing: {path}')
        # Detect a source added or removed during traversal, not just edits to
        # the files that happened to be present in the first directory walk.
        after = {}
        for name in sources:
            collect(REPO, name, 'Cellerator', after)
        collect(CELL, 'include', 'Cellerator/components/CellShard', after)
        collect(KDA, 'kda_experiments', 'KDA', after)
        if set(after) != {name for name in files if not name.startswith('replay/')}:
            raise RuntimeError('source membership changed while capturing; retry')
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    summary = {'schema': 'cgs.closure-receipt.v1', 'fingerprint': fingerprint,
               'archive': str(out), 'archive_sha256': digest(out), 'file_count': len(files),
               'archive_bytes': out.stat().st_size}
    print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()
