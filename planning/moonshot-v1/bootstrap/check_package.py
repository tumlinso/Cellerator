#!/usr/bin/env python3
"""Verify this archive's content hashes. No repository or Todo mutations."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path

def verify(root: Path) -> int:
    manifest=json.loads((root/'MANIFEST.json').read_text())
    if manifest.get('schema_version')!=1: raise ValueError('unsupported manifest')
    failures=[]
    for name,expected in manifest['files'].items():
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts: raise ValueError('unsafe manifest path')
        path=root/rel
        if path.is_symlink() or not path.is_file(): failures.append(name+': missing or symlink');continue
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected: failures.append(name+': hash mismatch')
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='MANIFEST.json'}
    extra=actual-set(manifest['files'])
    failures.extend(x+': unexpected file' for x in sorted(extra))
    if failures: raise ValueError('\n'.join(failures))
    return len(manifest['files'])

if __name__=='__main__':
    try: print(f'Package integrity passed: {verify(Path(__file__).resolve().parents[1])} files')
    except (ValueError,OSError,KeyError,json.JSONDecodeError) as e:
        print(f'Package integrity failed: {e}',file=sys.stderr);sys.exit(1)
