#!/usr/bin/env python3
"""Preserve NF1 dirty bytes/diffs outside source. Never edits Git or Todo state."""
from __future__ import annotations
import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
sys.dont_write_bytecode = True
from common import canonical, external, git, host_context, require, sha, sha_bytes, write_new


def run_bytes(root: Path, *args: str) -> bytes:
    return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True).stdout


def catalog(root: Path) -> list[dict]:
    records = []
    for block in run_bytes(root, 'worktree', 'list', '--porcelain', '-z').split(b'\0\0'):
        item = {}
        for line in block.split(b'\0'):
            if line:
                key, _, value = line.partition(b' '); item[os.fsdecode(key)] = os.fsdecode(value)
        if 'worktree' in item: records.append(item)
    return records


def dirty_files(root: Path) -> list[str]:
    raw = run_bytes(root, 'diff', '--name-only', '-z', 'HEAD')
    raw += run_bytes(root, 'ls-files', '--others', '--exclude-standard', '-z')
    result = sorted({os.fsdecode(x) for x in raw.split(b'\0') if x})
    for name in result:
        p = Path(name)
        require(not p.is_absolute() and '..' not in p.parts, 'Unsafe Git path')
    return result


def material(path: Path) -> tuple[str, bytes]:
    if path.is_symlink(): return 'symlink', os.fsencode(os.readlink(path))
    if path.is_file(): return 'file', path.read_bytes()
    if not path.exists(): return 'deleted', b''
    raise ValueError('Dirty submodule/directory requires explicit preservation: ' + str(path))


def capture(output: Path) -> dict:
    _, roots, snapshots = host_context(); output = external(output, list(roots.values()))
    require(not output.exists(), 'Capture destination must be new')
    output.mkdir(parents=True)
    inventory = {'format': 'nf1a-preservation-v1', 'passed': False, 'projects': {},
                 'meaning': 'Byte preservation and evidence inventory, not integration or test acceptance.'}
    archive = output / 'dirty-source.tar'
    with tarfile.open(archive, 'x') as tar:
        for project, root in roots.items():
            all_trees = catalog(root); trees = []; prefix = 'refs/heads/codex/' + ('ce' if project == 'cellerator' else 'gh') + '-nf1-'
            for index, record in enumerate(all_trees):
                if not record.get('branch', '').startswith(prefix): continue
                path = Path(record['worktree']).resolve()
                require(git(path, 'rev-parse', '--git-common-dir'), 'Worktree Git identity missing')
                start_head = git(path, 'rev-parse', 'HEAD')
                start_status = run_bytes(path, 'status', '--porcelain=v1', '-z', '--untracked-files=all')
                files = []; contents = {}
                for number, name in enumerate(dirty_files(path)):
                    kind, data = material(path / name); member = f'{project}/{index}/files/{number}'
                    contents[name] = (kind, sha_bytes(data))
                    if kind != 'deleted':
                        info = tarfile.TarInfo(member); info.size = len(data); info.mode = 0o600
                        tar.addfile(info, io.BytesIO(data))
                    files.append({'path': name, 'kind': kind, 'sha256': sha_bytes(data),
                                  'archive_member': member if kind != 'deleted' else None})
                for label, args in [('staged.patch', ('diff', '--binary', '--cached', 'HEAD')),
                                    ('unstaged.patch', ('diff', '--binary'))]:
                    data = run_bytes(path, *args); member = f'{project}/{index}/{label}'
                    info = tarfile.TarInfo(member); info.size = len(data); info.mode = 0o600; tar.addfile(info, io.BytesIO(data))
                require(start_head == git(path, 'rev-parse', 'HEAD'), 'Worktree moved during capture')
                require(start_status == run_bytes(path, 'status', '--porcelain=v1', '-z', '--untracked-files=all'), 'Worktree status changed during capture')
                require(set(contents) == set(dirty_files(path)), 'Dirty paths changed during capture')
                for name, expected in contents.items():
                    kind, data = material(path / name)
                    require((kind, sha_bytes(data)) == expected, 'Dirty bytes changed during capture: ' + name)
                trees.append({'branch': record.get('branch'), 'worktree': str(path), 'head': start_head,
                              'status_sha256': sha_bytes(start_status), 'files': files})
            snap = snapshots[project]
            run_id = ('CE' if project == 'cellerator' else 'GH') + '-NF1-RUN-V1'
            ws = [r for r in snap.todo_tables.get('workflow_workspaces', []) if r.get('run_id') == run_id]
            wsids = {r['id'] for r in ws}
            patches = [r for r in snap.todo_tables.get('workflow_patch_artifacts', []) if r.get('workspace_id') in wsids]
            # Only source/artifact identifiers are persisted, never claim/session capabilities.
            keep = ('id','run_id','lane_id','workspace_id','branch','base_commit','artifact_ref','artifact_kind','state','content_hash','diff_hash','integration_task_id')
            inventory['projects'][project] = {'project_uuid': snap.project_uuid, 'revision': snap.todo_revision,
                'registered_root': str(root), 'all_git_worktrees': all_trees, 'captured_nf1_worktrees': trees,
                'workspace_records': [{k:r[k] for k in keep if k in r} for r in ws],
                'patch_records': [{k:r[k] for k in keep if k in r} for r in patches]}
    inventory.update(archive=str(archive), archive_sha256=sha(archive), passed=True)
    path = output / 'preservation.json'; write_new(path, inventory)
    return {'status': 'captured', 'receipt': str(path), 'sha256': sha(path),
            'worktrees': sum(len(p['captured_nf1_worktrees']) for p in inventory['projects'].values())}

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    try: print(json.dumps(capture(a.output),sort_keys=True))
    except Exception as exc:
        print('capture stopped; partial external files retained: '+str(exc),file=sys.stderr); raise SystemExit(1)
