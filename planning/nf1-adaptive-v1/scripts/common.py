"""Small package utilities. This module has no workflow mutation operations."""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

PACKAGE = Path(__file__).resolve().parents[1]
SUCCESS = {'implemented', 'validated', 'no_change_required', 'evaluated_not_promoted'}
TERMINAL = {'done', 'superseded', 'cancelled', 'stale'}

def require(ok: object, message: str) -> None:
    if not ok:
        raise ValueError(message)

def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\n').encode()

def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

def sha(path: Path | str) -> str:
    return sha_bytes(Path(path).read_bytes())

def load(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_new(path: Path, value: Any, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode)
    with os.fdopen(fd, 'wb') as out:
        out.write(canonical(value)); out.flush(); os.fsync(out.fileno())

def external(path: Path, roots: list[Path]) -> Path:
    resolved = path.expanduser().resolve()
    require(not any(resolved == r.resolve() or resolved.is_relative_to(r.resolve()) for r in roots),
            'Evidence/review output must be outside both source repositories')
    return resolved

def verify_manifest(package: Path = PACKAGE) -> str:
    manifest = package / 'MANIFEST.sha256'
    entries: dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        digest, sep, name = line.partition('  ')
        p = Path(name)
        require(sep and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest), 'Malformed manifest digest')
        require(name not in entries and not p.is_absolute() and '..' not in p.parts, 'Unsafe/duplicate manifest path')
        target = package / p
        require(target.is_file() and not target.is_symlink(), 'Manifest target missing or symlinked: ' + name)
        require(target.resolve().is_relative_to(package.resolve()), 'Manifest target escapes package')
        require(sha(target) == digest, 'Manifest mismatch: ' + name)
        entries[name] = digest
    actual = {p.relative_to(package).as_posix() for p in package.rglob('*') if p.is_file() and p.name != 'MANIFEST.sha256'}
    require(actual == set(entries), 'Unsealed or missing package files: ' + repr(sorted(actual ^ set(entries))))
    return sha(manifest)

def git(repo: Path, *args: str) -> str:
    r = subprocess.run(['git', '-C', str(repo), *args], check=True, text=True, capture_output=True)
    return r.stdout.strip()

def clean_source(repo: Path) -> dict[str, Any]:
    """Require committed source; only known generated Todo projections may differ.

    The exclusions are returned explicitly. They never include code, build files,
    arbitrary state directories, or an operator-supplied ignore glob.
    """
    head = git(repo, 'rev-parse', 'HEAD')
    raw = subprocess.run(['git', '-C', str(repo), 'status', '--porcelain=v1', '-z', '--untracked-files=all'],
                         check=True, capture_output=True).stdout
    entries = raw.split(b'\0'); ignored = []; nonclean = []; i = 0
    while i < len(entries):
        item = entries[i]; i += 1
        if not item:
            continue
        status = item[:2].decode(); name = os.fsdecode(item[3:])
        names = [name]
        if 'R' in status or 'C' in status:
            require(i < len(entries), 'Truncated Git rename status')
            names.append(os.fsdecode(entries[i])); i += 1
        def generated(n: str) -> bool:
            return n in {'todos.md', 'todo-status.md', '.todo-orchestrator/state.snapshot.json'} or (n.startswith('todos/') and n.endswith('.md'))
        if all(generated(n) for n in names):
            ignored.extend(names)
        else:
            nonclean.extend(names)
    require(not nonclean, 'Commit or preserve source before publishing a gate: ' + repr(nonclean))
    return {'head': head, 'tree': git(repo, 'rev-parse', 'HEAD^{tree}'), 'excluded_generated_projection_changes': sorted(ignored)}

def host_context() -> tuple[Any, dict[str, Path], dict[str, Any]]:
    """Use the installed Project Control read boundary, never SQLite directly."""
    from project_control.config import load_config
    from project_control.registry import WorkspaceRegistry
    from project_control.mutation import build_mutation_snapshot
    cfg = load_config(); registry = WorkspaceRegistry(cfg)
    program = load(PACKAGE / 'machine/program.json')
    roots = {}; snapshots = {}
    for name, info in program['projects'].items():
        roots[name] = registry.repository(name).root
        snap = build_mutation_snapshot(cfg, name)
        require(snap.project_uuid == info['uuid'], 'Wrong Todo authority for ' + name)
        require(isinstance(snap.todo_revision, int) and not isinstance(snap.todo_revision, bool), 'Missing authority revision')
        require('tasks' in snap.todo_tables and 'workflow_runs' in snap.todo_tables,
                'Incomplete native read coverage for ' + name)
        require(snap.todo_semantic and snap.todo_workflow, 'Semantic workflow read unavailable for ' + name)
        snapshots[name] = snap
    return cfg, roots, snapshots

def rows(snapshot: Any, table: str) -> dict[str, dict[str, Any]]:
    require(table in snapshot.todo_tables, 'Missing authoritative table: ' + table)
    return {str(r['id']): r for r in snapshot.todo_tables[table] if r.get('id') is not None}

def successful(row: dict[str, Any]) -> bool:
    return row.get('status') == 'done' and row.get('result') in SUCCESS

def completed_prerequisites(outcome: dict[str, Any], snapshots: dict[str, Any]) -> list[dict[str, Any]]:
    observed = []
    for tid in outcome['completion_requires']:
        project = 'cellerator' if tid.startswith('CE-') else 'glasshelix'
        task = rows(snapshots[project], 'tasks').get(tid, {})
        require(successful(task), 'Completion prerequisite not successfully finished: ' + tid)
        observed.append({'project': project, 'task_id': tid, 'status': task['status'], 'result': task['result'],
                         'completion_commit': task.get('completion_commit'), 'completion_revision': task.get('completion_revision'),
                         'observed_revision': snapshots[project].todo_revision})
    return observed
