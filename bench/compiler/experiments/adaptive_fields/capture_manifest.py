#!/usr/bin/env python3
"""Capture experiment source identity without querying or executing a GPU."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def command(*args: str) -> str:
    return subprocess.run(args, cwd=ROOT, check=True, capture_output=True,
                          text=True).stdout.strip()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture() -> dict:
    source = {}
    for path in sorted(HERE.rglob('*')):
        relative = path.relative_to(HERE)
        if not path.is_file() or any(part in {'evidence', 'build', '__pycache__',
                                              'generated', 'cache', 'artifacts',
                                              'results', 'raw', 'workspace'}
                                     for part in relative.parts):
            continue
        if path.suffix in {'.py', '.cc', '.cpp', '.hh', '.h', '.cu', '.cuh',
                            '.md', '.json', '.toml', '.sh'}:
            source[str(path.relative_to(ROOT))] = digest(path)
    skills = [Path('/home/tumlinson/.agents/skills/cuda/SKILL.md'),
              Path('/home/tumlinson/.agents/skills/cpp-context-compiler/SKILL.md'),
              Path('/home/tumlinson/kda/docs/agent-entry.md')]
    return {
        'schema_version': 1,
        'captured_at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'repository': str(ROOT), 'git_head': command('git', 'rev-parse', 'HEAD'),
        'git_status': command('git', 'status', '--short'),
        'tracked_diff_sha256': hashlib.sha256(
            command('git', 'diff', '--binary', 'HEAD').encode()).hexdigest(),
        'python': platform.python_version(), 'platform': platform.platform(),
        'source_sha256': source,
        'consulted_guidance_sha256': {str(p): digest(p) for p in skills},
        'authority': {'project_control_next_task': 'no_work; no handle',
                      'scope': 'explicit user-authorized isolated experiments',
                      'gpu_admission': 'separate Device Scheduler receipts'},
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--snapshot-output', type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if HERE not in output.parents:
        parser.error('output must stay inside the adaptive_fields experiment')
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = capture()
    if args.snapshot_output:
        snapshot = args.snapshot_output.resolve()
        if HERE not in snapshot.parents:
            parser.error('source snapshot must stay inside the experiment')
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(snapshot, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
            for relative in manifest['source_sha256']:
                bundle.write(ROOT / relative, relative)
        manifest['experiment_source_snapshot'] = str(snapshot)
        manifest['experiment_source_snapshot_sha256'] = digest(snapshot)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
