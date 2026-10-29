"""Shared source identity and generated-output rules for the Cellerator map."""
from __future__ import annotations
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

OUTPUT_NAMES = {
    "path_map.jsonl", "path_map.csv", "routing_rules.json", "manifest.json",
    "nontracked_inventory.json", "validation.json",
}

def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=repo)

def split0(raw: bytes) -> list[str]:
    return [item for item in raw.decode().split("\0") if item]

def generated_paths(repo: Path, output: Path) -> set[str]:
    try:
        directory = output.resolve().relative_to(repo.resolve())
    except ValueError:
        return set()
    return {str(directory / name) for name in OUTPUT_NAMES}

def worktree_file_identity(repo: Path, relative_path: str) -> tuple[str, str]:
    path = repo / relative_path
    try:
        info = path.lstat()
    except FileNotFoundError:
        return "missing", "missing"
    if stat.S_ISLNK(info.st_mode):
        mode = "120000"
        data = os.fsencode(os.readlink(path))
    elif stat.S_ISREG(info.st_mode):
        mode = "100755" if info.st_mode & 0o111 else "100644"
        data = path.read_bytes()
    else:
        mode = f"unsupported:{info.st_mode}"
        data = b""
    return mode, hashlib.sha256(data).hexdigest()

def snapshot(repo: Path, output: Path, generator: Path, validator: Path) -> dict:
    generated = generated_paths(repo, output)
    index_raw = git(repo, "ls-files", "-s", "-z")
    tracked = []
    for entry in split0(index_raw):
        metadata, path = entry.split("\t", 1)
        mode, oid, stage = metadata.split()
        tracked.append({"path": path, "mode": mode, "oid": oid, "stage": stage})
    tracked_source = [entry for entry in tracked if entry["path"] not in generated]
    untracked_all = split0(git(repo, "ls-files", "--others", "--exclude-standard", "-z"))
    untracked = [path for path in untracked_all if path not in generated]
    source_rows = []
    for entry in tracked_source:
        mode, content_hash = worktree_file_identity(repo, entry["path"])
        source_rows.append({
            "path": entry["path"], "index_mode": entry["mode"], "index_oid": entry["oid"],
            "worktree_mode": mode, "content_sha256": content_hash,
        })
    for path in untracked:
        mode, content_hash = worktree_file_identity(repo, path)
        source_rows.append({
            "path": path, "index_mode": "untracked", "index_oid": "untracked",
            "worktree_mode": mode, "content_sha256": content_hash,
        })
    index_material = [
        {"path": entry["path"], "mode": entry["mode"], "oid": entry["oid"], "stage": entry["stage"]}
        for entry in tracked_source
    ]
    index_hash = hashlib.sha256(json.dumps(index_material, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    worktree_hash = hashlib.sha256(json.dumps(source_rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    tool_hashes = {
        "generator": hashlib.sha256(generator.read_bytes()).hexdigest(),
        "validator": hashlib.sha256(validator.read_bytes()).hexdigest(),
        "shared_snapshot": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    identity_material = {"index_sha256": index_hash, "worktree_sha256": worktree_hash, "toolchain_sha256": tool_hashes}
    identity = hashlib.sha256(json.dumps(identity_material, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    status_rows = git(repo, "status", "--porcelain=v1", "-z", "--untracked-files=no").decode().split("\0")
    dirty = sorted({item[3:] for item in status_rows if item} - generated)
    return {
        "generated_paths": generated, "tracked": tracked, "tracked_source": tracked_source,
        "untracked_source": untracked, "source_rows": source_rows,
        "source_index_sha256": index_hash, "source_worktree_sha256": worktree_hash,
        "dirty_tracked_paths": dirty, "toolchain_sha256": tool_hashes,
        "source_identity": identity,
        "head": git(repo, "rev-parse", "HEAD").decode().strip(),
    }
