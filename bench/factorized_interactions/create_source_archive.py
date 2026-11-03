#!/usr/bin/env python3
"""Create a deterministic single-file closure for Autobench data capture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile


sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_runner import source_manifest  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--label", default="source-closure-v2")
    args = parser.parse_args()

    source = args.source.resolve(strict=True)
    source_manifest_path = args.source_manifest.resolve(strict=True)
    data_root = args.data_root.resolve(strict=True)
    source_entries, source_hash = source_manifest(source)
    staged_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if (staged_manifest.get("schema") !=
            "cellerator.factorized_interactions.source_closure.v1" or
            staged_manifest.get("source_closure_sha256") != source_hash or
            staged_manifest.get("files") != source_entries):
        raise RuntimeError("copied source closure differs from its staged manifest")

    archive_path = data_root / f"{args.label}.tar"
    archive_manifest_path = data_root / f"{args.label}-manifest.json"
    stager_path = data_root / f"build-from-{args.label}.py"
    if archive_path.exists() or archive_manifest_path.exists() or stager_path.exists():
        raise FileExistsError(
            "archive outputs already exist; preserve them and use a new versioned label")

    with tarfile.open(archive_path, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for entry in source_entries:
            relative = entry["path"]
            path = source / relative
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"source closure member is not a regular file: {relative}")
            info = tarfile.TarInfo(relative)
            info.size = entry["bytes"]
            info.mode = 0o444
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            with path.open("rb") as stream:
                archive.addfile(info, stream)

    entry_by_path = {entry["path"]: entry for entry in source_entries}
    helper_name = "bench/factorized_interactions/build_from_archive.py"
    helper_source = source / helper_name
    if not helper_source.is_file():
        raise FileNotFoundError(f"archive helper is outside the frozen closure: {helper_name}")
    helper_entry = entry_by_path[helper_name]
    stager_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(helper_source, stager_path)
    if sha256_file(stager_path) != helper_entry["sha256"]:
        raise RuntimeError("build stager differs from the frozen source closure")
    stager_path.chmod(0o444)

    archive_manifest = {
        "schema": "cellerator.factorized_interactions.archived_source_closure.v1",
        "source_closure_sha256": source_hash,
        "file_count": len(source_entries),
        "source_bytes": sum(entry["bytes"] for entry in source_entries),
        "files": source_entries,
        "archive_path": archive_path.name,
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
        "build_stager_path": stager_path.name,
        "build_stager_sha256": sha256_file(stager_path),
    }
    archive_manifest_path.write_text(
        json.dumps(archive_manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    archive_manifest_path.chmod(0o444)
    archive_path.chmod(0o444)
    print(json.dumps({
        "source_closure_sha256": source_hash,
        "file_count": len(source_entries),
        "source_bytes": archive_manifest["source_bytes"],
        "archive": str(archive_path),
        "archive_bytes": archive_manifest["archive_bytes"],
        "archive_sha256": archive_manifest["archive_sha256"],
        "archive_manifest": str(archive_manifest_path),
        "build_stager": str(stager_path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
