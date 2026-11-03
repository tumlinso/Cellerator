#!/usr/bin/env python3
"""Verify and unpack one immutable source closure, then build its targets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_member_name(name: str) -> str:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or path.as_posix() != name:
        raise RuntimeError(f"unsafe source archive member name: {name!r}")
    return name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--configure-only", action="store_true",
                        help="verify configure dependencies without compiling targets")
    parser.add_argument("--native-only", action="store_true",
                        help="build the native candidate runner without Torch")
    args = parser.parse_args()
    if args.parallel < 1 or args.parallel > 32:
        parser.error("--parallel must be in 1..32")

    archive_path = args.archive.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != \
            "cellerator.factorized_interactions.archived_source_closure.v1":
        parser.error("source archive manifest has an unknown schema")
    if archive_path.name != manifest.get("archive_path"):
        parser.error("source archive path differs from its manifest")
    if manifest.get("build_stager_path") != Path(__file__).name or \
            manifest.get("build_stager_sha256") != sha256_file(Path(__file__).resolve(strict=True)):
        parser.error("build stager identity differs from its source archive manifest")
    if archive_path.stat().st_size != manifest.get("archive_bytes") or \
            sha256_file(archive_path) != manifest.get("archive_sha256"):
        parser.error("source archive bytes differ from the immutable manifest")

    entries = manifest.get("files")
    if not isinstance(entries, list) or len(entries) != manifest.get("file_count"):
        parser.error("source archive manifest has invalid file entries")
    names = [safe_member_name(entry.get("path", "")) for entry in entries]
    if names != sorted(names) or len(names) != len(set(names)):
        parser.error("source archive file entries are not unique and ordered")

    build = args.build.resolve()
    build.mkdir(parents=True, exist_ok=True)
    source = build / "source"
    source_manifest_path = build / "source-manifest.json"
    if source.exists() or source_manifest_path.exists():
        raise FileExistsError("build-local source extraction already exists")
    source.mkdir()

    with tarfile.open(archive_path, mode="r:") as archive:
        members = archive.getmembers()
        member_names = [safe_member_name(member.name) for member in members]
        if member_names != names:
            raise RuntimeError("source archive member list differs from its file manifest")
        for member, entry in zip(members, entries):
            if not member.isfile() or member.issym() or member.islnk():
                raise RuntimeError(f"source archive contains a nonregular member: {member.name}")
            if member.size != entry["bytes"]:
                raise RuntimeError(f"source archive size mismatch: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError(f"source archive member could not be read: {member.name}")
            destination = source.joinpath(*PurePosixPath(member.name).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            payload = stream.read()
            if len(payload) != entry["bytes"] or \
                    hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                raise RuntimeError(f"source archive content hash mismatch: {member.name}")
            destination.write_bytes(payload)
            destination.chmod(0o444)

    source_manifest = {
        "schema": "cellerator.factorized_interactions.source_closure.v1",
        "source_closure_sha256": manifest["source_closure_sha256"],
        "source_root_label": "verified source archive extracted by Autobench build",
        "file_count": len(entries),
        "files": entries,
    }
    source_manifest_path.write_text(
        json.dumps(source_manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    source_manifest_path.chmod(0o444)
    for directory in sorted((path for path in source.rglob("*") if path.is_dir()),
                            key=lambda path: len(path.parts), reverse=True):
        directory.chmod(0o555)
    source.chmod(0o555)

    factorized = source / "bench" / "factorized_interactions"
    build_script = (factorized / "candidate_integration" / "build_candidate_runner.py"
                    if args.native_only else factorized / "build_runner.py")
    build_output = build / "native_candidate" if args.native_only else build
    command = [sys.executable, str(build_script), "--source", str(source),
               "--build", str(build_output), "--parallel", str(args.parallel)]
    if args.configure_only:
        command.append("--configure-only")
    print(json.dumps({
        "source_closure_sha256": manifest["source_closure_sha256"],
        "archive_sha256": manifest["archive_sha256"],
        "file_count": len(entries),
        "build_argv": command,
    }, sort_keys=True), flush=True)
    subprocess.run(command, cwd=source, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
