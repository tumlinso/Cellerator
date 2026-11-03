#!/usr/bin/env python3
"""Materialize verified native source and previously compiled bytes; never compile."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def unpack(archive_path, entries, destination):
    names = [entry["path"] for entry in entries]
    if names != sorted(names) or len(names) != len(set(names)):
        raise ValueError("archive entries must be unique and sorted")
    with tarfile.open(archive_path, "r:") as archive:
        members = archive.getmembers()
        if [member.name for member in members] != names:
            raise ValueError("archive members differ from recorded manifest")
        for member, entry in zip(members, entries):
            rel = PurePosixPath(member.name)
            if rel.is_absolute() or ".." in rel.parts or rel.as_posix() != member.name \
                    or not member.isfile() or member.size != entry["bytes"]:
                raise ValueError(f"invalid archive member {member.name!r}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("unreadable archive member")
            payload = stream.read()
            if len(payload) != entry["bytes"] or hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                raise ValueError(f"archive member digest mismatch: {member.name}")
            path = destination.joinpath(*rel.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            path.chmod(0o555 if entry.get("executable") else 0o444)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.source_manifest.read_text())
    bundle = json.loads(args.bundle_manifest.read_text())
    if manifest.get("schema") != "cellerator.factorized_interactions.archived_source_closure.v1" or \
            bundle.get("schema") != "cellerator.factorized_interactions.prebuilt_bundle.v1":
        raise ValueError("unknown input manifest schema")
    if digest(args.source_archive) != bundle["source_archive_sha256"] or \
            digest(args.source_manifest) != bundle["source_manifest_sha256"] or \
            digest(args.bundle) != bundle["bundle_sha256"] or \
            manifest["source_closure_sha256"] != bundle["source_closure_sha256"]:
        raise ValueError("prebuilt bundle input identities differ")
    build = args.build.resolve()
    build.mkdir(parents=True, exist_ok=True)
    source = build / "source"
    native = build / "native_candidate" / "native_build"
    if source.exists() or native.exists():
        raise FileExistsError("prebuilt materialization target already exists")
    source.mkdir()
    native.mkdir(parents=True)
    unpack(args.source_archive, manifest["files"], source)
    unpack(args.bundle, bundle["files"], native)
    for name in ("factorized_interactions_build_manifest.json",
                 "factorized_interactions_baseline_build_manifest.json"):
        recorded = json.loads((native / name).read_text())
        if recorded["source_closure_sha256"] != bundle["source_closure_sha256"]:
            raise ValueError("native build manifest source closure differs")
        for item in recorded["binaries"].values():
            path = native / item["path"]
            if not path.is_file() or path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
                raise ValueError("native binary differs from original build manifest")
    for root in (source, native):
        for directory in sorted((p for p in root.rglob("*") if p.is_dir()),
                                key=lambda p: len(p.parts), reverse=True):
            directory.chmod(0o555)
        root.chmod(0o555)
    (build / "prebuilt-provenance.json").write_text(json.dumps(bundle, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": "verified_prebuilt_copy", "compiled": False,
                      "source_closure_sha256": bundle["source_closure_sha256"],
                      "original_build_fingerprint": bundle["original_build_fingerprint"]}, sort_keys=True))


if __name__ == "__main__":
    main()
