#!/usr/bin/env python3
"""Freeze the Cellerator scale-build source closure as a deterministic archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import tarfile
from pathlib import Path


SOURCE_PATTERNS = (
    "CMakeLists.txt",
    "cmake/**",
    "include/**",
    "src/**",
    "modules/**",
    "tools/**",
    "compat/**",
    "examples/CMakeLists.txt",
    "examples/biology/CMakeLists.txt",
    "examples/biology/biological_relations.cc",
    "bench/native_data/**",
    "bench/CMakeLists.txt",
    "bench/planner_targets.cmake",
    "bench/*.cc",
    "bench/*.cpp",
    "bench/*.cu",
    "bench/*.cuh",
    "bench/**/*.c",
    "bench/**/*.cc",
    "bench/**/*.cpp",
    "bench/**/*.cu",
    "bench/**/*.h",
    "bench/**/*.hh",
    "bench/**/*.hpp",
    "bench/**/*.cuh",
    "bench/**/*.cmake",
    "bench/**/CMakeLists.txt",
    "bench/quantized_matrix_bench.cu",
    "bench/sparse_ops_bench.cu",
    "bench/gene_candidate_discovery_bench.cu",
    "bench/math/feature_major_candidate_compare_bench.cu",
    "bench/architecture_evidence/real_regime_bench.cu",
    "tests/execution_dependency/gated_product_fixture.hh",
    "tests/native_data/reference/**",
    "tests/**",
)
INTERNAL_MANIFEST = "SOURCE-MANIFEST.json"


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_fingerprint(files: dict[str, dict[str, object]]) -> str:
    material = "\n".join(
        f"{path}\0{files[path]['sha256']}" for path in sorted(files)
    ).encode()
    return digest_bytes(material)


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve(strict=True)
    archive_path = args.archive.resolve()
    manifest_path = args.manifest.resolve()
    for output in (archive_path, manifest_path):
        if output.is_relative_to(source) and ".autobench" not in output.relative_to(source).parts:
            raise SystemExit("archive and manifest outputs must stay outside selected source paths")

    from autobench.snapshot import _files

    paths = _files(source, SOURCE_PATTERNS)
    if not paths:
        raise SystemExit("the declared Cellerator build source closure is empty")
    before = {}
    source_files: dict[str, dict[str, object]] = {}
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise SystemExit(f"source closure contains an unsupported path: {path}")
        relative = path.relative_to(source).as_posix()
        info = path.stat()
        before[relative] = (info.st_size, info.st_mtime_ns)
        source_files[relative] = {
            "sha256": digest_file(path),
            "size_bytes": info.st_size,
            "mode": stat.S_IMODE(info.st_mode),
        }

    manifest = {
        "schema": "cellerator.native_data.source_closure.v1",
        "fingerprint": manifest_fingerprint(source_files),
        "file_count": len(source_files),
        "files": source_files,
    }
    manifest_bytes = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode() + b"\n"

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_archive = archive_path.with_name(archive_path.name + ".tmp")
    with tarfile.open(temporary_archive, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path in paths:
            relative = path.relative_to(source).as_posix()
            entry = tarfile.TarInfo(relative)
            entry.size = int(source_files[relative]["size_bytes"])
            entry.mode = int(source_files[relative]["mode"])
            entry.uid = entry.gid = 0
            entry.uname = entry.gname = ""
            entry.mtime = 0
            entry.pax_headers = {}
            with path.open("rb") as stream:
                archive.addfile(entry, stream)
        manifest_entry = tarfile.TarInfo(INTERNAL_MANIFEST)
        manifest_entry.size = len(manifest_bytes)
        manifest_entry.mode = 0o644
        manifest_entry.uid = manifest_entry.gid = 0
        manifest_entry.uname = manifest_entry.gname = ""
        manifest_entry.mtime = 0
        manifest_entry.pax_headers = {}
        import io
        archive.addfile(manifest_entry, io.BytesIO(manifest_bytes))

    after_paths = _files(source, SOURCE_PATTERNS)
    after_names = [path.relative_to(source).as_posix() for path in after_paths]
    if after_names != [path.relative_to(source).as_posix() for path in paths]:
        temporary_archive.unlink(missing_ok=True)
        raise SystemExit("source file set changed while archiving; retry after source settles")
    after = {
        path.relative_to(source).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in after_paths
    }
    if before != after:
        temporary_archive.unlink(missing_ok=True)
        raise SystemExit("source file metadata changed while archiving")
    after_manifest = {
        path.relative_to(source).as_posix(): {
            "sha256": digest_file(path),
            "size_bytes": path.stat().st_size,
            "mode": stat.S_IMODE(path.stat().st_mode),
        }
        for path in after_paths
    }
    if after_manifest != source_files:
        temporary_archive.unlink(missing_ok=True)
        raise SystemExit("source file bytes changed while archiving")

    atomic_write(manifest_path, manifest_bytes)
    temporary_archive.replace(archive_path)
    print(json.dumps({
        "source_closure_fingerprint": manifest["fingerprint"],
        "file_count": manifest["file_count"],
        "archive": str(archive_path),
        "archive_sha256": digest_file(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "manifest": str(manifest_path),
        "manifest_sha256": digest_bytes(manifest_bytes),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
