#!/usr/bin/env python3
"""Copy a fixed source closure into the isolated nested Autobench workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


SOURCE_PREFIXES = (
    "cmake", "compat", "examples", "include", "modules", "profiles",
    "stdlib", "src", "tests", "third_party", "tools",
)
BENCH_SOURCE_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hh",
    ".hpp", ".hxx", ".cmake", ".txt",
}
EXACT_FILES = (
    "CMakeLists.txt",
    "bench/CMakeLists.txt",
    "bench/planner_targets.cmake",
    "bench/quantized_matrix_bench.cu",
    "bench/sparse_ops_bench.cu",
    "bench/gene_candidate_discovery_bench.cu",
    "bench/math/feature_major_candidate_compare_bench.cu",
    "bench/architecture_evidence/real_regime_bench.cu",
    "examples/native_cuda/factorized_interaction_training.cu",
    "planning/interaction-layout-proposal-2026-11-02/CONTRACT.md",
    "planning/interaction-layout-proposal-2026-11-02/PERFORMANCE_PLAN.md",
)
OWNED_SUFFIXES = {".cc", ".cu", ".json", ".md", ".py", ".toml", ".txt"}
OWNED_PREFIXES = (
    "bench/factorized_interactions",
    "tests/factorized_interactions",
    "tests/factorized_interactions_preparation",
    "bindings",
    "python/cellerator",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def closure_paths(source: Path, baseplane_root: Path) -> list[tuple[Path, str]]:
    selected: dict[str, Path] = {}
    for name in EXACT_FILES:
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(f"required closure file is missing: {name}")
        selected[name] = path
    for prefix in SOURCE_PREFIXES:
        for path in (source / prefix).rglob("*"):
            if path.is_symlink():
                raise RuntimeError(f"source tree contains symlink in closure: {path}")
            if path.is_file() and "__pycache__" not in path.parts and \
                    path.suffix != ".pyc":
                selected[path.relative_to(source).as_posix()] = path
    for prefix in OWNED_PREFIXES:
        root = source / prefix
        if not root.is_dir():
            raise FileNotFoundError(f"owned closure directory is missing: {prefix}")
        for path in root.rglob("*"):
            if path.is_symlink():
                raise RuntimeError(f"source tree contains symlink in closure: {path}")
            if path.is_file() and path.suffix in OWNED_SUFFIXES and \
                    "__pycache__" not in path.parts and path.suffix != ".pyc":
                selected[path.relative_to(source).as_posix()] = path

    bench_root = source / "bench"
    for path in bench_root.rglob("*"):
        if path.is_symlink():
            if path.name == "CMakeLists.txt" or path.suffix in BENCH_SOURCE_SUFFIXES:
                raise RuntimeError(f"benchmark source closure contains symlink: {path}")
            continue
        if path.is_file() and (path.name == "CMakeLists.txt" or
                               path.suffix in BENCH_SOURCE_SUFFIXES):
            selected[path.relative_to(source).as_posix()] = path

    if not (baseplane_root / "CMakeLists.txt").is_file():
        raise FileNotFoundError(
            f"Baseplane source root does not contain CMakeLists.txt: {baseplane_root}")
    baseplane_files = [baseplane_root / "CMakeLists.txt"]
    for prefix in ("cmake", "include", "src", "tests", "bench"):
        root = baseplane_root / prefix
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_symlink():
                raise RuntimeError(f"Baseplane source closure contains symlink: {path}")
            if path.is_file():
                baseplane_files.append(path)
    for path in baseplane_files:
        relative = path.relative_to(baseplane_root).as_posix()
        selected[f"external/Baseplane/{relative}"] = path
    return sorted(((path, relative) for relative, path in selected.items()),
                  key=lambda entry: entry[1])


def digest_manifest(entries: list[dict]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        relative = entry["path"].encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(int(entry["bytes"]).to_bytes(8, "little"))
        digest.update(bytes.fromhex(entry["sha256"]))
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--baseplane-root", type=Path, required=True,
                        help="read-only Baseplane dependency source tree")
    parser.add_argument("--workspace", type=Path, required=True,
                        help="isolated nested Autobench workspace")
    parser.add_argument("--label", default="source",
                        help="versioned immutable source directory label")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", args.label):
        parser.error("--label must be a simple versioned directory name")
    source = args.source.resolve(strict=True)
    baseplane_root = args.baseplane_root.resolve(strict=True)
    workspace = args.workspace.resolve(strict=True)
    data_root = workspace / ".autobench" / "data" / "factorized-interactions"
    staged_root = data_root / args.label
    manifest_path = data_root / f"{args.label}-manifest.json"
    if staged_root.exists() or manifest_path.exists():
        raise FileExistsError(
            "factorized-interactions source staging already exists; preserve it and "
            "create a new versioned staging directory instead of overwriting")

    paths = closure_paths(source, baseplane_root)
    entries = []
    for path, relative in paths:
        entries.append({"path": relative, "bytes": path.stat().st_size,
                        "sha256": sha256_file(path)})
    source_hash = digest_manifest(entries)

    staged_root.mkdir(parents=True)
    for (path, _), entry in zip(paths, entries):
        destination = staged_root / entry["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        copied_hash = sha256_file(destination)
        current_hash = sha256_file(path)
        if copied_hash != entry["sha256"] or current_hash != entry["sha256"]:
            raise RuntimeError(f"source changed while staging: {entry['path']}")
        destination.chmod(0o444)

    manifest = {
        "schema": "cellerator.factorized_interactions.source_closure.v1",
        "source_closure_sha256": source_hash,
        "source_root_label": "Cellerator checkout; copied without symlinks",
        "file_count": len(entries),
        "files": entries,
    }
    data_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n",
                             encoding="utf-8")
    manifest_path.chmod(0o444)
    for directory in sorted((path for path in staged_root.rglob("*") if path.is_dir()),
                            key=lambda path: len(path.parts), reverse=True):
        directory.chmod(0o555)
    staged_root.chmod(0o555)
    print(json.dumps({
        "source_closure": str(staged_root),
        "source_manifest": str(manifest_path),
        "source_closure_sha256": source_hash,
        "file_count": len(entries),
        "bytes": sum(entry["bytes"] for entry in entries),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
