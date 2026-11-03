#!/usr/bin/env python3
"""Derive a KDA manifest from an already source-bound Cellerator closure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from common import AdapterError, candidate_tree_identity, digest_file, write_json


REQUIRED_PROJECT_FILES = {
    "CMakeLists.txt",
    "bench/factorized_interactions/CMakeLists.txt",
    "bench/factorized_interactions/build_runner.py",
    "bench/factorized_interactions/native_runner.cu",
    "include/Cellerator/modeling/cuda/factorized_interaction.hh",
    "src/modeling/cuda/factorized_interaction.cu",
    "tests/factorized_interactions/reference.py",
    "tests/factorized_interactions/workloads.py",
}


def make_manifest(source_root: Path, closure_manifest_path: Path,
                  dependencies: list[dict[str, str]]) -> dict[str, Any]:
    source_root = source_root.resolve(strict=True)
    closure = json.loads(closure_manifest_path.read_text(encoding="utf-8"))
    files = closure.get("files")
    if not isinstance(files, list) or not files:
        files = closure.get("source_files")
    if not isinstance(files, list) or not files:
        raise AdapterError("closure manifest must contain a nonempty files/source_files array")
    declared: list[str] = []
    for row in files:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str) or \
                not isinstance(row.get("sha256"), str):
            raise AdapterError("closure file entries must include path and SHA256")
        declared.append(row["path"])
    if len(declared) != len(set(declared)):
        raise AdapterError("source closure manifest contains duplicate paths")
    missing_required = REQUIRED_PROJECT_FILES - set(declared)
    if missing_required:
        raise AdapterError(f"source closure is not a complete factorized-interaction project: {sorted(missing_required)}")
    actual_hash, actual_entries = candidate_tree_identity(source_root)
    actual_by_path = {entry["path"]: entry["sha256"] for entry in actual_entries}
    declared_by_path = {row["path"]: row["sha256"] for row in files}
    if actual_by_path != declared_by_path:
        missing = sorted(set(actual_by_path) - set(declared_by_path))
        absent = sorted(set(declared_by_path) - set(actual_by_path))
        changed = sorted(path for path in set(actual_by_path) & set(declared_by_path)
                         if actual_by_path[path] != declared_by_path[path])
        raise AdapterError(f"source tree and closure differ: missing={missing}, absent={absent}, changed={changed}")
    closure_hash = closure.get("source_closure_sha256")
    if not isinstance(closure_hash, str) or len(closure_hash) != 64:
        # KDA's own snapshot hash is still exact and will be rechecked at
        # registration; do not substitute it for a missing upstream identity.
        raise AdapterError("closure manifest must name its source_closure_sha256")
    normalized_dependencies = list(dependencies)
    normalized_dependencies.append({
        "name": "cellerator-source-closure",
        "revision": f"sha256:{closure_hash}",
    })
    normalized_dependencies.append({
        "name": "cellerator-source-tree",
        "revision": f"sha256:{actual_hash}",
    })
    if len({item.get("name") for item in normalized_dependencies}) != len(normalized_dependencies):
        raise AdapterError("candidate dependency names must be unique")
    return {
        "schema_version": 1,
        "files": [entry["path"] for entry in actual_entries],
        "dependencies": normalized_dependencies,
        "provenance": {
            "source_closure_manifest_sha256": digest_file(closure_manifest_path),
            "source_closure_sha256": closure_hash,
            "kda_source_tree_sha256": actual_hash,
            "file_count": len(actual_entries),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--closure-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dependency", action="append", default=[],
                        help="additional immutable dependency as NAME=REVISION")
    args = parser.parse_args(argv)
    dependencies = []
    for item in args.dependency:
        if "=" not in item:
            parser.error("--dependency must use NAME=REVISION")
        name, revision = item.split("=", 1)
        if not name or not revision:
            parser.error("--dependency name and revision must be nonempty")
        dependencies.append({"name": name, "revision": revision})
    try:
        manifest = make_manifest(args.source, args.closure_manifest, dependencies)
        write_json(args.output.resolve(), manifest, exclusive=True)
    except Exception as exc:
        parser.error(f"cannot create candidate manifest: {exc}")
    print(json.dumps({"manifest": str(args.output.resolve()),
                      "file_count": len(manifest["files"]),
                      "source_tree_sha256": manifest["provenance"]["kda_source_tree_sha256"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
