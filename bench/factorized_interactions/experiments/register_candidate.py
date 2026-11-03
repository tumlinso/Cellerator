#!/usr/bin/env python3
"""Checked wrapper around KDA candidate registration for full source trees."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from common import (
    AdapterError,
    candidate_tree_identity,
    load_variant_map,
)


DEFAULT_KDA_ROOT = Path("/home/tumlinson/kda")
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


def _checked_registration(*, task: Path, workspace: Path, candidate_id: str,
                          source_dir: Path, manifest_path: Path,
                          variant_map_path: Path, hypothesis: str,
                          change: str, baseline: bool,
                          kda_root: Path) -> dict[str, Any]:
    task = task.resolve(strict=True)
    source_dir = source_dir.resolve(strict=True)
    manifest_path = manifest_path.resolve(strict=True)
    variant_map_path = variant_map_path.resolve(strict=True)
    kda_root = kda_root.resolve(strict=True)
    if not source_dir.is_dir() or source_dir.is_symlink():
        raise AdapterError("candidate source must be a regular full source directory")
    variants = load_variant_map(variant_map_path)
    if candidate_id not in variants["variants"]:
        raise AdapterError(f"candidate {candidate_id!r} is outside this task's exact variant map")
    variant = variants["variants"][candidate_id]
    if variant.get("enabled") is not True:
        raise AdapterError(f"candidate {candidate_id!r} is not integrated and enabled")
    if not hypothesis.strip() or not change.strip():
        raise AdapterError("candidate hypothesis and change must be explicit")

    sys.path.insert(0, str(kda_root))
    try:
        from kda_experiments.store import read_candidate_manifest
    except ImportError as exc:
        raise AdapterError(f"cannot import KDA source-manifest validator from {kda_root}") from exc
    parsed = read_candidate_manifest(manifest_path, source_dir)
    files = set(parsed["files"])
    missing = REQUIRED_PROJECT_FILES - files
    if missing:
        raise AdapterError(f"candidate manifest is not a complete native project: {sorted(missing)}")
    tree_hash, tree_entries = candidate_tree_identity(source_dir)
    tree_by_path = {row["path"]: row["sha256"] for row in tree_entries}
    if set(tree_by_path) != files:
        raise AdapterError("candidate tree and KDA manifest file set differ")
    for row in tree_entries:
        if tree_by_path[row["path"]] != row["sha256"]:
            raise AdapterError(f"candidate file changed during source validation: {row['path']}")

    argv = [
        sys.executable, "-m", "kda_experiments",
        "--task", str(task), "--workspace", str(workspace.resolve()),
        "register", candidate_id, str(source_dir),
        "--manifest", str(manifest_path),
        "--hypothesis", hypothesis, "--change", change,
    ]
    if baseline:
        if candidate_id != variants["comparison"]["incumbent"]:
            raise AdapterError("only this task's declared incumbent may be registered as baseline")
        argv.append("--baseline")
    elif candidate_id == variants["comparison"]["incumbent"]:
        raise AdapterError("the declared incumbent must be registered with --baseline")
    experiment_group = variants["comparison"].get("experiment_group")
    if experiment_group:
        argv.extend(["--experiment-group", experiment_group])
    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(kda_root) + (os.pathsep + old_pythonpath if old_pythonpath else "")
    result = subprocess.run(argv, cwd=task.parent, env=env, text=True,
                            capture_output=True, check=False)
    if result.returncode != 0:
        raise AdapterError(f"KDA registration failed: {result.stderr[-3000:]}")
    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise AdapterError("KDA registration did not emit JSON") from exc
    if response.get("status") != "ok":
        raise AdapterError(f"KDA registration returned {response.get('status')!r}")
    return {
        "candidate_id": candidate_id,
        "status": "registered",
        "source_tree_sha256": tree_hash,
        "kda_manifest_sha256": parsed["manifest_sha256"],
        "kda_response": response,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--variant-map", type=Path, required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--change", required=True)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--kda-root", type=Path, default=DEFAULT_KDA_ROOT)
    args = parser.parse_args(argv)
    try:
        response = _checked_registration(
            task=args.task, workspace=args.workspace,
            candidate_id=args.candidate_id, source_dir=args.source,
            manifest_path=args.manifest, variant_map_path=args.variant_map,
            hypothesis=args.hypothesis, change=args.change,
            baseline=args.baseline, kda_root=args.kda_root,
        )
    except Exception as exc:
        parser.error(f"candidate registration refused: {exc}")
    print(json.dumps(response, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
