#!/usr/bin/env python3
"""Bind KDA-prepared candidate artifacts to an Autobench managed build."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

KDA_PACKAGE_ROOT = Path("/home/tumlinson/kda")
if str(KDA_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(KDA_PACKAGE_ROOT))


METADATA_NAME = "kda-autobench-build.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_kda(source: Path, task_path: Path, workspace: Path, args: list[str]) -> dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONPATH"] = "/home/tumlinson/kda" + os.pathsep + env.get("PYTHONPATH", "")
    env["CELLERATOR_KDA_SOURCE_ROOT"] = str(source)
    command = [sys.executable, "-m", "kda_experiments", "--task", str(task_path),
               "--workspace", str(workspace), *args]
    result = subprocess.run(command, cwd=source, env=env, check=False, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.returncode:
        raise RuntimeError(f"KDA command failed ({result.returncode}): {' '.join(command)}")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("KDA command did not return one JSON document") from exc
    if value.get("status") != "ok":
        raise RuntimeError(f"KDA command returned status {value.get('status')!r}")
    return value


def _candidate(source: Path, task_path: Path, workspace: Path, candidate_id: str) -> dict[str, Any]:
    value = _run_kda(source, task_path, workspace, ["inspect", candidate_id])
    item = value.get("candidate")
    if not isinstance(item, dict) or item.get("candidate_id") != candidate_id:
        raise RuntimeError(f"KDA inspect returned no candidate {candidate_id!r}")
    return item


def verify_source_closure(source: Path, task_path: Path, base_manifest_path: Path) -> dict[str, str]:
    manifest = _load(base_manifest_path)
    root = Path(manifest["source_root"]).resolve(strict=True)
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("KDA source manifest has no file hashes")
    for relative, expected in files.items():
        if sha256(root / relative) != expected or sha256(source / relative) != expected:
            raise RuntimeError(f"source snapshot differs from KDA baseline: {relative}")

    task = _load(task_path)
    task_dir_relative = task_path.relative_to(source).parent
    trusted_files: dict[str, str] = {}
    for trusted in task.get("trusted_inputs", []):
        if trusted.get("role") == "toolchain":
            continue
        raw = Path(trusted["path"])
        relative = raw.relative_to(root) if raw.is_absolute() else task_dir_relative / raw
        actual_path = root / relative
        snapshot_path = source / relative
        expected = sha256(actual_path)
        if sha256(snapshot_path) != expected:
            raise RuntimeError(f"trusted KDA input differs from snapshot: {relative}")
        trusted_files[relative.as_posix()] = expected
    return trusted_files


def _safe_artifact(build_dir: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise RuntimeError(f"KDA artifact path is unsafe: {relative}")
    path = build_dir / rel
    if path.is_symlink() or not path.is_file() or not path.resolve(strict=True).is_relative_to(build_dir.resolve(strict=True)):
        raise RuntimeError(f"KDA artifact is missing or unsafe: {relative}")
    return path


def prepare_into_autobench(
    *, source: Path, task_path: Path, workspace: Path, autobench_build: Path,
    base_manifest_path: Path, family: str, candidate_ids: list[str],
) -> None:
    trusted_files = verify_source_closure(source, task_path, base_manifest_path)
    result = _run_kda(source, task_path, workspace,
                      ["prepare", *candidate_ids, "--jobs", "1"])
    results = result.get("results")
    if not isinstance(results, list) or {row.get("candidate_id") for row in results} != set(candidate_ids):
        raise RuntimeError("KDA prepare did not return every requested candidate")

    output = autobench_build.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite nonempty Autobench build output: {output}")
    artifact_root = output / "prepared"
    bound_candidates = []
    for build_result in results:
        candidate_id = build_result["candidate_id"]
        if build_result.get("status") != "ok":
            raise RuntimeError(f"KDA candidate preparation failed: {candidate_id}")
        candidate = _candidate(source, task_path, workspace, candidate_id)
        kda_dir = Path(build_result["directory"]).resolve(strict=True)
        kda_manifest_path = kda_dir / "build.json"
        kda_manifest = _load(kda_manifest_path)
        if (kda_manifest.get("status") != "ok"
                or kda_manifest.get("candidate_id") != candidate_id
                or kda_manifest.get("snapshot_sha256") != candidate.get("snapshot_sha256")
                or kda_manifest.get("build_fingerprint") != build_result.get("fingerprint")
                or sha256(kda_manifest_path) != build_result.get("manifest_sha256")):
            raise RuntimeError(f"KDA build manifest does not match prepared candidate {candidate_id}")
        build_artifacts = kda_manifest.get("artifacts")
        if not isinstance(build_artifacts, list) or not build_artifacts:
            raise RuntimeError(f"KDA build has no artifacts: {candidate_id}")
        copied_artifacts = []
        for artifact in build_artifacts:
            relative = artifact["path"]
            source_artifact = _safe_artifact(kda_dir, relative)
            digest = sha256(source_artifact)
            if digest != artifact.get("sha256"):
                raise RuntimeError(f"KDA prepared artifact hash mismatch: {candidate_id}/{relative}")
            destination_relative = Path("prepared") / candidate_id / relative
            destination = output / destination_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_artifact, destination)
            if sha256(destination) != digest:
                raise RuntimeError(f"copied Autobench artifact hash mismatch: {candidate_id}/{relative}")
            copied_artifacts.append({"kda_path": relative,
                                     "autobench_path": destination_relative.as_posix(),
                                     "sha256": digest})
        manifest_copy_rel = Path("prepared") / candidate_id / "kda-build.json"
        manifest_copy = output / manifest_copy_rel
        manifest_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(kda_manifest_path, manifest_copy)
        bound_candidates.append({
            "candidate_id": candidate_id,
            "candidate_snapshot_sha256": candidate["snapshot_sha256"],
            "candidate_dependencies": candidate.get("dependencies", []),
            "candidate_source_manifest": candidate.get("source_manifest", []),
            "kda_build_fingerprint": build_result["fingerprint"],
            "kda_build_directory": str(kda_dir),
            "kda_build_manifest_sha256": build_result["manifest_sha256"],
            "autobench_build_manifest": manifest_copy_rel.as_posix(),
            "artifacts": copied_artifacts,
        })

    metadata = {
        "schema_version": 1,
        "family": family,
        "source_snapshot": str(source.resolve(strict=True)),
        "task_path": task_path.relative_to(source).as_posix(),
        "task_sha256": sha256(task_path),
        "base_source_manifest": base_manifest_path.relative_to(source).as_posix(),
        "base_source_manifest_sha256": sha256(base_manifest_path),
        "base_source_files": _load(base_manifest_path)["files"],
        "trusted_input_files": trusted_files,
        "candidates": bound_candidates,
    }
    metadata_path = output / METADATA_NAME
    metadata_path.write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "family": family,
                      "candidates": [item["candidate_id"] for item in bound_candidates],
                      "artifact_count": sum(len(item["artifacts"]) for item in bound_candidates),
                      "metadata": str(metadata_path)}, sort_keys=True), flush=True)


def verify_autobench_build(
    *, source: Path, task_path: Path, workspace: Path, autobench_build: Path,
    base_manifest_path: Path, family: str, candidate_ids: list[str],
) -> None:
    trusted_files = verify_source_closure(source, task_path, base_manifest_path)
    output = autobench_build.resolve(strict=True)
    metadata_path = output / METADATA_NAME
    metadata = _load(metadata_path)
    if (metadata.get("schema_version") != 1 or metadata.get("family") != family
            or metadata.get("source_snapshot") != str(source.resolve(strict=True))
            or metadata.get("task_path") != task_path.relative_to(source).as_posix()
            or metadata.get("task_sha256") != sha256(task_path)
            or metadata.get("base_source_manifest") != base_manifest_path.relative_to(source).as_posix()
            or metadata.get("base_source_manifest_sha256") != sha256(base_manifest_path)
            or metadata.get("base_source_files") != _load(base_manifest_path)["files"]
            or metadata.get("trusted_input_files") != trusted_files):
        raise RuntimeError("Autobench build metadata does not bind the current task/source snapshot")

    bound = {item.get("candidate_id"): item for item in metadata.get("candidates", [])}
    if set(bound) != set(candidate_ids):
        raise RuntimeError("Autobench build does not contain exactly the expected KDA candidates")
    expected_local_files = {METADATA_NAME}
    from kda_experiments.contracts import load_task, trusted_input_identities
    from kda_experiments.engine import (
        _build_fingerprint, _command_identities, _find_cached_build, package_fingerprint,
    )
    task, resolved_task = load_task(task_path)
    current_task_hash = sha256(resolved_task)
    trusted = trusted_input_identities(task)
    command_ids = _command_identities(task, resolved_task.parent, workspace)
    engine_fingerprint = package_fingerprint()

    for candidate_id in candidate_ids:
        candidate = _candidate(source, task_path, workspace, candidate_id)
        item = bound[candidate_id]
        if (item.get("candidate_snapshot_sha256") != candidate.get("snapshot_sha256")
                or item.get("candidate_dependencies") != candidate.get("dependencies", [])
                or item.get("candidate_source_manifest") != candidate.get("source_manifest", [])):
            raise RuntimeError(f"KDA candidate source identity changed: {candidate_id}")
        kda_dir = Path(item["kda_build_directory"]).resolve(strict=True)
        kda_manifest_path = kda_dir / "build.json"
        kda_manifest = _load(kda_manifest_path)
        if sha256(kda_manifest_path) != item.get("kda_build_manifest_sha256"):
            raise RuntimeError(f"prepared KDA build manifest changed: {candidate_id}")
        current_fingerprint = _build_fingerprint(
            task, candidate, resolved_task.parent, command_ids, trusted,
            engine_fingerprint, kda_manifest["compiler_target_probe"], current_task_hash,
        )
        if (current_fingerprint != item.get("kda_build_fingerprint")
                or kda_manifest.get("build_fingerprint") != current_fingerprint):
            raise RuntimeError(f"prepared KDA build is stale for current inputs: {candidate_id}")
        selected = _find_cached_build(workspace, candidate, current_fingerprint)
        if (selected is None or Path(selected["directory"]).resolve(strict=True) != kda_dir
                or selected.get("manifest_sha256") != item.get("kda_build_manifest_sha256")):
            raise RuntimeError(f"KDA would select a different prepared build: {candidate_id}")
        copied_manifest = output / item["autobench_build_manifest"]
        if sha256(copied_manifest) != item["kda_build_manifest_sha256"]:
            raise RuntimeError(f"Autobench copy of KDA build manifest changed: {candidate_id}")
        expected_local_files.add(item["autobench_build_manifest"])
        original_artifacts = {artifact["path"]: artifact["sha256"]
                              for artifact in kda_manifest["artifacts"]}
        copied_artifacts = {artifact["kda_path"]: artifact for artifact in item["artifacts"]}
        if set(original_artifacts) != set(copied_artifacts):
            raise RuntimeError(f"Autobench artifact list differs from KDA manifest: {candidate_id}")
        for relative, digest in original_artifacts.items():
            original = _safe_artifact(kda_dir, relative)
            artifact = copied_artifacts[relative]
            autobench_relative = artifact["autobench_path"]
            copied = _safe_artifact(output, autobench_relative)
            if (digest != artifact.get("sha256") or sha256(original) != digest
                    or sha256(copied) != digest):
                raise RuntimeError(f"prepared KDA binary artifact changed: {candidate_id}/{relative}")
            expected_local_files.add(autobench_relative)

    actual_local_files = set()
    for base, dirs, files in os.walk(output, followlinks=False):
        if any((Path(base) / directory).is_symlink() for directory in dirs):
            raise RuntimeError("symlink found in Autobench managed build output")
        for filename in files:
            path = Path(base) / filename
            if path.is_symlink() or not path.is_file():
                raise RuntimeError("unsafe file found in Autobench managed build output")
            actual_local_files.add(path.relative_to(output).as_posix())
    if actual_local_files != expected_local_files:
        raise RuntimeError("Autobench managed build contains missing or unbound files")

