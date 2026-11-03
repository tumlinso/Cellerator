"""Shared, dependency-light contracts for the KDA task adapters."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "bench" / "factorized_interactions"
EXPERIMENTS = BENCH / "experiments"
NATIVE_MANIFEST_DEFAULT = Path(
    "native_candidate/native_build/factorized_interactions_build_manifest.json"
)


class AdapterError(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def digest_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise AdapterError(f"expected a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AdapterError(f"expected a JSON object in {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any], *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if exclusive:
        try:
            with path.open("x", encoding="utf-8") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError as exc:
            raise AdapterError(f"refusing to overwrite existing evidence: {path}") from exc
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def safe_relative(value: str, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise AdapterError(f"{label} must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise AdapterError(f"{label} escapes its declared root: {value!r}")
    return path


def verify_regular_within(root: Path, relative: str, label: str) -> Path:
    rel = safe_relative(relative, label)
    candidate = root / Path(*rel.parts)
    if candidate.is_symlink():
        raise AdapterError(f"{label} may not be a symlink: {relative}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve(strict=True)) or not resolved.is_file():
        raise AdapterError(f"{label} is missing or outside its root: {relative}")
    return resolved


def candidate_tree_identity(source: Path) -> tuple[str, list[dict[str, str]]]:
    source = source.resolve(strict=True)
    if not source.is_dir():
        raise AdapterError(f"candidate source is not a directory: {source}")
    entries: list[dict[str, str]] = []
    for current, dirs, files in os.walk(source, followlinks=False):
        current_path = Path(current)
        if any((current_path / directory).is_symlink() for directory in dirs):
            raise AdapterError("candidate source tree contains a symlink directory")
        for name in files:
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise AdapterError(f"candidate source contains a nonregular file: {path}")
            entries.append({
                "path": path.relative_to(source).as_posix(),
                "sha256": digest_file(path),
            })
    entries.sort(key=lambda row: row["path"])
    if not entries:
        raise AdapterError("candidate source tree is empty")
    return digest_json(entries), entries


def load_variant_map(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if value.get("schema_version") != 1:
        raise AdapterError("variant map schema_version must equal 1")
    comparison = value.get("comparison")
    variants = value.get("variants")
    if not isinstance(comparison, dict) or not isinstance(variants, dict):
        raise AdapterError("variant map requires comparison and variants objects")
    incumbent = comparison.get("incumbent")
    challenger = comparison.get("challenger")
    if not isinstance(incumbent, str) or not isinstance(challenger, str) or incumbent == challenger:
        raise AdapterError("variant map requires distinct incumbent and challenger IDs")
    strongest_direct_control = comparison.get("strongest_direct_control")
    if strongest_direct_control != incumbent:
        raise AdapterError("the incumbent must be the explicitly declared strongest qualified direct control")
    if comparison.get("same_source_runner_required") is not True:
        raise AdapterError("all candidate modes must use the same source-bound runner snapshot")
    if set(variants) != {incumbent, challenger}:
        raise AdapterError("variant map must contain exactly the compared candidates")
    runner_iterations = value.get("runner_iterations")
    if isinstance(runner_iterations, bool) or not isinstance(runner_iterations, int) or \
            runner_iterations < 1:
        raise AdapterError("variant map runner_iterations must be a positive integer")
    builder = value.get("builder")
    if not isinstance(builder, dict):
        raise AdapterError("variant map must bind a trusted candidate build helper")
    script = builder.get("trusted_script")
    snapshot_script = builder.get("snapshot_script_relative_path")
    argv = builder.get("argv")
    if not isinstance(script, str) or not script or not isinstance(argv, list) or not argv or \
            any(not isinstance(item, str) or not item for item in argv):
        raise AdapterError("variant map builder requires trusted_script and argv")
    if not Path(script).is_absolute() or not Path(script).is_file() or Path(script).is_symlink():
        raise AdapterError("builder trusted_script must be an absolute regular file")
    if not isinstance(snapshot_script, str):
        raise AdapterError("variant map builder must identify its source-snapshot helper path")
    safe_relative(snapshot_script, "candidate snapshot build helper path")
    if not {"{source}", "{build}"}.issubset(argv):
        raise AdapterError("builder argv must use the exact source and build placeholders")
    if any("{" in item or "}" in item for item in argv
           if item not in {"{source}", "{build}"}):
        raise AdapterError("builder argv contains an unsupported placeholder")
    manifest_relative_path = builder.get("manifest_relative_path")
    if not isinstance(manifest_relative_path, str):
        manifest_relative_path = NATIVE_MANIFEST_DEFAULT.as_posix()
    safe_relative(manifest_relative_path, "native build manifest path")
    builder["manifest_relative_path"] = manifest_relative_path
    for candidate_id, variant in variants.items():
        if not isinstance(candidate_id, str) or not isinstance(variant, dict):
            raise AdapterError("each variant must be an object keyed by candidate ID")
        if variant.get("enabled") is not True:
            raise AdapterError(f"candidate route {candidate_id!r} is not integrated/enabled")
        route = variant.get("route")
        if route not in {"direct", "packet"}:
            raise AdapterError(f"unsupported native route for {candidate_id!r}: {route!r}")
        allowed = variant.get("allowed_backends")
        if not isinstance(allowed, list) or not allowed or any(
                x not in {"direct", "packet", "stable"} for x in allowed):
            raise AdapterError(f"variant {candidate_id!r} needs allowed_backends")
        variant.setdefault("native_manifest_relative_path", builder["manifest_relative_path"])
        safe_relative(variant["native_manifest_relative_path"], "native build manifest path")
        candidate_argument = variant.get("candidate_argument", candidate_id)
        if candidate_argument not in {"native_direct", "packet", "minimal_direct", "direct_unique_output"}:
            raise AdapterError(f"variant {candidate_id!r} has an unknown candidate_argument")
        if (candidate_argument == "packet") != (route == "packet"):
            raise AdapterError(f"variant {candidate_id!r} candidate_argument and route disagree")
        variant["candidate_argument"] = candidate_argument
        for key in ("allowed_implementations", "allowed_layouts"):
            allowed_values = variant.get(key)
            if allowed_values is not None and (
                    not isinstance(allowed_values, list) or not allowed_values or
                    any(not isinstance(item, str) or not item for item in allowed_values)):
                raise AdapterError(f"variant {candidate_id!r} has an invalid {key}")
        if variant.get("accepted_route", route) not in {
            "direct", "packet", "direct_unique_output", "minimal_direct"
        }:
            raise AdapterError(f"variant {candidate_id!r} has an unknown accepted route")
    if variants[incumbent].get("route") != "direct":
        raise AdapterError("strongest direct control must use a direct candidate route")
    return value


def attempt_dir_for_request(request_path: Path) -> Path:
    resolved = request_path.resolve(strict=True)
    for parent in resolved.parents:
        marker = parent / "batch-request.json"
        if marker.is_file() and not marker.is_symlink():
            return parent
    raise AdapterError("request path is not inside a KDA evaluation attempt")


def checked_build(candidate: Mapping[str, Any], build_map: Mapping[str, Any],
                  variant: Mapping[str, Any],
                  snapshot_record: Mapping[str, Any] | None = None
                  ) -> tuple[Path, Path, str]:
    build = build_map.get("build", build_map)
    if not isinstance(build, dict):
        raise AdapterError("candidate request is missing its KDA build reference")
    directory = Path(build.get("directory", "")).resolve(strict=True)
    if not directory.is_dir():
        raise AdapterError("candidate build directory is missing")
    artifact_rows = build.get("artifacts")
    if not isinstance(artifact_rows, list):
        raise AdapterError("candidate build reference has no artifact list")
    artifacts: dict[str, str] = {}
    for row in artifact_rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise AdapterError("candidate build artifact record is malformed")
        relative = row["path"]
        path = verify_regular_within(directory, relative, "build artifact")
        digest = digest_file(path)
        if row.get("sha256") != digest:
            raise AdapterError(f"build artifact hash mismatch: {relative}")
        artifacts[relative] = digest

    manifest_rel = variant["native_manifest_relative_path"]
    manifest_path = verify_regular_within(directory, manifest_rel, "native build manifest")
    manifest_rel_posix = PurePosixPath(manifest_rel).as_posix()
    if manifest_rel_posix not in artifacts or artifacts[manifest_rel_posix] != digest_file(manifest_path):
        raise AdapterError("native build manifest is not covered by KDA artifact identity")
    manifest = read_json(manifest_path)
    if manifest.get("schema") != "cellerator.factorized_interactions.build.v1":
        raise AdapterError("native build manifest schema is unknown")
    runner_info = manifest.get("binaries", {}).get("runner")
    if not isinstance(runner_info, dict) or not isinstance(runner_info.get("path"), str):
        raise AdapterError("native build manifest has no runner identity")
    runner_rel_native = safe_relative(runner_info["path"], "native runner path")
    runner_path = (manifest_path.parent / Path(*runner_rel_native.parts)).resolve(strict=True)
    if not runner_path.is_relative_to(directory) or runner_path.is_symlink() or not runner_path.is_file():
        raise AdapterError("native runner is missing or escapes its KDA build")
    runner_rel_kda = runner_path.relative_to(directory).as_posix()
    runner_hash = digest_file(runner_path)
    if runner_info.get("sha256") != runner_hash or artifacts.get(runner_rel_kda) != runner_hash:
        raise AdapterError("native runner hash is not covered by the native and KDA build manifests")
    library_info = manifest.get("binaries", {}).get("factorized_interaction_cuda")
    if not isinstance(library_info, dict) or not isinstance(library_info.get("path"), str):
        raise AdapterError("native build manifest has no linked Cellerator CUDA library identity")
    library_rel_native = safe_relative(library_info["path"], "native CUDA library path")
    library_path = (manifest_path.parent / Path(*library_rel_native.parts)).resolve(strict=True)
    if not library_path.is_relative_to(directory) or library_path.is_symlink() or not library_path.is_file():
        raise AdapterError("native CUDA library is missing or escapes its KDA build")
    library_rel_kda = library_path.relative_to(directory).as_posix()
    library_hash = digest_file(library_path)
    if library_info.get("sha256") != library_hash or artifacts.get(library_rel_kda) != library_hash:
        raise AdapterError("native CUDA library hash is not covered by the native and KDA build manifests")
    source_hash = candidate.get("snapshot_sha256")
    if snapshot_record is not None:
        if snapshot_record.get("snapshot_sha256") != source_hash:
            raise AdapterError("candidate snapshot differs from the registered KDA source identity")
        source_dir_value = snapshot_record.get("snapshot_dir")
        if not isinstance(source_dir_value, str):
            raise AdapterError("registered KDA candidate has no source snapshot directory")
        tree_hash, tree_entries = candidate_tree_identity(Path(source_dir_value))
        if tree_hash != source_hash:
            raise AdapterError("candidate source files differ from the registered KDA snapshot hash")
        native_files = manifest.get("source_files")
        if not isinstance(native_files, list):
            raise AdapterError("native build manifest has no source file identities")
        native_identity = sorted(
            ({"path": row.get("path"), "sha256": row.get("sha256")}
             for row in native_files if isinstance(row, dict)),
            key=lambda row: str(row["path"]),
        )
        if native_identity != tree_entries:
            raise AdapterError("native binary was not built from the registered KDA candidate snapshot")
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise AdapterError("candidate source snapshot hash is missing")
    return runner_path, manifest_path, runner_hash


def find_build_record(batch: Mapping[str, Any], candidate_id: str) -> Mapping[str, Any]:
    builds = batch.get("builds")
    if not isinstance(builds, dict) or candidate_id not in builds:
        raise AdapterError(f"KDA batch has no build for candidate {candidate_id!r}")
    return builds[candidate_id]


def workspace_for_attempt(attempt_dir: Path) -> Path:
    try:
        workspace = attempt_dir.parents[3]
    except IndexError as exc:
        raise AdapterError("KDA attempt path is not nested under its workspace") from exc
    if not (workspace / "candidates.jsonl").is_file():
        raise AdapterError("cannot identify the KDA workspace containing this attempt")
    return workspace
