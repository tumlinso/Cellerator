#!/usr/bin/env python3
"""Run a native-data CUDA binary from an immutable Autobench snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


INTERNAL_SOURCE_MANIFEST = "SOURCE-MANIFEST.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_tree_digest(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    paths = sorted(
        path for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
        and "__pycache__" not in path.relative_to(root).parts
        and path.suffix != ".pyc"
    )
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest(), len(paths)


def autobench_snapshot_fingerprint(manifest: dict[str, str]) -> str:
    encoded = "\n".join(f"{path}\0{manifest[path]}" for path in sorted(manifest))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _snapshot_input(source_root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str):
        raise SystemExit(f"external build manifest has no {label} path")
    posix = PurePosixPath(relative)
    if posix.is_absolute() or not posix.parts or ".." in posix.parts:
        raise SystemExit(f"external build manifest has an unsafe {label} path")
    path = source_root.joinpath(*posix.parts).resolve(strict=True)
    if source_root not in path.parents or path.is_symlink() or not path.is_file():
        raise SystemExit(f"{label} is not a regular file in the immutable packet snapshot")
    return path


def _validate_archived_source_closure(compile_snapshot: dict, source_root: Path) -> None:
    archive_path = _snapshot_input(
        source_root, compile_snapshot.get("source_archive_path"), "source archive"
    )
    manifest_path = _snapshot_input(
        source_root, compile_snapshot.get("manifest_path"), "source manifest"
    )
    if sha256_file(archive_path) != compile_snapshot.get("source_archive_sha256"):
        raise SystemExit("captured source archive differs from the external build manifest")
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != compile_snapshot.get("manifest_sha256"):
        raise SystemExit("captured per-file source manifest differs from the external build manifest")
    source_manifest = json.loads(manifest_bytes)
    files = source_manifest.get("files")
    if source_manifest.get("schema") != "cellerator.native_data.source_closure.v1":
        raise SystemExit("captured source manifest has an unsupported schema")
    if not isinstance(files, dict) or not files:
        raise SystemExit("captured source manifest has no file closure")
    if source_manifest.get("file_count") != len(files):
        raise SystemExit("captured source manifest file count is inconsistent")
    normalized = {}
    for relative, details in files.items():
        if not isinstance(relative, str) or not isinstance(details, dict):
            raise SystemExit("captured source manifest contains an invalid file record")
        posix = PurePosixPath(relative)
        if posix.is_absolute() or not posix.parts or ".." in posix.parts:
            raise SystemExit("captured source manifest contains an unsafe path")
        digest = details.get("sha256")
        size = details.get("size_bytes")
        mode = details.get("mode")
        if (not isinstance(digest, str) or len(digest) != 64 or
                isinstance(size, bool) or not isinstance(size, int) or size < 0 or
                isinstance(mode, bool) or not isinstance(mode, int)):
            raise SystemExit(f"invalid source closure details for {relative}")
        normalized[relative] = details
    if autobench_snapshot_fingerprint({key: value["sha256"] for key, value in normalized.items()}) != source_manifest.get("fingerprint"):
        raise SystemExit("captured source manifest fingerprint is invalid")
    if source_manifest.get("fingerprint") != compile_snapshot.get("fingerprint"):
        raise SystemExit("source closure fingerprint differs from the external build manifest")
    if source_manifest.get("file_count") != compile_snapshot.get("file_count"):
        raise SystemExit("source closure file count differs from the external build manifest")

    with tarfile.open(archive_path, mode="r:") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise SystemExit("captured source archive contains duplicate paths")
        records = {member.name: member for member in members}
        if set(records) != set(normalized) | {INTERNAL_SOURCE_MANIFEST}:
            raise SystemExit("captured source archive member list differs from its manifest")
        manifest_member = records[INTERNAL_SOURCE_MANIFEST]
        manifest_stream = archive.extractfile(manifest_member)
        if not manifest_member.isfile() or manifest_stream is None or manifest_stream.read() != manifest_bytes:
            raise SystemExit("embedded source manifest differs from the detached manifest")
        for relative, details in normalized.items():
            member = records[relative]
            if not member.isfile() or member.issym() or member.islnk():
                raise SystemExit(f"source archive member is not a regular file: {relative}")
            if member.size != details["size_bytes"]:
                raise SystemExit(f"source archive member size differs from manifest: {relative}")
            if member.mode & 0o777 != details["mode"] & 0o777:
                raise SystemExit(f"source archive member mode differs from manifest: {relative}")
            stream = archive.extractfile(member)
            if stream is None or hashlib.sha256(stream.read()).hexdigest() != details["sha256"]:
                raise SystemExit(f"source archive member hash differs from manifest: {relative}")


def verify_external_artifact(
    binary: Path, manifest_path: Path, source_root: Path, output_root: Path
) -> tuple[Path, dict, str]:
    if source_root not in binary.parents:
        raise SystemExit("externally built binary must be contained in the immutable packet snapshot")
    if source_root not in manifest_path.parents:
        raise SystemExit("external build manifest must be contained in the immutable packet snapshot")
    manifest_bytes = manifest_path.read_bytes()
    artifact = json.loads(manifest_bytes)
    if artifact.get("schema") != "cellerator.native_data.external_snapshot_build.v1":
        raise SystemExit("external binary has an unknown provenance schema")
    artifact_kind = artifact.get("artifact_kind")
    allowed_artifact_kinds = {
        "externally_built_autobench_snapshot_artifact",
        "externally_built_autobench_archived_source_artifact",
    }
    if artifact_kind not in allowed_artifact_kinds:
        raise SystemExit("external binary is not declared as a supported Autobench-snapshot build")
    if artifact.get("autobench_build_output") is not False:
        raise SystemExit("external binary provenance must not claim an Autobench build output")
    binary_hash = sha256_file(binary)
    if artifact.get("binary", {}).get("sha256") != binary_hash:
        raise SystemExit("snapshot binary hash differs from its external build manifest")

    compile_snapshot = artifact.get("compile_source_snapshot", {})
    compile_manifest = compile_snapshot.get("manifest")
    if artifact_kind == "externally_built_autobench_archived_source_artifact":
        _validate_archived_source_closure(compile_snapshot, source_root)
    else:
        if not isinstance(compile_manifest, dict) or not compile_manifest:
            raise SystemExit("external build manifest lacks its compile-source manifest")
        normalized_manifest = {str(path): str(digest) for path, digest in compile_manifest.items()}
        if autobench_snapshot_fingerprint(normalized_manifest) != compile_snapshot.get("fingerprint"):
            raise SystemExit("compile-source manifest does not match its Autobench snapshot fingerprint")
        for relative, expected_hash in normalized_manifest.items():
            path = source_root / relative
            if not path.is_file() or path.is_symlink():
                raise SystemExit(f"compiled source input is absent from the packet snapshot: {relative}")
            if sha256_file(path) != expected_hash:
                raise SystemExit(f"compiled source input changed since artifact build: {relative}")

    packet_manifest = {}
    for path in sorted(
        item for item in source_root.rglob("*")
        if item.is_file() and not item.is_symlink()
        and "__pycache__" not in item.relative_to(source_root).parts
        and item.suffix != ".pyc"
    ):
        packet_manifest[path.relative_to(source_root).as_posix()] = sha256_file(path)
    if autobench_snapshot_fingerprint(packet_manifest) != source_root.name:
        raise SystemExit("packet snapshot contents do not match the Autobench snapshot directory identity")

    staged_dir = output_root / "staged_external_binaries"
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged_binary = staged_dir / binary.name
    if staged_binary.exists():
        if sha256_file(staged_binary) != binary_hash:
            raise SystemExit("a different binary already occupies the staged output path")
    else:
        shutil.copyfile(binary, staged_binary)
    staged_binary.chmod(0o755)
    if sha256_file(staged_binary) != binary_hash:
        raise SystemExit("staged executable bytes differ from the immutable snapshot artifact")
    return staged_binary, artifact, hashlib.sha256(manifest_bytes).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--build-root", type=Path)
    parser.add_argument("--artifact-manifest", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--case", choices=("reference_active", "reference_inactive", "low", "high"), required=True)
    parser.add_argument("--mode", choices=("correctness", "timing"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device0", type=int, default=0)
    parser.add_argument("--device1", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--atol", type=float, default=2.0e-5)
    parser.add_argument("--rtol", type=float, default=2.0e-5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    binary = args.binary.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    output_path = args.output.resolve()
    snapshot_marker = (".autobench", "state", "snapshots")
    source_parts = source_root.parts
    if not any(tuple(source_parts[i:i + 3]) == snapshot_marker for i in range(len(source_parts) - 2)):
        raise SystemExit("GPU benchmark source must come from an immutable Autobench {snapshot}")
    artifact = None
    artifact_manifest_sha256 = None
    build_root = None
    executable = binary
    if args.artifact_manifest is not None:
        if args.build_root is not None:
            raise SystemExit("choose either a registered {build} binary or an external snapshot artifact")
        manifest_path = args.artifact_manifest.resolve(strict=True)
        executable, artifact, artifact_manifest_sha256 = verify_external_artifact(
            binary, manifest_path, source_root, output_path.parent
        )
    else:
        if args.build_root is None:
            raise SystemExit("a {build} root or --artifact-manifest is required")
        build_root = args.build_root.resolve(strict=True)
        if build_root not in binary.parents:
            raise SystemExit("benchmark binary must be inside the Autobench {build} artifact")
    if args.device0 == args.device1:
        raise SystemExit("two distinct Scheduler-visible CUDA devices are required")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_path.with_name(output_path.stem + ".raw.json")
    stderr_path = output_path.with_name(output_path.stem + ".stderr.txt")
    command = [
        str(executable), "--case", args.case, "--mode", args.mode,
        "--output", str(raw_path), "--device0", str(args.device0),
        "--device1", str(args.device1), "--warmup", str(args.warmup),
        "--samples", str(args.samples), "--learning-rate", repr(args.learning_rate),
    ]
    completed = subprocess.run(command, cwd=source_root, text=True, capture_output=True,
                               env=os.environ.copy(), check=False)
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise SystemExit(f"benchmark binary exited {completed.returncode}; see {stderr_path}")
    result = json.loads(raw_path.read_text(encoding="utf-8"))
    source_digest, source_count = source_tree_digest(source_root)
    artifact_identity = {
        "binary_sha256": sha256_file(binary),
        "snapshot_source_sha256": source_digest,
        "snapshot_file_count": source_count,
        "snapshot_root": str(source_root),
        "autobench_snapshot_fingerprint": source_root.name,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if artifact is not None:
        artifact_identity.update({
            "binary_artifact_kind": artifact["artifact_kind"],
            "binary_path_from_snapshot": binary.relative_to(source_root).as_posix(),
            "external_build_manifest_sha256": artifact_manifest_sha256,
            "compile_source_snapshot_sha256": artifact["compile_source_snapshot"]["fingerprint"],
            "compile_source_file_count": artifact["compile_source_snapshot"]["file_count"],
            "compile_source_archive_sha256": artifact["compile_source_snapshot"].get("source_archive_sha256"),
            "compile_source_manifest_sha256": artifact["compile_source_snapshot"].get("manifest_sha256"),
            "build_recipe_sha256": artifact["build_recipe"]["sha256"],
            "staged_executable_sha256": sha256_file(executable),
        })
    else:
        artifact_identity.update({
            "binary_artifact_kind": "autobench_build_output",
            "binary_path_from_build_root": binary.relative_to(build_root).as_posix(),
        })
    result["artifact_identity"] = artifact_identity
    if args.mode == "correctness" and args.case.startswith("reference_"):
        oracle = source_root / "tests/native_data/reference/exact_oracle.py"
        validator = source_root / "bench/native_data/validate_reference.py"
        validation_path = output_path.with_name(output_path.stem + ".validation.json")
        validation = subprocess.run(
            [sys.executable, str(validator), "--result", str(raw_path), "--oracle", str(oracle),
             "--output", str(validation_path), "--atol", repr(args.atol), "--rtol", repr(args.rtol)],
            cwd=source_root, text=True, capture_output=True, check=False)
        if validation.returncode:
            raise SystemExit(f"exact Fraction oracle rejected benchmark result: {validation.stdout} {validation.stderr}")
        result["correctness_validation"] = json.loads(validation_path.read_text(encoding="utf-8"))
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "backend": result["backend"],
        "case": result["case"],
        "mode": result["mode"],
        "binary_sha256": result["artifact_identity"]["binary_sha256"],
        "output": str(output_path),
        "validated": result.get("correctness_validation", {}).get("passed"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
