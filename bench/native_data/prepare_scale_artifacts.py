#!/usr/bin/env python3
"""Build scale binaries from an Autobench-captured source-closure archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


CUDA_ROOT = Path("/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9")
TORCH_ROOT = Path(
    "/home/tumlinson/Software/miniconda3/lib/python3.13/site-packages/torch"
)
TORCH_PYTHON = Path("/home/tumlinson/Software/miniconda3/bin/python")
BASEPLANE_ROOT = Path("/home/tumlinson/Baseplane")
ARTIFACT_RELATIVE_ROOT = Path(
    ".autobench/data/cellerator-native-data-gated-product-scale"
)
ARCHIVE_RELATIVE_PATH = ARTIFACT_RELATIVE_ROOT / "source-closure-v3.tar"
MANIFEST_RELATIVE_PATH = ARTIFACT_RELATIVE_ROOT / "source-closure-v3.manifest.json"
INTERNAL_MANIFEST = "SOURCE-MANIFEST.json"
TARGETS = {
    "aten": "cellerator_native_data_aten_bench",
    "specialist": "cellerator_native_data_specialist_bench",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def closure_fingerprint(files: dict[str, dict[str, object]]) -> str:
    return sha256_bytes(
        "\n".join(
            f"{name}\0{files[name]['sha256']}" for name in sorted(files)
        ).encode()
    )


def safe_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise SystemExit(f"unsafe source archive member path: {name!r}")
    return path


def extract_captured_archive(snapshot: Path, destination: Path) -> tuple[dict, bytes, str]:
    archive_path = (snapshot / ARCHIVE_RELATIVE_PATH).resolve(strict=True)
    if snapshot not in archive_path.parents:
        raise SystemExit("source archive must be contained in the captured snapshot")
    archive_hash = sha256_file(archive_path)
    detached_path = snapshot / MANIFEST_RELATIVE_PATH
    if detached_path.is_file():
        if detached_path.is_symlink() or snapshot not in detached_path.resolve().parents:
            raise SystemExit("detached source manifest is outside the captured snapshot")
        detached_bytes = detached_path.read_bytes()
    else:
        with tarfile.open(archive_path, mode="r:") as archive:
            embedded = archive.extractfile(INTERNAL_MANIFEST)
            if embedded is None:
                raise SystemExit("captured source archive has no embedded source manifest")
            detached_bytes = embedded.read()
    detached = json.loads(detached_bytes)
    if detached.get("schema") != "cellerator.native_data.source_closure.v1":
        raise SystemExit("unknown Cellerator source-closure manifest schema")
    files = detached.get("files")
    if not isinstance(files, dict) or not files:
        raise SystemExit("source-closure manifest has no per-file records")
    if detached.get("file_count") != len(files):
        raise SystemExit("source-closure file count is inconsistent")
    if closure_fingerprint(files) != detached.get("fingerprint"):
        raise SystemExit("source-closure manifest fingerprint is invalid")

    if destination.exists():
        if any(destination.iterdir()):
            raise SystemExit(f"extraction destination is not empty: {destination}")
    else:
        destination.mkdir(parents=True)

    with tarfile.open(archive_path, mode="r:") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise SystemExit("source archive contains duplicate member names")
        records = {member.name: member for member in members}
        manifest_member = records.get(INTERNAL_MANIFEST)
        if manifest_member is None or not manifest_member.isfile():
            raise SystemExit("source archive lacks its embedded per-file manifest")
        embedded_stream = archive.extractfile(manifest_member)
        if embedded_stream is None:
            raise SystemExit("source archive manifest bytes are unavailable")
        embedded_bytes = embedded_stream.read()
        if embedded_bytes != detached_bytes:
            raise SystemExit("detached manifest differs from the captured archive manifest")

        expected_names = set(files) | {INTERNAL_MANIFEST}
        if set(records) != expected_names:
            raise SystemExit("source archive members differ from the per-file manifest")
        for name in sorted(files):
            relative = safe_member(name)
            member = records[name]
            if not member.isfile() or member.issym() or member.islnk():
                raise SystemExit(f"source archive contains a non-regular member: {name}")
            details = files[name]
            if member.size != details.get("size_bytes"):
                raise SystemExit(f"source archive size mismatch for {name}")
            if member.mode & 0o777 != details.get("mode", 0o644) & 0o777:
                raise SystemExit(f"source archive mode mismatch for {name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise SystemExit(f"source archive member is unavailable: {name}")
            content = stream.read()
            if sha256_bytes(content) != details.get("sha256"):
                raise SystemExit(f"source archive hash mismatch for {name}")
            output = destination.joinpath(*relative.parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(content)
            output.chmod(int(details.get("mode", 0o644)) & 0o777)

    return detached, detached_bytes, archive_hash


def run(command: list[str], *, cwd: Path, log: Path, label: str) -> str:
    completed = subprocess.run(
        command, cwd=cwd, text=True, capture_output=True, check=False,
        env=os.environ.copy(),
    )
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"$ {label}\n")
        stream.write("argv = " + json.dumps(command) + "\n")
        stream.write(completed.stdout)
        stream.write(completed.stderr)
        if not completed.stdout.endswith("\n"):
            stream.write("\n")
        if not completed.stderr.endswith("\n"):
            stream.write("\n")
        stream.write(f"returncode = {completed.returncode}\n\n")
    if completed.returncode != 0:
        raise SystemExit(
            f"{label} failed with exit {completed.returncode}; see {log}"
        )
    return completed.stdout.strip()


def tool_version(command: list[str], *, cwd: Path, log: Path, label: str) -> str:
    return run(command, cwd=cwd, log=log, label=label)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-snapshot-root", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()

    snapshot = args.source_snapshot_root.resolve(strict=True)
    build = args.build_root.resolve()
    artifact_root = args.artifact_root.resolve()
    marker = (".autobench", "state", "snapshots")
    if not any(
        tuple(snapshot.parts[index:index + 3]) == marker
        for index in range(len(snapshot.parts) - 2)
    ):
        raise SystemExit("build inputs must come from an immutable Autobench snapshot")
    if build.exists() and any(build.iterdir()):
        raise SystemExit(f"build root must be fresh and empty: {build}")
    if not CUDA_ROOT.joinpath("bin/nvcc").is_file():
        raise SystemExit(f"pinned CUDA 12.9 compiler is unavailable at {CUDA_ROOT}")
    if not TORCH_ROOT.joinpath("share/cmake/Torch/TorchConfig.cmake").is_file():
        raise SystemExit(f"pinned Torch CMake package is unavailable at {TORCH_ROOT}")
    if not BASEPLANE_ROOT.joinpath("CMakeLists.txt").is_file():
        raise SystemExit(f"pinned Baseplane source checkout is unavailable at {BASEPLANE_ROOT}")

    build.mkdir(parents=True, exist_ok=True)
    extracted_source = build / "source"
    closure, manifest_bytes, archive_hash = extract_captured_archive(snapshot, extracted_source)
    artifact_root.mkdir(parents=True, exist_ok=True)
    detached_manifest_output = artifact_root / MANIFEST_RELATIVE_PATH.name
    if detached_manifest_output.exists():
        if sha256_file(detached_manifest_output) != sha256_bytes(manifest_bytes):
            raise SystemExit("a different source manifest already occupies the artifact path")
    else:
        temporary_manifest = detached_manifest_output.with_suffix(".json.tmp")
        temporary_manifest.write_bytes(manifest_bytes)
        temporary_manifest.replace(detached_manifest_output)
    build_log = artifact_root / "build-v4.log"
    build_log.write_text(
        "Native-data scale build from an Autobench-captured source archive\n"
        f"source_snapshot={snapshot}\n"
        f"source_archive={ARCHIVE_RELATIVE_PATH.as_posix()}\n"
        f"source_archive_sha256={archive_hash}\n"
        f"source_closure_fingerprint={closure['fingerprint']}\n\n",
        encoding="utf-8",
    )

    configure = [
        "cmake", "-S", str(extracted_source), "-B", str(build / "cmake"),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CXX_COMPILER=/usr/bin/g++-12",
        f"-DCMAKE_CUDA_COMPILER={CUDA_ROOT / 'bin/nvcc'}",
        "-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12",
        "-DCMAKE_CUDA_ARCHITECTURES=70",
        f"-DCMAKE_PREFIX_PATH={TORCH_ROOT / 'share/cmake'}",
        f"-DTorch_DIR={TORCH_ROOT / 'share/cmake/Torch'}",
        f"-DBASEPLANE_SOURCE_DIR={BASEPLANE_ROOT}",
        "-DCELLERATOR_ENABLE_CUDA=ON",
        "-DCELLERATOR_ENABLE_HARDWARE_PROBE=OFF",
        "-DCELLERATOR_AUTO_DETECT_CUDA_ARCHITECTURES=OFF",
        "-DCELLERATOR_BUILD_TESTS=OFF",
        "-DCELLERATOR_BUILD_NATIVE_DATA_TESTS=OFF",
        "-DCELLERATOR_BUILD_NATIVE_DATA_BENCHMARKS=ON",
        "-DCELLERATOR_ENABLE_PYTHON=OFF",
        "-DCELLERATOR_ENABLE_TORCH=OFF",
    ]
    run(configure, cwd=extracted_source, log=build_log, label="configure")
    build_argv = [
        "cmake", "--build", str(build / "cmake"),
        "--target", *TARGETS.values(), "--parallel", "2",
    ]
    run(build_argv, cwd=extracted_source, log=build_log, label="build")

    versions = {
        "cmake": tool_version(["cmake", "--version"], cwd=extracted_source, log=build_log, label="cmake version"),
        "cxx": tool_version(["/usr/bin/g++-12", "--version"], cwd=extracted_source, log=build_log, label="g++ version"),
        "cuda_compiler": tool_version([str(CUDA_ROOT / "bin/nvcc"), "--version"], cwd=extracted_source, log=build_log, label="nvcc version"),
    }
    baseplane_revision = subprocess.run(
        ["git", "-C", str(BASEPLANE_ROOT), "rev-parse", "HEAD"],
        text=True, capture_output=True, check=False,
    )
    baseplane_dirty = subprocess.run(
        ["git", "-C", str(BASEPLANE_ROOT), "status", "--porcelain"],
        text=True, capture_output=True, check=False,
    )
    if baseplane_revision.returncode != 0 or baseplane_dirty.returncode != 0:
        raise SystemExit("Baseplane dependency provenance could not be read")
    torch_version = subprocess.run(
        [str(TORCH_PYTHON), "-c", "import torch; print(torch.__version__); print(torch.version.cuda)"],
        text=True, capture_output=True, check=False,
    )
    if torch_version.returncode != 0:
        raise SystemExit("pinned Torch package version probe failed")
    versions["torch"] = torch_version.stdout.strip()

    archive_sha = sha256_file(snapshot / ARCHIVE_RELATIVE_PATH)
    manifest_sha = sha256_bytes(manifest_bytes)
    for backend, target in TARGETS.items():
        built = build / "cmake/bench/native_data" / target
        if not built.is_file():
            raise SystemExit(f"CMake target output is absent: {built}")
        help_probe = subprocess.run(
            [str(built), "--help"], text=True, capture_output=True, check=False,
        )
        if help_probe.returncode != 0 or "--case" not in help_probe.stdout:
            raise SystemExit(f"CPU-only CLI probe failed for {target}")

        destination = artifact_root / backend
        destination.mkdir(parents=True, exist_ok=True)
        binary = destination / target
        shutil.copyfile(built, binary)
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        recipe = {
            "configure_argv": configure,
            "build_argv": build_argv,
            "build_root": str(build),
            "build_log_sha256": sha256_file(build_log),
            "cpu_only_probe": {
                "argv": [target, "--help"],
                "returncode": help_probe.returncode,
                "stdout": help_probe.stdout,
                "stderr": help_probe.stderr,
            },
            "source_archive_snapshot_root": str(snapshot),
            "source_archive_path": ARCHIVE_RELATIVE_PATH.as_posix(),
            "source_archive_sha256": archive_sha,
            "source_manifest_path": MANIFEST_RELATIVE_PATH.as_posix(),
            "source_manifest_sha256": manifest_sha,
            "cuda_kernel_execution": False,
        }
        recipe_hash = sha256_bytes(
            json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
        )
        external_manifest = {
            "schema": "cellerator.native_data.external_snapshot_build.v1",
            "artifact_kind": "externally_built_autobench_archived_source_artifact",
            "autobench_build_output": False,
            "built_locally_for_cpu_preparation": True,
            "captured_utc": datetime.now(timezone.utc).isoformat(),
            "target": target,
            "binary": {
                "path": f"{ARTIFACT_RELATIVE_ROOT.as_posix()}/{backend}/{target}",
                "bytes": binary.stat().st_size,
                "sha256": sha256_file(binary),
                "source_mode": oct(binary.stat().st_mode & 0o777),
            },
            "compile_source_snapshot": {
                "autobench_snapshot_path": str(snapshot),
                "fingerprint": closure["fingerprint"],
                "file_count": closure["file_count"],
                "source_archive_path": ARCHIVE_RELATIVE_PATH.as_posix(),
                "source_archive_sha256": archive_sha,
                "manifest_path": MANIFEST_RELATIVE_PATH.as_posix(),
                "manifest_sha256": manifest_sha,
                "manifest_schema": closure["schema"],
            },
            "build_recipe": {"sha256": recipe_hash, **recipe},
            "toolchain": versions,
            "external_configuration_dependency": {
                "name": "Baseplane",
                "source_root": str(BASEPLANE_ROOT),
                "git_revision": baseplane_revision.stdout.strip(),
                "worktree_clean": not baseplane_dirty.stdout.strip(),
                "built_as_candidate_dependency": False,
            },
        }
        manifest_output = destination / "external_build_manifest.json"
        temporary = manifest_output.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(external_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(manifest_output)

    print(json.dumps({
        "source_archive_snapshot_root": str(snapshot),
        "source_archive_sha256": archive_hash,
        "source_manifest_sha256": sha256_bytes(manifest_bytes),
        "source_closure_fingerprint": closure["fingerprint"],
        "compile_source_file_count": closure["file_count"],
        "build_log": str(build_log),
        "artifacts": {
            backend: {
                "binary_sha256": sha256_file(artifact_root / backend / target),
                "manifest": str(artifact_root / backend / "external_build_manifest.json"),
                "manifest_sha256": sha256_file(artifact_root / backend / "external_build_manifest.json"),
            }
            for backend, target in TARGETS.items()
        },
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
