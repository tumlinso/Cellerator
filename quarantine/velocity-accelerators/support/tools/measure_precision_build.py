#!/usr/bin/env python3
"""Qualify Milestone 3 module rebuild isolation in disposable source copies.

This script only prepares a plan by default. Pass --run after the host has a
quiescence window to configure/build isolated copies and collect timings.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import statistics
import subprocess
import sys
import time
import uuid


FORMAT = "velocity-cppm.module-build-qualification"
VERSION = 1
JOBS = 2
COMMAND_TIMEOUT_S = 120
WARM_PAIRS = 5
CUDA_COMPILER = Path(
    "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc"
)
ALL_SCOPES = ("cpu_pair", "cpu_avx2", "cuda_private", "anndata_private")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def source_files(base: Path, selected: tuple[str, ...]) -> dict[str, str]:
    files: dict[str, str] = {}
    for name in selected:
        path = base / name
        candidates = sorted(path.rglob("*") if path.is_dir() else [path])
        for candidate in candidates:
            if candidate.is_file():
                files[candidate.relative_to(base).as_posix()] = file_hash(candidate)
    return files


def tree_identity(base: Path, selected: tuple[str, ...], key_files=()) -> dict:
    files = source_files(base, selected)
    digest = hashlib.sha256()
    for name, value in sorted(files.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return {
        "root": str(base.resolve()),
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "key_file_sha256": {name: files[name] for name in key_files if name in files},
    }


def identity(path: Path) -> dict:
    stat = path.stat()
    return {"sha256": file_hash(path), "mtime_ns": stat.st_mtime_ns}


def identities(paths: list[Path], base: Path) -> dict[str, dict]:
    return {
        path.relative_to(base).as_posix(): identity(path)
        for path in sorted(paths)
    }


def bmi_paths(build: Path) -> list[Path]:
    return sorted(build.rglob("*.pcm"))


def consumer_object_paths(build: Path) -> list[Path]:
    paths = []
    for path in build.rglob("*.cpp.o"):
        parts = path.relative_to(build).parts
        if "examples" in parts and any(part.endswith("example.dir") for part in parts):
            paths.append(path)
    return sorted(paths)


def copy_velocity_source(origin: Path, destination: Path) -> None:
    before = velocity_source_identity(origin)
    destination.mkdir(parents=True)
    for name in ("src", "cmake", "examples"):
        shutil.copytree(origin / name, destination / name)
    for name in ("CMakeLists.txt", "THIRD_PARTY_NOTICES.md"):
        shutil.copy2(origin / name, destination / name)
    copied = velocity_source_identity(destination)
    after = velocity_source_identity(origin)
    if before["sha256"] != copied["sha256"] or before["sha256"] != after["sha256"]:
        raise RuntimeError(f"source changed while making isolated copy of {origin}")


def velocity_source_identity(base: Path) -> dict:
    selected = ("src", "cmake", "examples", "CMakeLists.txt", "THIRD_PARTY_NOTICES.md")
    keys = (
        "CMakeLists.txt",
        "src/velocity.cppm",
        "src/velocity.cpp",
        "src/precision.cpp",
        "src/detail/cpu_kernels.cpp",
        "src/cuda.cpp",
        "src/cuda/kernels.cu",
        "examples/steady_state.cpp",
    )
    return tree_identity(base, selected, keys)


def anndata_source_identity(base: Path) -> dict:
    selected = ("src", "cmake", "examples", "CMakeLists.txt", "LICENSE")
    keys = (
        "CMakeLists.txt",
        "src/anndata.cppm",
        "src/anndata.core.cppm",
        "src/h5ad.cpp",
        "src/zarr_codecs.cpp",
        "examples/foundation.cpp",
    )
    return tree_identity(base, selected, keys)


class CommandError(RuntimeError):
    pass


class RunContext:
    def __init__(self, root: Path, run_dir: Path):
        self.root = root
        self.run_dir = run_dir
        self.commands: list[dict] = []
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    def execute(self, name: str, argv: list[str], cwd: Path, timeout_s=COMMAND_TIMEOUT_S) -> float:
        safe_name = "".join(
            character if character.isalnum() or character in "._-" else "-"
            for character in name
        ).strip(".-") or "command"
        log_path = self.run_dir / "logs" / f"{len(self.commands) + 1:03d}-{safe_name}.log"
        record = {
            "name": name,
            "log_name": safe_name,
            "argv": [str(item) for item in argv],
            "command": shlex.join([str(item) for item in argv]),
            "cwd": str(cwd.resolve()),
            "timeout_s": timeout_s,
            "log": log_path.relative_to(self.root).as_posix(),
        }
        started = time.perf_counter()
        try:
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    [str(item) for item in argv],
                    cwd=cwd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=timeout_s,
                )
            elapsed = time.perf_counter() - started
            record.update(elapsed_s=elapsed, returncode=completed.returncode)
            self.commands.append(record)
            if completed.returncode:
                raise CommandError(f"{name} exited {completed.returncode}; see {record['log']}")
            return elapsed
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - started
            record.update(elapsed_s=elapsed, timed_out=True)
            self.commands.append(record)
            raise CommandError(f"{name} timed out after {timeout_s}s; see {record['log']}")
        except OSError as error:
            elapsed = time.perf_counter() - started
            record.update(elapsed_s=elapsed, execution_error=str(error))
            self.commands.append(record)
            raise CommandError(f"{name} could not start: {error}") from error


def capture_ids(build: Path) -> dict:
    return {
        "public_bmis": identities(bmi_paths(build), build),
        "consumer_objects": identities(consumer_object_paths(build), build),
    }


def ids_unchanged(before: dict, after: dict) -> bool:
    return before == after


def require_nonempty_artifacts(build: Path, include_consumers=True) -> dict:
    current = capture_ids(build)
    if not current["public_bmis"]:
        raise RuntimeError(f"no public module BMI files found under {build}")
    if include_consumers and not current["consumer_objects"]:
        raise RuntimeError(f"no example consumer objects found under {build}")
    return current


def append_private_comment(source: Path, marker: str) -> dict:
    before = file_hash(source)
    with source.open("ab") as stream:
        stream.write((f"\n// {marker}\n").encode("utf-8"))
    return {"path": str(source), "before_sha256": before, "edited_sha256": file_hash(source), "marker": marker}


def configure_velocity(ctx: RunContext, source: Path, build: Path, prefix: Path, extra=()) -> float:
    argv = [
        "cmake", "-S", str(source), "-B", str(build), "-G", "Ninja",
        "-DCMAKE_CXX_COMPILER=clang++-18",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_PREFIX_PATH={prefix.resolve()}",
        "-DVELOCITY_BUILD_TESTS=OFF",
        "-DVELOCITY_BUILD_QUALIFICATION=OFF",
        "-DVELOCITY_ENABLE_SANITIZERS=OFF",
        *extra,
    ]
    return ctx.execute("configure-" + build.parent.name, argv, ctx.root)


def build_velocity(ctx: RunContext, build: Path, name: str, target=None) -> float:
    argv = ["cmake", "--build", str(build), "--parallel", str(JOBS), "--verbose"]
    if target:
        argv.extend(("--target", target))
    return ctx.execute(name, argv, ctx.root)


def run_private_probe(
    ctx: RunContext,
    build: Path,
    source: Path,
    expected_object_suffix: str,
    marker: str,
    name: str,
) -> dict:
    before = require_nonempty_artifacts(build)
    private_object_before = {
        path.relative_to(build).as_posix(): identity(path)
        for path in build.rglob("*")
        if path.is_file() and path.relative_to(build).as_posix().endswith(expected_object_suffix)
    }
    if not private_object_before:
        raise RuntimeError(f"private object not found for {source}: {expected_object_suffix}")
    edit = append_private_comment(source, marker)
    elapsed = build_velocity(ctx, build, name)
    after = capture_ids(build)
    private_object_after = {
        path.relative_to(build).as_posix(): identity(path)
        for path in build.rglob("*")
        if path.is_file() and path.relative_to(build).as_posix().endswith(expected_object_suffix)
    }
    private_object_rebuilt = bool(private_object_before) and private_object_before != private_object_after
    bmis_unchanged = ids_unchanged(before["public_bmis"], after["public_bmis"])
    consumers_unchanged = ids_unchanged(before["consumer_objects"], after["consumer_objects"])
    result = {
        "source_edit": edit,
        "private_rebuild_s": elapsed,
        "private_object_before": private_object_before,
        "private_object_after": private_object_after,
        "private_object_rebuilt": private_object_rebuilt,
        "before": before,
        "after": after,
        "public_bmis_unchanged": bmis_unchanged,
        "consumer_objects_unchanged": consumers_unchanged,
        "passed": private_object_rebuilt and bmis_unchanged and consumers_unchanged,
    }
    return result


def configure_and_build_pair(ctx: RunContext, label: str, origin: Path, prefix: Path, extra=()) -> dict:
    base = ctx.run_dir / label
    source, build = base / "source", base / "build"
    base.mkdir(parents=True)
    copy_velocity_source(origin, source)
    configure_s = configure_velocity(ctx, source, build, prefix, extra)
    cold_s = build_velocity(ctx, build, f"cold-build-{label}")
    artifacts = require_nonempty_artifacts(build)
    return {
        "source": velocity_source_identity(source),
        "configure_s": configure_s,
        "cold_build_s": cold_s,
        "cold_total_s": configure_s + cold_s,
        "public_bmi_count": len(artifacts["public_bmis"]),
        "consumer_object_count": len(artifacts["consumer_objects"]),
        "_source_path": source,
        "_build_path": build,
    }


def run_cpu_pair(ctx: RunContext, root: Path, inputs: dict) -> dict:
    scope = {"status": "running"}
    try:
        baseline_origin = root / "build/milestone3/baseline/source"
        baseline_prefix = root / "build/milestone3/baseline/anndata-prefix"
        baseline_build_artifact = root / "build/milestone3/baseline/build"
        manifest_path = root / "build/milestone3/baseline/source-identity.json"
        threshold = float(json.loads((root / "planning/milestone-3.json").read_text())[
            "gates"]["warm_consumer_max_baseline_ratio"]
        )
        manifest = json.loads(manifest_path.read_text())
        actual_baseline = source_files(
            baseline_origin,
            ("src", "cmake", "examples", "CMakeLists.txt", "THIRD_PARTY_NOTICES.md"),
        )
        if actual_baseline != manifest["files"]:
            raise RuntimeError("frozen baseline source no longer matches source-identity.json")

        baseline = configure_and_build_pair(
            ctx, "cpu-pair/baseline", baseline_origin, baseline_prefix,
        )
        current = configure_and_build_pair(
            ctx, "cpu-pair/current-scalar", root, root / "build/deps/install",
            ("-DVELOCITY_CPU_ISA=scalar", "-DVELOCITY_ENABLE_CUDA=OFF"),
        )
        baseline_source = baseline["_source_path"] / "examples/steady_state.cpp"
        current_source = current["_source_path"] / "examples/steady_state.cpp"
        consumer_bytes = baseline_source.read_bytes()
        current_initial_sha = file_hash(current_source)
        current_source.write_bytes(consumer_bytes)
        consumer_sha = sha256_bytes(consumer_bytes)
        warm: list[dict] = []
        order_log: list[str] = []
        for pair_index in range(1, WARM_PAIRS + 1):
            marker = f"milestone-3 warm consumer sample {pair_index}"
            sample_bytes = consumer_bytes + f"\n// {marker}\n".encode("utf-8")
            baseline_source.write_bytes(sample_bytes)
            current_source.write_bytes(sample_bytes)
            order = ("baseline", "current") if pair_index % 2 else ("current", "baseline")
            pair_result = {"pair": pair_index, "order": list(order), "samples": {}}
            for side in order:
                condition = baseline if side == "baseline" else current
                build = condition["_build_path"]
                before = capture_ids(build)
                elapsed = build_velocity(
                    ctx,
                    build,
                    f"warm-{side}-{pair_index:02d}",
                    "velocity_example",
                )
                after = capture_ids(build)
                target_suffix = "CMakeFiles/velocity_example.dir/examples/steady_state.cpp.o"
                changed = [
                    path for path in before["consumer_objects"]
                    if before["consumer_objects"][path] != after["consumer_objects"].get(path)
                ]
                expected_changed = any(path.endswith(target_suffix) for path in changed)
                only_expected_changed = (
                    len(changed) == 1 and changed[0].endswith(target_suffix)
                )
                sample = {
                    "elapsed_s": elapsed,
                    "log": ctx.commands[-1]["log"],
                    "public_bmis_unchanged": ids_unchanged(before["public_bmis"], after["public_bmis"]),
                    "recompiled_consumer_object": expected_changed,
                    "only_expected_consumer_object_changed": only_expected_changed,
                    "changed_consumer_objects": changed,
                }
                pair_result["samples"][side] = sample
                order_log.append(side)
            warm.append(pair_result)

        baseline_samples = [item["samples"]["baseline"]["elapsed_s"] for item in warm]
        current_samples = [item["samples"]["current"]["elapsed_s"] for item in warm]
        baseline_median = statistics.median(baseline_samples)
        current_median = statistics.median(current_samples)
        ratio = current_median / baseline_median if baseline_median else None
        warm_passed = (
            ratio is not None
            and ratio <= threshold
            and all(
                side_sample["public_bmis_unchanged"]
                and side_sample["recompiled_consumer_object"]
                and side_sample["only_expected_consumer_object_changed"]
                for pair in warm
                for side_sample in pair["samples"].values()
            )
        )
        consumer_isolation = run_private_probe(
            ctx,
            current["_build_path"],
            current["_source_path"] / "src/velocity.cpp",
            "CMakeFiles/velocity.dir/src/velocity.cpp.o",
            "milestone-3 private velocity implementation rebuild probe",
            "private-velocity-rebuild",
        )
        for condition in (baseline, current):
            condition.pop("_source_path", None)
            condition.pop("_build_path", None)
        scope.update(
            status="complete",
            baseline_cold=baseline,
            current_scalar_cold=current,
            warm_consumer={
                "example": "examples/steady_state.cpp",
                "source_sha256": consumer_sha,
                "initial_current_example_sha256": current_initial_sha,
                "alternating_pairs": WARM_PAIRS,
                "invocation_order": order_log,
                "samples": warm,
                "baseline_median_s": baseline_median,
                "current_median_s": current_median,
                "current_over_baseline_ratio": ratio,
                "maximum_ratio": threshold,
                "passed": warm_passed,
            },
            private_velocity_implementation=consumer_isolation,
            prebuilt_baseline_build={
                "path": str(baseline_build_artifact),
                "used_for_timing": False,
                "reason": "Preserved in place; fresh copies of the exact frozen baseline source were measured.",
            },
            passed=warm_passed and consumer_isolation["passed"],
        )
    except Exception as error:
        scope.update(status="failed", error=str(error))
    return scope


def run_cpu_avx2(ctx: RunContext, root: Path, inputs: dict) -> dict:
    scope = {"status": "running"}
    try:
        condition = configure_and_build_pair(
            ctx,
            "cpu-avx2/current",
            root,
            root / "build/deps/install",
            ("-DVELOCITY_CPU_ISA=avx2", "-DVELOCITY_ENABLE_CUDA=OFF"),
        )
        private = run_private_probe(
            ctx,
            condition["_build_path"],
            condition["_source_path"] / "src/detail/cpu_kernels.cpp",
            "CMakeFiles/velocity.dir/src/detail/cpu_kernels.cpp.o",
            "milestone-3 private AVX2 CPU kernel rebuild probe",
            "private-avx2-kernel-rebuild",
        )
        condition.pop("_source_path", None)
        condition.pop("_build_path", None)
        scope.update(status="complete", avx2_cold=condition, private_cpu_kernel=private,
                     passed=private["passed"])
    except Exception as error:
        scope.update(status="failed", error=str(error))
    return scope


def run_cuda_private(ctx: RunContext, root: Path) -> dict:
    scope = {"status": "running", "gpu_execution": False}
    try:
        if not CUDA_COMPILER.is_file() or not os.access(CUDA_COMPILER, os.X_OK):
            raise RuntimeError(f"required NVCC executable unavailable: {CUDA_COMPILER}")
        source, build = ctx.run_dir / "cuda-private/source", ctx.run_dir / "cuda-private/build"
        source.parent.mkdir(parents=True)
        copy_velocity_source(root, source)
        source_snapshot = velocity_source_identity(source)
        configure_s = configure_velocity(
            ctx,
            source,
            build,
            root / "build/milestone3/cuda-deps",
            (
                "-DVELOCITY_ENABLE_CUDA=ON",
                "-DVELOCITY_CPU_ISA=scalar",
                f"-DCMAKE_CUDA_COMPILER={CUDA_COMPILER}",
                "-DCMAKE_CUDA_ARCHITECTURES=70",
            ),
        )
        cold_s = build_velocity(ctx, build, "cold-build-cuda")
        artifacts = require_nonempty_artifacts(build)
        cuda_probes = []
        for relative, object_suffix, marker, name in (
            (
                "src/cuda/kernels.cu",
                "CMakeFiles/velocity_cuda.dir/src/cuda/kernels.cu.o",
                "milestone-3 private CUDA kernel rebuild probe",
                "private-cuda-kernel-rebuild",
            ),
            (
                "src/cuda.cpp",
                "CMakeFiles/velocity_cuda.dir/src/cuda.cpp.o",
                "milestone-3 private CUDA host implementation rebuild probe",
                "private-cuda-host-rebuild",
            ),
        ):
            cuda_probes.append(run_private_probe(
                ctx, build, source / relative, object_suffix, marker, name
            ))
        scope.update(
            status="complete",
            source=source_snapshot,
            configure_s=configure_s,
            cold_build_s=cold_s,
            cold_total_s=configure_s + cold_s,
            public_bmi_count=len(artifacts["public_bmis"]),
            consumer_object_count=len(artifacts["consumer_objects"]),
            nvcc=str(CUDA_COMPILER),
            cuda_architecture=70,
            private_sources=cuda_probes,
            passed=all(probe["passed"] for probe in cuda_probes),
        )
    except Exception as error:
        scope.update(status="failed", error=str(error))
    return scope


def copy_anndata_source(origin: Path, destination: Path) -> None:
    before = anndata_source_identity(origin)
    destination.mkdir(parents=True)
    for name in ("src", "cmake", "examples"):
        shutil.copytree(origin / name, destination / name)
    for name in ("CMakeLists.txt", "LICENSE"):
        shutil.copy2(origin / name, destination / name)
    copied = anndata_source_identity(destination)
    after = anndata_source_identity(origin)
    if before["sha256"] != copied["sha256"] or before["sha256"] != after["sha256"]:
        raise RuntimeError(f"source changed while making isolated copy of {origin}")


def run_anndata_private(ctx: RunContext, root: Path) -> dict:
    scope = {"status": "running"}
    try:
        origin = Path("/home/tumlinson/anndata.cppm")
        dependency_prefix = origin / "build/dependencies/usr"
        source, build = ctx.run_dir / "anndata-private/source", ctx.run_dir / "anndata-private/build"
        source.parent.mkdir(parents=True)
        copy_anndata_source(origin, source)
        source_snapshot = anndata_source_identity(source)
        argv = [
            "cmake", "-S", str(source), "-B", str(build), "-G", "Ninja",
            "-DCMAKE_CXX_COMPILER=clang++-18",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_PREFIX_PATH={dependency_prefix.resolve()}",
            f"-DANNDATA_DEPENDENCY_PREFIX={dependency_prefix.resolve()}",
            "-DANNDATA_BUILD_TESTS=OFF",
            "-DANNDATA_BUILD_EXAMPLES=ON",
            "-DANNDATA_BUILD_BENCHMARKS=OFF",
            "-DANNDATA_ENABLE_HDF5=ON",
            "-DANNDATA_ENABLE_ZARR=ON",
            "-DANNDATA_ENABLE_CUDA=OFF",
        ]
        configure_s = ctx.execute("configure-anndata-private", argv, ctx.root)
        cold_s = ctx.execute(
            "cold-build-anndata-private",
            ["cmake", "--build", str(build), "--parallel", str(JOBS), "--verbose"],
            ctx.root,
        )
        before_initial = capture_ids(build)
        foundation = {
            "path": next(
                iter(
                    path for path in before_initial["consumer_objects"]
                    if path.endswith("CMakeFiles/anndata_example.dir/examples/foundation.cpp.o")
                ),
                None,
            )
        }
        if not foundation["path"]:
            raise RuntimeError("anndata foundation consumer object was not built")
        probes = []
        for relative, object_suffix, marker, name in (
            (
                "src/h5ad.cpp",
                "CMakeFiles/anndata.dir/src/h5ad.cpp.o",
                "milestone-3 private HDF5 codec rebuild probe",
                "private-hdf5-codec-rebuild",
            ),
            (
                "src/zarr_codecs.cpp",
                "CMakeFiles/anndata.dir/src/zarr_codecs.cpp.o",
                "milestone-3 private Zarr codec rebuild probe",
                "private-zarr-codec-rebuild",
            ),
        ):
            probe = run_private_probe(ctx, build, source / relative, object_suffix, marker, name)
            probes.append(probe)
        scope.update(
            status="complete",
            source=source_snapshot,
            configure_s=configure_s,
            cold_build_s=cold_s,
            cold_total_s=configure_s + cold_s,
            hdf5_enabled=True,
            zarr_enabled=True,
            dependency_prefix=tree_identity(dependency_prefix, ("include", "lib", "share")),
            public_bmi_count=len(before_initial["public_bmis"]),
            foundation_consumer_object=foundation,
            private_sources=probes,
            passed=all(probe["passed"] for probe in probes),
        )
    except Exception as error:
        scope.update(status="failed", error=str(error))
    return scope


def command_version(argv: list[str]) -> str | None:
    try:
        result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, check=False, timeout=10)
        if result.returncode == 0:
            return result.stdout.splitlines()[0] if result.stdout.splitlines() else ""
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def environment_identity() -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.splitlines()[0],
        "cmake": command_version(["cmake", "--version"]),
        "ninja": command_version(["ninja", "--version"]),
        "clang": command_version(["clang++-18", "--version"]),
        "clang_scan_deps": command_version(["clang-scan-deps-18", "--version"]),
        "nvcc": command_version([str(CUDA_COMPILER), "--version"]),
    }


def prepare_inputs(root: Path) -> dict:
    velocity_snapshot = velocity_source_identity(root)
    anndata_root = Path("/home/tumlinson/anndata.cppm")
    return {
        "velocity_live_source": velocity_snapshot,
        "baseline_velocity_source": velocity_source_identity(root / "build/milestone3/baseline/source"),
        "baseline_source_manifest": str(root / "build/milestone3/baseline/source-identity.json"),
        "prebuilt_baseline_build": str(root / "build/milestone3/baseline/build"),
        "prebuilt_baseline_build_artifacts": require_nonempty_artifacts(
            root / "build/milestone3/baseline/build"
        ),
        "baseline_anndata_prefix": tree_identity(
            root / "build/milestone3/baseline/anndata-prefix", ("include", "lib", "share")
        ),
        "current_cpu_anndata_prefix": tree_identity(root / "build/deps/install", ("include", "lib", "share")),
        "current_cuda_anndata_prefix": tree_identity(
            root / "build/milestone3/cuda-deps", ("include", "lib", "share")
        ),
        "anndata_live_source": anndata_source_identity(anndata_root),
        "anndata_dependency_prefix": tree_identity(
            anndata_root / "build/dependencies/usr", ("include", "lib", "share")
        ),
    }


def print_dry_run(root: Path, scopes: tuple[str, ...], output: Path) -> None:
    plan = {
        "execution": "plan_only",
        "required_to_build": "--run",
        "root": str(root),
        "output": str(output),
        "scopes": list(scopes),
        "build_timeout_s": COMMAND_TIMEOUT_S,
        "jobs": JOBS,
        "warm_pairs": WARM_PAIRS if "cpu_pair" in scopes else 0,
        "commands": {
            "cpu_pair": ["fresh baseline configure/build", "fresh current scalar configure/build",
                         "5 alternating paired steady_state consumer rebuilds", "private velocity.cpp rebuild"],
            "cpu_avx2": ["fresh current AVX2 configure/build", "private cpu_kernels.cpp rebuild"],
            "cuda_private": ["fresh CUDA-enabled configure/build with explicit NVCC, SM70",
                             "private CUDA .cu and host implementation rebuilds; no GPU execution"],
            "anndata_private": ["fresh full HDF5/Zarr configure/build with foundation consumer",
                                "private h5ad.cpp and zarr_codecs.cpp rebuilds"],
        },
        "note": "No configure, build, benchmark timing, or output file is produced without --run.",
    }
    print(json.dumps(plan, indent=2))


def append_result(output: Path, run: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    document = {"format": FORMAT, "version": VERSION, "runs": []}
    if output.exists():
        existing = json.loads(output.read_text())
        if existing.get("format") != FORMAT or existing.get("version") != VERSION:
            raise RuntimeError(f"refusing to replace unrelated result file: {output}")
        document = existing
    document["updated_utc"] = utc_now()
    document["runs"].append(run)
    temporary = output.with_name(output.name + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n")
    os.replace(temporary, output)


def validate_output_target(output: Path) -> None:
    if not output.exists():
        return
    existing = json.loads(output.read_text())
    if existing.get("format") != FORMAT or existing.get("version") != VERSION:
        raise RuntimeError(f"refusing to replace unrelated result file: {output}")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true",
                        help="execute isolated configure/build/timing work; requires a quiescence window")
    parser.add_argument("--scope", action="append", choices=ALL_SCOPES,
                        help="select a scope (repeatable); default is all scopes")
    parser.add_argument("--output", type=Path,
                        default=root / "benchmarks/results/module-build-milestone-3.json")
    args = parser.parse_args()
    scopes = tuple(args.scope or ALL_SCOPES)
    output = args.output.resolve()
    if not args.run:
        print_dry_run(root, scopes, output)
        return 0

    validate_output_target(output)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = root / "build/milestone3/build-qualification" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    ctx = RunContext(root, run_dir)
    live_before = velocity_source_identity(root)
    inputs = prepare_inputs(root)
    run = {
        "run_id": run_id,
        "started_utc": utc_now(),
        "requested_scopes": list(scopes),
        "conditions": {
            "build_jobs": JOBS,
            "command_timeout_s": COMMAND_TIMEOUT_S,
            "warm_recompile_pairs": WARM_PAIRS,
            "build_type": "Release",
            "generator": "Ninja",
            "compiler": "clang++-18",
            "cold_build_definition": "fresh isolated build directory; OS filesystem cache uncontrolled",
            "build_only": True,
            "runtime_samples": False,
            "gpu_execution": False,
            "concurrent_measurements": "Root grants quiescence before --run; script does not infer host-wide idleness.",
        },
        "environment": environment_identity(),
        "inputs": inputs,
        "artifact_directory": str(run_dir.relative_to(root)),
        "scopes": {},
        "commands": ctx.commands,
    }
    runners = {
        "cpu_pair": lambda: run_cpu_pair(ctx, root, inputs),
        "cpu_avx2": lambda: run_cpu_avx2(ctx, root, inputs),
        "cuda_private": lambda: run_cuda_private(ctx, root),
        "anndata_private": lambda: run_anndata_private(ctx, root),
    }
    for scope_name in scopes:
        try:
            run["scopes"][scope_name] = runners[scope_name]()
        except Exception as error:
            run["scopes"][scope_name] = {"status": "failed", "error": str(error)}
        run["commands"] = ctx.commands
    live_after = velocity_source_identity(root)
    run["live_source_after_sha256"] = live_after["sha256"]
    run["live_source_unchanged"] = live_before["sha256"] == live_after["sha256"]
    run["prebuilt_baseline_build_artifacts_after"] = require_nonempty_artifacts(
        root / "build/milestone3/baseline/build"
    )
    run["prebuilt_baseline_build_unchanged"] = (
        run["inputs"]["prebuilt_baseline_build_artifacts"]
        == run["prebuilt_baseline_build_artifacts_after"]
    )
    run["ended_utc"] = utc_now()
    run["execution_status"] = "complete" if all(
        value.get("status") == "complete" for value in run["scopes"].values()
    ) else "partial_or_failed"
    run["requested_scope_gates_passed"] = (
        run["execution_status"] == "complete"
        and run["live_source_unchanged"]
        and run["prebuilt_baseline_build_unchanged"]
        and all(value.get("passed", False) for value in run["scopes"].values())
    )
    run["overall_status"] = run["execution_status"]
    run["native_workflow_acceptance"] = "not_asserted; root retains final acceptance"
    append_result(output, run)
    print(json.dumps({
        "run_id": run_id,
        "overall_status": run["overall_status"],
        "requested_scope_gates_passed": run["requested_scope_gates_passed"],
        "scopes": {name: value.get("status") for name, value in run["scopes"].items()},
        "output": str(output),
        "artifact_directory": str(run_dir),
    }, indent=2))
    return 0 if run["requested_scope_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
