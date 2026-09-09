#!/usr/bin/env python3
"""Build real T03 conformance from clean test source and a pinned production owner."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess


def run(argv, cwd=None):
    result = subprocess.run(argv, cwd=cwd, check=True, capture_output=True, text=True, timeout=900)
    return {"argv": list(map(str, argv)), "stdout": result.stdout, "stderr": result.stderr}


def identity(root, *, tracked_only=False):
    if run(["git", "status", "--porcelain=v1", "--untracked-files=no" if tracked_only else "--untracked-files=all"], root)["stdout"]:
        raise ValueError(f"source must be clean: {root}")
    commit = run(["git", "rev-parse", "HEAD"], root)["stdout"].strip()
    paths = run(["git", "ls-files"], root)["stdout"].splitlines()
    material = {}
    for path in paths:
        file = root / path
        if file.is_file() and (file.suffix in {".hh", ".h", ".hpp", ".cuh", ".cc", ".cpp", ".cu", ".cmake"} or file.name == "CMakeLists.txt"):
            material[path] = hashlib.sha256(file.read_bytes()).hexdigest()
    return {"root": str(root), "commit": commit, "material_sha256": material}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--owner-root", type=Path, required=True)
    parser.add_argument("--owner-commit", required=True)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[3]
    owner = args.owner_root.resolve()
    source_before, owner_before = identity(source), identity(owner)
    baseplane = Path("/home/tumlinson/Baseplane")
    baseplane_before = identity(baseplane, tracked_only=True)
    if owner_before["commit"] != args.owner_commit:
        raise ValueError("production owner differs from authoritative dependency pin")
    binding_bytes = args.bindings.read_bytes()
    binding = json.loads(binding_bytes)
    build, evidence = Path(binding["build_dir"]).resolve(), Path(binding["evidence_dir"]).resolve()
    if build.is_relative_to(source) or evidence.is_relative_to(source):
        raise ValueError("build and evidence must be external")
    evidence.mkdir(parents=True, exist_ok=True)
    toolkit = "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9"
    commands = [run([toolkit + "/bin/nvcc", "--version"]), run(["/usr/bin/g++-12", "--version"]), run(["cmake", "-S", str(source / "tests/native_foundation/conformance"), "-B", str(build),
        "-DNF1_OWNER_ROOT=" + str(owner), "-DCELLERATOR_ENABLE_CUDA=ON", "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CUDA_COMPILER=" + toolkit + "/bin/nvcc", "-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12",
        "-DCMAKE_CUDA_ARCHITECTURES=70", "-DCUDAToolkit_ROOT=" + toolkit,
        "-DCELLERATOR_ENABLE_HARDWARE_PROBE=OFF", "-DBASEPLANE_SOURCE_DIR=/home/tumlinson/Baseplane",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"])]
    cache = build / "CMakeCache.txt"
    entries = dict(line.split("=", 1) for line in cache.read_text().splitlines() if "=" in line and not line.startswith(("#", "//")))
    if Path(entries["CMAKE_HOME_DIRECTORY:INTERNAL"]).resolve() != source / "tests/native_foundation/conformance":
        raise ValueError("CMake home must be the claimed test source")
    if Path(entries["NF1_OWNER_ROOT:PATH"]).resolve() != owner:
        raise ValueError("CMake production owner changed")
    commands.append(run(["cmake", "--build", str(build), "--target", "ce_nf1_t03", "--clean-first", "--parallel", str(args.jobs)]))
    commands.append(run(["ctest", "--test-dir", str(build), "--show-only=json-v1"]))
    if (source_before != identity(source) or owner_before != identity(owner)
            or baseplane_before != identity(baseplane, tracked_only=True) or binding_bytes != args.bindings.read_bytes()):
        raise ValueError("source, owner or bindings changed during build")
    receipt = {"kind": "nf1-pinned-owner-build-v1", "source": source_before, "production_owner": owner_before, "baseplane_dependency": baseplane_before,
        "build_dir": str(build), "commands": commands, "bindings_sha256": hashlib.sha256(binding_bytes).hexdigest(),
        "cmake_cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
        "compile_commands_sha256": hashlib.sha256((build / "compile_commands.json").read_bytes()).hexdigest(),
        "executable_sha256": hashlib.sha256((build / "ce_nf1_t03").read_bytes()).hexdigest(),
        "qualification": "compiled_only_gpu_and_sanitizer_not_yet_executed"}
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = evidence / ("build-" + source_before["commit"] + "-" + stamp + ".json")
    with destination.open("x") as output:
        json.dump(receipt, output, indent=2)
        output.write("\n")
    print(destination)


if __name__ == "__main__":
    main()
