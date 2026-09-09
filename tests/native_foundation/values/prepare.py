#!/usr/bin/env python3
"""Prepare clean-source leased CUDA tests and immutable external build evidence."""
# Adapted from GlassHelix 93988ed2fc7e179fa47fe71736935718e0d90538.
# Build preparation only; no oracle code shared.
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess


def run(argv, cwd=None):
    result = subprocess.run(argv, cwd=cwd, check=True, text=True, capture_output=True, timeout=300)
    return {"argv": [str(x) for x in argv], "stdout": result.stdout, "stderr": result.stderr}


def prepare():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--target", choices=["ce_nf1_v01", "ce_nf1_v02", "ce_nf1_v03", "ce_nf1_v04", "ce_nf1_v05", "ce_nf1_v06", "ce_nf1_v07"], default="ce_nf1_v01")
    parser.add_argument("--reuse-clean-build-receipt", type=Path,
                        help="Reuse a verified build only after test-list, launcher or documentation changes")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[3]
    bindings_path = args.bindings.resolve()
    binding_bytes = bindings_path.read_bytes()
    bindings = json.loads(binding_bytes)
    build = Path(bindings["build_dir"]).resolve()
    evidence = Path(bindings["evidence_dir"]).resolve()
    if build.is_relative_to(source) or evidence.is_relative_to(source):
        raise ValueError("build and evidence must be outside source")
    if not evidence.is_dir():
        raise ValueError("evidence directory must exist")
    status = run(["git", "status", "--porcelain=v1", "--untracked-files=all"], source)["stdout"]
    if status:
        raise ValueError("qualification requires committed clean source")
    head = run(["git", "rev-parse", "HEAD"], source)["stdout"].strip()
    dependency = Path("/home/tumlinson/Baseplane")
    dependency_head = run(["git", "rev-parse", "HEAD"], dependency)["stdout"].strip()
    if run(["git", "status", "--porcelain=v1", "--untracked-files=no"], dependency)["stdout"]:
        raise ValueError("Baseplane tracked dependency must be clean")
    reused = None
    if args.reuse_clean_build_receipt:
        prior_bytes = args.reuse_clean_build_receipt.read_bytes()
        prior = json.loads(prior_bytes)
        if prior['source_root'] != str(source) or prior['build_dir'] != str(build) or prior['dependency_commit'] != dependency_head:
            raise ValueError("prior clean build has different owners")
        for file, key in [(build / args.target, 'executable_sha256'), (build / 'CMakeCache.txt', 'cmake_cache_sha256'),
                          (build / 'compile_commands.json', 'compile_commands_sha256')]:
            if hashlib.sha256(file.read_bytes()).hexdigest() != prior[key]:
                raise ValueError("prior build artifact changed")
        run(['git', 'merge-base', '--is-ancestor', prior['source_commit'], head], source)
        changed = run(['git', 'diff', '--name-only', prior['source_commit'], head], source)['stdout'].splitlines()
        for name in changed:
            allowed = (name.startswith('tests/native_foundation/values/') and
                       (name.endswith('.py') or name.endswith('/CMakeLists.txt')))
            if not allowed and name != 'include/Cellerator/execution/native_value_instance/CAPABILITY.md':
                raise ValueError("compiled-source changes require a clean-first build: " + name)
        reused = {'path': str(args.reuse_clean_build_receipt.resolve()),
                  'sha256': hashlib.sha256(prior_bytes).hexdigest(), 'source_commit': prior['source_commit']}
    commands = [run(["cmake", "--version"]),
                run(["cmake", "-S", str(source / "tests/native_foundation/values"), "-B", str(build),
                     "-DCELLERATOR_ENABLE_CUDA=ON", "-DCMAKE_BUILD_TYPE=Release",
                     "-DCMAKE_CUDA_COMPILER=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc",
                     "-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12", "-DCMAKE_CUDA_ARCHITECTURES=70",
                     "-DCUDAToolkit_ROOT=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9",
                     "-DCELLERATOR_ENABLE_HARDWARE_PROBE=OFF", "-DBASEPLANE_SOURCE_DIR=/home/tumlinson/Baseplane",
                     "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                     "-DNF1_VALUES_RU1_REGRESSIONS=" + ("ON" if args.target == "ce_nf1_v07" else "OFF")])]
    cache = build / "CMakeCache.txt"
    entries = dict(line.split("=", 1) for line in cache.read_text().splitlines()
                   if "=" in line and not line.startswith(("#", "//")))
    if Path(entries["CMAKE_HOME_DIRECTORY:INTERNAL"]).resolve() != source / "tests/native_foundation/values":
        raise ValueError("build source differs from dispatched worktree")
    commands.append(run(["cmake", "--build", str(build), "--target", args.target] +
                        ([] if reused else ["--clean-first"]) + ["--parallel", "2"]))
    commands.append(run(["ctest", "--test-dir", str(build), "--show-only=json-v1"]))
    if run(["git", "rev-parse", "HEAD"], source)["stdout"].strip() != head or run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"], source)["stdout"]:
        raise ValueError("source changed during build")
    if binding_bytes != bindings_path.read_bytes():
        raise ValueError("bindings changed during build")
    if run(["git", "rev-parse", "HEAD"], dependency)["stdout"].strip() != dependency_head or run(["git", "status", "--porcelain=v1", "--untracked-files=no"], dependency)["stdout"]:
        raise ValueError("Baseplane changed during build")
    record = {"dependency_root": str(dependency), "dependency_commit": dependency_head,
              "reused_clean_build_receipt": reused,
              "compile_commands_sha256": hashlib.sha256((build / "compile_commands.json").read_bytes()).hexdigest(),
              "executable_sha256": hashlib.sha256((build / args.target).read_bytes()).hexdigest(),"kind": "nf1-cuda-build-v1", "source_root": str(source), "source_commit": head,
              "source_clean": True, "build_dir": str(build), "bindings_sha256": hashlib.sha256(binding_bytes).hexdigest(),
              "cmake_cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
              "commands": commands, "qualification": "prepared_for_native_gate_not_yet_passed"}
    destination = evidence / ("build-" + head + "-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    with destination.open("x") as output:
        json.dump(record, output, indent=2)
        output.write("\n")
    print(destination)


if __name__ == "__main__":
    prepare()
