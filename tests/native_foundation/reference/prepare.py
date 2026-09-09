#!/usr/bin/env python3
"""Prepare clean-source independent host tests and immutable external build evidence."""
# Adapted from GlassHelix 93988ed2fc7e179fa47fe71736935718e0d90538.
# Only host configure options and executable target differ; no oracle code shared.
import argparse
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
    commands = [run(["cmake", "--version"]),
                run(["cmake", "-S", str(source), "-B", str(build),
                     "-DCELLERATOR_BUILD_NATIVE_FOUNDATION_TESTS=ON", "-DCELLERATOR_ENABLE_CUDA=OFF", "-DCMAKE_BUILD_TYPE=Release"])]
    cache = build / "CMakeCache.txt"
    entries = dict(line.split("=", 1) for line in cache.read_text().splitlines()
                   if "=" in line and not line.startswith(("#", "//")))
    if Path(entries["CMAKE_HOME_DIRECTORY:INTERNAL"]).resolve() != source:
        raise ValueError("build source differs from dispatched worktree")
    commands.append(run(["cmake", "--build", str(build), "--target", "ce_nf1_t01", "--parallel", "2"]))
    commands.append(run(["ctest", "--test-dir", str(build), "--show-only=json-v1"]))
    if run(["git", "rev-parse", "HEAD"], source)["stdout"].strip() != head or run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"], source)["stdout"]:
        raise ValueError("source changed during build")
    if binding_bytes != bindings_path.read_bytes():
        raise ValueError("bindings changed during build")
    record = {"kind": "nf1-independent-build-v1", "source_root": str(source), "source_commit": head,
              "source_clean": True, "build_dir": str(build), "bindings_sha256": hashlib.sha256(binding_bytes).hexdigest(),
              "cmake_cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
              "commands": commands, "qualification": "prepared_for_native_gate_not_yet_passed"}
    destination = evidence / ("build-" + head + ".json")
    with destination.open("x") as output:
        json.dump(record, output, indent=2)
        output.write("\n")
    print(destination)


if __name__ == "__main__":
    prepare()
