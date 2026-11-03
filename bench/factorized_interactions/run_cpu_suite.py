#!/usr/bin/env python3
"""Run the CPU oracle, wire, host guard and preparation checks in a build."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], cwd: Path, env: dict[str, str], *,
        timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, env=env, text=True,
                          capture_output=True, check=False, timeout=timeout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    build = args.build.resolve(strict=True)
    args.output = args.output.resolve()
    manifest = build / "factorized_interactions_build_manifest.json"
    if not manifest.is_file():
        raise RuntimeError("source-bound build manifest is missing")
    manifest_bytes = manifest.read_bytes()
    metadata = json.loads(manifest_bytes)
    output_root = args.output.parent
    scratch = output_root / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    host_test_output = scratch / "host-tests"
    host_test_output.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TMPDIR"] = str(scratch / "temp")
    env["TEMP"] = env["TMPDIR"]
    env["TMP"] = env["TMPDIR"]
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    test_path = str(source / "tests" / "factorized_interactions")
    env["PYTHONPATH"] = test_path + os.pathsep + env.get("PYTHONPATH", "")
    python = run([
        sys.executable, "-m", "unittest", "discover", "-s",
        str(source / "tests" / "factorized_interactions"), "-p", "test_*.py", "-v",
    ], source, env)
    (host_test_output / "python-tests.stdout.log").write_text(
        python.stdout, encoding="utf-8")
    (host_test_output / "python-tests.stderr.log").write_text(
        python.stderr, encoding="utf-8")
    if python.returncode:
        sys.stderr.write(python.stderr)
        return python.returncode
    python_match = re.search(r"Ran (\d+) tests? in ([0-9.]+)s", python.stderr)
    if not python_match or "OK" not in python.stderr:
        raise RuntimeError("Python oracle suite did not report a complete passing run")

    host_tests = (
        ("host_contract_test", "cuda_host_contract"),
        ("preparation_test", "preparation"),
        ("pattern_binding_test", "pattern_binding"),
    )
    host_results = []
    for binary_key, label in host_tests:
        entry = metadata.get("binaries", {}).get(binary_key)
        if not isinstance(entry, dict):
            raise RuntimeError(f"build manifest omits host test binary {binary_key}")
        binary = (build / entry["path"]).resolve(strict=True)
        if not binary.is_file() or binary.stat().st_size != entry.get("bytes") or \
                sha256_file(binary) != entry.get("sha256"):
            raise RuntimeError(f"host test binary differs from build manifest: {binary_key}")
        result = run([str(binary)], host_test_output, env, timeout=60)
        (host_test_output / f"{label}.stdout.log").write_text(
            result.stdout, encoding="utf-8")
        (host_test_output / f"{label}.stderr.log").write_text(
            result.stderr, encoding="utf-8")
        if result.returncode:
            sys.stderr.write(result.stdout)
            sys.stderr.write(result.stderr)
            return result.returncode
        host_results.append({
            "name": label,
            "status": "passed",
            "binary_sha256": entry["sha256"],
        })

    result = {
        "schema": "cellerator.factorized_interactions.host_suite.v2",
        "python_tests": int(python_match.group(1)),
        "python_status": "passed",
        "host_binary_tests": host_results,
        "host_binary_tests_passed": len(host_results),
        "host_binary_tests_status": "passed",
        "build_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_closure_sha256": metadata["source_closure_sha256"],
        "runner_sha256": metadata["binaries"]["runner"]["sha256"],
    }
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
