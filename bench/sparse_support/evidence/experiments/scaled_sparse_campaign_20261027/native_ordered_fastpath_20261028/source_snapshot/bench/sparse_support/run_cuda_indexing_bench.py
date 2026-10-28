#!/usr/bin/env python3
"""Verify a frozen sparse fixture and invoke the native indexing benchmark.

Host-oracle runs are allowed without scheduler admission. CUDA timings require
an admitted Device Scheduler job and always retain the exact fixture manifest,
generator, binary, command, and measured output identities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]
GENERATOR = REPO / "bench/sparse_support/scaled_fixtures.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle(bundle: Path) -> tuple[dict, str, int, int]:
    manifest_path = bundle / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != "cellerator.scaled_sparse_fixture_bundle.v1":
        raise RuntimeError(f"unrecognized fixture bundle: {manifest_path}")
    verified: dict[str, str] = {}
    verified_bytes = 0
    for descriptor in manifest.get("arrays", {}).values():
        name = descriptor["file"]
        expected = descriptor["sha256"]
        if name in verified:
            if verified[name] != expected:
                raise RuntimeError(f"aliases disagree about array identity for {name}")
            continue
        path = bundle / name
        if path.stat().st_size != int(descriptor["nbytes"]):
            raise RuntimeError(f"fixture array byte count changed: {name}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"fixture array hash mismatch: {name}")
        verified[name] = actual
        verified_bytes += path.stat().st_size
    return manifest, hashlib.sha256(manifest_bytes).hexdigest(), verified_bytes, len(verified)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--edges", type=int, required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--entities", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--host-oracle-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.host_oracle_only and os.environ.get(
            "CELLERATOR_SPARSE_SUPPORT_BENCH_ADMITTED") != "1":
        parser.error("CUDA timing requires the admitted Device Scheduler job")
    binary = args.binary.resolve(strict=True)
    bundle = args.fixture_dir.resolve(strict=True)
    before, manifest_sha, checked_bytes, unique_array_count = verify_bundle(bundle)
    generator_sha = sha256_file(GENERATOR)
    binary_sha = sha256_file(binary)
    command = [str(binary), "--case-id", args.case_id, "--edges", str(args.edges),
               "--cells", str(args.cells), "--entities", str(args.entities),
               "--fixture-dir", str(bundle)]
    if args.host_oracle_only:
        command.extend(["--host-oracle-only", "--warmup", "0", "--samples", "1",
                        "--iterations", "1"])
    else:
        command.extend(["--warmup", str(args.warmup), "--samples", str(args.samples),
                        "--iterations", str(args.iterations)])
    proc = subprocess.run(command, cwd=REPO, text=True, capture_output=True,
                          check=False, env=os.environ.copy())
    if proc.returncode != 0:
        raise RuntimeError(f"native benchmark failed ({proc.returncode})\n{proc.stderr}")
    after, after_manifest_sha, after_checked_bytes, after_unique_array_count = verify_bundle(bundle)
    after_binary_sha = sha256_file(binary)
    after_generator_sha = sha256_file(GENERATOR)
    if (manifest_sha != after_manifest_sha or before != after
            or checked_bytes != after_checked_bytes
            or unique_array_count != after_unique_array_count
            or binary_sha != after_binary_sha or generator_sha != after_generator_sha):
        raise RuntimeError("fixture bundle changed while benchmark was running")
    output_lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if len(output_lines) != 1:
        raise RuntimeError("native benchmark did not emit exactly one JSON result")
    result = json.loads(output_lines[0])
    result["input_identity"] = {
        "fixture_manifest_sha256": manifest_sha,
        "fixture_generator_sha256": generator_sha,
        "fixture_verified_unique_array_bytes": checked_bytes,
        "fixture_unique_array_files": unique_array_count,
        "benchmark_binary_sha256": binary_sha,
        "benchmark_source_sha256": sha256_file(Path(__file__).resolve()),
        "fixture_generator_source_sha256_at_run": generator_sha,
        "command": command,
    }
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
