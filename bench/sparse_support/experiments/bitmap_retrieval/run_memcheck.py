#!/usr/bin/env python3
"""Capture a bounded memcheck run for the frozen bitmap retrieval probe."""

import argparse
import hashlib
import json
import pathlib
import subprocess
import time


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=pathlib.Path)
    parser.add_argument("--sanitizer", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    sanitizer = args.sanitizer.resolve()
    output = args.output.resolve()
    if output.exists():
        parser.error(f"refusing to overwrite evidence: {output}")
    command = [
        str(sanitizer), "--tool", "memcheck", "--error-exitcode", "99",
        str(binary), "--samples", "3", "--iterations", "1",
    ]
    result = subprocess.run(command, check=False, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    record = {
        "schema": "cellerator.sparse_support.bitmap_retrieval_memcheck.v1",
        "started_unix_ns": time.time_ns(),
        "command": command,
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "sanitizer": str(sanitizer),
        "sanitizer_sha256": sha256(sanitizer),
        "runner": str(pathlib.Path(__file__).resolve()),
        "runner_sha256": sha256(pathlib.Path(__file__).resolve()),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "completed_unix_ns": time.time_ns(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output), "binary_sha256": record["binary_sha256"],
        "returncode": result.returncode,
    }, sort_keys=True))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
