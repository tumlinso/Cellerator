#!/usr/bin/env python3
"""Run bounded CUDA sanitizer checks on a frozen sparse-support binary.

Invoke only inside an admitted Device Scheduler job. No GPU selection,
allocation, build, or admission is performed by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--sanitizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.expanduser().resolve()
    sanitizer = args.sanitizer.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not binary.is_file() or not sanitizer.is_file():
        parser.error("binary or sanitizer executable is missing")
    if output.exists():
        parser.error(f"refusing to overwrite evidence: {output}")

    result: dict[str, object] = {
        "schema": "cellerator.sparse_support.cuda_sanitizer_run.v1",
        "status": "running",
        "started_unix_ns": time.time_ns(),
        "binary_path": str(binary),
        "binary_sha256": sha256(binary),
        "sanitizer_path": str(sanitizer),
        "sanitizer_sha256": sha256(sanitizer),
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "tools": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    for tool in ("memcheck", "racecheck"):
        command = [
            str(sanitizer), "--tool", tool, "--error-exitcode", "99",
            str(binary), "--case-id", "source_skew_0_9_sanitized",
            "--edges", "2176", "--cells", "64", "--entities", "256",
            "--gate-density", "0.5", "--source-skew", "0.9",
            "--dirty-fraction", "0.01", "--seed", "20261027",
            "--warmup", "0", "--samples", "3", "--iterations", "1",
        ]
        completed = subprocess.run(command, check=False, text=True,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        record = {
            "tool": tool,
            "command": command,
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        result["tools"].append(record)
        result["status"] = "running"
        result["updated_unix_ns"] = time.time_ns()
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        if completed.returncode != 0:
            result["status"] = "failed"
    result["status"] = "complete" if all(
        item["exit_code"] == 0 for item in result["tools"]
    ) else "failed"
    result["completed_unix_ns"] = time.time_ns()
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
