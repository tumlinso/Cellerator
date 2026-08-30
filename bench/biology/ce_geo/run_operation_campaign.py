#!/usr/bin/env python3
"""Build, run, and validate the leased CE-GEO-118 operation campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
BINARY = ROOT / "build/ceGeoBiologyOperationCampaign"


def run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=ROOT, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compile-only", action="store_true")
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    BINARY.parent.mkdir(parents=True, exist_ok=True)
    compiled = run([
        "nvcc", "-std=c++17", "-arch=sm_70", "-O3", "-lineinfo",
        "-Xcompiler=-Wall,-Wextra,-Werror", "-I.", "-Iinclude",
        "bench/biology/ce_geo/operation_campaign.cu",
        "src/compute/architecture/providers/nvidia/sm70/relation_apply_n64.cu",
        "-lcudart", "-o", str(BINARY),
    ])
    if compiled.stderr:
        print(compiled.stderr, file=sys.stderr, end="")
    if arguments.compile_only:
        print(json.dumps({"compile_valid": 1}, sort_keys=True))
        return 0
    executed = run([str(BINARY), "--output", str(output), "--warmups", "3",
                    "--repeats", "11"])
    if executed.stderr:
        print(executed.stderr, file=sys.stderr, end="")
    validated = run([
        sys.executable, "bench/ce_geo/harness/run.py", "--campaign",
        "biology-operations", "--output", str(output),
    ])
    print(json.dumps(json.loads(validated.stdout), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
