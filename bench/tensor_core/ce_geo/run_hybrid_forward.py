#!/usr/bin/env python3
"""Build, run, and validate the CE-GEO-113 hybrid-forward campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
BINARY = ROOT / "build/ceGeoHybridForward"


def run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=ROOT, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=None)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "bench/ce_geo/evidence/sm70_forward_complete_cost.jsonl",
    )
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    BINARY.parent.mkdir(parents=True, exist_ok=True)
    run([
        "nvcc", "-std=c++17", "-arch=sm_70", "-O3", "-lineinfo",
        "-Xcompiler=-Wall,-Wextra,-Werror", "-I.", "-Iinclude",
        "bench/tensor_core/ce_geo/hybrid_forward.cu",
        "src/compute/architecture/providers/nvidia/sm70/relation_apply_hybrid.cu",
        "src/compute/architecture/providers/nvidia/sm70/relation_apply_n64.cu",
        "src/compute/architecture/providers/nvidia/sm70/residual.cu",
        "src/compute/architecture/providers/nvidia/sm70/value_pack.cu",
        "src/compute/projection/mma_residual_builder.cc",
        "-lcusparse", "-lcudart", "-o", str(BINARY),
    ])
    run([str(BINARY), "--output", str(output), "--warmups", "3",
         "--repeats", "11"])
    validated = run([
        sys.executable, "bench/ce_geo/harness/run.py",
        "--campaign", "sm70-hybrid-forward", "--output", str(output),
    ])
    metric = json.loads(validated.stdout)
    print(json.dumps(metric, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
