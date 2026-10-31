#!/usr/bin/env python3
"""Run the isolated biological-meaning smoke inside an admitted scheduler job."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[4]
REPO_SOURCES = (
    "python/cellerator/biology.py",
    "python/cellerator/torch/features.py",
    "python/cellerator/torch/readouts.py",
    "python/cellerator/torch/relation_attention.py",
    "python/cellerator/torch/relations.py",
    "python/cellerator/torch/programs.py",
    "tests/framework/test_biological_gpu.py",
    "planning/nn-biological-meaning-2026-10-31/gpu/scripts/qualify.py",
)
GUIDANCE_SOURCES = (
    Path("/home/tumlinson/.agents/skills/cuda/SKILL.md"),
    Path("/home/tumlinson/.agents/skills/cuda/references/execution/host-execution.md"),
    Path("/home/tumlinson/.agents/skills/cuda/references/specialized/torch-extensions.md"),
    Path("/home/tumlinson/.agents/skills/cuda/references/architectures/volta/router.md"),
    Path("/home/tumlinson/kda/docs/agent-entry.md"),
    Path("/home/tumlinson/kda/prompts/basic-flow.md"),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    return {relative: digest(ROOT / relative) for relative in REPO_SOURCES}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CELLERATOR_TEST_CUDA") != "1":
        raise SystemExit("refusing CUDA runner outside explicit Device Scheduler admission")

    sys.path.insert(0, str(ROOT / "python"))
    import cellerator
    import torch

    initial_hashes = source_hashes()
    guidance_hashes = {str(path): digest(path) for path in GUIDANCE_SOURCES}
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.discover(
            str(ROOT / "tests/framework"), pattern="test_biological_gpu.py"
        )
    )
    device = torch.device("cuda", 0)
    torch.cuda.synchronize(device)
    capability = torch.cuda.get_device_capability(device)
    final_hashes = source_hashes()
    sources_unchanged = initial_hashes == final_hashes
    expected_v100 = capability == (7, 0) and "V100" in torch.cuda.get_device_name(device)
    success = result.wasSuccessful() and sources_unchanged and expected_v100
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
    info = {
        "purpose": "correctness smoke only; no benchmark or biological validity claim",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "success": success,
        "sources_unchanged_during_run": sources_unchanged,
        "expected_v100_sm70": expected_v100,
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device),
        "device_capability": capability,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cellerator_path": list(cellerator.__path__),
        "git_head": head,
        "source_sha256_before": initial_hashes,
        "source_sha256_after": final_hashes,
        "guidance_sha256": guidance_hashes,
        "test_failures": [
            {"test": str(test), "traceback": traceback}
            for test, traceback in result.failures
        ],
        "test_errors": [
            {"test": str(test), "traceback": traceback}
            for test, traceback in result.errors
        ],
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))
    return int(not success)


if __name__ == "__main__":
    raise SystemExit(main())
