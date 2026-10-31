#!/usr/bin/env python3
"""Run serial KDA builds or one admitted attention CUDA comparison packet."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts/validation"))
from kda_autobench_artifacts import prepare_into_autobench, verify_autobench_build


KDA_REL = Path("planning/v100-framework-2026-10-29/attention/launch_geometry")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True,
                        help="Autobench snapshot containing the trusted sources and KDA task")
    parser.add_argument("--workspace", type=Path, required=True,
                        help="persistent KDA experiment workspace")
    parser.add_argument("--autobench-build", type=Path, required=True,
                        help="managed Autobench build output for prepared binaries")
    parser.add_argument("--mode", choices=("prepare", "evaluate"), required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    task_path = source / KDA_REL / "task.json"
    workspace = args.workspace.resolve()
    base_manifest_path = source / KDA_REL / "base-source-manifest.json"
    if args.mode == "prepare":
        prepare_into_autobench(
            source=source, task_path=task_path, workspace=workspace,
            autobench_build=args.autobench_build,
            base_manifest_path=base_manifest_path,
            family="attention", candidate_ids=["baseline", "one_cta_per_row"],
        )
        return 0
    verify_autobench_build(
        source=source, task_path=task_path, workspace=workspace,
        autobench_build=args.autobench_build,
        base_manifest_path=base_manifest_path,
        family="attention", candidate_ids=["baseline", "one_cta_per_row"],
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = "/home/tumlinson/kda" + os.pathsep + env.get("PYTHONPATH", "")
    env["CELLERATOR_KDA_SOURCE_ROOT"] = str(source)
    command = [sys.executable, "-m", "kda_experiments", "--task", str(task_path),
               "--workspace", str(workspace), "evaluate", "--candidates",
               "one_cta_per_row", "--incumbent", "baseline"]
    print("$", " ".join(command), flush=True)
    return subprocess.run(command, cwd=source, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
