#!/usr/bin/env python3
"""Run KDA CPU preparation or GPU evaluation from an admitted Autobench packet."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from kda_autobench_artifacts import prepare_into_autobench, verify_autobench_build


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True,
                        help="Autobench source snapshot containing the KDA task")
    parser.add_argument("--workspace", type=Path, required=True,
                        help="persistent KDA experiment workspace")
    parser.add_argument("--autobench-build", type=Path, required=True,
                        help="managed Autobench build output for prepared binaries")
    parser.add_argument("--mode", choices=("prepare", "evaluate"), required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    workspace = args.workspace.resolve()
    task = source / "planning/v100-framework-2026-10-29/kda/packed_embedding/task.json"
    if not task.is_file():
        raise FileNotFoundError(task)
    env = dict(os.environ)
    env["PYTHONPATH"] = "/home/tumlinson/kda" + os.pathsep + env.get("PYTHONPATH", "")
    env["CELLERATOR_KDA_SOURCE_ROOT"] = str(source)
    if args.mode == "prepare":
        prepare_into_autobench(
            source=source, task_path=task, workspace=workspace,
            autobench_build=args.autobench_build,
            base_manifest_path=source / "planning/v100-framework-2026-10-29/kda/packed_embedding/base-source-manifest.json",
            family="packed_embedding", candidate_ids=["baseline", "warp_backward"],
        )
    else:
        verify_autobench_build(
            source=source, task_path=task, workspace=workspace,
            autobench_build=args.autobench_build,
            base_manifest_path=source / "planning/v100-framework-2026-10-29/kda/packed_embedding/base-source-manifest.json",
            family="packed_embedding", candidate_ids=["baseline", "warp_backward"],
        )
        command = [sys.executable, "-m", "kda_experiments", "--task", str(task),
                   "--workspace", str(workspace)]
        job = env.get("DEVICE_SCHEDULER_JOB_ID")
        attempt = env.get("DEVICE_SCHEDULER_ATTEMPT_ID")
        visible = [value.strip() for value in env.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
        if not job or not attempt or len(visible) != 1 or not visible[0].startswith("GPU-"):
            raise RuntimeError("KDA evaluation requires one UUID-bound, scheduler-admitted Autobench attempt")
        command += ["evaluate", "--candidates", "warp_backward", "--incumbent", "baseline"]
        print("$", " ".join(command), flush=True)
        return subprocess.run(command, cwd=source, env=env, check=False).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
