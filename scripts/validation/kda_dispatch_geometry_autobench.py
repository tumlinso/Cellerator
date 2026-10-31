#!/usr/bin/env python3
"""Run serial KDA builds or an admitted dispatch CUDA comparison packet."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from kda_autobench_artifacts import prepare_into_autobench, verify_autobench_build


ROOT = Path("/home/tumlinson/Cellerator")
KDA_REL = Path("planning/v100-framework-2026-10-29/dispatch/launch_geometry")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True,
                        help="Autobench snapshot containing the source closure and KDA task")
    parser.add_argument("--workspace", type=Path, required=True,
                        help="persistent KDA experiment workspace")
    parser.add_argument("--autobench-build", type=Path, required=True,
                        help="managed Autobench build output for prepared binaries")
    parser.add_argument("--mode", choices=("prepare", "evaluate"), required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    workspace = args.workspace.resolve()
    task_path = source / KDA_REL / "task.json"
    if not task_path.is_file():
        raise FileNotFoundError(task_path)
    if args.mode == "prepare":
        prepare_into_autobench(
            source=source, task_path=task_path, workspace=workspace,
            autobench_build=args.autobench_build,
            base_manifest_path=source / KDA_REL / "base-source-manifest.json",
            family="dispatch", candidate_ids=["baseline", "one_cta_per_selection"],
        )
        return 0
    verify_autobench_build(
        source=source, task_path=task_path, workspace=workspace,
        autobench_build=args.autobench_build,
        base_manifest_path=source / KDA_REL / "base-source-manifest.json",
        family="dispatch", candidate_ids=["baseline", "one_cta_per_selection"],
    )
    task = json.loads(task_path.read_text(encoding="utf-8"))
    manifest = json.loads((source / KDA_REL / "base-source-manifest.json").read_text(encoding="utf-8"))
    actual_root = Path(manifest["source_root"]).resolve(strict=True)
    for trusted in task["trusted_inputs"]:
        if trusted["role"] == "toolchain":
            continue
        path = Path(trusted["path"])
        relative = path.relative_to(ROOT) if path.is_absolute() else task_path.parent.relative_to(source) / path
        if sha(actual_root / relative) != sha(source / relative):
            raise RuntimeError(f"trusted task input differs from the captured Autobench snapshot: {relative}")
        job = os.environ.get("DEVICE_SCHEDULER_JOB_ID")
        attempt = os.environ.get("DEVICE_SCHEDULER_ATTEMPT_ID")
        visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
        if not job or not attempt or len(visible) != 1 or not visible[0].startswith("GPU-"):
            raise RuntimeError("KDA evaluation requires one UUID-bound, scheduler-admitted Autobench attempt")

    env = dict(os.environ)
    env["PYTHONPATH"] = "/home/tumlinson/kda" + os.pathsep + env.get("PYTHONPATH", "")
    env["CELLERATOR_KDA_SOURCE_ROOT"] = str(source)
    command = [sys.executable, "-m", "kda_experiments", "--task", str(task_path),
               "--workspace", str(workspace)]
    command += ["evaluate", "--candidates", "one_cta_per_selection", "--incumbent", "baseline"]
    print("$", " ".join(command), flush=True)
    return subprocess.run(command, cwd=source, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
