#!/usr/bin/env python3
"""Evaluate the registered geometry candidates in the admitted GPU job."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
KDA_ROOT = Path("/home/tumlinson/kda")
PYTHON = "/home/tumlinson/Software/miniconda3/bin/python3"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path,
                        default=HERE / "evidence/kda_workspace")
    parser.add_argument("--task", type=Path, default=HERE / "kda_task.json")
    args = parser.parse_args()
    if os.environ.get("CELLERATOR_SPARSE_SUPPORT_BENCH_ADMITTED") != "1":
        parser.error("run only inside the admitted Device Scheduler job")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(KDA_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    command = [PYTHON, "-m", "kda_experiments", "--task", str(args.task.resolve()),
               "--workspace", str(args.workspace.resolve()), "evaluate",
               "--candidates", "simt", "canonical", "reordered", "hybrid8", "hybrid16",
               "--incumbent", "simt"]
    proc = subprocess.run(command, cwd=REPO, text=True, capture_output=True,
                          env=env, check=False)
    (output / "kda_stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (output / "kda_stderr.txt").write_text(proc.stderr, encoding="utf-8")
    result_path = None
    if proc.returncode == 0:
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("KDA evaluator exited successfully without a result envelope")
        result = json.loads(lines[-1])
        result_path = output / "kda_result.json"
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
    else:
        (output / "kda_exit.json").write_text(json.dumps({
            "exit_code": proc.returncode, "command": command,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"exit_code": proc.returncode, "output_dir": str(output),
                      "result_path": str(result_path) if result_path else None},
                     sort_keys=True))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
