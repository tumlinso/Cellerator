#!/usr/bin/env python3
"""Run the bounded KDA packed-versus-SIMT task inside a scheduled GPU job."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CELLERATOR_SPARSE_SUPPORT_BENCH_ADMITTED") != "1":
        parser.error("this runner requires the admitted Device Scheduler job")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "kda_experiments",
               "--task", str(args.task.resolve()),
               "--workspace", str(args.workspace.resolve()),
               "evaluate", "--candidates", "simt", "packed", "--incumbent", "simt"]
    proc = subprocess.run(command, cwd=Path(__file__).resolve().parents[2],
                          text=True, capture_output=True, env=os.environ.copy(),
                          check=False)
    (output_dir / "kda_stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (output_dir / "kda_stderr.txt").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode == 0:
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("KDA evaluator exited successfully without a result envelope")
        envelope = json.loads(lines[-1])
        (output_dir / "kda_result.json").write_text(
            json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        (output_dir / "kda_exit.json").write_text(json.dumps({
            "exit_code": proc.returncode,
            "command": command,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"exit_code": proc.returncode,
                      "output_dir": str(output_dir),
                      "result_path": str(output_dir / "kda_result.json")
                      if proc.returncode == 0 else None}, sort_keys=True))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
