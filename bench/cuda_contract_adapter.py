#!/usr/bin/env python3
"""Run one existing benchmark and expose one numeric result as JSON."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric-pattern", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a benchmark command is required after --")

    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    sys.stderr.write(completed.stderr)
    sys.stderr.write(completed.stdout)
    if completed.returncode != 0:
        return completed.returncode

    match = re.search(args.metric_pattern, completed.stdout, re.MULTILINE | re.DOTALL)
    if match is None or match.lastindex != 1:
        print("cuda_contract_adapter: metric pattern did not capture one value", file=sys.stderr)
        return 2
    try:
        metric = float(match.group(1))
    except ValueError:
        print("cuda_contract_adapter: captured metric is not numeric", file=sys.stderr)
        return 2
    print(json.dumps({"metric": metric}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
