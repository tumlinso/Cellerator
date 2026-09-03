#!/usr/bin/env python3
"""Canonical terminal front-end for this pre-ledger package.

The wrapper has read-only subcommands and explicit mutating subcommands.
It never guesses Todo-Orchestrator verbs: it executes only command forms
captured in evidence/todo_cli_resolution.json during package construction.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
RESOLUTION = PACKAGE / "evidence" / "todo_cli_resolution.json"
PLAN = PACKAGE / "machine" / "cellerator-compiler-part1.todo-plan.json"
VALIDATOR = PACKAGE / "scripts" / "validate_package.py"
SOURCE_ROOT_DEFAULT = Path(os.environ.get("CELLERATOR_SOURCE_ROOT", PACKAGE.parents[2] if len(PACKAGE.parents) > 2 else "."))

MUTATING = {"apply", "activate"}
CONFIRMATIONS = {
    "apply": "APPLY-CELLERATOR-COMPILER-PART1",
    "activate": "ACTIVATE-CE-CCP1-RUN-V1",
}

def run(argv: list[str], cwd: Path) -> int:
    print("+", " ".join(shlex.quote(x) for x in argv))
    return subprocess.run(argv, cwd=str(cwd)).returncode

def run_shell(command: str, cwd: Path) -> int:
    env = {
        **os.environ,
        "PLAN": str(PLAN.resolve()),
        "PROJECT_ROOT": str(cwd.resolve()),
        "RUN_ID": "CE-CCP1-RUN-V1",
    }
    print("+", command)
    return subprocess.run(["bash", "-lc", command], cwd=str(cwd), env=env).returncode

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate, preview, or manually bootstrap the proposed Cellerator compiler Part One Todo program."
    )
    parser.add_argument("operation", choices=["validate", "preview", "collisions", "verify", "apply", "activate"])
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT_DEFAULT)
    parser.add_argument("--confirm", default=None)
    parser.add_argument("--show-command", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    resolution = json.loads(RESOLUTION.read_text())
    commands = resolution.get("commands", {})

    # Package validation is mandatory before every operation.
    validation = [
        sys.executable, str(VALIDATOR),
        "--package-root", str(PACKAGE),
        "--source-root", str(source_root),
        "--require-manifest",
    ]
    if args.operation in MUTATING:
        validation.append("--check-live-preconditions")
    rc = run(validation, source_root)
    if rc:
        print("Package validation failed. No Todo command was executed.", file=sys.stderr)
        return rc

    command = commands.get(args.operation)
    if args.operation == "validate":
        # The package validator is itself the canonical first validation.
        # Run the installed Project Control/Todo validator too when one was resolved.
        command = commands.get("validate")
        if not command:
            print("Package validation passed. No separate installed plan-validator command was resolved.")
            return 0

    if not command:
        print(
            f"No exact installed command was resolved for {args.operation!r}. "
            f"Inspect {RESOLUTION} and rerun tooling discovery; the wrapper refuses to guess.",
            file=sys.stderr,
        )
        return 2

    if args.show_command:
        print(command)
        return 0

    if args.operation in MUTATING:
        required = CONFIRMATIONS[args.operation]
        if args.confirm != required:
            print(
                f"{args.operation} is mutating. Re-run with --confirm {required!r} "
                "only after revalidating the live Project Control preview.",
                file=sys.stderr,
            )
            return 3

    return run_shell(command, source_root)

if __name__ == "__main__":
    raise SystemExit(main())
