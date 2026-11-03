#!/usr/bin/env python3
"""Run KDA's direct backend inside an existing Autobench Scheduler attempt.

The recorded context is an observation, not a Scheduler allocation or cleanup
receipt. Acceptance requires independently matching the completed outer receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import uuid
from typing import Mapping


SCHEMA = "cellerator.factorized_interactions.outer_execution_context.v1"
CONTEXT_FILE = "outer-execution-context.json"
TRUSTED_ID = "in_job_entrypoint"


class ContextError(ValueError):
    """The current process cannot establish the requested in-job context."""


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scheduler_unit(attempt_id: str) -> str:
    # Same deterministic naming as Device Scheduler's systemd adapter.
    digest = hashlib.sha256(attempt_id.encode("utf-8")).hexdigest()[:32]
    return f"device-scheduler-attempt-{digest}.service"


def validate_context(env: Mapping[str, str], cgroup_text: str,
                     expected_uuid: str) -> dict[str, object]:
    if not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", expected_uuid):
        raise ContextError("expected UUID must be one explicit NVIDIA GPU UUID")
    try:
        uuid.UUID(expected_uuid[4:])
    except ValueError as exc:
        raise ContextError("invalid expected GPU UUID") from exc
    job_id = env.get("DEVICE_SCHEDULER_JOB_ID", "").strip()
    attempt_id = env.get("DEVICE_SCHEDULER_ATTEMPT_ID", "").strip()
    if not job_id or not attempt_id:
        raise ContextError("missing inherited Device Scheduler job or attempt ID")
    if env.get("AUTOBENCH_SCHEDULER_JOB_ID", "").strip() != job_id or \
            env.get("AUTOBENCH_SCHEDULER_ATTEMPT_ID", "").strip() != attempt_id:
        raise ContextError("Autobench and Device Scheduler job/attempt IDs differ")
    if env.get("CUDA_VISIBLE_DEVICES", "").strip() != expected_uuid:
        raise ContextError("CUDA visibility does not match the single expected GPU UUID")

    lines = [line.split(":", 2) for line in cgroup_text.splitlines()]
    unified = [fields[2] for fields in lines
               if len(fields) == 3 and fields[0] == "0" and fields[1] == ""]
    if len(unified) != 1:
        raise ContextError("expected one cgroup v2 process membership")
    cgroup = unified[0]
    path = PurePosixPath(cgroup)
    unit_name = scheduler_unit(attempt_id)
    if not path.is_absolute() or ".." in path.parts or path.parts.count(unit_name) != 1:
        raise ContextError("process is outside the exact Scheduler attempt cgroup")
    return {
        "schema": SCHEMA,
        "status": "observed",
        "scheduler_job_id": job_id,
        "scheduler_attempt_id": attempt_id,
        "device_uuid": expected_uuid,
        "cgroup": cgroup,
        "unit_name": unit_name,
    }


def validate_task(task: Mapping[str, object], task_path: Path,
                  expected_uuid: str, entrypoint: Path) -> None:
    execution = task.get("execution")
    target = task.get("target")
    binding = task.get("factorized_interaction")
    if not isinstance(execution, dict) or execution.get("mode") != "direct":
        raise ContextError("KDA task must use its direct backend")
    if not isinstance(target, dict) or target.get("kind") != "cuda" or \
            target.get("expected_device_uuid") != expected_uuid:
        raise ContextError("KDA CUDA target and requested GPU UUID do not match")
    if not isinstance(binding, dict) or binding.get("expected_device_uuid") != expected_uuid:
        raise ContextError("task and requested GPU UUID do not match")
    inputs = task.get("trusted_inputs")
    matches = [item for item in inputs if isinstance(item, dict) and
               item.get("id") == TRUSTED_ID] if isinstance(inputs, list) else []
    if len(matches) != 1 or not isinstance(matches[0].get("path"), str):
        raise ContextError("task must include this entrypoint once as a trusted input")
    path = Path(matches[0]["path"])
    if not path.is_absolute():
        path = task_path.parent / path
    if path.is_symlink() or path.resolve(strict=True) != entrypoint.resolve(strict=True):
        raise ContextError("task entrypoint identity does not match the executing file")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--expected-uuid", required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--incumbent", required=True)
    args = parser.parse_args(argv)
    try:
        task_path = args.task.resolve(strict=True)
        workspace = args.workspace.resolve(strict=True)
        if not workspace.is_dir() or not task_path.is_file():
            raise ContextError("task must be a file and workspace a directory")
        if args.incumbent in args.candidates or len(args.candidates) != len(set(args.candidates)):
            raise ContextError("challenger candidates must be unique and exclude the incumbent")
        task = json.loads(task_path.read_text(encoding="utf-8"))
        if not isinstance(task, dict):
            raise ContextError("KDA task must be a JSON object")
        entrypoint = Path(__file__).resolve(strict=True)
        validate_task(task, task_path, args.expected_uuid, entrypoint)
        context = validate_context(os.environ, Path("/proc/self/cgroup").read_text(encoding="ascii"),
                                   args.expected_uuid)
        context.update({"task_sha256": digest_file(task_path),
                        "entrypoint_sha256": digest_file(entrypoint),
                        "pid": os.getpid()})
        record_path = workspace / CONTEXT_FILE
        # Exclusive creation prevents accidental replacement of earlier evidence.
        with record_path.open("x", encoding="utf-8") as record:
            json.dump(context, record, sort_keys=True, indent=2)
            record.write("\n")
            record.flush()
            os.fsync(record.fileno())
        command = [sys.executable, "-m", "kda_experiments", "--task", str(task_path),
                   "--workspace", str(workspace), "evaluate", "--candidates",
                   *args.candidates, "--incumbent", args.incumbent]
        # exec preserves the original process/cgroup and inherited environment.
        os.execv(sys.executable, command)
    except (ContextError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"in-job KDA context rejected: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
