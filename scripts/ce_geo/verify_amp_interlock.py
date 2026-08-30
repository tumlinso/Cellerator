#!/usr/bin/env python3
"""Verify that the loaded CE-AMP workflow remains permission-gated and dormant."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


RUN_ID = "CE-AMP-RUN-V1"
PERMISSION_ID = "CE-AMP-PERMISSION"
CHECKPOINT_ID = "CE-GEO-COMPLETE"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def database_path(root: Path) -> Path:
    candidates = sorted((root / ".git" / "todo-orchestrator").glob("*/state.sqlite3"))
    require(len(candidates) == 1, f"expected one authoritative todo database, found {len(candidates)}")
    return candidates[0]


def has_interlock(task: dict) -> bool:
    dependencies = task.get("depends_on", [])
    checkpoint = any(
        item.get("type") == "checkpoint" and item.get("checkpoint_id") == CHECKPOINT_ID
        for item in dependencies
    )
    permission = any(
        item.get("type") == "decision"
        and item.get("decision_id") == PERMISSION_ID
        and item.get("operator") == "equals"
        and item.get("value") == "granted"
        for item in dependencies
    )
    return checkpoint and permission


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    plan_path = root / "ce-geo-plan.json"
    with plan_path.open(encoding="utf-8") as stream:
        plan = json.load(stream)

    runs = {run["id"]: run for run in plan.get("runs", [])}
    require(RUN_ID in runs, "CE-AMP run is not loaded")
    amp_run = runs[RUN_ID]
    require(amp_run.get("root_task_id") == "CE-AMP-00", "unexpected CE-AMP root task")
    plan_tasks = {task["id"]: task for task in plan.get("tasks", []) if task["id"].startswith("CE-AMP-")}
    require(len(plan_tasks) == 10, f"expected 10 CE-AMP tasks, found {len(plan_tasks)}")
    plan_lanes = {lane["id"]: lane for lane in amp_run.get("lanes", [])}
    require(len(plan_lanes) == 7, f"expected 7 CE-AMP lanes, found {len(plan_lanes)}")

    lane_heads = {}
    for lane_id, lane in plan_lanes.items():
        tasks = lane.get("tasks", [])
        require(tasks, f"{lane_id}: lane has no tasks")
        head = tasks[0]
        require(head in plan_tasks, f"{lane_id}: unknown head task {head}")
        require(has_interlock(plan_tasks[head]), f"{head}: missing checkpoint or human permission interlock")
        lane_heads[lane_id] = head

    database = database_path(root)
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        run = connection.execute(
            "SELECT id, root_task_id, status FROM workflow_runs WHERE id = ?", (RUN_ID,)
        ).fetchone()
        require(run is not None and run["root_task_id"] == "CE-AMP-00", "authoritative CE-AMP run is missing")

        lane_rows = list(
            connection.execute(
                "SELECT id, role, state FROM workflow_lanes WHERE run_id = ? ORDER BY id", (RUN_ID,)
            )
        )
        require({row["id"] for row in lane_rows} == set(plan_lanes), "authoritative lane inventory differs")

        lane_task_rows = list(
            connection.execute(
                "SELECT lane_id, position, task_id, state FROM workflow_lane_tasks "
                "WHERE lane_id GLOB 'CE-AMP-*' ORDER BY lane_id, position"
            )
        )
        require({row["task_id"] for row in lane_task_rows} == set(plan_tasks), "authoritative task queue differs")
        require(all(row["state"] == "queued" for row in lane_task_rows), "a CE-AMP lane task was activated")

        task_rows = list(
            connection.execute(
                "SELECT id, status, result, completion_commit FROM tasks WHERE id GLOB 'CE-AMP-*' ORDER BY id"
            )
        )
        require(len(task_rows) == 10, "authoritative CE-AMP task inventory differs")
        require(all(row["status"] == "planned" for row in task_rows), "a CE-AMP task has started")
        require(all(row["result"] is None and row["completion_commit"] is None for row in task_rows), "CE-AMP has terminal residue")

        permission = connection.execute(
            "SELECT value_json FROM decisions WHERE id = ?", (PERMISSION_ID,)
        ).fetchone()
        require(permission is not None and json.loads(permission["value_json"]) == "not_granted", "CE-AMP permission changed")
        checkpoint = connection.execute(
            "SELECT state FROM checkpoints WHERE id = ?", (CHECKPOINT_ID,)
        ).fetchone()
        require(checkpoint is not None, "CE-GEO completion checkpoint is missing")

        claim_count = connection.execute(
            "SELECT COUNT(*) FROM claims WHERE task_id GLOB 'CE-AMP-*'"
        ).fetchone()[0]
        dispatch_count = connection.execute(
            "SELECT COUNT(*) FROM workflow_dispatches WHERE lane_id GLOB 'CE-AMP-*'"
        ).fetchone()[0]
        capability_count = connection.execute(
            "SELECT COUNT(*) FROM workflow_capabilities WHERE run_id = ?", (RUN_ID,)
        ).fetchone()[0]
        require(claim_count == 0, "CE-AMP claim history is not empty")
        require(dispatch_count == 0, "CE-AMP dispatch history is not empty")
        require(capability_count == 0, "CE-AMP capability history is not empty")
    finally:
        connection.close()

    report = {
        "schema": "CELLERATOR-CE-GEO-AMP-INTERLOCK-AUDIT/1",
        "task_id": "CE-GEO-126",
        "validated": True,
        "run": {"id": RUN_ID, "root_task_id": "CE-AMP-00", "status": run["status"]},
        "inventory": {"lanes": len(plan_lanes), "tasks": len(plan_tasks)},
        "lane_heads": dict(sorted(lane_heads.items())),
        "interlock": {
            "checkpoint": CHECKPOINT_ID,
            "checkpoint_state_at_audit": checkpoint["state"],
            "decision": PERMISSION_ID,
            "required_value": "granted",
            "actual_value": "not_granted",
            "controller": "human only",
        },
        "activity": {
            "actionable_candidates": 0,
            "claims": claim_count,
            "dispatches": dispatch_count,
            "capabilities": capability_count,
            "started_tasks": 0,
        },
        "plan": {"path": "ce-geo-plan.json", "sha256": sha256(plan_path)},
    }
    output = (root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("CE-AMP interlock valid: 7 lane heads, 10 planned tasks, 0 actionable candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
