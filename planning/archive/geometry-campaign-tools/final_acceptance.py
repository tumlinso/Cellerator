#!/usr/bin/env python3
"""Publish the deterministic final CE-GEO Volta acceptance audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


FINAL_TASKS = {"CE-GEO-125", "CE-GEO-126", "CE-GEO-127"}
REQUIRED_EVIDENCE = (
    "baseline/baseline.json",
    "full_volta_validation.json",
    "fusion_evaluation.json",
    "micro/final_volta_microarchitecture.json",
    "numerical_policy.json",
    "preprint/summary.json",
    "sanitizer/campaign.json",
    "sm70_forward_disposition.json",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    require(isinstance(value, dict), f"{path}: expected a JSON object")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def authority_database(root: Path) -> Path:
    candidates = sorted((root / ".git" / "todo-orchestrator").glob("*/state.sqlite3"))
    require(len(candidates) == 1, f"expected one authoritative todo database, found {len(candidates)}")
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = Path.cwd().resolve()
    plan_path = (root / args.plan).resolve()
    plan = load_json(plan_path)
    plan_tasks = {task["id"]: task for task in plan.get("tasks", [])}
    required_ids = sorted(
        task_id
        for task_id in plan_tasks
        if task_id.startswith("CE-GEO-") and task_id != "CE-GEO-00" and task_id not in FINAL_TASKS
    )

    database = authority_database(root)
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        rows = {
            row["id"]: row
            for row in connection.execute(
                "SELECT id, status, result, completion_commit FROM tasks WHERE id GLOB 'CE-GEO-*'"
            )
        }
        require(set(required_ids) <= rows.keys(), "authoritative database is missing required CE-GEO tasks")
        for task_id in required_ids:
            row = rows[task_id]
            require(row["status"] == "done", f"{task_id}: expected done, got {row['status']}")
            allowed = plan_tasks[task_id].get("result_policy", {}).get("allowed_dispositions", [])
            require(row["result"] in allowed, f"{task_id}: disposition {row['result']!r} is not allowed")
            require(row["completion_commit"], f"{task_id}: missing completion commit")

        gate_rows = list(
            connection.execute(
                "SELECT id, status, valid FROM gates "
                "WHERE task_id GLOB 'CE-GEO-*' AND task_id NOT IN ('CE-GEO-125','CE-GEO-126','CE-GEO-127')"
            )
        )
        require(gate_rows, "no CE-GEO gates found")
        for row in gate_rows:
            require(row["status"] == "passed" and row["valid"] == 1, f"{row['id']}: gate is not valid")

        static_gate = connection.execute(
            "SELECT status, valid FROM gates WHERE id = 'CE-GEO-107-STATIC'"
        ).fetchone()
        require(static_gate is not None, "CE-GEO-only static audit gate is missing")
        require(static_gate["status"] == "passed" and static_gate["valid"] == 1, "static audit failed")

        interface_rows = list(
            connection.execute(
                "SELECT id, state, version FROM interfaces WHERE owner_task_id GLOB 'CE-GEO-*' ORDER BY id"
            )
        )
        planned_interfaces = {item["id"] for item in plan.get("interfaces", [])}
        require({row["id"] for row in interface_rows} == planned_interfaces, "interface inventory differs")
        require(all(row["state"] == "frozen" for row in interface_rows), "not every interface is frozen")

        checkpoint_rows = list(
            connection.execute(
                "SELECT id, state FROM checkpoints "
                "WHERE id GLOB 'CE-GEO-*' AND id NOT IN ('CE-GEO-VOLTA-COMPLETE','CE-GEO-COMPLETE') "
                "ORDER BY id"
            )
        )
        require(checkpoint_rows, "no prerequisite checkpoints found")
        require(all(row["state"] == "reached" for row in checkpoint_rows), "a prerequisite checkpoint is pending")
    finally:
        connection.close()

    evidence_root = root / "bench" / "ce_geo" / "evidence"
    evidence_hashes = {}
    for relative in REQUIRED_EVIDENCE:
        path = evidence_root / relative
        require(path.is_file(), f"missing evidence artifact: {relative}")
        evidence_hashes[relative] = sha256(path)

    validation = load_json(evidence_root / "full_volta_validation.json")
    require(validation.get("validated") is True, "full Volta validation is not accepted")
    require(validation.get("summary", {}).get("passed_validation_command_count") == 22, "acceptance is not 22/22")
    sanitizer = load_json(evidence_root / "sanitizer" / "campaign.json")
    require(sanitizer.get("validated") is True, "sanitizer campaign is not accepted")
    require(sanitizer.get("summary", {}).get("passed_run_count") == 22, "sanitizer is not 22/22")
    forward = load_json(evidence_root / "sm70_forward_disposition.json")
    require(forward.get("promotion", {}).get("accepted") is True, "N=64 hybrid promotion is absent")
    require(forward.get("measured_regime", {}).get("logical_edges") == 2176, "promotion regime changed")
    fusion = load_json(evidence_root / "fusion_evaluation.json")
    require(fusion.get("disposition") == "evaluated_not_promoted", "fusion disposition changed")
    preprint = load_json(evidence_root / "preprint" / "summary.json")
    require(preprint.get("accepted_for_promotion") is False, "ablations were unexpectedly promoted")
    require(preprint.get("measurement_count") == 20, "ablation evidence is incomplete")

    disposition_counts: dict[str, int] = {}
    for task_id in required_ids:
        disposition = rows[task_id]["result"]
        disposition_counts[disposition] = disposition_counts.get(disposition, 0) + 1

    report = {
        "schema": "CELLERATOR-CE-GEO-FINAL-VOLTA-ACCEPTANCE/1",
        "task_id": "CE-GEO-125",
        "validated": True,
        "checkpoint": "CE-GEO-VOLTA-COMPLETE",
        "authority": str(database.relative_to(root)),
        "plan": {"path": str(plan_path.relative_to(root)), "sha256": sha256(plan_path)},
        "tasks": {
            "required_terminal_count": len(required_ids),
            "dispositions": dict(sorted(disposition_counts.items())),
        },
        "gates": {"passed_valid_count": len(gate_rows), "static_contract_audit": "passed"},
        "interfaces": [dict(row) for row in interface_rows],
        "prerequisite_checkpoints": [dict(row) for row in checkpoint_rows],
        "evidence_sha256": evidence_hashes,
        "acceptance": {
            "commands": "22/22",
            "sanitizer_runs": "22/22",
            "promoted_regime": "sm70 N64 exact hybrid cover only",
            "fusion": "evaluated_not_promoted",
            "ablations": "20 measurements; zero promotions",
        },
    }
    output = (root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"CE-GEO final Volta acceptance passed: {len(required_ids)} tasks, {len(gate_rows)} gates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
