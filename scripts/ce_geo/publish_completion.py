#!/usr/bin/env python3
"""Publish the deterministic CE-GEO closure record."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


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


def database_path(root: Path) -> Path:
    candidates = sorted((root / ".git" / "todo-orchestrator").glob("*/state.sqlite3"))
    require(len(candidates) == 1, f"expected one authoritative todo database, found {len(candidates)}")
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = Path.cwd().resolve()
    evidence_root = root / "bench" / "ce_geo" / "evidence"
    final_acceptance_path = evidence_root / "final_volta_acceptance.json"
    amp_audit_path = evidence_root / "amp_interlock_audit.json"
    full_validation_path = evidence_root / "full_volta_validation.json"
    sanitizer_path = evidence_root / "sanitizer" / "campaign.json"

    final_acceptance = load_json(final_acceptance_path)
    amp_audit = load_json(amp_audit_path)
    full_validation = load_json(full_validation_path)
    sanitizer = load_json(sanitizer_path)
    require(final_acceptance.get("validated") is True, "final Volta acceptance is not validated")
    require(amp_audit.get("validated") is True, "CE-AMP interlock audit is not validated")
    require(full_validation.get("validated") is True, "full validation evidence is not validated")
    require(sanitizer.get("validated") is True, "sanitizer evidence is not validated")
    require(amp_audit.get("interlock", {}).get("actual_value") == "not_granted", "CE-AMP is open")
    require(amp_audit.get("activity", {}).get("actionable_candidates") == 0, "CE-AMP is actionable")

    database = database_path(root)
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        prerequisite_rows = list(
            connection.execute(
                "SELECT id, status, result, completion_commit FROM tasks "
                "WHERE id GLOB 'CE-GEO-*' AND id NOT IN ('CE-GEO-00','CE-GEO-127') ORDER BY id"
            )
        )
        require(len(prerequisite_rows) == 107, f"expected 107 prerequisite tasks, found {len(prerequisite_rows)}")
        require(all(row["status"] == "done" for row in prerequisite_rows), "a CE-GEO prerequisite is not terminal")
        require(all(row["completion_commit"] for row in prerequisite_rows), "a prerequisite lacks a completion commit")

        closing_task = connection.execute(
            "SELECT status FROM tasks WHERE id = 'CE-GEO-127'"
        ).fetchone()
        require(closing_task is not None and closing_task["status"] in {"in_progress", "done"}, "CE-GEO-127 is not active")

        volta_checkpoint = connection.execute(
            "SELECT state FROM checkpoints WHERE id = 'CE-GEO-VOLTA-COMPLETE'"
        ).fetchone()
        require(volta_checkpoint is not None and volta_checkpoint["state"] == "reached", "Volta checkpoint is not reached")

        permission = connection.execute(
            "SELECT value_json FROM decisions WHERE id = 'CE-AMP-PERMISSION'"
        ).fetchone()
        require(permission is not None and json.loads(permission["value_json"]) == "not_granted", "CE-AMP permission changed")
        amp_started = connection.execute(
            "SELECT COUNT(*) FROM tasks WHERE id GLOB 'CE-AMP-*' AND status != 'planned'"
        ).fetchone()[0]
        amp_claims = connection.execute(
            "SELECT COUNT(*) FROM claims WHERE task_id GLOB 'CE-AMP-*'"
        ).fetchone()[0]
        require(amp_started == 0 and amp_claims == 0, "CE-AMP is not dormant")
    finally:
        connection.close()

    dispositions: dict[str, int] = {}
    for row in prerequisite_rows:
        dispositions[row["result"]] = dispositions.get(row["result"], 0) + 1

    report = {
        "schema": "CELLERATOR-CE-GEO-COMPLETION/1",
        "task_id": "CE-GEO-127",
        "validated": True,
        "closing_disposition": "validated",
        "checkpoint_to_reach": "CE-GEO-COMPLETE",
        "volta_checkpoint": "reached",
        "terminal_prerequisites": {
            "count": len(prerequisite_rows),
            "dispositions": dict(sorted(dispositions.items())),
        },
        "evidence": {
            "final_volta_acceptance": {
                "path": str(final_acceptance_path.relative_to(root)),
                "sha256": sha256(final_acceptance_path),
            },
            "amp_interlock_audit": {
                "path": str(amp_audit_path.relative_to(root)),
                "sha256": sha256(amp_audit_path),
            },
            "full_volta_validation": {
                "path": str(full_validation_path.relative_to(root)),
                "sha256": sha256(full_validation_path),
            },
            "sanitizer_campaign": {
                "path": str(sanitizer_path.relative_to(root)),
                "sha256": sha256(sanitizer_path),
            },
        },
        "ce_amp": {
            "permission": "not_granted",
            "loaded_tasks": 10,
            "started_tasks": amp_started,
            "claims": amp_claims,
            "actionable_candidates": 0,
            "required_for_ce_geo_completion": False,
        },
        "limits": [
            "The promoted sm70 N64 exact hybrid cover remains evidence-scoped.",
            "Whole-exchange fusion and the final ablation package remain evaluated_not_promoted.",
            "CE-AMP remains a separately runnable extension controlled only by explicit human permission.",
        ],
    }
    output = (root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("CE-GEO closure record valid: 107 terminal prerequisites; CE-AMP dormant")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
