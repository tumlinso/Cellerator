#!/usr/bin/env python3
"""Validate the machine-readable CE-GEO bootstrap preflight record."""

import json
import pathlib
import sys


def fail(message: str) -> None:
    raise ValueError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_preflight.py <preflight.json>", file=sys.stderr)
        return 2

    path = pathlib.Path(sys.argv[1])
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        require(record.get("schema_version") == 1, "unsupported schema_version")
        require(record.get("campaign") == "CE-GEO", "wrong campaign")
        require(record.get("task_id") == "CE-GEO-01", "wrong task")

        source = record["source"]
        require(len(source["commit"]) == 40, "invalid source commit")
        require(source["branch"] == "main", "unexpected source branch")
        require(source["dirty_before_claim"] is False, "pre-claim tree was dirty")
        require(source["git_diff_check"] == "pass", "git diff check did not pass")

        workflow = record["workflow"]
        require(workflow["workflow_front_door"] == "coding-workflow", "wrong workflow front door")
        require(workflow["run_id"] == "CE-GEO-RUN-V1", "wrong run")
        require(workflow["lane_id"] == "CE-GEO-L-COORD", "wrong lane")
        require(workflow["claimed_task"] == "CE-GEO-01", "wrong claimed task")
        require(workflow["ce_amp"]["permission_value"] == "not_granted", "Ampere permission is not closed")
        require(workflow["ce_amp"]["required_checkpoint"] == "CE-GEO-COMPLETE", "Ampere checkpoint interlock missing")
        require(workflow["ce_amp"]["actionable"] is False, "Ampere unexpectedly actionable")

        coordination = record["coordination_entities"]
        locks = set(coordination["named_locks"])
        require("cuda-benchmark-mutex" in locks, "repository benchmark mutex missing")
        require("ce-geo-integration" in locks, "integration lock missing")
        resource_ids = {item["id"] for item in coordination["resource_classes"]}
        require("accelerator" in resource_ids, "generic accelerator resource missing")

        accelerators = record["accelerators"]
        require(accelerators["leases_held"] == [], "preflight unexpectedly holds an accelerator lease")
        require(all(device["compute_capability"] == "7.0" for device in accelerators["devices"]), "non-Volta device in Volta inventory")

        coexistence = record["coexistence"]
        require(coexistence["ce_ptr_run"] == "CE-PTR-RUN", "CE-PTR run not recorded")
        require(coexistence["active_ce_ptr_claims"] == [], "unresolved CE-PTR ownership")

        additive = record["additive_identity_proof"]
        require(additive["ce_geo_prefix_preexisting_collision"] is False, "CE-GEO ID collision")
        require(additive["ce_amp_prefix_preexisting_collision"] is False, "CE-AMP ID collision")
        require(additive["historical_runs_modified"] is False, "historical run mutation")
        require(additive["historical_interfaces_revised"] is False, "historical interface mutation")
        require(additive["project_reinitialized"] is False, "project reinitialization")

        rollback = record["rollback"]
        require(all(rollback.get(key) for key in ("source_changes", "workflow_state", "claims_and_resources", "artifacts")), "incomplete rollback procedure")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"CE-GEO preflight validation failed: {error}", file=sys.stderr)
        return 1

    print("CE-GEO preflight validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
