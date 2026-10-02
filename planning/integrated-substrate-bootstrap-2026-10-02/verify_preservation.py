#!/usr/bin/env python3
"""Verify additive import against the live native read-only authority export."""
import argparse, collections, json, os, pathlib, subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo-root", required=True, type=pathlib.Path)
args = parser.parse_args()
root = args.repo_root.resolve()
folder = root / "planning/integrated-substrate-bootstrap-2026-10-02"
before = json.loads((folder / "results/authority-export.json").read_text())["data"]["state"]
command = ["python", "/home/tumlinson/.agents/skills/todo-orchestrator/scripts/todo.py", "export", "--repo-root", str(root), "--json"]
result = subprocess.run(command, env=dict(os.environ, TODO_ORCHESTRATOR_READ_ONLY="1"), capture_output=True, text=True, check=True)
(folder / "results/authority-after-import.json").write_text(result.stdout)
after = json.loads(result.stdout)["data"]["state"]
roots = {task["id"] for task in json.loads((folder / "composition-receipt.json").read_text())["existing_task_definitions"]}
old_tables, new_tables = before["tables"], after["tables"]
def normalize(table, row):
    row = dict(row)
    if table == "tasks" and row["id"] in roots:
        for key in ("revision", "updated_at", "version"):
            row.pop(key, None)
    if table == "gates" and row["task_id"] in roots:
        row.pop("revision", None)
    return json.dumps(row, sort_keys=True, separators=(",", ":"))
errors = []
for table, rows in old_tables.items():
    old = collections.Counter(normalize(table, row) for row in rows)
    new = collections.Counter(normalize(table, row) for row in new_tables[table])
    missing = old - new
    if missing:
        errors.append({"table": table, "changed_or_removed_records": sum(missing.values())})
for row in old_tables["tasks"]:
    if row["id"] in roots:
        current = next(item for item in new_tables["tasks"] if item["id"] == row["id"])
        if current["version"] != row["version"] + 1:
            errors.append({"table": "tasks", "task": row["id"], "error": "expected one exact plan replay version increment"})
receipt = {"status": "passed" if not errors else "failed", "before_revision": before["project_revision"], "after_revision": after["project_revision"], "existing_tasks_checked": len(old_tables["tasks"]), "existing_interfaces_checked": len(old_tables["interfaces"]), "existing_runs_checked": len(old_tables["workflow_runs"]), "permitted_metadata_updates": sorted(roots), "errors": errors}
(folder / "preservation-verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, sort_keys=True))
raise SystemExit(1 if errors else 0)
