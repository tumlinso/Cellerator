#!/usr/bin/env python3
"""Validate the non-authoritative Cellerator compiler Part One pre-ledger package.

This validator is intentionally read-only with respect to Todo Orchestrator.
It checks package consistency, graph correctness, live-ID collisions, and the
recorded source/authority preconditions. It never applies a plan or starts a run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

APPLY_PLAN = Path("machine/cellerator-compiler-part1.todo-plan.json")
PRELEDGER = Path("machine/proposed_todos.json")
PROPOSED_CSV = Path("machine/proposed_todos.csv")
DEPENDENCY_CSV = Path("machine/dependency_edges.csv")
INTERFACE_CATALOG = Path("machine/interface_catalog.json")
PLAN_SUMMARY = Path("machine/plan_summary.json")
CHECKPOINT_CSV = Path("machine/checkpoints.csv")
BARRIER_CATALOG = Path("machine/barriers.json")
LANE_CATALOG = Path("machine/lanes.json")
WORKSTREAM_CATALOG = Path("machine/workstreams.json")
HUMAN_TASK_DIR = Path("proposed-todos")
MANIFEST = Path("MANIFEST.sha256")

class ValidationError(RuntimeError):
    pass

def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValidationError(f"missing required JSON: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid JSON in {path}: {exc}") from exc

def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)

def unique(records: Iterable[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        require(isinstance(record, dict), f"{label}[{index}] is not an object")
        value = record.get(key)
        require(isinstance(value, str) and value, f"{label}[{index}] has no non-empty {key}")
        require(value not in result, f"duplicate {label} {key}: {value}")
        result[value] = record
    return result

def checkpoint_map(tasks: list[dict[str, Any]]) -> dict[str, tuple[str, dict[str, Any]]]:
    result: dict[str, tuple[str, dict[str, Any]]] = {}
    for task in tasks:
        for checkpoint in task.get("checkpoints", []) or []:
            cid = checkpoint.get("id")
            require(isinstance(cid, str) and cid, f"task {task['id']} has invalid checkpoint")
            require(cid not in result, f"duplicate checkpoint ID: {cid}")
            result[cid] = (task["id"], checkpoint)
    return result

def task_edges(tasks: list[dict[str, Any]], checkpoints: dict[str, tuple[str, dict[str, Any]]]) -> list[tuple[str, str, str]]:
    edges: list[tuple[str, str, str]] = []
    task_ids = {task["id"] for task in tasks}
    for task in tasks:
        child = task["id"]
        parent = task.get("parent_id")
        if parent is not None:
            require(parent in task_ids, f"task {child} has missing parent {parent}")
        for dep in task.get("depends_on", []) or []:
            dtype = dep.get("type")
            if dtype == "task":
                source = dep.get("task_id")
                require(source in task_ids, f"task {child} depends on missing task {source}")
                edges.append((source, child, "task"))
            elif dtype == "checkpoint":
                source_cp = dep.get("checkpoint_id")
                require(source_cp in checkpoints, f"task {child} depends on missing checkpoint {source_cp}")
                source = checkpoints[source_cp][0]
                edges.append((source, child, f"checkpoint:{source_cp}"))
            else:
                raise ValidationError(f"task {child} has unsupported dependency type {dtype!r}")
    return edges

def verify_acyclic(task_ids: Iterable[str], edges: Iterable[tuple[str, str, str]]) -> None:
    ids = list(task_ids)
    outgoing: dict[str, list[str]] = defaultdict(list)
    indegree = {task_id: 0 for task_id in ids}
    for source, target, _ in edges:
        if source == target:
            raise ValidationError(f"self-cycle at {source}")
        outgoing[source].append(target)
        indegree[target] += 1
    queue = deque(sorted(task_id for task_id, degree in indegree.items() if degree == 0))
    seen = 0
    while queue:
        node = queue.popleft()
        seen += 1
        for target in outgoing[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if seen != len(ids):
        cycle_nodes = sorted(task_id for task_id, degree in indegree.items() if degree)
        raise ValidationError("task/checkpoint-expanded dependency graph is cyclic; residual nodes: "
                              + ", ".join(cycle_nodes[:30]))

def verify_reachability(tasks: list[dict[str, Any]], root_id: str) -> None:
    children: dict[str, list[str]] = defaultdict(list)
    for task in tasks:
        parent = task.get("parent_id")
        if parent:
            children[parent].append(task["id"])
    reachable = set()
    queue = deque([root_id])
    while queue:
        node = queue.popleft()
        if node in reachable:
            continue
        reachable.add(node)
        queue.extend(children[node])
    missing = sorted({task["id"] for task in tasks} - reachable)
    require(not missing, "tasks not reachable through parent hierarchy from root: " + ", ".join(missing[:30]))

def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))
    except FileNotFoundError as exc:
        raise ValidationError(f"missing CSV: {path}") from exc

def scan_live_ids(source_root: Path, package_root: Path) -> set[str]:
    """Collect live task-like IDs without reading this package as authority."""
    ids: set[str] = set()
    task_heading = re.compile(r"^#\s+([A-Z][A-Z0-9-]+):", re.MULTILINE)
    candidate_files: list[Path] = []
    todo_dir = source_root / "todos"
    if todo_dir.is_dir():
        candidate_files.extend(todo_dir.glob("*.md"))
    orch = source_root / ".todo-orchestrator"
    if orch.is_dir():
        candidate_files.extend(orch.glob("*.json"))
    candidate_files.extend(source_root.glob("*plan*.json"))
    package_resolved = package_root.resolve()
    for path in candidate_files:
        try:
            if package_resolved in path.resolve().parents:
                continue
            if path.stat().st_size > 100_000_000:
                continue
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        if path.suffix == ".md":
            ids.update(task_heading.findall(text))
            continue
        try:
            doc = json.loads(text)
        except Exception:
            continue
        if isinstance(doc, dict):
            for record in doc.get("tasks", []) or []:
                if isinstance(record, dict) and isinstance(record.get("id"), str):
                    ids.add(record["id"])
    return ids

def git_head(source_root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ).stdout.strip()
    except Exception:
        return None

def verify_manifest(package_root: Path) -> tuple[int, list[str]]:
    path = package_root / MANIFEST
    if not path.exists():
        return 0, ["MANIFEST.sha256 is absent"]
    mismatches: list[str] = []
    checked = 0
    listed: set[str] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            digest, rel = line.split("  ", 1)
        except ValueError:
            mismatches.append(f"malformed manifest line: {line!r}")
            continue
        if rel in listed:
            mismatches.append(f"duplicate manifest member: {rel}")
            continue
        listed.add(rel)
        target = package_root / rel
        if not target.is_file():
            mismatches.append(f"missing manifest member: {rel}")
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        checked += 1
        if actual != digest:
            mismatches.append(f"hash mismatch: {rel}")
    expected = {
        p.relative_to(package_root).as_posix()
        for p in package_root.rglob("*")
        if p.is_file()
        and p.name != MANIFEST.name
        and not p.name.endswith((".tar.gz", ".zip"))
    }
    omitted = sorted(expected - listed)
    extra = sorted(listed - expected)
    if omitted:
        mismatches.append("unlisted package members: " + ", ".join(omitted[:30]))
    if extra:
        mismatches.append("manifest lists excluded/unknown members: " + ", ".join(extra[:30]))
    return checked, mismatches

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source-root", type=Path, default=None,
                        help="Cellerator repository root for collision/live-precondition checks")
    parser.add_argument("--task-id", default=None,
                        help="Additionally require this proposed task ID")
    parser.add_argument("--check-live-preconditions", action="store_true")
    parser.add_argument("--require-manifest", action="store_true")
    parser.add_argument("--json-report", type=Path, default=None)
    args = parser.parse_args()

    root = args.package_root.resolve()
    source_root = args.source_root.resolve() if args.source_root else None
    errors: list[str] = []
    warnings: list[str] = []
    facts: dict[str, Any] = {}

    try:
        plan = load_json(root / APPLY_PLAN)
        preledger = load_json(root / PRELEDGER)
        interfaces_doc = load_json(root / INTERFACE_CATALOG)
        summary = load_json(root / PLAN_SUMMARY)
        barriers_doc = load_json(root / BARRIER_CATALOG)
        lanes_doc = load_json(root / LANE_CATALOG)
        workstreams_doc = load_json(root / WORKSTREAM_CATALOG)

        require(plan.get("schema_version") == 3, "apply-ready plan is not schema version 3")
        require(plan.get("project", {}).get("workspace") == "cellerator",
                "apply-ready plan targets the wrong workspace")
        tasks = plan.get("tasks")
        require(isinstance(tasks, list) and tasks, "apply-ready plan has no tasks")
        task_by_id = unique(tasks, "id", "task")
        checkpoints = checkpoint_map(tasks)
        edges = task_edges(tasks, checkpoints)
        verify_acyclic(task_by_id, edges)

        runs = plan.get("runs") or []
        run_by_id = unique(runs, "id", "run")
        require(len(run_by_id) == 1, "Part One plan must define exactly one first-class run")
        run = next(iter(run_by_id.values()))
        root_id = run.get("root_task_id")
        require(root_id in task_by_id, f"run root task is missing: {root_id}")
        verify_reachability(tasks, root_id)

        interface_by_id = unique(plan.get("interfaces") or [], "id", "interface")
        for iid, interface in interface_by_id.items():
            owner = interface.get("owner_task_id")
            require(owner in task_by_id, f"interface {iid} owner task is missing: {owner}")
        for task in tasks:
            for consumed in task.get("consumes_interfaces", []) or []:
                iid = consumed.get("id")
                require(iid in interface_by_id,
                        f"task {task['id']} consumes missing interface {iid}")
            for checkpoint in task.get("checkpoints", []) or []:
                for published in checkpoint.get("publishes_interfaces", []) or []:
                    iid = published.get("id")
                    require(iid in interface_by_id,
                            f"checkpoint {checkpoint['id']} publishes missing interface {iid}")
                    require(interface_by_id[iid]["owner_task_id"] == task["id"],
                            f"checkpoint {checkpoint['id']} publishes interface {iid} owned by "
                            f"{interface_by_id[iid]['owner_task_id']}, not {task['id']}")

        barrier_by_id = unique(plan.get("barriers") or [], "id", "barrier")
        for bid, barrier in barrier_by_id.items():
            for requirement in barrier.get("requirements", []) or []:
                rtype = requirement.get("type")
                rid = requirement.get("id")
                if rtype == "task":
                    require(rid in task_by_id, f"barrier {bid} requires missing task {rid}")
                elif rtype == "checkpoint":
                    require(rid in checkpoints, f"barrier {bid} requires missing checkpoint {rid}")
                else:
                    raise ValidationError(f"barrier {bid} has unsupported requirement {rtype!r}")

        # Lanes must cover every task exactly once.
        lane_records = run.get("lanes") or []
        lane_by_id = unique(lane_records, "id", "lane")
        membership: dict[str, list[str]] = defaultdict(list)
        for lane_id, lane in lane_by_id.items():
            parent_lane = lane.get("parent_lane_id")
            if parent_lane is not None:
                require(parent_lane in lane_by_id, f"lane {lane_id} has missing parent lane {parent_lane}")
            workspace = lane.get("workspace")
            if workspace:
                mode = workspace.get("mode")
                require(mode in {"exclusive", "read_shared", "contract_split", "isolated_merge"},
                        f"lane {lane_id} uses unsupported workspace mode {mode!r}")
                integration = workspace.get("integration_task_id")
                if integration is not None:
                    require(integration in task_by_id,
                            f"lane {lane_id} names missing integration task {integration}")
            for task_id in lane.get("tasks", []) or []:
                require(task_id in task_by_id, f"lane {lane_id} contains missing task {task_id}")
                membership[task_id].append(lane_id)
        missing_lane = sorted(set(task_by_id) - set(membership))
        duplicate_lane = sorted(task_id for task_id, owners in membership.items() if len(owners) != 1)
        require(not missing_lane, "tasks missing from lanes: " + ", ".join(missing_lane[:30]))
        require(not duplicate_lane, "tasks assigned to multiple lanes: " + ", ".join(duplicate_lane[:30]))

        # Human and pre-ledger representations.
        pre_tasks = preledger.get("tasks")
        require(isinstance(pre_tasks, list), "pre-ledger catalog has no tasks array")
        pre_by_id = unique(pre_tasks, "id", "pre-ledger task")
        require(set(pre_by_id) == set(task_by_id),
                "pre-ledger and apply-ready task ID sets differ")

        human_files = sorted((root / HUMAN_TASK_DIR).glob("*.md"))
        human_ids = set()
        heading = re.compile(r"^#\s+([A-Z][A-Z0-9-]+):", re.MULTILINE)
        for path in human_files:
            matches = heading.findall(path.read_text(errors="ignore"))
            require(len(matches) == 1, f"human proposed Todo {path.name} has {len(matches)} ID headings")
            human_ids.add(matches[0])
        require(len(human_files) == len(tasks),
                f"human Todo count {len(human_files)} != machine count {len(tasks)}")
        require(human_ids == set(task_by_id), "human and machine task ID sets differ")

        csv_tasks = read_csv(root / PROPOSED_CSV)
        csv_task_ids = [row.get("id", "") for row in csv_tasks]
        require(len(csv_task_ids) == len(set(csv_task_ids)), "proposed_todos.csv has duplicate IDs")
        require(set(csv_task_ids) == set(task_by_id),
                "proposed_todos.csv and apply-ready plan task IDs differ")

        dep_csv = read_csv(root / DEPENDENCY_CSV)
        json_edge_set = {(source, target, kind) for source, target, kind in edges}
        csv_edge_set = {
            (row.get("source_task_id", ""), row.get("target_task_id", ""),
             row.get("dependency_kind", ""))
            for row in dep_csv
        }
        require(json_edge_set == csv_edge_set,
                "dependency_edges.csv does not exactly match JSON dependencies")

        csv_checkpoints = read_csv(root / CHECKPOINT_CSV)
        require({row.get("checkpoint_id", "") for row in csv_checkpoints} == set(checkpoints),
                "checkpoint CSV does not match plan checkpoints")

        # Cross-machine catalogs.
        catalog_interfaces = interfaces_doc.get("interfaces", interfaces_doc if isinstance(interfaces_doc, list) else [])
        catalog_interface_ids = {record["id"] for record in catalog_interfaces}
        require(catalog_interface_ids == set(interface_by_id),
                "interface catalog does not match apply-ready plan")
        catalog_barriers = barriers_doc.get("barriers", barriers_doc if isinstance(barriers_doc, list) else [])
        require({record["id"] for record in catalog_barriers} == set(barrier_by_id),
                "barrier catalog does not match apply-ready plan")
        catalog_lanes = lanes_doc.get("lanes", lanes_doc if isinstance(lanes_doc, list) else [])
        require({record["id"] for record in catalog_lanes} == set(lane_by_id),
                "lane catalog does not match apply-ready plan")

        if args.task_id:
            require(args.task_id in task_by_id, f"requested task is absent: {args.task_id}")

        facts.update({
            "schema_version": plan["schema_version"],
            "task_count": len(tasks),
            "human_todo_count": len(human_files),
            "workstream_count": len(workstreams_doc.get("workstreams", [])),
            "lane_count": len(lane_by_id),
            "interface_count": len(interface_by_id),
            "barrier_count": len(barrier_by_id),
            "checkpoint_count": len(checkpoints),
            "dependency_edge_count": len(edges),
            "run_count": len(run_by_id),
            "root_task_id": root_id,
        })
        expected = summary.get("counts", {})
        for key, actual_key in [
            ("tasks", "task_count"),
            ("workstreams", "workstream_count"),
            ("lanes", "lane_count"),
            ("interfaces", "interface_count"),
            ("barriers", "barrier_count"),
            ("checkpoints", "checkpoint_count"),
        ]:
            if key in expected:
                require(expected[key] == facts[actual_key],
                        f"plan_summary count {key}={expected[key]} != {facts[actual_key]}")

        if source_root:
            live_ids = scan_live_ids(source_root, root)
            collisions = sorted(set(task_by_id) & live_ids)
            require(not collisions, "proposed task IDs collide with live authority: "
                    + ", ".join(collisions[:50]))
            facts["live_task_id_count_scanned"] = len(live_ids)
            facts["live_collision_count"] = 0
            current_head = git_head(source_root)
            facts["current_source_head"] = current_head
            recorded_head = plan.get("project", {}).get("baseline_commit")
            if args.check_live_preconditions:
                require(current_head == recorded_head,
                        f"Git HEAD changed: recorded {recorded_head}, current {current_head}")
            elif current_head and current_head != recorded_head:
                warnings.append(
                    f"Git HEAD differs from recorded snapshot: {recorded_head} -> {current_head}; "
                    "regenerate or re-preview before applying."
                )

        checked, manifest_errors = verify_manifest(root)
        facts["manifest_members_checked"] = checked
        if args.require_manifest:
            require((root / MANIFEST).exists(), "manifest is required but absent")
        require(not manifest_errors or not (root / MANIFEST).exists(),
                "; ".join(manifest_errors))
        if manifest_errors and (root / MANIFEST).exists():
            raise ValidationError("; ".join(manifest_errors))
        if not (root / MANIFEST).exists():
            warnings.append("MANIFEST.sha256 not yet generated")

    except ValidationError as exc:
        errors.append(str(exc))
    except Exception as exc:
        errors.append(f"unexpected validation failure: {type(exc).__name__}: {exc}")

    report = {
        "schema_version": 1,
        "package": "cellerator-compiler-preledger-v1",
        "validated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "passed" if not errors else "failed",
        "facts": facts,
        "warnings": warnings,
        "errors": errors,
    }
    output = json.dumps(report, indent=2, sort_keys=True)
    if args.json_report:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(output + "\n")
    print(output)
    return 0 if not errors else 1

if __name__ == "__main__":
    raise SystemExit(main())
