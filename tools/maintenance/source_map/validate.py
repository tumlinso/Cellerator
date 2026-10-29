#!/usr/bin/env python3
"""Validate source-map path coverage and its source/content identity."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import snapshot, OUTPUT_NAMES

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo",required=True,help="Cellerator repository checkout")
parser.add_argument("--output",default="docs/development/source_map",help="Map directory relative to --repo")
parser.add_argument("--migration-manifest",help="Audit declared existing destinations in this source-migrations.json")
args=parser.parse_args()
repo=Path(args.repo).resolve()
out=Path(args.output)
out=(repo/out).resolve() if not out.is_absolute() else out.resolve()
manifest=json.loads((out/"manifest.json").read_text())
rules=json.loads((out/"routing_rules.json").read_text())
rows=list(csv.DictReader((out/"path_map.csv").open()))
errors=[]

def migration_destinations(manifest_path: str) -> list[str]:
    candidate=Path(manifest_path)
    candidate=(repo/candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        candidate.relative_to(repo)
    except ValueError as exc:
        raise ValueError("migration manifest must be inside the repository") from exc
    data=json.loads(candidate.read_text())
    if not isinstance(data,dict):
        raise ValueError("migration manifest must contain an object")
    destinations=[]
    for section in ("path_migrations","source_extractions","compatibility_facades"):
        rows=data.get(section,[])
        if not isinstance(rows,list):
            raise ValueError(f"{section} must be a list")
        for row in rows:
            if not isinstance(row,dict) or row.get("target_path_exists",True) is False:
                continue
            value=row.get("new_path")
            if not value:
                continue
            path=Path(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe migration destination: {value}")
            destinations.append(path.as_posix())
    return sorted(set(destinations))

destinations_to_audit=[]
if args.migration_manifest:
    try:
        destinations_to_audit=migration_destinations(args.migration_manifest)
    except (OSError,ValueError,json.JSONDecodeError) as exc:
        errors.append(f"migration manifest audit failed: {exc}")

payload=dict(manifest)
recorded_manifest_hash=payload.pop("manifest_sha256",None)
actual_manifest_hash=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()
if recorded_manifest_hash!=actual_manifest_hash:
    errors.append("manifest content digest mismatch; regenerate")

def resolve_tool(value):
    candidate=Path(value)
    return candidate if candidate.is_absolute() else repo/candidate
tools=manifest["toolchain_paths"]
current=snapshot(repo,out,resolve_tool(tools["generator"]),resolve_tool(tools["validator"]))
if destinations_to_audit:
    tracked_paths={entry["path"] for entry in current["tracked_source"]}
    for destination in destinations_to_audit:
        if destination not in tracked_paths:
            errors.append(f"intended migration destination is not tracked: {destination}")
if current["source_identity"]!=manifest["source_identity"]:
    errors.append("authoritative source identity changed; regenerate the map")
if current["source_index_sha256"]!=manifest["source_index_sha256"]:
    errors.append("non-generated Git index changed since capture")
if current["source_worktree_sha256"]!=manifest["source_worktree_sha256"]:
    errors.append("source worktree bytes or non-ignored untracked inputs changed")
if current["toolchain_sha256"]!=manifest["toolchain_sha256"]:
    errors.append("generator, validator, or shared source-snapshot code changed")
if manifest.get("rules_sha256")!=hashlib.sha256((out/"routing_rules.json").read_bytes()).hexdigest():
    errors.append("routing rules changed; regenerate the map")
expected_output_hashes=OUTPUT_NAMES-{"manifest.json"}
if set(manifest.get("generated_output_sha256",{}))!=expected_output_hashes:
    errors.append("manifest generated-output hash catalog is incomplete")
for name,expected_hash in manifest.get("generated_output_sha256",{}).items():
    path=out/name
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected_hash:
        errors.append(f"generated output changed; regenerate: {name}")
if set(current["generated_paths"])!=set(manifest["generated_paths"]):
    errors.append("generated output path set changed")

expected={row["path"]:{"state":"tracked","mode":row["mode"],"oid":row["oid"]} for row in current["tracked_source"]}
for path in current["untracked_source"]:
    expected[path]={"state":"untracked","mode":"untracked","oid":None}
for path in current["generated_paths"]:
    expected[path]={"state":"generated_metadata","mode":"generated","oid":None}
source_by_path={row["path"]:row for row in current["source_rows"]}
seen=set()
for row in rows:
    path=row["path"]
    if path in seen:
        errors.append(f"duplicate path: {path}")
    seen.add(path)
    route=rules.get(row["rule_id"])
    if path not in expected:
        errors.append(f"stale or unexpected map path: {path}")
        continue
    if route is None:
        errors.append(f"missing routing rule {row['rule_id']}: {path}")
        continue
    for key in ("role","current_owner","intended_destination","disposition","maturity","consumers","tests_or_evidence","archive_reference"):
        if not str(route.get(key," ")).strip():
            errors.append(f"missing {key}: {path}")
    if route["disposition"]=="hold_unclassified_review":
        errors.append(f"unclassified path needs an explicit routing rule: {path}")
    exp=expected[path]
    if not str(row.get("archive_reference", "")).strip():
        errors.append(f"missing archive reference: {path}")
    if row["source_state"]!=exp["state"] or row["git_mode"]!=exp["mode"] or (row["blob_oid"] or None)!=exp["oid"]:
        errors.append(f"source path/state/mode/blob mismatch: {path}")
    if exp["state"] in {"tracked","untracked"}:
        source=source_by_path.get(path)
        if not source or row["worktree_mode"]!=source["worktree_mode"] or row["content_sha256"]!=source["content_sha256"]:
            errors.append(f"source content identity mismatch: {path}")
    elif row["worktree_mode"] not in ("",None) or row["content_sha256"] not in ("",None):
        errors.append(f"generated metadata must not include its own content identity: {path}")
for path in sorted(set(expected)-seen):
    errors.append(f"path lacks a route-map row: {path}")
if len(rows)!=len(expected):
    errors.append("path-map row count differs from current routed path count")
if manifest.get("tracked_source_count")!=len(current["tracked_source"]):
    errors.append("tracked source count differs")
if manifest.get("untracked_source_paths")!=current["untracked_source"]:
    errors.append("untracked source path list differs")

report={
    "ok":not errors,"routed_paths":len(rows),"rules":len(rules),
    "source_identity":current["source_identity"],
    "capture_head_provenance_only":manifest["capture_head_provenance_only"],
    "errors":errors,
    "scope":"path coverage, explicit route, source/index/worktree identity, generated-output integrity, and optional migration-destination tracking audit; not semantic, build, test, biological, or performance acceptance",
}
print(json.dumps(report,indent=2))
sys.exit(bool(errors))
