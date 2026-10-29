#!/usr/bin/env python3
import csv
import pathlib
import sys


def read(path, columns):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows or list(rows[0]) != columns:
        raise AssertionError(f"schema mismatch: {path}")
    return rows


root = pathlib.Path(sys.argv[1])
candidate_rows = read(root / "profiler_candidate_matrix.tsv", [
    "candidate_id", "operation", "provider", "capability", "stage_count",
    "requires_measurement", "fallback", "value_modes", "order_modes"])
stage_rows = read(root / "profiler_stage_manifest.tsv", [
    "candidate_id", "stage_id", "correlation_id", "static_name", "stage_kind",
    "launch_index", "logical_work", "physical_work", "useful_work", "padded_work",
    "relation_bytes", "dense_input_bytes", "output_bytes", "value_pack_bytes",
    "residual_edges", "graph_capture", "numerics"])
receipt_rows = read(root / "profiler_resource_receipts.tsv", [
    "candidate_id", "receipt_kind", "query_performed", "compile_arch",
    "registers_per_thread", "static_shared_bytes", "max_threads_per_block",
    "observed_device", "observed_toolchain"])
marker_rows = read(root / "profiling_marker_manifest.tsv", [
    "stage_id", "correlation_id", "static_name", "default_enabled",
    "dynamic_string", "allocation"])

candidates = {row["candidate_id"]: row for row in candidate_rows}
assert len(candidates) == len(candidate_rows)
stages = {}
correlations = set()
names = set()
for row in stage_rows:
    assert row["candidate_id"] in candidates
    assert row["stage_id"] not in stages
    stages[row["stage_id"]] = row
    assert row["correlation_id"] not in correlations
    correlations.add(row["correlation_id"])
    assert row["static_name"] not in names
    names.add(row["static_name"])
    logical = int(row["logical_work"])
    physical = int(row["physical_work"])
    useful = int(row["useful_work"])
    padded = int(row["padded_work"])
    assert physical == useful + padded
    assert logical >= 0 and physical >= 0

for candidate_id, candidate in candidates.items():
    owned = [row for row in stage_rows if row["candidate_id"] == candidate_id]
    assert len(owned) == int(candidate["stage_count"])
    assert sorted(int(row["launch_index"]) for row in owned) == list(range(len(owned)))

receipts = {row["candidate_id"]: row for row in receipt_rows}
assert set(receipts) == set(candidates)
for row in receipts.values():
    assert row["receipt_kind"] == "synthetic_build_contract"
    assert row["query_performed"] == "false"
    assert row["compile_arch"] == "sm_70"
    assert all(row[key] == "not_observed" for key in
               ("registers_per_thread", "static_shared_bytes", "max_threads_per_block"))
    assert row["observed_device"] == "none" and row["observed_toolchain"] == "none"

markers = {row["stage_id"]: row for row in marker_rows}
assert set(markers) == set(stages)
for stage_id, marker in markers.items():
    stage = stages[stage_id]
    assert marker["correlation_id"] == stage["correlation_id"]
    assert marker["static_name"] == stage["static_name"]
    assert marker["default_enabled"] == "false"
    assert marker["dynamic_string"] == "false"
    assert marker["allocation"] == "false"
