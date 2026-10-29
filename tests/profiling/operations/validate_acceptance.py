#!/usr/bin/env python3
import hashlib
import json
import pathlib
import sys


bench_dir = pathlib.Path(sys.argv[1])
document = json.loads(
    (bench_dir / "profiler_readiness_acceptance_v1.json").read_text(encoding="utf-8"))
assert document["schema"] == "ce-exop-profiler-readiness-acceptance-v1"
assert document["disposition"] == "synthetic_validated"
for relative, expected in document["fixture_hashes"].items():
    actual = hashlib.sha256((bench_dir / relative).read_bytes()).hexdigest()
    assert actual == expected, f"hash mismatch for {relative}: {actual}"

properties = set(document["validated_properties"])
assert "stable_unique_candidate_stage_symbol_and_correlation_identity" in properties
assert "generic_cellshard_facing_export_without_transport_or_storage_ownership" in properties
assert "global_64_local_compact_identity_fixture" in properties

evidence = document["execution_evidence"]
assert evidence["host_reference"] and evidence["strict_cpp17"]
assert evidence["asan_ubsan"] and evidence["sm70_compile_only"]
for forbidden in (
        "cuda_runtime_executed", "gpu_resource_lease_used", "nsight_capture_executed",
        "timing_executed"):
    assert evidence[forbidden] is False
assert len(document["limits"]) == 3
