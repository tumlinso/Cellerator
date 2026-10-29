#!/usr/bin/env python3
import json
import pathlib
import sys


path = pathlib.Path(sys.argv[1])
document = json.loads(path.read_text(encoding="utf-8"))
assert document["schema"] == "cellerator-generic-partition-export-v1"
exports = document["exports"]
assert len(exports) >= 2

forbidden = {
    "cellshard_callback", "scheduler", "transport", "rdma", "storage_path",
    "upload", "download", "worker_placement"
}


def reject_forbidden(value):
    if isinstance(value, dict):
        assert not forbidden.intersection(value)
        for child in value.values():
            reject_forbidden(child)
    elif isinstance(value, list):
        for child in value:
            reject_forbidden(child)


reject_forbidden(document)
candidate_ids = set()
for export in exports:
    candidate_ids.add(export["candidate_id"])
    assert export["provider_id"] and export["capability_id"]
    partition = export["semantic_partition"]
    assert partition["global_extent"] > 2**32
    assert partition["local_extent"] <= 2**16
    assert partition["local_index_width_bits"] == 16
    assert partition["partition_id"] > 2**32
    assert export["input_order"] and export["output_order"]
    stages = export["stage_graph"]
    stage_ids = [stage["stage_id"] for stage in stages]
    assert len(stage_ids) == len(set(stage_ids))
    seen = set()
    for stage in stages:
        assert set(stage["predecessors"]).issubset(seen)
        seen.add(stage["stage_id"])
    assert all(value >= 0 for value in export["costs"].values())
    assert export["graph_capture"] in {"compatible", "incompatible"}
    assert export["device_requirements"]["architecture"] == "sm_70"
    assert isinstance(export["communication_boundary"]["present"], bool)

assert len(candidate_ids) == len(exports)
