#!/usr/bin/env python3
import json
import pathlib
import sys


document = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert document["schema"] == "ce-exop-deferred-profiling-requirements-v1"
assert document["generated_from"] == "synthetic_profiler_fixtures_only"
assert document["claims"] == {
    "hardware_results": False,
    "performance_promotion": False,
    "real_data_speedup": False,
    "universal_winner": False,
    "preprint_readiness": False,
}

campaigns = {item["id"]: item for item in document["deferred_campaigns"]}
assert set(campaigns) == {"deep_nsight", "biological_data", "preprint", "ce_amp"}
assert campaigns["deep_nsight"]["state"] == "deferred"
assert campaigns["deep_nsight"]["authorization"] == "CE-EXOP-DEEP-PROFILING=authorized"
assert campaigns["deep_nsight"]["resource_requirements"] == [
    "accelerator:any", "cuda-benchmark-mutex"]
assert "exact_correctness_before_timing" in campaigns["deep_nsight"]["evidence_required"]
assert campaigns["biological_data"]["state"] == "deferred"
assert "dataset_provenance" in campaigns["biological_data"]["evidence_required"]
assert campaigns["preprint"]["state"] == "deferred"
assert "end_to_end_cost_accounting" in campaigns["preprint"]["evidence_required"]
assert campaigns["ce_amp"]["state"] == "blocked"
assert campaigns["ce_amp"]["authorization"] == "CE-AMP-PERMISSION=granted"
assert campaigns["ce_amp"]["evidence_required"] == ["CE-EXOP-COMPLETE"]
assert "only" in document["synthetic_evidence_limit"]
