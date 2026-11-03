#!/usr/bin/env python3
"""Compare one specialist JSON result with the prepared ATen baseline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

FIELDS = ("output", "dx", "dz", "dtheta", "dphi", "dtau",
          "theta_after_update", "post_update_output")
SPECIALIST_BACKEND = "cellerator_distributed_gated_product"


def compare(reference_path: Path, candidate_path: Path, atol: float, rtol: float) -> dict:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if reference.get("backend") != "aten_eager_two_device":
        raise ValueError("reference result is not the declared ATen two-device baseline")
    if candidate.get("backend") != SPECIALIST_BACKEND:
        raise ValueError(f"candidate backend must be {SPECIALIST_BACKEND!r}")
    if reference.get("mode") != "correctness" or candidate.get("mode") != "correctness":
        raise ValueError("both results must be correctness runs with collected values")
    if reference.get("case") not in ("low_activity_disjoint_32", "high_activity_overlap_32"):
        raise ValueError("comparison is restricted to the two declared scale workloads")
    if candidate.get("case") != reference.get("case"):
        raise ValueError("candidate and baseline case identities differ")
    ref_work = reference.get("workload", {})
    cand_work = candidate.get("workload", {})
    for key in ("batch", "coordinates", "conditions", "parameters", "groups",
                "outputs", "active_groups", "group_owners", "value_storage_dtype",
                "regulator_dtype", "compute_dtype", "parameter_dtype", "output_dtype", "vjp_dtype"):
        if ref_work.get(key) != cand_work.get(key):
            raise ValueError(f"candidate/baseline workload mismatch for {key}")
    ref_config = reference.get("configuration", {})
    cand_config = candidate.get("configuration", {})
    for key in ("device0", "device1", "learning_rate"):
        if ref_config.get(key) != cand_config.get(key):
            raise ValueError(f"candidate/baseline configuration mismatch for {key}")
    for document, label in ((reference, "baseline"), (candidate, "candidate")):
        placement = document.get("placement", {})
        if not placement.get("owner_devices") or not placement.get("canonical_devices"):
            raise ValueError(f"{label} result is missing explicit source/canonical placement metadata")
    for key in ("owner_devices", "canonical_devices"):
        if reference["placement"].get(key) != candidate["placement"].get(key):
            raise ValueError(f"candidate/baseline placement mismatch for {key}")
    ref_identity = reference.get("workload_identity")
    cand_identity = candidate.get("workload_identity")
    if not ref_identity or ref_identity != cand_identity:
        raise ValueError("canonical workload identity differs between backend runs")
    coordinates = int(ref_work["coordinates"])
    conditions = int(ref_work["conditions"])
    parameters = int(ref_work["parameters"])
    batch = int(ref_work["batch"])
    expected_extents = {
        "output": batch * coordinates,
        "dx": batch * coordinates,
        "dz": batch * conditions * 2,
        "dtheta": parameters,
        "dphi": parameters,
        "dtau": parameters,
        "theta_after_update": parameters,
        "post_update_output": batch * coordinates,
    }
    ref_devices = reference.get("devices", [])
    cand_devices = candidate.get("devices", [])
    if len(ref_devices) != 2 or len(cand_devices) != 2:
        raise ValueError("paired comparison requires exactly two visible devices")
    for index, (left_device, right_device) in enumerate(zip(ref_devices, cand_devices)):
        for key in ("visible_index", "name", "uuid", "compute_capability", "memory_bytes"):
            if left_device.get(key) != right_device.get(key):
                raise ValueError(f"device {index} mismatch for {key}")

    ref_values = reference.get("correctness_values", {})
    cand_values = candidate.get("correctness_values", {})
    passed = True
    field_results = {}
    for name in FIELDS:
        left = ref_values.get(name)
        right = cand_values.get(name)
        expected_extent = expected_extents[name]
        if (not isinstance(left, list) or not isinstance(right, list) or
                len(left) != expected_extent or len(right) != expected_extent):
            field_results[name] = {"passed": False, "reason": "extent mismatch"}
            passed = False
            continue
        max_abs = 0.0
        max_rel = 0.0
        field_passed = True
        for expected, observed in zip(left, right):
            expected = float(expected)
            observed = float(observed)
            if not math.isfinite(expected) or not math.isfinite(observed):
                field_passed = False
                continue
            error = abs(observed - expected)
            relative = error / max(abs(expected), 1.0e-30)
            if error > atol + rtol * abs(expected):
                field_passed = False
            max_abs = max(max_abs, error)
            max_rel = max(max_rel, relative)
        field_results[name] = {
            "passed": field_passed,
            "elements": len(left),
            "max_abs_error": max_abs,
            "max_relative_error": max_rel,
        }
        passed = passed and field_passed
    return {
        "schema": "cellerator.native_data.backend_comparison.v1",
        "case": reference["case"],
        "baseline_backend": reference["backend"],
        "candidate_backend": candidate["backend"],
        "baseline_binary_sha256": reference.get("artifact_identity", {}).get("binary_sha256"),
        "candidate_binary_sha256": candidate.get("artifact_identity", {}).get("binary_sha256"),
        "workload_identity": ref_identity,
        "atol": atol,
        "rtol": rtol,
        "passed": passed,
        "fields": field_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, required=True)
    parser.add_argument("--rtol", type=float, required=True)
    args = parser.parse_args()
    report = compare(args.reference, args.candidate, args.atol, args.rtol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"case": report["case"], "passed": report["passed"],
                      "fields": report["fields"]}, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
