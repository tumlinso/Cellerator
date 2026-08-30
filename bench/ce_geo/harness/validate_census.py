#!/usr/bin/env python3
"""Validate CE-GEO-116 structural-census coverage without inferring GPU speed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_COVERAGE = (
    "pbmc3k_available",
    "developmental_embryo_available",
    "heart_synthetic_surrogate_present",
    "uniform_random_negative_control_present",
    "heart_real_checked_unavailable",
    "perturbation_checked_unavailable",
    "multiome_checked_unavailable",
    "regulatory_checked_unavailable",
    "trajectory_checked_unavailable",
    "pbmc3k_negative_control_rejected",
)


def load_records(path: Path) -> list[dict[str, object]]:
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"line {number} is not a JSON object")
        records.append(value)
    return records


def validate(records: list[dict[str, object]]) -> dict[str, object]:
    summaries = [record for record in records
        if record.get("record_type") == "coverage_summary"]
    if len(summaries) != 1:
        raise ValueError("exactly one coverage_summary record is required")
    summary = summaries[0]
    if summary.get("schema") != "CELLERATOR-CE-GEO-TILEABILITY/1" \
            or summary.get("campaign_id") != "CE-GEO-116-biology-tileability":
        raise ValueError("coverage summary identity is invalid")
    provenance_records = [record for record in records
        if record.get("record_type") == "provenance"]
    if len(provenance_records) != 1:
        raise ValueError("exactly one provenance record is required")
    provenance = provenance_records[0]
    if provenance.get("measurement_domain") != "cpu_structural_census":
        raise ValueError("census measurement domain is not CPU structural")
    if provenance.get("controller_evidence_id") \
            != "CE-GEO-116-tileability-census-v1" \
            or provenance.get("benchmark_mutex") is not True \
            or provenance.get("uncontaminated") is not True:
        raise ValueError("census provenance is incomplete")
    for key in REQUIRED_COVERAGE:
        if summary.get(key) is not True:
            raise ValueError(f"required census fact is absent: {key}")
    if summary.get("accepted_for_promotion") is not False:
        raise ValueError("structural census must not claim performance promotion")
    expected_counts = {
        "available_cases": 4,
        "checked_unavailable_cases": 5,
        "synthetic_cases": 2,
        "negative_control_cases": 2,
    }
    for key, expected in expected_counts.items():
        if provenance.get(key) != expected:
            raise ValueError(f"coverage count mismatch: {key}")

    pbmc = [record for record in records
        if record.get("case_id") == "pbmc3k_support_512"]
    if len(pbmc) != 1 or pbmc[0].get("control_role") != "negative_control" \
            or pbmc[0].get("tileability_qualified") is not False \
            or not isinstance(pbmc[0].get("occupied_tiles"), int) \
            or pbmc[0]["occupied_tiles"] <= 0:
        raise ValueError("PBMC3K negative-control measurement is invalid")
    occupancy = pbmc[0].get("scalar_occupancy")
    if not isinstance(occupancy, (int, float)) or not 0.0 < occupancy < 1.0:
        raise ValueError("PBMC3K occupancy is invalid")
    return {
        "census_valid": True,
        "campaign_id": summary["campaign_id"],
        "measurement_domain": provenance["measurement_domain"],
        "pbmc3k_tileability_qualified": False,
        "accepted_for_promotion": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True, type=Path)
    arguments = parser.parse_args()
    print(json.dumps(validate(load_records(arguments.evidence)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
