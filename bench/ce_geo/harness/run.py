#!/usr/bin/env python3
"""Validate controller-produced CE-GEO campaign evidence for workflow gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"line {line_number} is not a JSON object")
        records.append(value)
    if not records:
        raise ValueError("campaign evidence is empty")
    return records


def validate_campaign(campaign: str, records: list[dict[str, object]]) -> dict[str, object]:
    campaign_token = campaign.replace("-", "").lower()
    identities = [str(record.get("campaign_id", "")) for record in records]
    if not any(campaign_token in identity.replace("-", "").lower()
               for identity in identities):
        raise ValueError(f"campaign identity does not match {campaign}")

    provenance = next((record for record in records
        if record.get("controller_evidence_id")), None)
    if provenance is None:
        raise ValueError("controller evidence identity is missing")
    if provenance.get("benchmark_mutex") is not True:
        raise ValueError("benchmark mutex acquisition is not recorded")
    if provenance.get("uncontaminated") is not True:
        raise ValueError("campaign is absent or contaminated")

    measurements = [record for record in records
        if record.get("correctness_passed") is True]
    if not measurements:
        raise ValueError("no correctness-passing measurement record exists")
    complete = [float(record["complete_ns"]) for record in measurements
        if isinstance(record.get("complete_ns"), (int, float))]
    if not complete or not all(math.isfinite(value) and value > 0.0
                               for value in complete):
        raise ValueError("consumer-complete timing is missing or invalid")

    return {
        "evidence_valid": 1,
        "campaign": campaign,
        "controller_evidence_id": provenance["controller_evidence_id"],
        "measurement_records": len(measurements),
        "minimum_complete_ns": min(complete),
        "promotion_claimed": any(
            record.get("accepted_for_promotion") is True for record in records),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    print(json.dumps(validate_campaign(
        arguments.campaign, load_records(arguments.output)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
