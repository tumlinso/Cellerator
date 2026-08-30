#!/usr/bin/env python3

from __future__ import annotations

import unittest

from validate_census import REQUIRED_COVERAGE, validate


def valid_records() -> list[dict[str, object]]:
    summary: dict[str, object] = {
        "record_type": "coverage_summary",
        "schema": "CELLERATOR-CE-GEO-TILEABILITY/1",
        "campaign_id": "CE-GEO-116-biology-tileability",
        "accepted_for_promotion": False,
    }
    summary.update({key: True for key in REQUIRED_COVERAGE})
    return [{
        "record_type": "provenance",
        "measurement_domain": "cpu_structural_census",
        "controller_evidence_id": "CE-GEO-116-tileability-census-v1",
        "benchmark_mutex": True,
        "uncontaminated": True,
        "available_cases": 4,
        "checked_unavailable_cases": 5,
        "synthetic_cases": 2,
        "negative_control_cases": 2,
    }, summary, {
        "case_id": "pbmc3k_support_512",
        "control_role": "negative_control",
        "tileability_qualified": False,
        "occupied_tiles": 1,
        "scalar_occupancy": 0.03,
    }]


class CensusValidationTest(unittest.TestCase):
    def test_accepts_complete_nonpromotion_census(self) -> None:
        self.assertTrue(validate(valid_records())["census_valid"])

    def test_rejects_missing_real_data_disclosure(self) -> None:
        records = valid_records()
        records[1]["heart_real_checked_unavailable"] = False
        with self.assertRaisesRegex(ValueError, "heart_real_checked_unavailable"):
            validate(records)

    def test_rejects_promotion_claim(self) -> None:
        records = valid_records()
        records[1]["accepted_for_promotion"] = True
        with self.assertRaisesRegex(ValueError, "must not claim"):
            validate(records)


if __name__ == "__main__":
    unittest.main()
