#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from run import load_records, validate_campaign


class CampaignGateTest(unittest.TestCase):
    def test_accepts_uncontaminated_controller_evidence(self) -> None:
        records = [
            {"campaign_id": "CE-GEO-111-value-pack-residual",
             "controller_evidence_id": "evidence-1", "benchmark_mutex": True,
             "uncontaminated": True, "accepted_for_promotion": False},
            {"campaign_id": "CE-GEO-111-value-pack-residual",
             "correctness_passed": True, "complete_ns": 42.0},
        ]
        result = validate_campaign("value-pack-residual", records)
        self.assertEqual(result["evidence_valid"], 1)
        self.assertFalse(result["promotion_claimed"])

    def test_rejects_missing_controller_provenance(self) -> None:
        with self.assertRaisesRegex(ValueError, "controller evidence"):
            validate_campaign("example", [
                {"campaign_id": "example", "correctness_passed": True,
                 "complete_ns": 1.0},
            ])

    def test_loads_json_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.jsonl"
            path.write_text('{"campaign_id":"one"}\n{"complete_ns":2}\n',
                            encoding="utf-8")
            self.assertEqual(len(load_records(path)), 2)


if __name__ == "__main__":
    unittest.main()
