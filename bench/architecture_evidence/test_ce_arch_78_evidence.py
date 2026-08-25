#!/usr/bin/env python3
"""Contract checks for the committed CE-ARCH-78 V100 evidence."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


EVIDENCE = Path(__file__).with_name("ce_arch_78_v100.json")


class CeArch78EvidenceTest(unittest.TestCase):
    def test_evidence_is_correct_and_low_variance(self) -> None:
        evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
        self.assertEqual(evidence["schema"], "CE-ARCH-78-EVIDENCE/1")
        self.assertEqual(evidence["device"], "Tesla V100-SXM2-16GB")
        self.assertEqual(evidence["sm"], 70)
        self.assertEqual(evidence["fixture"]["output_effect"], "accumulate")
        self.assertEqual(evidence["fixture"]["samples"], 9)
        self.assertEqual(evidence["fixture"]["uses_per_sample"], 100)
        for value in evidence["median_ns_per_use"].values():
            self.assertGreater(value, 0.0)
        for value in evidence["mad_percent"].values():
            self.assertLess(value, 1.0)

    def test_recorded_crossover_uses_total_amortized_cost(self) -> None:
        evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
        timing = evidence["median_ns_per_use"]
        tolerance = evidence["selection"]["practical_tolerance_percent"] / 100.0

        def winner(reuse: int) -> str:
            fused_total = timing["fused"] * reuse
            materialized_total = (
                timing["first_materialized"]
                + timing["cached_materialized"] * (reuse - 1)
            )
            return "materialized" if materialized_total < fused_total * (
                1.0 - tolerance
            ) else "fused"

        self.assertEqual(winner(23), "fused")
        self.assertEqual(winner(24), "materialized")
        self.assertEqual(
            evidence["selection"]["fused_last_winning_reuse"], 23
        )
        self.assertEqual(
            evidence["selection"]["materialized_first_winning_reuse"], 24
        )


if __name__ == "__main__":
    unittest.main()
