#!/usr/bin/env python3
"""Contract checks for the committed CE-ARCH-76 V100 evidence."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


EVIDENCE = Path(__file__).with_name("ce_arch_76_v100.jsonl")
CANDIDATES = {"row_masked", "csr", "feature_major"}
REGIMES = {"high_sharing", "medium_sharing", "low_sharing"}
DENSE_WIDTHS = {1, 2, 4, 8, 16}


def load_records() -> list[dict[str, object]]:
    with EVIDENCE.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


class CeArch76EvidenceTest(unittest.TestCase):
    def test_complete_equivalent_candidate_grid(self) -> None:
        records = load_records()
        self.assertEqual(len(records), 45)
        self.assertEqual(
            {(item["candidate"], item["regime"], item["n"])
             for item in records},
            {(candidate, regime, n)
             for candidate in CANDIDATES
             for regime in REGIMES
             for n in DENSE_WIDTHS},
        )
        for item in records:
            self.assertEqual(item["schema"], "CE-ARCH-76-EVIDENCE/1")
            self.assertTrue(item["correct"])
            self.assertEqual(item["output_effect"], "overwrite")
            self.assertEqual(item["input_order"], "packed-row-major")
            self.assertEqual(item["output_order"], "execution-row-major")
            self.assertEqual(item["rows"], 65536)
            self.assertEqual(item["features"], 32768)
            self.assertEqual(item["nnz"], 2097152)
            self.assertEqual(item["warmups"], 3)
            self.assertEqual(item["repeats"], 11)
            self.assertEqual(item["expected_reuse"], 8)
            self.assertEqual(item["device"], "Tesla V100-SXM2-16GB")
            self.assertEqual(item["sm"], 70)
            self.assertGreater(item["median_total_ms"], 0.0)
            self.assertLess(item["mad_percent"], 2.0)

    def test_each_comparison_uses_identical_work(self) -> None:
        groups: dict[tuple[str, int], list[dict[str, object]]] = {}
        for item in load_records():
            groups.setdefault((str(item["regime"]), int(item["n"])), []).append(item)
        self.assertEqual(len(groups), 15)
        invariant_fields = (
            "rows", "features", "nnz", "n", "warmups", "repeats",
            "device", "sm", "cuda_driver", "cuda_runtime", "expected_reuse",
            "value_bytes", "output_bytes", "output_effect", "input_order",
            "output_order",
        )
        for records in groups.values():
            self.assertEqual({item["candidate"] for item in records}, CANDIDATES)
            for field in invariant_fields:
                self.assertEqual(len({item[field] for item in records}), 1, field)

    def test_persistent_cost_is_separate_from_steady_state(self) -> None:
        for item in load_records():
            one_time = sum(float(item[field]) for field in (
                "query_ms", "projection_build_ms", "value_pack_ms",
                "backend_prepare_ms",
            ))
            expected = float(item["median_total_ms"]) + one_time / 8.0
            self.assertAlmostEqual(float(item["amortized_total_ms"]), expected, places=6)
            if item["candidate"] == "row_masked":
                self.assertEqual(item["projection_build_ms"], 0)
                self.assertEqual(item["value_pack_ms"], 0)


if __name__ == "__main__":
    unittest.main()
