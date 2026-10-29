#!/usr/bin/env python3
"""Host-only regression for full-E memory admission estimates."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from packed_training_benchmark import estimate_scaled_case_memory_bytes  # noqa: E402


class ScaledCaseMemoryPreflightTests(unittest.TestCase):
    def test_large_event_case_includes_full_edge_value_refresh(self):
        estimate = estimate_scaled_case_memory_bytes(
            candidate_items=82_959,
            candidate_workset_bytes=2_986_776,
            cells=8,
            entities=20_000,
            support_edges=1_280_000,
            width=64,
            tile_count=20_000,
        )

        self.assertEqual(estimate["candidate_feature_workspace_bytes"], 169_900_032)
        self.assertEqual(estimate["dense_context_and_output_bytes"], 245_760_000)
        self.assertEqual(estimate["global_edge_value_refresh_bytes"], 983_040_000)
        self.assertEqual(
            estimate["estimated_numerical_workspace_bytes_before_ce_geo"],
            1_401_686_808,
        )
        self.assertEqual(estimate["estimated_packed_active_tile_workspace_bytes"], 42_475_008)
        self.assertEqual(estimate["estimated_peak_bytes_before_admission"], 1_444_161_816)


if __name__ == "__main__":
    unittest.main()
