"""Policy arithmetic checks using scenario-only costs, never GPU measurements."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).with_name("run_trace.py")
SPEC = importlib.util.spec_from_file_location("adaptive_trace", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CostPolicyTests(unittest.TestCase):
    def test_switch_requires_remaining_savings_to_exceed_setup(self):
        steps = []
        for index, (epoch, density) in enumerate(
                [(1, "low")] * 16 + [(2, "high")] * 16 + [(2, "low")] * 16):
            steps.append({"step": index, "structure_epoch": epoch,
                          "density_class": density})
        table = {
            "schema": "cellerator.adaptive_fields.e7_cost_table.v1",
            "evidence_kind": "scenario_test_only",
            "initial_candidate": "scan",
            "policy_evaluation_ms": 0.01,
            "candidates": {
                "scan": {"prepare_ms": 0.1, "switch_ms": 0.02,
                         "step_ms": {"low": 0.1, "high": 0.8}},
                "ordered_postings": {"prepare_ms": 0.5, "switch_ms": 0.05,
                                     "step_ms": {"low": 0.2, "high": 0.4}},
            },
        }
        result = MODULE.replay_cost_policy(steps, table)
        decisions = result["decisions"]
        self.assertEqual([row["block_start_step"] for row in decisions], [0, 16, 32])
        self.assertEqual(decisions[1]["selected_candidate"], "scan")
        self.assertEqual(decisions[2]["selected_candidate"], "ordered_postings")
        self.assertGreater(decisions[2]["net_estimated_savings_ms"], 0)
        self.assertEqual(result["cost_table_evidence_kind"], "scenario_test_only")

    def test_switch_is_rejected_when_savings_do_not_cover_cost(self):
        steps = [{"structure_epoch": 1, "density_class": "low"} for _ in range(32)]
        table = {
            "schema": "cellerator.adaptive_fields.e7_cost_table.v1",
            "evidence_kind": "scenario_test_only",
            "initial_candidate": "scan",
            "policy_evaluation_ms": 0.01,
            "candidates": {
                "scan": {"prepare_ms": 0.1, "switch_ms": 0.02,
                         "step_ms": {"low": 0.1, "high": 0.8}},
                "ordered_postings": {"prepare_ms": 5.0, "switch_ms": 0.05,
                                     "step_ms": {"low": 0.09, "high": 0.4}},
            },
        }
        result = MODULE.replay_cost_policy(steps, table)
        self.assertTrue(all(row["selected_candidate"] == "scan"
                            for row in result["decisions"]))
        self.assertLessEqual(result["decisions"][1]["net_estimated_savings_ms"], 0)


if __name__ == "__main__":
    unittest.main()
