import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parent


class CeArch84EvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "ce_arch_84_v100.jsonl"
        cls.records = [json.loads(line) for line in path.read_text().splitlines()]

    def test_complete_equivalent_candidate_grid(self):
        self.assertEqual(27, len(self.records))
        grouped = {}
        for record in self.records:
            self.assertEqual("CE-ARCH-84-EVIDENCE/1", record["schema"])
            self.assertTrue(record["correct"])
            self.assertEqual("overwrite", record["output_effect"])
            self.assertEqual("packed-row-major", record["input_order"])
            self.assertEqual("execution-row-major", record["output_order"])
            self.assertEqual("Tesla V100-SXM2-16GB", record["device"])
            self.assertEqual(70, record["sm"])
            self.assertEqual(8, record["expected_reuse"])
            grouped.setdefault((record["regime"], record["n"]), []).append(record)
        self.assertEqual(
            {(regime, n) for regime in
             ("high_sharing", "medium_sharing", "low_sharing")
             for n in (17, 32, 64)}, set(grouped))
        for records in grouped.values():
            self.assertEqual(
                {"row_masked", "csr", "feature_major_cta"},
                {record["candidate"] for record in records})
            common = (records[0]["rows"], records[0]["features"],
                      records[0]["nnz"], records[0]["n"],
                      records[0]["value_bytes"], records[0]["output_bytes"])
            for record in records[1:]:
                self.assertEqual(common, (
                    record["rows"], record["features"], record["nnz"],
                    record["n"], record["value_bytes"], record["output_bytes"]))

    def test_retention_and_fallback_regimes_are_measured(self):
        groups = {}
        for record in self.records:
            groups.setdefault((record["regime"], record["n"]), []).append(record)
        steady_winner = {
            key: min(records, key=lambda item: item["median_total_ms"])["candidate"]
            for key, records in groups.items()
        }
        for regime in ("high_sharing", "medium_sharing", "low_sharing"):
            self.assertEqual("feature_major_cta", steady_winner[(regime, 32)])
            self.assertEqual("feature_major_cta", steady_winner[(regime, 64)])
        self.assertEqual("csr", steady_winner[("low_sharing", 17)])
        self.assertEqual("row_masked", steady_winner[("medium_sharing", 17)])
        for records in groups.values():
            self.assertEqual("row_masked", min(records,
                key=lambda item: item["amortized_total_ms"])["candidate"])

    def test_variance_and_cost_accounting_are_truthful(self):
        self.assertLess(max(record["mad_percent"] for record in self.records), 2.0)
        for record in self.records:
            if record["candidate"] == "feature_major_cta":
                self.assertEqual(0, record["dynamic_input_pack_ms"])
                self.assertEqual(0, record["output_order_ms"])
                self.assertGreater(record["projection_build_ms"], 0)
                self.assertGreater(record["value_pack_ms"], 0)
            if record["candidate"] == "row_masked":
                self.assertEqual(0, record["projection_build_ms"])


if __name__ == "__main__":
    unittest.main()
