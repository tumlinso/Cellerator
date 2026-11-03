from __future__ import annotations

from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import bounded_validation as bounded  # noqa: E402
import run_qualification_partition as qualification  # noqa: E402


class QualificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "build/source"
        self.source.mkdir(parents=True)
        self.packet = self.root / "packet"
        self.artifact = self.packet / "cases"
        self.output = self.packet / "qualification.json"
        self.uuid = "GPU-12345678-1234-5678-9abc-def012345678"
        self.context = {"scheduler_job_id": "job", "scheduler_attempt_id": "attempt",
                        "device_uuid": self.uuid, "cgroup": "/job", "unit_name": "job"}
        self.build = {"source_closure_sha256": "a" * 64, "runner_sha256": "b" * 64,
                      "library_sha256": "c" * 64, "build_manifest_sha256": "d" * 64,
                      "validator_sha256": bounded.VALIDATOR_SHA256}
        self.calls = []
        self.fact_calls = []

    def facts(self, source, workload, activation, parameter):
        self.fact_calls.append((workload, activation, parameter))
        return {"strongest_direct_mode": "minimal_direct", "stable": False, "unique": False}

    def invoke_case(self, **kwargs):
        self.calls.append(kwargs)
        case = kwargs["case_id"]
        evidence = kwargs["artifact_root"] / "evidence" / f"{case}.json"
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_text(json.dumps({"case_id": case, "validated": True}))
        return {"evidence_path": str(evidence),
                "evidence_relative_path": evidence.relative_to(self.packet).as_posix(),
                "input_case_sha256": "e" * 64, "output_result_sha256": "f" * 64,
                "peak_scratch_bytes": 1234,
                "summary": {"native_execution": {"selected_backend": "direct",
                                                   "actual_implementation": kwargs["mode"],
                                                   "structural_layout": "canonical_direct"}}}

    def execute(self, kind, role=None, partition=None):
        module = mock.Mock()
        module.correctness_workload_ids.return_value = qualification.AUXILIARY_IDS
        with mock.patch.object(bounded, "_source_module", return_value=module), \
             mock.patch.object(bounded, "verify_build", return_value=self.build), \
             mock.patch.object(bounded, "read_context", return_value=self.context), \
             mock.patch.object(bounded, "workload_facts", side_effect=self.facts), \
             mock.patch.object(bounded, "run_validation", side_effect=self.invoke_case), \
             redirect_stderr(io.StringIO()):
            return qualification.run(source=self.source, runner=self.root / "runner",
                                     build_manifest=self.root / "manifest", expected_uuid=self.uuid,
                                     artifact_dir=self.artifact, output=self.output,
                                     kind=kind, role=role, partition=partition)

    def test_dtype_exactly_covers_eight_storage_triples_and_four_modes(self):
        report = self.execute("dtype")
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["expected_case_count"], 32)
        self.assertEqual(report["passed_case_count"], 32)
        combinations = {(call["activation"], call["parameter"], call["output_dtype"], call["mode"])
                        for call in self.calls}
        self.assertEqual(len(combinations), 32)
        self.assertTrue(all(call["repeat_vjp"] and call["iterations"] == 0 for call in self.calls))
        self.assertTrue(all(call["packet_root"] == self.packet for call in self.calls))
        self.assertEqual(json.loads(self.output.read_text())["status"], "pass")
        self.assertEqual(len(self.fact_calls), 4)  # independent output dtype does not affect structure

    def test_auxiliary_retains_complete_corpus_and_explicit_fp32_fixture(self):
        report = self.execute("aux", "strongest-direct", "regular")
        self.assertEqual(report["status"], "pass")
        regular = [name for name in qualification.AUXILIARY_IDS
                   if name != "correctness_packet_high_fanin_1m"]
        self.assertEqual([row["workload"] for row in report["records"]], regular)
        self.assertEqual(len(self.calls), 11)
        self.assertEqual(report["full_auxiliary_corpus_count"], 12)
        self.assertFalse(report["full_auxiliary_corpus_qualified_by_this_report"])
        fma = next(row for row in self.calls if row["workload"] == "correctness_fma_product_residual_cancellation")
        self.assertEqual((fma["activation"], fma["parameter"]), ("fp32", "fp32"))
        self.assertTrue(all(row["mode"] == "minimal_direct" for row in self.calls))
        self.assertTrue(all(row["evidence_relative_path"].startswith("cases/evidence/")
                            for row in report["records"]))

    def test_auxiliary_partitions_are_disjoint_and_cover_the_full_corpus(self):
        ordinary = qualification._plan("aux", "packet", "regular")
        high = qualification._plan("aux", "packet", "high-fanin-1m")
        self.assertEqual([row["workload"] for row in high], ["correctness_packet_high_fanin_1m"])
        self.assertEqual({row["workload"] for row in ordinary + high}, set(qualification.AUXILIARY_IDS))
        self.assertEqual(len(ordinary) + len(high), 12)
        with self.assertRaises(bounded.ValidationFailure):
            qualification._plan("aux", "packet", None)

    def test_failed_case_stops_and_writes_no_success_qualification(self):
        original = self.invoke_case
        def fail(**kwargs):
            if len(self.calls) == 2:
                raise bounded.ValidationFailure("fixed validator failed; wires retained")
            return original(**kwargs)
        self.invoke_case = fail
        report = self.execute("dtype")
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["passed_case_count"], 2)
        self.assertEqual(len(report["failures"]), 1)

    def test_rejects_writes_inside_shared_build_before_creating_dirs(self):
        illegal = self.source.parent / "new-output"
        with self.assertRaisesRegex(bounded.ValidationFailure, "outside the shared build"):
            qualification.run(source=self.source, runner=self.root / "runner",
                              build_manifest=self.root / "manifest", expected_uuid=self.uuid,
                              artifact_dir=illegal / "cases", output=illegal / "report.json",
                              kind="dtype", role=None)
        self.assertFalse(illegal.exists())


if __name__ == "__main__":
    unittest.main()
