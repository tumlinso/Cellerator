"""CPU-only tests for the complete-chain sanitizer Autobench adapter."""
import hashlib
import importlib.util
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/sanitize_chain.py"
SPEC = importlib.util.spec_from_file_location("cgs_sanitize_chain", SCRIPT)
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class SanitizeChainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.archive = self.root / "closure.tar.gz"
        self.archive.write_bytes(b"frozen archive fixture")
        self.archive_sha = adapter.sha256(self.archive)
        self.binary_source = self.root / "qualified-cgs-study"
        self.binary_source.write_bytes(b"qualified binary fixture")
        self.binary_sha = adapter.sha256(self.binary_source)
        self.original_receipt = self.root / "original-build.json"
        write_json(self.original_receipt, {
            "schema": "cgs.build.v1",
            "suite": "kda",
            "target": "cgs_study",
            "archive_sha256": self.archive_sha,
            "binary_sha256": self.binary_sha,
            "binary_relative_path": "cmake/cgs_study",
        })
        self.build_dir = self.root / "adapter-build"
        args = SimpleNamespace(
            archive=self.archive,
            archive_sha256=self.archive_sha,
            binary=self.binary_source,
            receipt=self.original_receipt,
            build=self.build_dir,
        )
        adapter.build(args)

    def tearDown(self):
        self.tmp.cleanup()

    def test_build_verifies_archive_and_binary_and_restores_owner_execute(self):
        receipt = json.loads((self.build_dir / "build.json").read_text())
        staged = self.build_dir / "cgs_study"
        self.assertEqual(receipt["schema"], "cgs.build.v1")
        self.assertEqual(receipt["archive_sha256"], self.archive_sha)
        self.assertEqual(receipt["binary_sha256"], self.binary_sha)
        self.assertEqual(receipt["binary_relative_path"], "cgs_study")
        self.assertTrue(staged.stat().st_mode & stat.S_IXUSR)
        self.assertEqual(adapter.sha256(staged), self.binary_sha)

    def test_build_rejects_archive_and_binary_identity_mismatches(self):
        other_archive = self.root / "other.tar.gz"
        other_archive.write_bytes(b"different archive")
        args = SimpleNamespace(
            archive=other_archive,
            archive_sha256=self.archive_sha,
            binary=self.binary_source,
            receipt=self.original_receipt,
            build=self.root / "bad-archive-build",
        )
        with self.assertRaisesRegex(adapter.AdapterError, "source archive bytes"):
            adapter.build(args)

        other_binary = self.root / "other-binary"
        other_binary.write_bytes(b"not the qualified binary")
        args = SimpleNamespace(
            archive=self.archive,
            archive_sha256=self.archive_sha,
            binary=other_binary,
            receipt=self.original_receipt,
            build=self.root / "bad-binary-build",
        )
        with self.assertRaisesRegex(adapter.AdapterError, "original build receipt"):
            adapter.build(args)

    def campaign_record(self, case, uuid="GPU-test"):
        return {
            "schema": "cgs.campaign.v1",
            "status": "ok",
            "accuracy_accepted": True,
            "validation_scope": "first-and-last-complete-steps",
            "descriptor_validation": "exact-work-membership-and-raw-group-identity",
            "workload_sha256": "a" * 64,
            "device_uuid": uuid,
            "case": case["case"],
            "gate": case["gate"],
            "body": case["body"],
            "backend": case["backend"],
            "precision": case["precision"],
            "condition_update": case["condition_update"],
            "tile": case["tile"],
            "stages": case["stages"],
        }

    def test_run_executes_exact_six_cases_and_retains_validated_outputs(self):
        fake_sanitizer = self.root / "compute-sanitizer"
        fake_sanitizer.write_text("test placeholder")
        output = self.root / "packet"
        calls = []

        def fake_run(argv, env, text, capture_output, check):
            calls.append((argv, env.copy()))
            self.assertTrue(capture_output)
            self.assertEqual(text, True)
            self.assertFalse(check)
            campaign_path = Path(argv[argv.index("--output") + 1])
            case = next(
                item for item in adapter.expected_cases()
                if item["gate"] in campaign_path.parent.name and item["body"] in campaign_path.parent.name
            )
            write_json(campaign_path, self.campaign_record(case))
            return SimpleNamespace(returncode=0, stdout="ERROR SUMMARY: 0 errors\n", stderr="")

        args = SimpleNamespace(build=self.build_dir, uuid="GPU-test", sanitizer="memcheck", output=output)
        with patch.object(adapter, "SANITIZER", fake_sanitizer), patch.object(adapter, "quiescence", return_value=[]), \
             patch.dict(os.environ, {
                 "AUTOBENCH_SCHEDULER_JOB_ID": "scheduler-job",
                 "AUTOBENCH_SCHEDULER_ATTEMPT_ID": "attempt-1",
             }, clear=True), \
             patch.object(adapter.subprocess, "run", side_effect=fake_run):
            metrics = adapter.run(args)

        self.assertTrue(metrics["success"])
        self.assertEqual(metrics["schema"], "cgs.packet.v1")
        self.assertEqual(metrics["suite"], "sanitize-chain")
        self.assertEqual(metrics["case_count"], 6)
        self.assertEqual(len(calls), 6)
        self.assertEqual({r["gate"] for r in metrics["cases"]}, {"supplied", "ste", "teacher"})
        self.assertEqual({r["body"] for r in metrics["cases"]}, {"product", "linear"})
        for row, (argv, env) in zip(metrics["cases"], calls):
            self.assertEqual(row["status"], "pass")
            self.assertEqual(argv[0], str(fake_sanitizer))
            self.assertEqual(argv[1:7], ["--tool", "memcheck", "--target-processes", "all", "--error-exitcode", "99"])
            self.assertEqual(argv[7], str(self.build_dir / "cgs_study"))
            option_values = dict(zip(argv[8::2], argv[9::2]))
            self.assertEqual(option_values, {
                "--mode": "campaign",
                "--case": adapter.CASE_ID,
                "--backend": "cohort-shared",
                "--gate": row["gate"],
                "--body": row["body"],
                "--precision": "fp16",
                "--condition-update": "none" if row["gate"] == "ste" else "incremental",
                "--tile": "64",
                "--stages": "2",
                "--warmups": "0",
                "--repeats": "1",
                "--output": str(output / row["campaign_path"]),
            })
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "GPU-test")
            case_dir = output / Path(row["campaign_path"]).parent
            self.assertTrue((output / row["campaign_path"]).is_file())
            self.assertTrue((case_dir / "sanitizer.stdout").is_file())
            self.assertTrue((case_dir / "sanitizer.stderr").is_file())
        self.assertEqual(json.loads((output / "metrics.json").read_text()), metrics)

    def test_run_records_failed_cases_and_keeps_invalid_raw_json_and_logs(self):
        fake_sanitizer = self.root / "compute-sanitizer"
        fake_sanitizer.write_text("test placeholder")
        output = self.root / "failed-packet"
        count = 0

        def fake_run(argv, **kwargs):
            nonlocal count
            count += 1
            campaign_path = Path(argv[argv.index("--output") + 1])
            if count == 2:
                campaign_path.write_text("{invalid json")
                return SimpleNamespace(returncode=99, stdout="", stderr="ERROR SUMMARY: 1 error\n")
            case = adapter.expected_cases()[count - 1]
            write_json(campaign_path, self.campaign_record(case))
            return SimpleNamespace(returncode=0, stdout="ERROR SUMMARY: 0 errors\n", stderr="")

        args = SimpleNamespace(build=self.build_dir, uuid="GPU-test", sanitizer="racecheck", output=output)
        with patch.object(adapter, "SANITIZER", fake_sanitizer), patch.object(adapter, "quiescence", return_value=[]), \
             patch.dict(os.environ, {
                 "AUTOBENCH_SCHEDULER_JOB_ID": "scheduler-job",
                 "AUTOBENCH_SCHEDULER_ATTEMPT_ID": "attempt-2",
                 "CUDA_VISIBLE_DEVICES": "GPU-test",
             }), \
             patch.object(adapter.subprocess, "run", side_effect=fake_run):
            metrics = adapter.run(args)

        self.assertFalse(metrics["success"])
        self.assertFalse(metrics["cases"][1]["success"])
        failed_json = output / metrics["cases"][1]["campaign_path"]
        failed_stdout = output / metrics["cases"][1]["sanitizer_stdout_path"]
        failed_stderr = output / metrics["cases"][1]["sanitizer_stderr_path"]
        self.assertEqual(failed_json.read_text(), "{invalid json")
        self.assertTrue(failed_stdout.is_file())
        self.assertIn("1 error", failed_stderr.read_text())
        self.assertEqual(len(list(output.glob("*/campaign.json"))), 6)
        self.assertEqual(json.loads((output / "metrics.json").read_text())["success"], False)

    def test_run_rejects_missing_admission_and_uuid_conflict_before_launch(self):
        fake_sanitizer = self.root / "compute-sanitizer"
        fake_sanitizer.write_text("test placeholder")
        args = SimpleNamespace(build=self.build_dir, uuid="GPU-requested", sanitizer="initcheck",
                               output=self.root / "unstarted")
        with patch.object(adapter, "SANITIZER", fake_sanitizer), \
             patch.dict(os.environ, {}, clear=True), \
             patch.object(adapter.subprocess, "run") as proc:
            with self.assertRaisesRegex(adapter.AdapterError, "Scheduler job and attempt IDs"):
                adapter.run(args)
            proc.assert_not_called()

        with patch.object(adapter, "SANITIZER", fake_sanitizer), \
             patch.dict(os.environ, {
                 "AUTOBENCH_SCHEDULER_JOB_ID": "job",
                 "AUTOBENCH_SCHEDULER_ATTEMPT_ID": "attempt",
                 "CUDA_VISIBLE_DEVICES": "GPU-other",
             }), \
             patch.object(adapter.subprocess, "run") as proc:
            with self.assertRaisesRegex(adapter.AdapterError, "conflicts"):
                adapter.run(args)
            proc.assert_not_called()


if __name__ == "__main__":
    unittest.main()
