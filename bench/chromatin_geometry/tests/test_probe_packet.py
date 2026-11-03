import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/probe_packet.py"
SPEC = importlib.util.spec_from_file_location("cgs_probe_packet_test", SCRIPT)
packet = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packet)


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_archive(path: Path, files: dict[str, bytes]) -> str:
    manifest = {"files": {name: {"size": len(data), "sha256": digest_bytes(data)}
                           for name, data in files.items()}}
    all_files = {**files, "SOURCE-MANIFEST.json": json.dumps(manifest).encode()}
    with tarfile.open(path, "w:gz") as archive:
        for name, data in all_files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
    return packet.sha256(path)


def campaign_record(case_id: str, backend: str, uuid: str, **changes) -> dict:
    data = {
        "schema": "cgs.campaign.v1", "status": "ok", "accuracy_accepted": True,
        "validation_scope": "first-and-last-complete-steps",
        "descriptor_validation": ("exact-gate-planes-no-materialized-workset" if backend == packet.ARMS[0]
                                  else "exact-work-membership-and-raw-group-identity"),
        "device_uuid": uuid, "case": case_id, "backend": backend,
        "gate": "supplied", "body": "product", "precision": "fp16",
        "condition_update": "none", "input": "synthetic",
        "reuse_horizon": packet.CASES[case_id]["reuse_horizon"], "tile": 32, "stages": 1,
        "cell_tiles_per_cta": 2, "workload_sha256": packet.CASES[case_id]["workload_sha256"],
        "workload_content_sha256": packet.COMMON_CONTENT_SHA256,
        "cell_order": "contiguous" if backend == packet.ARMS[0] else "exact-program-cohorts",
        "per_lane_group_mask": backend == packet.ARMS[0],
        "program_sorting": backend == packet.ARMS[1],
        "numerical_contract": {"input_storage": "fp16", "accumulation": "fp32"},
        "resident_complete_horizon_ms": [2.0, 2.2, 1.8],
        "recurrent_complete_horizon_ms": [2.0, 2.2, 1.8],
        "output_recovery_ms": [0.1, 0.1, 0.1],
        "cold_execution_preparation_ms": 3.0,
        "output_and_vjp_sha256": ["a" * 64, "b" * 64, "c" * 64],
    }
    data.update(changes)
    return data


class ProbePacketTests(unittest.TestCase):
    def test_fixed_six_round_order_is_balanced_ab_ba(self):
        orders = packet.expected_orders()
        self.assertEqual(orders, [list(packet.ARMS), list(reversed(packet.ARMS))] * 3)
        self.assertEqual(len(orders), 6)
        self.assertTrue(all(set(order) == set(packet.ARMS) for order in orders))
        self.assertEqual(sum(order == list(reversed(orders[index - 1]))
                             for index, order in enumerate(orders) if index % 2 == 1), 3)
        self.assertEqual(sum(order.index(packet.ARMS[0]) < order.index(packet.ARMS[1]) for order in orders), 3)

    @staticmethod
    def paired_rounds(control, candidate):
        return [{"arms": {packet.ARMS[0]: {"resident_median_ms": a},
                          packet.ARMS[1]: {"resident_median_ms": b}}}
                for a, b in zip(control, candidate)]

    def test_paired_statistic_rejects_false_gain_from_separate_arm_medians(self):
        rows = self.paired_rounds([1, 100, 100, 1, 100, 100], [1, 1, 100, 1, 1, 100])
        result = packet.paired_timing_summary(rows)
        self.assertEqual(result["median_paired_ratio_cohort_over_masked"], 1)
        self.assertAlmostEqual(result["median_cohort_latency_over_masked_latency"], 0.01)
        self.assertFalse(result["practical_latency_reduction_over_5_percent"])
        self.assertFalse(result["grouping_beneficial_for_probe"])

    def test_paired_statistic_finds_gain_hidden_by_separate_arm_medians(self):
        rows = self.paired_rounds([1, 2, 3, 4, 100, 200], [0.9, 1.8, 2.7, 40, 90, 180])
        result = packet.paired_timing_summary(rows)
        self.assertAlmostEqual(result["median_paired_ratio_cohort_over_masked"], 0.9)
        self.assertGreater(result["median_cohort_latency_over_masked_latency"], 1)
        self.assertTrue(result["practical_latency_reduction_over_5_percent"])
        self.assertEqual(result["grouping_beneficial_for_probe"],
                         result["paired_95_interval_supports_improvement"])

    def test_gain_requires_both_practical_size_and_paired_interval(self):
        certain = packet.paired_timing_summary(self.paired_rounds([1] * 6, [0.9] * 6))
        self.assertTrue(certain["grouping_beneficial_for_probe"])
        self.assertEqual(certain["paired_bootstrap_95_ratio_interval"], [0.9, 0.9])
        boundary = packet.paired_timing_summary(self.paired_rounds([100] * 6, [95] * 6))
        self.assertFalse(boundary["practical_latency_reduction_over_5_percent"])
        self.assertFalse(boundary["grouping_beneficial_for_probe"])
        noisy = packet.paired_timing_summary(self.paired_rounds([1] * 6, [0.9, 0.9, 0.9, 1.1, 1.1, 0.9]))
        self.assertTrue(noisy["practical_latency_reduction_over_5_percent"])
        self.assertFalse(noisy["paired_95_interval_supports_improvement"])
        self.assertFalse(noisy["grouping_beneficial_for_probe"])

    def test_paired_statistic_rejects_missing_rounds_and_invalid_times(self):
        with self.assertRaisesRegex(packet.PacketError, "six"):
            packet.paired_timing_summary(self.paired_rounds([1] * 5, [1] * 5))
        for bad in [0, -1, float("nan"), float("inf"), True]:
            with self.assertRaisesRegex(packet.PacketError, "finite and positive"):
                packet.paired_timing_summary(self.paired_rounds([1] * 6, [bad] * 6))

    def test_campaign_validation_pins_semantics_and_rejects_bad_digest_or_metrics(self):
        case_id = next(iter(packet.CASES))
        good = campaign_record(case_id, packet.ARMS[0], "GPU-test")
        packet.validate_campaign(good, case_id, packet.ARMS[0], "GPU-test", timing=True)
        with self.assertRaisesRegex(packet.PacketError, "workload_sha256"):
            packet.validate_campaign({**good, "workload_sha256": "0" * 64}, case_id,
                                     packet.ARMS[0], "GPU-test", timing=False)
        with self.assertRaisesRegex(packet.PacketError, "descriptor validation"):
            packet.validate_campaign({**good, "descriptor_validation": "unchecked"}, case_id,
                                     packet.ARMS[0], "GPU-test", timing=False)
        with self.assertRaisesRegex(packet.PacketError, "resident_complete_horizon_ms"):
            packet.validate_campaign({**good, "resident_complete_horizon_ms": [2.0]}, case_id,
                                     packet.ARMS[0], "GPU-test", timing=True)
        with self.assertRaisesRegex(packet.PacketError, "recovery"):
            packet.validate_campaign({**good, "output_recovery_ms": [0.0, 0.1, 0.1]}, case_id,
                                     packet.ARMS[0], "GPU-test", timing=True)

    def test_build_copies_two_binaries_and_binds_both_original_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "closure.tar.gz"
            archive_sha = write_archive(archive, {
                "Cellerator/bench/chromatin_geometry/scripts/run_packet.py": b"# guarded\n",
                "Cellerator/bench/chromatin_geometry/scripts/kda_driver.py": b"# tracked\n",
            })
            study = root / "study-bin"
            qualifier = root / "qualifier-bin"
            study.write_bytes(b"qualified study binary")
            qualifier.write_bytes(b"qualified test binary")
            study_receipt = root / "study-build.json"
            qualifier_receipt = root / "qualifier-build.json"
            study_receipt.write_text(json.dumps({
                "schema": "cgs.build.v1", "target": "cgs_study", "suite": "kda",
                "archive_sha256": archive_sha, "binary_sha256": packet.sha256(study),
            }))
            qualifier_receipt.write_text(json.dumps({
                "schema": "cgs.build.v1", "target": "cgs_grouping_probe_qualification", "suite": "probe",
                "archive_sha256": archive_sha, "binary_sha256": packet.sha256(qualifier),
            }))
            build_dir = root / "build"
            result = packet.build(argparse.Namespace(
                archive=archive, archive_sha256=archive_sha, binary=study, receipt=study_receipt,
                qualification_binary=qualifier, qualification_receipt=qualifier_receipt, build=build_dir,
            ))
            self.assertEqual(result["schema"], "cgs.grouping-probe-build.v1")
            self.assertEqual(packet.sha256(build_dir / "cgs_study"), packet.sha256(study))
            self.assertEqual(packet.sha256(build_dir / "cgs_grouping_probe_qualification"), packet.sha256(qualifier))
            self.assertEqual(packet.validate_build(build_dir)[0]["archive_sha256"], archive_sha)
            self.assertTrue((build_dir / "original-cgs-study-build.json").is_file())
            self.assertTrue((build_dir / "original-qualification-build.json").is_file())

    def test_build_rejects_receipt_mismatch_and_nonempty_output_without_deleting_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "closure.tar.gz"
            archive_sha = write_archive(archive, {
                "Cellerator/bench/chromatin_geometry/scripts/run_packet.py": b"# guarded\n",
                "Cellerator/bench/chromatin_geometry/scripts/kda_driver.py": b"# tracked\n",
            })
            study, qualifier = root / "study", root / "qualifier"
            study.write_bytes(b"study")
            qualifier.write_bytes(b"qualifier")
            receipt = root / "receipt.json"
            receipt.write_text(json.dumps({"schema": "cgs.build.v1", "target": "cgs_study",
                                           "archive_sha256": archive_sha, "binary_sha256": "f" * 64}))
            other = root / "other.json"
            other.write_text(json.dumps({"schema": "cgs.build.v1", "target": "cgs_grouping_probe_qualification",
                                         "archive_sha256": archive_sha, "binary_sha256": packet.sha256(qualifier)}))
            with self.assertRaisesRegex(packet.PacketError, "binary bytes"):
                packet.build(argparse.Namespace(archive=archive, archive_sha256=archive_sha, binary=study,
                                                receipt=receipt, qualification_binary=qualifier,
                                                qualification_receipt=other, build=root / "build"))
            occupied = root / "occupied"
            occupied.mkdir()
            marker = occupied / "keep.txt"
            marker.write_text("preserve")
            good = root / "good.json"
            good.write_text(json.dumps({"schema": "cgs.build.v1", "target": "cgs_study",
                                        "archive_sha256": archive_sha, "binary_sha256": packet.sha256(study)}))
            with self.assertRaisesRegex(packet.PacketError, "new or empty"):
                packet.build(argparse.Namespace(archive=archive, archive_sha256=archive_sha, binary=study,
                                                receipt=good, qualification_binary=qualifier,
                                                qualification_receipt=other, build=occupied))
            self.assertEqual(marker.read_text(), "preserve")

    def test_qualification_runs_standalone_then_all_sanitizers_then_four_oracles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "probe-packet"
            output.mkdir()
            study, qualifier = root / "cgs_study", root / "cgs_grouping_probe_qualification"
            study.write_text("binary")
            qualifier.write_text("qualifier")
            build_info = {
                "archive_sha256": "d" * 64,
                "cgs_study": {"binary_sha256": "a" * 64, "original_receipt_sha256": "b" * 64, "path": "cgs_study"},
                "qualification_binary": {"binary_sha256": "c" * 64, "original_receipt_sha256": "e" * 64,
                                          "path": "cgs_grouping_probe_qualification"},
            }
            calls = []

            def fake_run(argv, env, text, capture_output, check):
                calls.append(list(argv))
                if Path(argv[0]) == qualifier:
                    return argparse.Namespace(returncode=0, stdout=json.dumps({
                        "schema": "cgs.grouping-probe-qualification.v1", "status": "pass",
                        "device_uuid": "GPU-test",
                    }), stderr="")
                if Path(argv[0]) == packet.SANITIZER:
                    return argparse.Namespace(returncode=0, stdout="ERROR SUMMARY: 0 errors", stderr="")
                case_id = argv[argv.index("--case") + 1]
                backend = argv[argv.index("--backend") + 1]
                raw = Path(argv[argv.index("--output") + 1])
                raw.write_text(json.dumps(campaign_record(case_id, backend, "GPU-test")))
                return argparse.Namespace(returncode=0, stdout="native pass", stderr="")

            with patch.dict(os.environ, {"AUTOBENCH_SCHEDULER_JOB_ID": "qual-job",
                                         "AUTOBENCH_SCHEDULER_ATTEMPT_ID": "qual-attempt"}), \
                    patch.object(packet.subprocess, "run", side_effect=fake_run):
                result = packet.run_qualification(build_info, study, qualifier, output, "GPU-test")
            self.assertEqual(result["status"], "pass")
            self.assertEqual(len(calls), 1 + len(packet.SANITIZERS) + len(packet.CASES) * len(packet.ARMS))
            self.assertEqual(calls[0], [str(qualifier)])
            for call, sanitizer in zip(calls[1:1 + len(packet.SANITIZERS)], packet.SANITIZERS):
                self.assertIn("--target-processes", call)
                self.assertIn("all", call)
                self.assertEqual(call[call.index("--tool") + 1], sanitizer)
                self.assertEqual(call[call.index("--error-exitcode") + 1], "99")
            proof = json.loads((output / "qualification-proof.json").read_text())
            self.assertEqual(proof["status"], "pass")
            self.assertEqual(set(proof["campaigns"]), set(packet.CASES))
            self.assertTrue((output / "raw/qualification/sanitizer-racecheck.stderr.txt").is_file())

    def test_qualification_stops_on_first_unclean_sanitizer_and_preserves_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "probe-packet"
            output.mkdir()
            study, qualifier = root / "study", root / "qualifier"
            study.write_text("study")
            qualifier.write_text("qualifier")
            build_info = {"archive_sha256": "d" * 64,
                          "cgs_study": {"binary_sha256": "a" * 64, "original_receipt_sha256": "b" * 64},
                          "qualification_binary": {"binary_sha256": "c" * 64, "original_receipt_sha256": "e" * 64}}
            calls = []

            def fake_run(argv, env, text, capture_output, check):
                calls.append(list(argv))
                if Path(argv[0]) == qualifier:
                    return argparse.Namespace(returncode=0, stdout=json.dumps({
                        "schema": "cgs.grouping-probe-qualification.v1", "status": "pass"}), stderr="")
                return argparse.Namespace(returncode=0, stdout="no summary", stderr="sanitizer output retained")

            with patch.dict(os.environ, {"AUTOBENCH_SCHEDULER_JOB_ID": "qual-job",
                                         "AUTOBENCH_SCHEDULER_ATTEMPT_ID": "qual-attempt"}), \
                    patch.object(packet.subprocess, "run", side_effect=fake_run):
                with self.assertRaisesRegex(packet.PacketError, "memcheck"):
                    packet.run_qualification(build_info, study, qualifier, output, "GPU-test")
            self.assertEqual(len(calls), 2)
            self.assertEqual((output / "raw/qualification/sanitizer-memcheck.stderr.txt").read_text(),
                             "sanitizer output retained")
            progress = json.loads((output / "raw/qualification/qualification-progress.json").read_text())
            self.assertEqual(progress["status"], "fail")
            self.assertFalse((output / "qualification-proof.json").exists())

    def test_nested_output_and_scheduler_uuid_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(packet.PacketError, "dedicated"):
                packet.packet_directory(root / "output")
            nested = root / "probe-packet"
            nested.mkdir()
            marker = nested / "existing.log"
            marker.write_text("preserve")
            with self.assertRaisesRegex(packet.PacketError, "preserving"):
                packet.packet_directory(nested)
            self.assertEqual(marker.read_text(), "preserve")
            with patch.dict(os.environ, {"AUTOBENCH_SCHEDULER_JOB_ID": "job",
                                         "AUTOBENCH_SCHEDULER_ATTEMPT_ID": "attempt",
                                         "CUDA_VISIBLE_DEVICES": "GPU-other"}):
                with self.assertRaisesRegex(packet.PacketError, "exactly match"):
                    packet.scheduler_identity("GPU-test")

    def test_timing_collects_six_paired_rounds_and_keeps_cold_recovery_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "probe-packet"
            output.mkdir()
            source = root / "source"
            (source / "Cellerator/bench/chromatin_geometry/scripts").mkdir(parents=True)
            (source / "Cellerator/bench/chromatin_geometry/scripts/run_packet.py").write_text("# guard\n")
            (source / "Cellerator/bench/chromatin_geometry/scripts/kda_driver.py").write_text("# driver\n")
            build_info = {
                "archive_sha256": "d" * 64,
                "cgs_study": {"binary_sha256": "a" * 64, "original_receipt_sha256": "b" * 64},
                "qualification_binary": {"binary_sha256": "c" * 64, "original_receipt_sha256": "e" * 64},
            }
            proof = {
                "schema": "cgs.grouping-probe-proof.v1", "status": "pass",
                "archive_sha256": "d" * 64, "binary_sha256": "a" * 64,
                "qualification_binary_sha256": "c" * 64,
                "cgs_study_build_receipt_sha256": "b" * 64,
                "qualification_build_receipt_sha256": "e" * 64,
                "device_uuid": "GPU-test", "scheduler_job_id": "qualification-job",
                "scheduler_attempt_id": "qualification-attempt",
                "campaigns": {case_id: {"workload_sha256": values["workload_sha256"],
                                          "workload_content_sha256": packet.COMMON_CONTENT_SHA256,
                                          "arms": {arm: {"raw_path": f"{arm}.json", "raw_sha256": "f" * 64}
                                                   for arm in packet.ARMS}}
                              for case_id, values in packet.CASES.items()},
                "sanitizers": {tool: {"status": "pass", "zero_error_summary": True,
                                      "stdout_sha256": "1" * 64, "stderr_sha256": "2" * 64}
                               for tool in packet.SANITIZERS},
                "qualification_checks": [{"status": "pass"}],
            }
            proof_path = root / "qualification-proof.json"
            proof_path.write_text(json.dumps(proof))
            invocation_count = 0
            driver = SimpleNamespace()

            def fake_tracked(argv, events, lock_path, metadata, raw_path):
                return packet.subprocess.CompletedProcess(argv, 0, "native stdout", "native stderr")

            def fake_run_one(binary, backend, case, warmups, repeats, raw_path):
                nonlocal invocation_count
                invocation_count += 1
                self.assertEqual((warmups, repeats), (1, 3))
                self.assertEqual(case["input"]["campaign_workload_sha256"],
                                 packet.CASES[case["id"]]["workload_sha256"])
                data = campaign_record(case["id"], backend, "GPU-test")
                base = 1.0 if backend == packet.ARMS[0] else 1.2
                data.update({"resident_complete_horizon_ms": [base, base + 0.1, base - 0.1],
                             "recurrent_complete_horizon_ms": [base, base + 0.1, base - 0.1],
                             "output_recovery_ms": [0.2, 0.2, 0.2],
                             "cold_execution_preparation_ms": 4.0})
                Path(raw_path).write_text(json.dumps(data))
                argv = [str(binary), "--mode", "campaign", "--case", case["id"], "--backend", backend,
                        "--gate", "supplied", "--body", "product", "--precision", "fp16",
                        "--condition-update", "none", "--tile", "32", "--stages", "1",
                        "--warmups", "1", "--repeats", "3", "--output", str(raw_path)]
                driver.run_tracked(argv, os.environ["CGS_LIFETIME_EVENTS"], os.environ["CGS_PROCESS_LOCK"],
                                   {"device_uuid": "GPU-test", "binary_sha256": "a" * 64,
                                    "backend": backend}, raw_path)
                return data

            driver.run_tracked = fake_tracked
            driver.run_one = fake_run_one
            run_packet = SimpleNamespace()

            with patch.dict(os.environ, {"AUTOBENCH_SCHEDULER_JOB_ID": "timing-job",
                                         "AUTOBENCH_SCHEDULER_ATTEMPT_ID": "timing-attempt",
                                         "CUDA_VISIBLE_DEVICES": "GPU-test",
                                         "CGS_LIFETIME_EVENTS": str(root / "lifetimes.jsonl"),
                                         "CGS_PROCESS_LOCK": str(root / "lock")}):
                with patch.object(packet, "_load_module", side_effect=[run_packet, driver]):
                    result = packet.run_timing(build_info, root / "cgs_study", source, output,
                                               "GPU-test", next(iter(packet.CASES)), proof_path)
            self.assertEqual(result["status"], "pass")
            self.assertEqual(invocation_count, 12)
            self.assertEqual(len(result["rounds"]), 6)
            self.assertEqual(sum(row["order"].index(packet.ARMS[0]) < row["order"].index(packet.ARMS[1])
                                 for row in result["rounds"]), 3)
            self.assertAlmostEqual(result["median_paired_ratio_cohort_over_masked"], 1.2)
            first = result["rounds"][0]["arms"][packet.ARMS[0]]
            self.assertEqual(first["cold_execution_preparation_ms"], 4.0)
            self.assertEqual(first["output_recovery_ms"], [0.2, 0.2, 0.2])
            self.assertEqual(first["resident_complete_horizon_ms"], [1.0, 1.1, 0.9])
            self.assertTrue((output / first["raw_path"]).is_file())
            self.assertTrue((output / first["stdout_path"]).is_file())

    def test_bad_timing_identity_fails_without_retry_and_retains_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "probe-packet"
            output.mkdir()
            source = root / "source"
            (source / "Cellerator/bench/chromatin_geometry/scripts").mkdir(parents=True)
            build_info = {
                "archive_sha256": "d" * 64,
                "cgs_study": {"binary_sha256": "a" * 64, "original_receipt_sha256": "b" * 64},
                "qualification_binary": {"binary_sha256": "c" * 64, "original_receipt_sha256": "e" * 64},
            }
            proof = {
                "schema": "cgs.grouping-probe-proof.v1", "status": "pass",
                "archive_sha256": "d" * 64, "binary_sha256": "a" * 64,
                "qualification_binary_sha256": "c" * 64,
                "cgs_study_build_receipt_sha256": "b" * 64,
                "qualification_build_receipt_sha256": "e" * 64,
                "device_uuid": "GPU-test", "scheduler_job_id": "qual-job", "scheduler_attempt_id": "qual-attempt",
                "campaigns": {case_id: {"workload_sha256": values["workload_sha256"],
                                          "workload_content_sha256": packet.COMMON_CONTENT_SHA256,
                                          "arms": {arm: {"raw_path": "raw.json", "raw_sha256": "f" * 64}
                                                   for arm in packet.ARMS}}
                              for case_id, values in packet.CASES.items()},
                "sanitizers": {tool: {"status": "pass", "zero_error_summary": True}
                               for tool in packet.SANITIZERS},
                "qualification_checks": [{"status": "pass"}],
            }
            proof_path = root / "proof.json"
            proof_path.write_text(json.dumps(proof))
            driver = SimpleNamespace()
            calls = []

            def fake_run_one(binary, backend, case, warmups, repeats, raw_path):
                calls.append(backend)
                data = campaign_record(case["id"], backend, "GPU-test", workload_content_sha256="0" * 64)
                Path(raw_path).write_text(json.dumps(data))
                driver.run_tracked([str(binary)], os.environ["CGS_LIFETIME_EVENTS"],
                                   os.environ["CGS_PROCESS_LOCK"], {}, raw_path)
                return data

            driver.run_tracked = lambda argv, events, lock, metadata, raw: packet.subprocess.CompletedProcess(
                argv, 0, "output", "")
            driver.run_one = fake_run_one
            with patch.dict(os.environ, {"AUTOBENCH_SCHEDULER_JOB_ID": "time-job",
                                         "AUTOBENCH_SCHEDULER_ATTEMPT_ID": "time-attempt",
                                         "CUDA_VISIBLE_DEVICES": "GPU-test",
                                         "CGS_LIFETIME_EVENTS": str(root / "events"),
                                         "CGS_PROCESS_LOCK": str(root / "lock")}):
                with patch.object(packet, "_load_module", side_effect=[SimpleNamespace(), driver]):
                    with self.assertRaisesRegex(packet.PacketError, "workload_content_sha256"):
                        packet.run_timing(build_info, root / "cgs_study", source, output,
                                          "GPU-test", next(iter(packet.CASES)), proof_path)
            self.assertEqual(len(calls), 1)
            progress = json.loads((output / "timing-progress.json").read_text())
            self.assertEqual(progress["status"], "fail")
            failed_arm = progress["rounds"][0]["arms"][calls[0]]
            self.assertEqual(failed_arm["status"], "failed")
            self.assertTrue((output / failed_arm["raw_path"]).is_file())
            self.assertTrue((output / f"raw/timing/round-00/{calls[0]}.stderr.txt").is_file())

    def test_timing_guard_contamination_keeps_packet_unqualified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build = root / "build"
            build.mkdir()
            (build / "build.json").write_text("{}")
            study = root / "cgs_study"
            qualifier = root / "qualifier"
            source = root / "source"
            proof = root / "qualification-proof.json"
            study.write_text("study")
            qualifier.write_text("qualifier")
            source.mkdir()
            proof.write_text("{}")
            build_info = {
                "archive_sha256": "d" * 64,
                "cgs_study": {"binary_sha256": "a" * 64, "original_receipt_sha256": "b" * 64,
                              "path": "cgs_study"},
                "qualification_binary": {"binary_sha256": "c" * 64,
                                          "original_receipt_sha256": "e" * 64,
                                          "path": "cgs_grouping_probe_qualification"},
            }

            def fake_guard(command, env, uuid, output, binary_sha256):
                retained_proof = Path(command[command.index("--qualification-proof") + 1])
                self.assertEqual(retained_proof, output / "qualification-proof.json")
                self.assertEqual(retained_proof.read_bytes(), proof.read_bytes())
                (output / "collection.json").write_text(json.dumps({"schema": "cgs.grouping-probe-timing.v1",
                                                                     "status": "pass"}))
                (output / "serial-telemetry.json").write_text(json.dumps({"clean": False}))
                return packet.subprocess.CompletedProcess(command, 0, "", ""), False

            run_packet = SimpleNamespace(run_serial_monitored=fake_guard)
            args = argparse.Namespace(phase="r1", build=build, uuid="GPU-test",
                                      output=root / "output/probe-packet", suite="kda",
                                      qualification_proof=proof)
            with patch.dict(os.environ, {"AUTOBENCH_SCHEDULER_JOB_ID": "timing-job",
                                         "AUTOBENCH_SCHEDULER_ATTEMPT_ID": "timing-attempt",
                                         "CUDA_VISIBLE_DEVICES": "GPU-test"}), \
                    patch.object(packet, "validate_build", return_value=(build_info, study, qualifier, source)), \
                    patch.object(packet, "_load_module", return_value=run_packet):
                result = packet.run(args)
            self.assertFalse(result["success"])
            self.assertFalse(result["serial_telemetry_clean"])
            self.assertEqual(result["status"], "fail")
            self.assertEqual(result["qualification_proof_path"], "qualification-proof.json")
            self.assertEqual(result["qualification_proof_sha256"], packet.sha256(proof))
            self.assertEqual((root / "output/probe-packet/qualification-proof.json").read_bytes(),
                             proof.read_bytes())
            self.assertTrue((root / "output/probe-packet/metrics.json").is_file())

            original_copy = packet.shutil.copyfile

            def corrupt_proof_copy(src, dst, *, follow_symlinks=True):
                if Path(src) == proof:
                    Path(dst).write_bytes(b"tampered proof")
                    return dst
                return original_copy(src, dst, follow_symlinks=follow_symlinks)

            bad_args = argparse.Namespace(**vars(args))
            bad_args.output = root / "tampered-output/probe-packet"
            with patch.dict(os.environ, {"AUTOBENCH_SCHEDULER_JOB_ID": "timing-job",
                                         "AUTOBENCH_SCHEDULER_ATTEMPT_ID": "timing-attempt",
                                         "CUDA_VISIBLE_DEVICES": "GPU-test"}), \
                    patch.object(packet, "validate_build", return_value=(build_info, study, qualifier, source)), \
                    patch.object(packet, "_load_module", return_value=run_packet), \
                    patch.object(packet.shutil, "copyfile", side_effect=corrupt_proof_copy), \
                    patch.object(run_packet, "run_serial_monitored") as monitor:
                with self.assertRaisesRegex(packet.PacketError, "retained qualification proof"):
                    packet.run(bad_args)
                monitor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
