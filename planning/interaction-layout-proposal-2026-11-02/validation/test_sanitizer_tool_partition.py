from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest


SCRIPT = Path(__file__).with_name("sanitizer_tool_partition.py")
SPEC = importlib.util.spec_from_file_location("fi_sanitizer_partition_test", SCRIPT)
assert SPEC and SPEC.loader
partition = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(partition)
DISPATCH_FIXTURE = Path(__file__).with_name("sanitizer-minimal-dispatch-fixture.json")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    return digest(path.read_bytes())


def contract_evidence() -> dict[str, dict[str, float]]:
    return {name: {"max_error_to_allowance": 0.5}
            for name in partition.EVIDENCE_KEYS}


class SanitizerPartitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dispatch_fixture = json.loads(DISPATCH_FIXTURE.read_text(encoding="utf-8"))
        self.tmp = tempfile.TemporaryDirectory(prefix="fi-sanitizer-partition-")
        self.root = Path(self.tmp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.suite_path = self.source / "bench/factorized_interactions/run_sanitizer_suite.py"
        self.validator_path = self.source / "bench/factorized_interactions/validate_runner.py"
        self.context_path = self.source / "bench/factorized_interactions/experiments/in_job_kda.py"
        for path, content in ((self.suite_path, b"# fixed test suite\n"),
                              (self.validator_path, b"# independent validator\n"),
                              (self.context_path, b"# context guard\n")):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        self.source_entries = []
        for path in (self.suite_path, self.validator_path, self.context_path):
            self.source_entries.append({
                "path": path.relative_to(self.source).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_digest(path),
            })
        self.source_entries.sort(key=lambda row: row["path"])
        self.source_closure, _ = partition.source_entries_digest(self.source_entries)

        self.native = self.root / "native_build"
        cmake = self.native / "cmake"
        cmake.mkdir(parents=True)
        self.baseline_runner = cmake / "baseline_runner"
        self.candidate_runner = cmake / "candidate_runner"
        self.component_archive = cmake / "libfactorized.a"
        self.baseline_runner.write_bytes(b"baseline binary bytes")
        self.candidate_runner.write_bytes(b"candidate binary bytes")
        self.component_archive.write_bytes(b"static component bytes")
        self.candidate_manifest = self.native / "factorized_interactions_build_manifest.json"
        self.baseline_manifest = self.native / "factorized_interactions_baseline_build_manifest.json"
        common = {
            "schema": "cellerator.factorized_interactions.build.v1",
            "architecture": "sm_70",
            "fast_math": False,
            "source_closure_sha256": self.source_closure,
            "source_files": self.source_entries,
        }
        candidate = {
            **common,
            "binaries": {
                "runner": {"path": "cmake/candidate_runner",
                           "bytes": self.candidate_runner.stat().st_size,
                           "sha256": file_digest(self.candidate_runner)},
                "factorized_interaction_cuda": {
                    "path": "cmake/libfactorized.a",
                    "bytes": self.component_archive.stat().st_size,
                    "sha256": file_digest(self.component_archive)},
            },
        }
        self.candidate_manifest.write_text(json.dumps(candidate, sort_keys=True),
                                           encoding="utf-8")
        baseline = {
            **common,
            "binaries": {
                "runner": {"path": "cmake/baseline_runner",
                           "bytes": self.baseline_runner.stat().st_size,
                           "sha256": file_digest(self.baseline_runner)},
                "factorized_interaction_cuda": {
                    "path": "cmake/libfactorized.a",
                    "bytes": self.component_archive.stat().st_size,
                    "sha256": file_digest(self.component_archive)},
            },
            "candidate_runner": {
                "binary_path": "cmake/candidate_runner",
                "binary_sha256": file_digest(self.candidate_runner),
                "manifest_path": self.candidate_manifest.name,
                "manifest_sha256": file_digest(self.candidate_manifest),
            },
        }
        self.baseline_manifest.write_text(json.dumps(baseline, sort_keys=True),
                                          encoding="utf-8")

        self.tool_home = self.root / "compute-sanitizer"
        self.tool_home.mkdir()
        self.tool_binary = self.tool_home / "compute-sanitizer"
        self.tool_binary.write_text(
            "#!/usr/bin/env python3\n"
            "print('NVIDIA (R) Compute Sanitizer')\n"
            "print('Version 2025.2.1.0 (build 35969825) (public-release)')\n",
            encoding="utf-8")
        self.tool_binary.chmod(0o755)
        (self.tool_home / "support.dat").write_bytes(b"captured support fixture")
        self.toolhome_original = partition.TOOL_HOME_DEFAULT
        partition.TOOL_HOME_DEFAULT = self.tool_home

        self.job_id = "job-test-1"
        self.attempt_id = "attempt-test-1"
        self.uuid = "GPU-6c1cac7f-a360-0aef-ba98-2828bfd1db1a"
        self.unit = partition.load_trusted_module
        expected_unit = "device-scheduler-attempt-" + hashlib.sha256(
            self.attempt_id.encode()).hexdigest()[:32] + ".service"
        self.cgroup = f"0::/user.slice/user@1000.service/app.slice/{expected_unit}\n"
        self.env = os.environ.copy()
        self.env.update({
            "DEVICE_SCHEDULER_JOB_ID": self.job_id,
            "DEVICE_SCHEDULER_ATTEMPT_ID": self.attempt_id,
            "AUTOBENCH_SCHEDULER_JOB_ID": self.job_id,
            "AUTOBENCH_SCHEDULER_ATTEMPT_ID": self.attempt_id,
            "CUDA_VISIBLE_DEVICES": self.uuid,
            "NVCOMPILER_COMPUTE_SANITIZER_HOME": str(self.tool_home),
        })
        self.output_root = self.root / "output"
        self.output_root.mkdir()
        self.artifact_dir = self.output_root / "cases"
        self.output_path = self.output_root / "partition.json"
        self.fake_suite = _FakeSuite(self)

    def tearDown(self) -> None:
        partition.TOOL_HOME_DEFAULT = self.toolhome_original
        self.tmp.cleanup()

    def _context_module(self):
        owner = self

        def validate_context(env, cgroup_text, expected_uuid):
            if env.get("DEVICE_SCHEDULER_JOB_ID") != owner.job_id or \
                    env.get("AUTOBENCH_SCHEDULER_JOB_ID") != owner.job_id or \
                    env.get("DEVICE_SCHEDULER_ATTEMPT_ID") != owner.attempt_id or \
                    env.get("AUTOBENCH_SCHEDULER_ATTEMPT_ID") != owner.attempt_id or \
                    env.get("CUDA_VISIBLE_DEVICES") != expected_uuid or \
                    expected_uuid != owner.uuid or owner.cgroup not in cgroup_text:
                raise ValueError("fixture outer context mismatch")
            unit = owner.cgroup.strip().split("/")[-1]
            return {"schema": "cellerator.factorized_interactions.outer_execution_context.v1",
                    "status": "observed", "scheduler_job_id": owner.job_id,
                    "scheduler_attempt_id": owner.attempt_id,
                    "device_uuid": owner.uuid, "cgroup": owner.cgroup.strip()[3:],
                    "unit_name": unit}

        return types.SimpleNamespace(validate_context=validate_context)

    def _module_loader(self, name, path):
        if name == "factorized_interaction_outer_context":
            return self._context_module()
        if name == "factorized_interaction_sanitizer_suite":
            return self.fake_suite
        return partition.load_trusted_module(name, path)

    def execute(self, *, tool="memcheck", env=None):
        return partition.execute_partition(
            source=self.source, tool=tool,
            runner=self.baseline_runner, build_manifest=self.baseline_manifest,
            expected_uuid=self.uuid, output_root=self.output_root,
            artifact_dir=self.artifact_dir, output_path=self.output_path,
            environ=self.env if env is None else env, cgroup_text=self.cgroup,
            module_loader=self._module_loader)

    def test_success_runs_one_tool_and_records_all_fourteen_cases(self):
        report = self.execute(tool="racecheck")
        self.assertEqual(report["schema"], partition.SCHEMA)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["tool"], "racecheck")
        self.assertEqual(report["expected_tools"], list(partition.EXPECTED_TOOLS))
        self.assertEqual(report["expected_case_count"], 14)
        self.assertEqual(report["case_count"], 14)
        self.assertEqual(report["passed"], 14)
        self.assertFalse(report["timed"])
        self.assertFalse(report["repeat_vjp_checked"])
        self.assertEqual(report["outer_receipt_verification"], "pending")
        self.assertEqual(len(report["artifact_inventory"]), 14)
        self.assertIsNotNone(self.fake_suite.last_report)
        candidate_rows = [row for row in self.fake_suite.last_report["results"]
                          if "candidate" in row]
        self.assertEqual(len(candidate_rows), 2)
        row_keys = {"tool", "route", "candidate", "workload", "exit_code", "validation"}
        validation_keys = {
            "schema", "validated", "timed", "workload", "route", "activation_storage",
            "parameter_storage", "output_storage", "gradient_mode", "scheduler_job_id",
            "build_provenance", "contract_evidence", "native_execution",
        }
        for row in candidate_rows:
            self.assertEqual(set(row), row_keys)
            self.assertEqual(set(row["validation"]), validation_keys)
            actual = self.dispatch_fixture[row["candidate"]]["dtype32_case"]
            native = row["validation"]["native_execution"]
            for field in ("selected_backend", "actual_implementation", "structural_layout"):
                self.assertEqual(native[field], actual[field])
        self.assertEqual(self.fake_suite.TOOLS, ("racecheck",))
        self.assertEqual(self.fake_suite.COMPUTE_SANITIZER, str(self.tool_binary))
        self.assertEqual(report["component_archive_sha256"], file_digest(self.component_archive))
        self.assertEqual(report["shared_library_sha256"], file_digest(self.component_archive))
        self.assertTrue(self.output_path.is_file())

    def test_dispatch_fixture_matches_actual_accepted_dtype32_report(self):
        fixture = self.dispatch_fixture
        report_relative = Path(fixture["source_dtype_report"]["path"])
        report_path = next((parent / report_relative
                            for parent in Path(__file__).resolve().parents
                            if (parent / report_relative).is_file()), None)
        if report_path is None:
            self.skipTest("source dtype report is not packaged beside this portable fixture")
        self.assertEqual(file_digest(report_path), fixture["source_dtype_report"]["sha256"])
        source_report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(source_report["status"], "pass")
        self.assertEqual(source_report["expected_case_count"], 32)
        for mode in ("minimal_direct", "direct_unique_output"):
            case = fixture[mode]
            expected = case["dtype32_case"]
            row = next(row for row in source_report["records"]
                       if row["mode"] == mode and row["activation"] == "fp16" and
                       row["parameter"] == "fp16" and
                       row["output"] == expected["output"])
            for field, value in expected.items():
                self.assertEqual(row[field], value, (mode, field))
            summary_path = report_path.parent / case["validator_summary_relative_path"]
            self.assertEqual(file_digest(summary_path), case["validator_summary_sha256"])
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["native_execution"], case["native_execution"])
            self.assertEqual(summary["native_execution"]["candidate"], mode)
            self.assertEqual(summary["native_execution"]["selected_backend"], "direct")
            self.assertEqual(summary["native_execution"]["structural_layout"], "canonical_direct")
        minimal = fixture["minimal_direct"]["dtype32_case"]
        self.assertEqual((minimal["selected_backend"], minimal["actual_implementation"],
                          minimal["structural_layout"]),
                         ("direct", "direct", "canonical_direct"))

    def test_rejects_wrong_visible_device_before_running_suite(self):
        bad_env = dict(self.env)
        bad_env["CUDA_VISIBLE_DEVICES"] = "GPU-00000000-0000-0000-0000-000000000000"
        with self.assertRaises(partition.PartitionError):
            self.execute(env=bad_env)
        self.assertFalse(self.fake_suite.called)
        self.assertFalse(self.output_path.exists())

    def test_rejects_unmanifested_source_file(self):
        (self.source / "bench/factorized_interactions/rogue.py").write_text("bad\n")
        with self.assertRaisesRegex(partition.PartitionError, "source tree file set differs"):
            self.execute()
        self.assertFalse(self.fake_suite.called)

    def test_rejects_other_tool_results_and_preserves_failure_evidence(self):
        self.fake_suite.extra_tool = True
        with self.assertRaisesRegex(partition.PartitionError, "exactly the requested sanitizer"):
            self.execute(tool="initcheck")
        self.assertTrue(self.output_path.is_file())
        report = json.loads(self.output_path.read_text())
        self.assertEqual(report["status"], "fail")
        self.assertTrue(report["errors"])
        self.assertTrue((self.output_root / "fixed-sanitizer-suite.json").is_file())

    def test_rejects_candidate_metadata_that_did_not_select_unique_output(self):
        self.fake_suite.bad_unique_dispatch = True
        with self.assertRaisesRegex(partition.PartitionError, "intended layout"):
            self.execute()
        report = json.loads(self.output_path.read_text())
        self.assertEqual(report["status"], "fail")

    def test_rejects_minimal_direct_candidate_that_uses_packet_layout(self):
        self.fake_suite.bad_minimal_layout = True
        with self.assertRaisesRegex(partition.PartitionError, "intended layout"):
            self.execute()
        report = json.loads(self.output_path.read_text())
        self.assertEqual(report["status"], "fail")

    def test_rejects_native_build_mutation_and_preserves_failure_report(self):
        self.fake_suite.mutate_build = self.baseline_runner
        with self.assertRaisesRegex(partition.PartitionError, "native build changed"):
            self.execute()
        report = json.loads(self.output_path.read_text())
        self.assertEqual(report["status"], "fail")


class _FakeSuite:
    def __init__(self, owner: SanitizerPartitionTests) -> None:
        self.owner = owner
        self.TOOLS = partition.EXPECTED_TOOLS
        self.COMPUTE_SANITIZER = ""
        self.called = False
        self.extra_tool = False
        self.bad_unique_dispatch = False
        self.bad_minimal_layout = False
        self.mutate_build: Path | None = None
        self.last_report: dict[str, object] | None = None

    def main(self) -> int:
        self.called = True
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--runner", type=Path, required=True)
        parser.add_argument("--build-manifest", type=Path, required=True)
        parser.add_argument("--expected-uuid", required=True)
        parser.add_argument("--artifact-dir", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args(sys.argv[1:])
        self.assert_one_tool()
        baseline_manifest_bytes = args.build_manifest.read_bytes()
        baseline = json.loads(baseline_manifest_bytes)
        candidate_manifest_path = args.build_manifest.parent / baseline["candidate_runner"]["manifest_path"]
        candidate_manifest_bytes = candidate_manifest_path.read_bytes()
        candidate = json.loads(candidate_manifest_bytes)
        tool_version = subprocess.run(
            [self.COMPUTE_SANITIZER, "--version"], check=True, text=True,
            capture_output=True).stdout.strip() + "\n"
        # The wrapper strips the combined output, while the original suite strips
        # its own combined stdout/stderr. Keep the same normalized string.
        tool_version = tool_version.strip()
        result_rows = []
        baseline_cases = (
            ("correctness_ordered_tied_zero", "fp16", "fp16", "fp16"),
            ("correctness_context_cancellation_middle", "fp32", "fp32", "fp32"),
            ("correctness_context_cancellation_tail", "fp32", "fp32", "fp32"),
            ("correctness_fma_product_residual_cancellation", "fp32", "fp32", "fp32"),
            ("a4_m256_u32_fanin64_tied", "fp16", "fp16", "fp16"),
            ("correctness_empty_mechanisms", "fp16", "fp16", "fp16"),
        )
        for tool in self.TOOLS:
            tools = [tool, "racecheck"] if self.extra_tool else [tool]
            for route in ("packet", "direct"):
                for workload, activation, parameter, output in baseline_cases:
                    backend = route
                    native = {
                        "device_uuid": args.expected_uuid,
                        "scheduler_job_id": self.owner.job_id,
                        "gradient_mode": "overwrite",
                        "requested_route": route,
                        "selected_backend": backend,
                    }
                    validation = self._validation(
                        workload, route, activation, parameter, output, native,
                        baseline["source_closure_sha256"],
                        baseline["binaries"]["runner"]["sha256"],
                        digest(baseline_manifest_bytes))
                    result_rows.append({"tool": tool, "route": route,
                                        "workload": workload, "exit_code": 0,
                                        "validation": validation})
                    log = args.artifact_dir / f"case-{len(result_rows):02d}.log"
                    log.parent.mkdir(parents=True, exist_ok=True)
                    log.write_text("fake sanitizer log\n", encoding="utf-8")
            for candidate_name, workload, activation, parameter, output in (
                    ("direct_unique_output", "packet_a4_m32_u16", "fp16", "fp16", "fp32"),
                    ("minimal_direct", "correctness_ordered_tied_zero", "fp16", "fp16", "fp16")):
                fixture = self.owner.dispatch_fixture[candidate_name]
                actual = fixture["dtype32_case"]["actual_implementation"]
                if self.bad_unique_dispatch and candidate_name == "direct_unique_output":
                    actual = "direct"
                layout = ("full_packet" if self.bad_minimal_layout and
                          candidate_name == "minimal_direct" else
                          fixture["dtype32_case"]["structural_layout"])
                native = dict(fixture["native_execution"])
                native.update({"device_uuid": args.expected_uuid,
                               "scheduler_job_id": self.owner.job_id,
                               "gradient_mode": "overwrite",
                               "requested_route": "direct",
                               "candidate": candidate_name,
                               "actual_implementation": actual,
                               "structural_layout": layout})
                if candidate_name == "minimal_direct":
                    facts = self.owner.dispatch_fixture["sanitizer_workload_facts"]
                    native.update({"cells": facts["cells"], "channels": facts["channels"],
                                   "mechanisms": facts["mechanisms"], "outputs": facts["outputs"],
                                   "structural_maxima": facts["maxima"],
                                   "terms": facts["total_terms"]})
                    native.pop("memory_bytes", None)
                validation = self._validation(
                    workload, "direct", activation, parameter, output, native,
                    candidate["source_closure_sha256"],
                    candidate["binaries"]["runner"]["sha256"],
                    digest(candidate_manifest_bytes))
                result_rows.append({"tool": tool, "route": "direct",
                                    "candidate": candidate_name,
                                    "workload": workload, "exit_code": 0,
                                    "validation": validation})
                log = args.artifact_dir / f"case-{len(result_rows):02d}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text("fake sanitizer log\n", encoding="utf-8")
            if self.mutate_build is not None:
                self.mutate_build.write_bytes(self.mutate_build.read_bytes() + b"mutation")
        suite_report = {
            "schema": "cellerator.factorized_interactions.sanitizer_suite.v1",
            "scheduler_job_id": self.owner.job_id,
            "device_uuid": args.expected_uuid,
            "compute_sanitizer": self.COMPUTE_SANITIZER,
            "compute_sanitizer_version": tool_version,
            "build_manifest_sha256": digest(baseline_manifest_bytes),
            "source_closure_sha256": baseline["source_closure_sha256"],
            "runner_binary_sha256": baseline["binaries"]["runner"]["sha256"],
            "candidate_build_manifest_sha256": digest(candidate_manifest_bytes),
            "candidate_runner_binary_sha256": candidate["binaries"]["runner"]["sha256"],
            "timed": False,
            "case_count": len(result_rows),
            "passed": len(result_rows),
            "results": result_rows,
        }
        if self.extra_tool:
            suite_report["results"][0]["tool"] = "racecheck"
        self.last_report = suite_report
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(suite_report, sort_keys=True), encoding="utf-8")
        print(json.dumps({"case_count": suite_report["case_count"]}))
        return 0

    def assert_one_tool(self) -> None:
        if len(self.TOOLS) != 1:
            raise AssertionError(f"expected one patched tool, got {self.TOOLS}")

    def _validation(self, workload, route, activation, parameter, output, native,
                    source_hash, runner_hash, manifest_hash):
        return {
            "schema": "cellerator.factorized_interactions.validation.v1",
            "validated": True,
            "timed": False,
            "workload": workload,
            "route": route,
            "activation_storage": activation,
            "parameter_storage": parameter,
            "output_storage": output,
            "gradient_mode": "overwrite",
            "scheduler_job_id": self.owner.job_id,
            "build_provenance": {
                "source_closure_sha256": source_hash,
                "runner_binary_sha256": runner_hash,
                "build_manifest_sha256": manifest_hash,
            },
            "contract_evidence": contract_evidence(),
            "native_execution": native,
        }


if __name__ == "__main__":
    unittest.main()
