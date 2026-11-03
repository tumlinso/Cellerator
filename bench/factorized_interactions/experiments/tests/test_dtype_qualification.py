from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from run_dtype_qualification import (  # noqa: E402
    EXPECTED_CASES,
    MODES,
    QualificationError,
    check_build,
    dtype_matrix,
    expected_identity,
    mode_route,
    run_matrix,
    source_closure_digest,
    validate_summary,
)


JOB_CONTEXT = {
    "scheduler_job_id": "outer-job",
    "scheduler_attempt_id": "outer-attempt",
    "device_uuid": "GPU-21131915-1488-23af-38dd-1743ae1f5cc8",
    "cgroup": "/example/device-scheduler-attempt-x.service",
    "unit_name": "device-scheduler-attempt-x.service",
}


class DtypeQualificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runner = self.root / "build" / "runner"
        self.runner.parent.mkdir()
        self.runner.write_bytes(b"immutable test runner")
        self.manifest = self.root / "build" / "manifest.json"
        source_entries = [{
            "path": "bench/factorized_interactions/candidate_runner.cu",
            "bytes": 6,
            "sha256": hashlib.sha256(b"source").hexdigest(),
        }]
        source_hash, _ = source_closure_digest(source_entries)
        self.manifest.write_text(json.dumps({
            "schema": "cellerator.factorized_interactions.build.v1",
            "architecture": "sm_70", "fast_math": False,
            "source_closure_sha256": source_hash,
            "source_files": source_entries,
            "binaries": {"runner": {
                "path": "runner",
                "sha256": hashlib.sha256(self.runner.read_bytes()).hexdigest(),
            }},
        }), encoding="utf-8")
        self.build = check_build(self.runner, self.manifest)

    def fake_summary(self, mode: str, activation: str, parameter: str,
                     output: str) -> dict:
        backend, implementation, layout = expected_identity(mode)
        route = mode_route(mode)
        evidence = {name: {"max_abs_error": 0.0} for name in (
            "y", "dx", "dz", "dalpha", "dbias", "dgamma", "dx_repeat",
            "dz_repeat", "dalpha_repeat", "dbias_repeat", "dgamma_repeat")}
        return {
            "schema": "cellerator.factorized_interactions.validation.v1",
            "validated": True, "timed": False,
            "workload": "packet_a4_m32_u16", "route": route,
            "activation_storage": activation, "parameter_storage": parameter,
            "output_storage": output, "gradient_mode": "overwrite",
            "scheduler_job_id": "outer-job",
            "build_provenance": {
                "source_closure_sha256": self.build["source_closure_sha256"],
                "runner_binary_sha256": self.build["runner_sha256"],
                "build_manifest_sha256": self.build["build_manifest_sha256"],
            },
            "native_execution": {
                "candidate": mode, "requested_route": route,
                "selected_backend": backend,
                "actual_implementation": implementation,
                "structural_layout": layout,
                "device_uuid": JOB_CONTEXT["device_uuid"],
                "scheduler_job_id": "outer-job",
            },
            "contract_evidence": evidence,
            "input_case_sha256": "b" * 64,
        }

    def test_dtype_matrix_and_mode_routes_cover_exact_fixed_product(self) -> None:
        rows = dtype_matrix()
        self.assertEqual(len(rows), 8)
        self.assertEqual(len({tuple(row.values()) for row in rows}), 8)
        self.assertEqual(EXPECTED_CASES, 32)
        self.assertEqual(mode_route("packet"), "packet")
        for mode in MODES[1:]:
            self.assertEqual(mode_route(mode), "direct")

    def test_expected_backend_and_layout_are_mode_specific(self) -> None:
        self.assertEqual(expected_identity("packet"), ("packet", "packet", "full_packet"))
        self.assertEqual(expected_identity("native_direct"), ("direct", "direct", "full_packet"))
        self.assertEqual(expected_identity("minimal_direct"), ("direct", "direct", "canonical_direct"))
        self.assertEqual(expected_identity("direct_unique_output"),
                         ("direct", "direct_unique_output", "canonical_direct"))

    def test_build_is_bound_to_manifest_source_binary_and_sm70(self) -> None:
        bad = self.runner.read_bytes() + b"changed"
        self.runner.write_bytes(bad)
        with self.assertRaisesRegex(QualificationError, "SHA256"):
            check_build(self.runner, self.manifest)
        self.runner.write_bytes(b"immutable test runner")
        manifest = json.loads(self.manifest.read_text())
        manifest["fast_math"] = True
        self.manifest.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(QualificationError, "non-fast-math"):
            check_build(self.runner, self.manifest)

    def test_build_rejects_tampered_source_closure(self) -> None:
        manifest = json.loads(self.manifest.read_text())
        manifest["source_files"][0]["path"] = "../outside.cc"
        self.manifest.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(QualificationError, "escapes"):
            check_build(self.runner, self.manifest)

    def test_summary_rejects_wrong_dtype_or_candidate_dispatch(self) -> None:
        dtype = dtype_matrix()[0]
        summary = self.fake_summary("direct_unique_output", **{
            "activation": dtype["activation"], "parameter": dtype["parameter"],
            "output": dtype["output"],
        })
        record = validate_summary(summary, mode="direct_unique_output", dtype=dtype,
                                  device_uuid=JOB_CONTEXT["device_uuid"],
                                  scheduler_job_id="outer-job", build=self.build)
        self.assertEqual(record["status"], "pass")
        bad_dtype = dict(summary, activation_storage="fp32")
        with self.assertRaisesRegex(QualificationError, "activation_storage"):
            validate_summary(bad_dtype, mode="direct_unique_output", dtype=dtype,
                             device_uuid=JOB_CONTEXT["device_uuid"],
                             scheduler_job_id="outer-job", build=self.build)
        bad_dispatch = json.loads(json.dumps(summary))
        bad_dispatch["native_execution"]["actual_implementation"] = "direct"
        with self.assertRaisesRegex(QualificationError, "actual_implementation"):
            validate_summary(bad_dispatch, mode="direct_unique_output", dtype=dtype,
                             device_uuid=JOB_CONTEXT["device_uuid"],
                             scheduler_job_id="outer-job", build=self.build)

    def test_runner_invokes_fixed_validator_for_all_32_pairs(self) -> None:
        calls: list[list[str]] = []

        def invoke(command, **kwargs):
            calls.append(command)
            def value(flag: str) -> str:
                return command[command.index(flag) + 1]
            mode = value("--runner").split("--candidate ", 1)[1]
            summary = self.fake_summary(
                mode, value("--activation-dtype"), value("--parameter-dtype"),
                value("--output-dtype"),
            )
            return subprocess.CompletedProcess(command, 0,
                stdout=json.dumps(summary), stderr="")

        output = self.root / "result.json"
        result = run_matrix(
            runner=self.runner, build_manifest=self.manifest,
            expected_uuid=JOB_CONTEXT["device_uuid"],
            artifact_dir=self.root / "cases", output=output,
            reference_cache_dir=self.root / "cases" / "reference-cache",
            invoke=invoke, context_reader=lambda _uuid: dict(JOB_CONTEXT),
        )
        self.assertEqual(len(calls), 32)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["passed_case_count"], 32)
        self.assertFalse(result["timing_performed"])
        self.assertTrue(all("--iterations" in call and call[call.index("--iterations") + 1] == "0"
                            for call in calls))
        self.assertTrue(all("--repeat-vjp" in call for call in calls))
        self.assertTrue(all(call[1] == str(HERE.parent / "validate_runner.py")
                            for call in calls))
        self.assertTrue(all(call[call.index("--reference-cache-dir") + 1] ==
                            str(self.root / "cases" / "reference-cache") for call in calls))
        self.assertEqual(json.loads(output.read_text())["status"], "pass")

    def test_subprocess_failure_is_retained_and_suite_continues(self) -> None:
        calls = 0

        def invoke(command, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return subprocess.CompletedProcess(command, 1, stdout="partial", stderr="failed")
            def value(flag: str) -> str:
                return command[command.index(flag) + 1]
            mode = value("--runner").split("--candidate ", 1)[1]
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(self.fake_summary(
                mode, value("--activation-dtype"), value("--parameter-dtype"),
                value("--output-dtype"))), stderr="")

        result = run_matrix(
            runner=self.runner, build_manifest=self.manifest,
            expected_uuid=JOB_CONTEXT["device_uuid"],
            artifact_dir=self.root / "failed-cases", output=self.root / "failed.json",
            invoke=invoke, context_reader=lambda _uuid: dict(JOB_CONTEXT),
        )
        self.assertEqual(calls, 32)
        self.assertEqual(result["status"], "fail")
        self.assertEqual(len(result["failures"]), 1)
        failure = result["failures"][0]
        self.assertEqual(Path(failure["stdout"]).read_text(), "partial")
        self.assertEqual(Path(failure["stderr"]).read_text(), "failed")


if __name__ == "__main__":
    unittest.main()
