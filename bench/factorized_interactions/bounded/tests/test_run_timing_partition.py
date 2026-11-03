from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


_SOURCE_ROOT_ENV = "CELLERATOR_TEST_SOURCE_ROOT"
_configured_source_root = os.environ.get(_SOURCE_ROOT_ENV)
ROOT = (Path(_configured_source_root).resolve(strict=True) if _configured_source_root
        else Path(__file__).resolve().parents[4])
if not (ROOT / "bench" / "factorized_interactions" / "validate_runner.py").is_file():
    raise RuntimeError(f"{_SOURCE_ROOT_ENV} must identify the accepted Cellerator source tree")
BOUNDED = Path(__file__).resolve().parents[1]
if str(BOUNDED) not in sys.path:
    sys.path.insert(0, str(BOUNDED))

import bounded_validation as bounded  # noqa: E402
import run_timing_partition as timing  # noqa: E402


UUID = "GPU-6c1cac7f-a360-0aef-ba98-2828bfd1db1a"
WORKLOAD = "packet_a4_m32_u16"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class TimingPartitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.packet_root = self.root / "attempt-output"
        self.packet_root.mkdir()
        self.source = ROOT
        self.artifact_dir = self.packet_root / "artifacts"
        self.output = self.packet_root / "timing.json"
        self.qualification_root = self.root / "qualification-snapshot"
        self.qualification_root.mkdir()
        self.bundle = self.qualification_root / "qualification.json"
        self.runner, self.manifest, self.build = self._make_build()
        self.context = {
            "schema": "cellerator.factorized_interactions.outer_execution_context.v1",
            "status": "observed",
            "scheduler_job_id": "job-timing-test",
            "scheduler_attempt_id": "attempt-timing-test",
            "device_uuid": UUID,
            "cgroup": "/system.slice/device-scheduler-attempt-test.service",
            "unit_name": "device-scheduler-attempt-test.service",
        }
        self.facts = bounded.workload_facts(self.source, WORKLOAD, "fp16", "fp16")
        workloads = bounded._source_module(
            self.source, "tests/factorized_interactions", "workloads",
        )
        self.performance_ids = list(workloads.performance_workload_ids())
        identity = timing._identity_from_build(self.build)
        self.qualification = {
            "path": str(self.bundle.resolve()),
            "sha256": "",
            "schema": timing.QUALIFICATION_SCHEMA,
            "status": "validated_pending_independent_review",
            "device_uuid": UUID,
            "build": {
                "source_closure_sha256": identity["source_closure_sha256"],
                "candidate_runner_sha256": identity["candidate_runner_sha256"],
                "candidate_manifest_sha256": identity["candidate_manifest_sha256"],
                "baseline_runner_sha256": identity["baseline_runner_sha256"],
                "baseline_manifest_sha256": identity["baseline_manifest_sha256"],
                "library_sha256": identity["library_sha256"],
                "validator_sha256": identity["validator_sha256"],
            },
            "coverage": {"dtype": 32, "aux_packet": 12,
                         "aux_strongest_direct": 12, "sanitizer": {
                             name: 14 for name in
                             ("memcheck", "racecheck", "initcheck", "synccheck")}},
            "evidence_root": ".",
            "sanitizer_wrapper_path": "sanitizer/fixed_wrapper.py",
            "sanitizer_wrapper_sha256": "",
            "sanitizer_tool_identity": {"suite_sha256": "d" * 64},
            "reports": [],
        }
        wrapper = self.qualification_root / self.qualification["sanitizer_wrapper_path"]
        wrapper.parent.mkdir(parents=True)
        wrapper.write_text("# fixed test wrapper\n", encoding="utf-8")
        self.qualification["sanitizer_wrapper_sha256"] = timing.sha256_file(wrapper)
        kinds = (
            "dtype", "aux_packet_regular", "aux_packet_high_fanin_1m",
            "aux_strongest_direct_regular", "aux_strongest_direct_high_fanin_1m",
            "sanitizer_memcheck", "sanitizer_racecheck", "sanitizer_initcheck",
            "sanitizer_synccheck",
        )
        for index, kind in enumerate(kinds):
            directory = self.qualification_root / "reports" / kind
            directory.mkdir(parents=True)
            paths = {}
            for name in ("report", "receipt", "pointer"):
                path = directory / f"{name}.json"
                path.write_text(json.dumps({"kind": kind, "part": name}) + "\n",
                                encoding="utf-8")
                paths[f"{name}_path"] = path.relative_to(self.qualification_root).as_posix()
                paths[f"{name}_sha256"] = timing.sha256_file(path)
            self.qualification["reports"].append({"kind": kind, **paths,
                                                   "case_count": 32 if index == 0 else 14})
        self.bundle.write_text(json.dumps(self.qualification) + "\n", encoding="utf-8")
        self.qualification["sha256"] = timing.sha256_file(self.bundle)

    def _make_build(self):
        build_root = self.root / "build" / "native_candidate" / "native_build"
        candidate_root = build_root / "candidate_native" / "native_build"
        candidate_cmake = candidate_root / "cmake"
        base_cmake = build_root / "cmake"
        candidate_cmake.mkdir(parents=True)
        base_cmake.mkdir(parents=True)
        candidate_runner = candidate_cmake / "candidate-runner"
        baseline_runner = base_cmake / "baseline-runner"
        candidate_runner.write_bytes(b"candidate runner bytes")
        baseline_runner.write_bytes(b"baseline runner bytes")
        library = candidate_cmake / "libfactorized.a"
        baseline_library = base_cmake / "libfactorized.a"
        library.write_bytes(b"shared library bytes")
        baseline_library.write_bytes(library.read_bytes())
        source_files = []
        candidate_manifest = candidate_root / "factorized_interactions_build_manifest.json"
        baseline_manifest = build_root / "factorized_interactions_build_manifest.json"
        candidate_data = {
            "schema": "cellerator.factorized_interactions.build.v1",
            "architecture": "sm_70", "fast_math": False,
            "source_closure_sha256": timing.EXPECTED_SOURCE_CLOSURE_SHA256,
            "source_files": source_files,
            "binaries": {
                "runner": {"path": "cmake/candidate-runner", "bytes": candidate_runner.stat().st_size,
                           "sha256": timing.sha256_file(candidate_runner)},
                "factorized_interaction_cuda": {
                    "path": "cmake/libfactorized.a", "bytes": library.stat().st_size,
                    "sha256": timing.sha256_file(library)},
            },
        }
        baseline_data = {
            "schema": "cellerator.factorized_interactions.build.v1",
            "architecture": "sm_70", "fast_math": False,
            "source_closure_sha256": timing.EXPECTED_SOURCE_CLOSURE_SHA256,
            "source_files": source_files,
            "binaries": {
                "runner": {"path": "cmake/baseline-runner", "bytes": baseline_runner.stat().st_size,
                           "sha256": timing.sha256_file(baseline_runner)},
                "factorized_interaction_cuda": {
                    "path": "cmake/libfactorized.a", "bytes": baseline_library.stat().st_size,
                    "sha256": timing.sha256_file(baseline_library)},
            },
        }
        candidate_data["candidate_runner_provenance"] = {"base_runner_sha256": "1" * 64}
        candidate_manifest.write_text(json.dumps(candidate_data), encoding="utf-8")
        baseline_data["candidate_runner"] = {
            "manifest_path": "candidate_native/native_build/factorized_interactions_build_manifest.json",
            "binary_path": "candidate_native/native_build/cmake/candidate-runner",
            "manifest_sha256": timing.sha256_file(candidate_manifest),
            "binary_sha256": timing.sha256_file(candidate_runner),
        }
        baseline_manifest.write_text(json.dumps(baseline_data), encoding="utf-8")
        build = {
            "runner_path": str(candidate_runner.resolve()),
            "runner_sha256": timing.sha256_file(candidate_runner),
            "build_manifest_path": str(candidate_manifest.resolve()),
            "build_manifest_sha256": timing.sha256_file(candidate_manifest),
            "source_closure_sha256": timing.EXPECTED_SOURCE_CLOSURE_SHA256,
            "library_path": str(library.resolve()),
            "library_sha256": timing.sha256_file(library),
            "validator_sha256": timing.EXPECTED_VALIDATOR_SHA256,
        }
        return candidate_runner, candidate_manifest, build

    def _verify_build(self, _source, _runner, _manifest):
        return dict(self.build)

    def _context_reader(self, _source, _uuid):
        return dict(self.context)

    def _fake_qualification_validator(self, bundle_path, source, candidate_manifest,
                                     baseline_manifest, expected_uuid):
        self.assertEqual(Path(source), self.source.resolve())
        self.assertEqual(Path(candidate_manifest), self.manifest.resolve())
        self.assertEqual(Path(baseline_manifest),
                         Path(self.build["build_manifest_path"]).parents[2] /
                         "factorized_interactions_build_manifest.json")
        self.assertEqual(expected_uuid, UUID)
        return json.loads(Path(bundle_path).read_text(encoding="utf-8"))

    def _fake_run_validation(self, calls, *, fail_at=None, peak=1024,
                             change_input_at=None, mutate_wrapper_at=None,
                             mutate_sidecar_at=None, sidecar_path=None):
        def invoke(**kwargs):
            index = len(calls)
            calls.append(kwargs)
            case_id = kwargs["case_id"]
            cache_dir = kwargs["cache_dir"]
            kwargs["cache_existed_before"] = cache_dir.exists()
            owner = bounded._owned_cache(
                kwargs["artifact_root"], cache_dir,
                context=kwargs["context"], build=kwargs["build"],
                helper_hash=bounded.sha256_file(Path(bounded.__file__).resolve()),
            )
            kwargs["cache_owner_sha256"] = timing.sha256_file(owner)
            case_dir = kwargs["artifact_root"] / "wire" / case_id
            if fail_at == index:
                case_dir.mkdir(parents=True)
                (case_dir / "failed.numeric.case").write_bytes(b"retained failure wire")
                raise RuntimeError("fixed validator rejected an oracle-bound numeric mismatch")
            case_dir.mkdir(parents=True)
            input_payload = b"same fixed workload input"
            if change_input_at == index:
                input_payload = b"changed fixed workload input"
            input_digest = _sha(input_payload)
            output_digest = _sha(b"same fixed serialized output")
            native = {
                "candidate": kwargs["mode"],
                "requested_route": "packet" if kwargs["mode"] == "packet" else "direct",
                "selected_backend": kwargs["expected_identity"][0],
                "actual_implementation": kwargs["expected_identity"][1],
                "structural_layout": kwargs["expected_identity"][2],
                "device_uuid": UUID,
                "scheduler_job_id": self.context["scheduler_job_id"],
            }
            metrics = {
                "schema": "cellerator.factorized_interactions.metrics.v1",
                "device_uuid": UUID,
                "scheduler_job_id": self.context["scheduler_job_id"],
                "iterations": timing.ITERATIONS,
                "selected_backend": kwargs["expected_identity"][0],
                "phase_ms": {timing.PRIMARY_METRIC: 0.125,
                             "warm_forward_vjp_batch_host_wall": 3.75},
                "samples_ms": {name: [0.01] * timing.ITERATIONS for name in
                               ("forward", "backward", "forward_vjp", "parameter_refresh")},
            }
            summary = {
                "schema": "cellerator.factorized_interactions.validation.v1",
                "validated": True, "timed": True,
                "workload": kwargs["workload"],
                "route": native["requested_route"],
                "activation_storage": kwargs["activation"],
                "parameter_storage": kwargs["parameter"],
                "output_storage": kwargs["output_dtype"],
                "gradient_mode": "overwrite",
                "scheduler_job_id": self.context["scheduler_job_id"],
                "input_case_sha256": input_digest,
                "contract_evidence": {name: {"max_error_to_allowance": 0.0} for name in
                                       ("y", "dx", "dz", "dalpha", "dbias", "dgamma")},
                "build_provenance": {
                    "source_closure_sha256": self.build["source_closure_sha256"],
                    "runner_binary_sha256": self.build["runner_sha256"],
                    "build_manifest_sha256": self.build["build_manifest_sha256"],
                },
                "native_execution": native,
                "metrics": metrics,
            }
            evidence_dir = kwargs["artifact_root"] / "evidence"
            evidence_dir.mkdir(exist_ok=True)
            evidence_path = evidence_dir / f"{case_id}.json"
            evidence_path.write_text("{}\n", encoding="utf-8")
            stdout_path = evidence_dir / f"{case_id}.stdout"
            stderr_path = evidence_dir / f"{case_id}.stderr"
            stdout_path.write_text("{}\n", encoding="utf-8")
            stderr_path.write_text("\n", encoding="utf-8")
            if mutate_wrapper_at == index:
                wrapper = self.qualification_root / self.qualification["sanitizer_wrapper_path"]
                wrapper.write_text("# modified during test\n", encoding="utf-8")
            if mutate_sidecar_at == index:
                if sidecar_path is None:
                    raise AssertionError("test sidecar mutation needs a path")
                sidecar_path.write_bytes(b"modified sidecar bytes\n")
            return {
                "schema": "cellerator.factorized_interactions.bounded_case.v1",
                "case_id": case_id,
                "summary": summary,
                "wire_inventory": [
                    {"path": "input.case", "bytes": 10, "sha256": input_digest},
                    {"path": "output.result", "bytes": 11, "sha256": output_digest},
                ],
                "input_case_sha256": input_digest,
                "output_result_sha256": output_digest,
                "context": dict(self.context),
                "build": dict(self.build),
                "validator_sha256": self.build["validator_sha256"],
                "stdout_sha256": timing.sha256_file(stdout_path),
                "stderr_sha256": timing.sha256_file(stderr_path),
                "peak_scratch_bytes": peak,
                "peak_measurement": "50ms sampled whole packet tree plus post-write evidence allowance",
                "success_wires_removed": True,
                "evidence_path": str(evidence_path),
                "evidence_relative_path": evidence_path.relative_to(self.packet_root).as_posix(),
                "stdout_relative_path": stdout_path.relative_to(self.packet_root).as_posix(),
                "stderr_relative_path": stderr_path.relative_to(self.packet_root).as_posix(),
            }
        return invoke

    def _collect(self, phase, calls, *, fail_at=None, peak=1024,
                 change_input_at=None, mutate_wrapper_at=None,
                 mutate_sidecar_at=None, harness_paths=None):
        os.environ["AUTOBENCH_SCHEDULER_JOB_ID"] = self.context["scheduler_job_id"]
        os.environ["AUTOBENCH_SCHEDULER_ATTEMPT_ID"] = self.context["scheduler_attempt_id"]
        return timing.collect_partition(
            source=self.source,
            runner=self.runner,
            build_manifest=self.manifest,
            workload=WORKLOAD,
            phase=phase,
            expected_uuid=UUID,
            qualification_bundle=self.bundle,
            artifact_dir=self.artifact_dir,
            output=self.output,
            verify_build=self._verify_build,
            context_reader=self._context_reader,
            run_validation=self._fake_run_validation(
                calls, fail_at=fail_at, peak=peak, change_input_at=change_input_at,
                mutate_wrapper_at=mutate_wrapper_at,
                mutate_sidecar_at=mutate_sidecar_at,
                sidecar_path=(harness_paths or {}).get("recorded_receipts_sha256")),
            bundle_validator=self._fake_qualification_validator,
            facts=self.facts,
            performance_ids=self.performance_ids,
            expected_mode_identity=bounded.expected_mode_identity,
            harness_paths=harness_paths,
        )

    def test_confirmation_continues_abba_and_descriptive_arm_rotates(self) -> None:
        self.assertEqual(timing.pair_order("exploration", 0),
                         (["strongest_direct", "packet"], 0))
        self.assertEqual(timing.pair_order("exploration", 1),
                         (["packet", "strongest_direct"], 1))
        self.assertEqual(timing.pair_order("confirmation", 0),
                         (["packet", "strongest_direct"], 1))
        self.assertEqual(timing.pair_order("confirmation", 1),
                         (["strongest_direct", "packet"], 2))
        all_positions = [timing.pair_order(phase, i)[1]
                         for phase in ("exploration", "confirmation")
                         for i in range(7)]
        self.assertEqual([all_positions.count(pos) for pos in range(3)], [5, 5, 4])

    def test_qualification_gate_requires_verifier_acceptance_mapping(self) -> None:
        with self.assertRaisesRegex(timing.TimingPartitionError, "did not accept"):
            timing.validate_qualification_bundle(
                self.bundle,
                source=self.source,
                build_identity=timing._identity_from_build(self.build),
                expected_uuid=UUID,
                bundle_validator=lambda *_args: {},
            )

    def test_portable_qualification_paths_survive_bundle_relocation(self) -> None:
        relocated_root = self.root / "relocated-snapshot"
        shutil.copytree(self.qualification_root, relocated_root)
        relocated_bundle = relocated_root / "qualification.json"
        summary = timing.validate_qualification_bundle(
            relocated_bundle,
            source=self.source,
            build_identity=timing._identity_from_build(self.build),
            expected_uuid=UUID,
            bundle_validator=self._fake_qualification_validator,
        )
        self.assertEqual(summary["evidence_root"], ".")
        self.assertEqual(summary["reports"][0]["report_path"],
                         "reports/dtype/report.json")
        self.assertEqual(summary["sanitizer_wrapper_path"], "sanitizer/fixed_wrapper.py")
        timing._verify_qualification_snapshot(summary)

    def test_collects_exactly_twenty_one_unique_validated_calls(self) -> None:
        calls = []
        report = self._collect("exploration", calls)
        self.assertEqual(len(calls), 21)
        ids = [call["case_id"] for call in calls]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(report["schema"], timing.SCHEMA)
        self.assertEqual(report["status"], "observations_complete_pending_independent_review")
        self.assertEqual(report["completed_blocks"], 7)
        self.assertEqual(report["completed_observations"], 21)
        self.assertEqual(report["max_observed_peak_scratch_bytes"] < timing.MAX_ATTEMPT_BYTES, True)
        self.assertEqual(report["attempt_peak_bound_bytes"], max(
            report["max_observed_peak_scratch_bytes"], report["projected_complete_attempt_bytes"]))
        encoded_report = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        self.assertEqual(report["projected_complete_attempt_bytes"],
                         timing._tree_bytes(self.packet_root) + len(encoded_report))
        self.assertIn("50 ms", report["scratch_measurement"]["method"])
        self.assertIn("miss shorter transient peaks",
                      report["scratch_measurement"]["timing_note"])
        harness = report["harness_provenance"]
        self.assertEqual(harness["schema"], timing.HARNESS_PROVENANCE_SCHEMA)
        self.assertEqual(set(harness["before"]), {
            "timing_collector_sha256", "bounded_validation_sha256",
            "recorded_qualification_sha256", "recorded_receipts_sha256",
            "sanitizer_wrapper_sha256",
        })
        self.assertEqual(harness["before"], harness["after"])
        self.assertTrue(harness["unchanged"])
        for block in report["blocks"]:
            expected_pair, expected_position = timing.pair_order("exploration", block["round_index"])
            expected_order = list(expected_pair)
            expected_order.insert(expected_position, "native_direct")
            self.assertEqual(block["paired_order"], expected_pair)
            self.assertEqual(block["execution_order"], expected_order)
            self.assertEqual(list(block["arms"]), expected_order)
            self.assertEqual(len({arm["input_case_sha256"] for arm in block["arms"].values()}), 1)
            self.assertEqual(len({arm["output_result_sha256"] for arm in block["arms"].values()}), 1)
            for arm in block["arms"].values():
                self.assertEqual(len(arm["metrics"]["samples_ms"]["forward"]), 30)
                self.assertEqual(set(arm["validated_summary"]["contract_evidence"]),
                                 {"y", "dx", "dz", "dalpha", "dbias", "dgamma"})
                self.assertEqual(arm["binary_identity"]["runner_binary_sha256"],
                                 report["build"]["candidate_runner_sha256"])
                self.assertEqual(arm["execution_context"]["scheduler_attempt_id"],
                                 self.context["scheduler_attempt_id"])

    def test_first_validator_call_creates_and_owns_the_fresh_cache(self) -> None:
        calls = []
        self._collect("exploration", calls)
        self.assertFalse(calls[0]["cache_existed_before"])
        self.assertTrue(all(call["cache_existed_before"] for call in calls[1:]))
        self.assertEqual(len({call["cache_owner_sha256"] for call in calls}), 1)
        owner = self.artifact_dir / ".bounded-cache-owner.json"
        cache = self.artifact_dir / "reference-cache"
        self.assertTrue(owner.is_file())
        self.assertTrue(cache.is_dir())
        metadata = json.loads(owner.read_text(encoding="utf-8"))
        self.assertEqual(metadata["cache_relative_path"], "reference-cache")
        self.assertEqual(metadata["scheduler_attempt_id"], self.context["scheduler_attempt_id"])

    def test_validator_failure_stops_and_retains_failing_wire(self) -> None:
        calls = []
        with self.assertRaises(timing.PartitionRunFailure) as raised:
            self._collect("confirmation", calls, fail_at=1)
        self.assertEqual(len(calls), 2)
        partial = raised.exception.partial_report
        self.assertEqual(partial["status"], "failed_incomplete")
        self.assertEqual(partial["completed_observations"], 1)
        self.assertEqual(list(partial["blocks"][0]["arms"]), ["packet"])
        failed_id = calls[1]["case_id"]
        retained = self.artifact_dir / "wire" / failed_id / "failed.numeric.case"
        self.assertTrue(retained.is_file())
        self.assertEqual(retained.read_bytes(), b"retained failure wire")

    def test_attempt_scratch_limit_stops_before_next_run(self) -> None:
        calls = []
        with self.assertRaises(timing.PartitionRunFailure):
            self._collect("exploration", calls, peak=timing.MAX_ATTEMPT_BYTES)
        self.assertEqual(len(calls), 1)

    def test_input_bytes_must_remain_fixed_across_all_paired_blocks(self) -> None:
        calls = []
        with self.assertRaises(timing.PartitionRunFailure) as raised:
            self._collect("exploration", calls, change_input_at=3)
        self.assertEqual(len(calls), 4)
        self.assertIn("input bytes changed", str(raised.exception))
        self.assertEqual(raised.exception.partial_report["completed_observations"], 3)

    def test_sidecar_mutation_during_execution_invalidates_report(self) -> None:
        calls = []
        with self.assertRaises(timing.PartitionRunFailure) as raised:
            self._collect("exploration", calls, mutate_wrapper_at=0)
        self.assertEqual(len(calls), 21)
        provenance = raised.exception.partial_report["harness_provenance"]
        self.assertFalse(provenance["unchanged"])
        self.assertNotEqual(provenance["before"]["sanitizer_wrapper_sha256"],
                            provenance["after"]["sanitizer_wrapper_sha256"])
        self.assertIn("qualification sidecar changed", raised.exception.partial_report[
            "failure"]["message"])

    def test_receipt_helper_mutation_during_execution_invalidates_report(self) -> None:
        harness_paths = timing._harness_component_paths(self.bundle)
        temporary_receipts = self.root / "recorded_receipts.py"
        temporary_receipts.write_bytes(harness_paths["recorded_receipts_sha256"].read_bytes())
        harness_paths["recorded_receipts_sha256"] = temporary_receipts
        calls = []
        with self.assertRaises(timing.PartitionRunFailure) as raised:
            self._collect("confirmation", calls, mutate_sidecar_at=0,
                          harness_paths=harness_paths)
        self.assertEqual(len(calls), 21)
        provenance = raised.exception.partial_report["harness_provenance"]
        self.assertFalse(provenance["unchanged"])
        self.assertNotEqual(provenance["before"]["recorded_receipts_sha256"],
                            provenance["after"]["recorded_receipts_sha256"])


if __name__ == "__main__":
    unittest.main()
