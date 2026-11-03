from __future__ import annotations

import copy
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import recorded_qualification as recorded  # noqa: E402


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _contract(*, bad_ratio: bool = False) -> dict[str, dict[str, float]]:
    names = ["y", "dx", "dz", "dalpha", "dbias", "dgamma"]
    names += [name + "_repeat" for name in names[1:]]
    return {name: {"max_error_to_allowance": 1.01 if bad_ratio and name == "dgamma" else 0.1}
            for name in names}


class RecordedQualificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.uuid = "GPU-12345678-1234-5678-9abc-def012345678"
        self.context = {
            "scheduler_job_id": "scheduler-job-17",
            "scheduler_attempt_id": "scheduler-attempt-4",
            "device_uuid": self.uuid,
            "cgroup": "/system.slice/autobench-job.scope",
            "unit_name": "autobench-job.scope",
        }
        self.identity = {
            "source_closure_sha256": "a" * 64,
            "candidate_runner_sha256": "b" * 64,
            "candidate_manifest_sha256": "c" * 64,
            "baseline_runner_sha256": "d" * 64,
            "baseline_manifest_sha256": "e" * 64,
            "library_sha256": "f" * 64,
            "validator_sha256": recorded.VALIDATOR_SHA256,
        }
        self.build = {
            "source_closure_sha256": self.identity["source_closure_sha256"],
            "runner_sha256": self.identity["candidate_runner_sha256"],
            "build_manifest_sha256": self.identity["candidate_manifest_sha256"],
            "library_sha256": self.identity["library_sha256"],
            "validator_sha256": self.identity["validator_sha256"],
        }
        self.fixture_serial = 0

    def _fixture_dir(self, label: str, parent: Path | None = None) -> Path:
        self.fixture_serial += 1
        parent = self.root if parent is None else parent
        parent.mkdir(parents=True, exist_ok=True)
        path = parent / f"{label}-{self.fixture_serial:03d}"
        path.mkdir(parents=True)
        return path

    def _dtype_report(self, *, context: dict | None = None,
                      bad_ratio: bool = False,
                      fixture_parent: Path | None = None) -> tuple[dict, Path]:
        context = context or self.context
        report_path = self._fixture_dir("dtype", fixture_parent) / "dtype-report.json"
        rows = []
        modes = {
            "packet": ("packet", "packet", "full_packet"),
            "native_direct": ("direct", "direct", "full_packet"),
            "minimal_direct": ("direct", "direct", "canonical_direct"),
            "direct_unique_output": ("direct", "direct_unique_output", "canonical_direct"),
        }
        for activation, parameter, output in itertools.product(("fp16", "fp32"), repeat=3):
            source_bytes = f"{activation}|{parameter}|{output}|fixed-input".encode()
            input_digest = _sha(source_bytes)
            for mode, implementation in modes.items():
                case = report_path.parent / "cases" / f"{mode}.a{activation}.p{parameter}.o{output}"
                case.mkdir(parents=True)
                (case / "input.case").write_bytes(source_bytes)
                output_bytes = f"{mode}|{activation}|{parameter}|{output}|result".encode()
                (case / "output.result").write_bytes(output_bytes)
                ratios = _contract(bad_ratio=bad_ratio and mode == "packet" and
                                   activation == parameter == output == "fp16")
                summary = {
                    "validated": True,
                    "timed": False,
                    "workload": "packet_a4_m32_u16",
                    "contract_evidence": ratios,
                    "input_case_sha256": input_digest,
                    "input_case_shards": 1,
                    "input_case_bytes": len(source_bytes),
                    "output_bytes": len(output_bytes),
                    "scheduler_job_id": context["scheduler_job_id"],
                    "native_execution": {
                        "candidate": mode,
                        "device_uuid": self.uuid,
                        "scheduler_job_id": context["scheduler_job_id"],
                        "selected_backend": implementation[0],
                        "actual_implementation": implementation[1],
                        "structural_layout": implementation[2],
                    },
                }
                _write_json(case / "validator.stdout.json", summary)
                (case / "validator.stderr.log").write_bytes(b"")
                rows.append({
                    "activation": activation,
                    "parameter": parameter,
                    "output": output,
                    "mode": mode,
                    "workload": "packet_a4_m32_u16",
                    "status": "pass",
                    "selected_backend": implementation[0],
                    "actual_implementation": implementation[1],
                    "structural_layout": implementation[2],
                    "source_closure_sha256": self.identity["source_closure_sha256"],
                    "runner_sha256": self.identity["candidate_runner_sha256"],
                    "build_manifest_sha256": self.identity["candidate_manifest_sha256"],
                    "validator_sha256": self.identity["validator_sha256"],
                    "input_case_sha256": input_digest,
                    "contract_evidence": ratios,
                })
        report = {
            "schema": "cellerator.factorized_interactions.dtype_qualification.v1",
            "status": "pass",
            "timing_performed": False,
            "expected_case_count": 32,
            "passed_case_count": 32,
            "failures": [],
            "validator_sha256": self.identity["validator_sha256"],
            "scheduler_context": context,
            "build": self.build.copy(),
            "records": rows,
        }
        return report, report_path

    def _bounded_dtype_report(self, context: dict,
                              fixture_parent: Path | None = None) -> tuple[dict, Path]:
        report_path = self._fixture_dir("bounded-dtype", fixture_parent) / "qualification.json"
        records = []
        modes = {
            "packet": ("packet", "packet", "full_packet"),
            "native_direct": ("direct", "direct", "full_packet"),
            "minimal_direct": ("direct", "direct", "canonical_direct"),
            "direct_unique_output": ("direct", "direct_unique_output", "canonical_direct"),
        }
        for activation, parameter, output in itertools.product(("fp16", "fp32"), repeat=3):
            for mode, implementation in modes.items():
                case_id = f"case-{len(records):03d}"
                workload = "packet_a4_m32_u16"
                input_digest = _sha(f"{case_id}|input".encode())
                output_digest = _sha(f"{case_id}|output".encode())
                summary = {
                    "validated": True,
                    "timed": False,
                    "workload": workload,
                    "activation_storage": activation,
                    "parameter_storage": parameter,
                    "output_storage": output,
                    "input_case_sha256": input_digest,
                    "scheduler_job_id": context["scheduler_job_id"],
                    "contract_evidence": _contract(),
                    "native_execution": {
                        "candidate": mode,
                        "device_uuid": self.uuid,
                        "scheduler_job_id": context["scheduler_job_id"],
                        "selected_backend": implementation[0],
                        "actual_implementation": implementation[1],
                        "structural_layout": implementation[2],
                        "cells": 1,
                        "channels": 3,
                        "mechanisms": 32,
                        "outputs": 32,
                        "structural_maxima": {
                            "output_mechanisms": 1,
                            "input_roles": 1,
                            "context_terms": 1,
                            "alpha_mechanisms": 1,
                            "bias_mechanisms": 1,
                            "gamma_terms": 1,
                            "mechanism_terms": 4,
                        },
                    },
                }
                case_root = report_path.parent / "cases"
                evidence_path = case_root / "evidence" / f"{case_id}.json"
                validated_path = evidence_path.with_name(case_id + ".validated.json")
                stdout_path = case_root / "streams" / f"{case_id}.stdout.json"
                stderr_path = case_root / "streams" / f"{case_id}.stderr.log"
                _write_json(stdout_path, summary)
                stderr_path.parent.mkdir(parents=True, exist_ok=True)
                stderr_path.write_bytes(b"")
                inventory = [
                    {"path": f"{case_id}.case", "bytes": 10, "sha256": input_digest},
                    {"path": f"{case_id}.result", "bytes": 11, "sha256": output_digest},
                ]
                common = {
                    "schema": "cellerator.factorized_interactions.bounded_case.v1",
                    "status": "pass",
                    "case_id": case_id,
                    "workload": workload,
                    "mode": mode,
                    "summary": summary,
                    "wire_inventory": inventory,
                    "input_case_sha256": input_digest,
                    "output_result_sha256": output_digest,
                    "context": context.copy(),
                    "build": self.build.copy(),
                    "validator_sha256": self.identity["validator_sha256"],
                    "bounded_validation_sha256": recorded.BOUNDED_VALIDATION_SHA256,
                    "cache_owner_relative_path": "reference-cache/owner.json",
                    "cache_owner_sha256": "7" * 64,
                    "evidence_path": str(evidence_path),
                    "stdout_sha256": _sha(stdout_path.read_bytes()),
                    "stderr_sha256": _sha(stderr_path.read_bytes()),
                    "validated_relative_path": validated_path.relative_to(report_path.parent).as_posix(),
                    "stdout_relative_path": stdout_path.relative_to(report_path.parent).as_posix(),
                    "stderr_relative_path": stderr_path.relative_to(report_path.parent).as_posix(),
                    "peak_scratch_bytes": 1024,
                    "scratch_limit_bytes": 128 * 1024 * 1024,
                    "peak_measurement": "50ms sampled whole packet tree plus post-write evidence allowance",
                }
                _write_json(validated_path, common | {
                    "cleanup_pending": True,
                    "success_wires_removed": False,
                })
                evidence = common | {
                    "cleanup_pending": False,
                    "success_wires_removed": True,
                    "validated_record_sha256": _sha(validated_path.read_bytes()),
                    "evidence_relative_path": evidence_path.relative_to(report_path.parent).as_posix(),
                }
                _write_json(evidence_path, evidence)
                records.append({
                    "case_id": case_id,
                    "workload": workload,
                    "mode": mode,
                    "activation": activation,
                    "parameter": parameter,
                    "output": output,
                    "selected_backend": implementation[0],
                    "actual_implementation": implementation[1],
                    "structural_layout": implementation[2],
                    "input_case_sha256": input_digest,
                    "output_result_sha256": output_digest,
                    "evidence_relative_path": evidence["evidence_relative_path"],
                    "evidence_sha256": _sha(evidence_path.read_bytes()),
                    "peak_scratch_bytes": 1024,
                    "status": "pass",
                })
        report = {
            "schema": "cellerator.factorized_interactions.bounded_qualification.v1",
            "status": "pass",
            "kind": "dtype",
            "role": "dtype-all",
            "partition": None,
            "full_auxiliary_corpus_count": len(recorded.AUXILIARY_IDS),
            "full_auxiliary_corpus_qualified_by_this_report": False,
            "partition_workloads": [],
            "expected_case_count": 32,
            "passed_case_count": 32,
            "scheduler_context": context,
            "build": self.build.copy(),
            "records": records,
            "failures": [],
            "scratch_limit_bytes": 128 * 1024 * 1024,
            "peak_scratch_bytes": 2048,
            "successful_wire_policy": "digested and removed only after durable fixed-validator evidence",
            "harness": {"qualification_sha256": recorded.QUALIFICATION_DRIVER_SHA256,
                        "bounded_validation_sha256": recorded.BOUNDED_VALIDATION_SHA256},
        }
        return report, report_path

    def _aux_report(self, role: str, partition: str, *, context: dict | None = None,
                    direct_mode: str = "minimal_direct",
                    bad_ratio: bool = False,
                    fixture_parent: Path | None = None) -> tuple[dict, Path]:
        context = context or self.context
        kind = "aux_packet" if role == "packet" else "aux_strongest_direct"
        count = 11 if partition == "regular" else 1
        report_path = self._fixture_dir(f"{role}-{partition}", fixture_parent) / "qualification.json"
        records = []
        by_workload = {item[0]: item for item in recorded._expected_aux_plan(role)}
        plan = [by_workload[name] for name in recorded.AUXILIARY_IDS if
                (name == "correctness_packet_high_fanin_1m") == (partition == "high-fanin-1m")]
        self.assertEqual(len(plan), count)
        for index, (workload, activation, parameter, output) in enumerate(plan):
            case_id = f"case-{index:03d}"
            mode = "packet" if role == "packet" else direct_mode
            stable_workloads = {
                "correctness_fma_product_residual_cancellation",
                "correctness_context_cancellation_middle",
                "correctness_context_cancellation_tail",
                "correctness_high_fanin_100k",
                "correctness_packet_high_fanin_fp32_100k",
                "correctness_packet_high_fanin_1m",
            }
            stable = workload in stable_workloads
            implementation = (("stable", "stable", "full_packet") if stable else
                              ("packet", "packet", "full_packet") if role == "packet" else
                              ("direct", direct_mode, "canonical_direct"))
            mechanisms = (1_000_000 if workload == "correctness_packet_high_fanin_1m" else
                          100_000 if workload in {"correctness_high_fanin_100k",
                                                 "correctness_packet_high_fanin_fp32_100k"} else 2)
            outputs = 1 if mechanisms > 2 else 1
            max_terms = (8193 if workload in {"correctness_context_cancellation_middle",
                                               "correctness_context_cancellation_tail"} else
                         8192 if workload == "correctness_fma_product_residual_cancellation" else 1)
            output_fanin = mechanisms if mechanisms > 2 else 2
            ratios = _contract(bad_ratio=bad_ratio and index == 0)
            input_digest = _sha(f"{case_id}|input".encode())
            output_digest = _sha(f"{case_id}|output".encode())
            summary = {
                "validated": True,
                "timed": False,
                "workload": workload,
                "activation_storage": activation,
                "parameter_storage": parameter,
                "output_storage": output,
                "input_case_sha256": input_digest,
                "scheduler_job_id": context["scheduler_job_id"],
                "contract_evidence": ratios,
                "native_execution": {
                    "candidate": mode,
                    "device_uuid": self.uuid,
                    "scheduler_job_id": context["scheduler_job_id"],
                    "selected_backend": implementation[0],
                    "actual_implementation": implementation[1],
                    "structural_layout": implementation[2],
                    # Multiple mechanisms share one output, so the strongest
                    # direct route is minimal_direct unless direct_mode is
                    # deliberately made inconsistent by a negative test.
                    "cells": 1,
                    "channels": 3,
                    "mechanisms": mechanisms,
                    "outputs": outputs,
                    "structural_maxima": {
                        "output_mechanisms": output_fanin,
                        "input_roles": 1,
                        "context_terms": max_terms,
                        "alpha_mechanisms": 1,
                        "bias_mechanisms": 1,
                        "gamma_terms": 1,
                        "mechanism_terms": max_terms,
                    },
                },
            }
            case_root = report_path.parent / "cases"
            evidence_path = case_root / "evidence" / f"{case_id}.json"
            validated_path = evidence_path.with_name(case_id + ".validated.json")
            stdout_path = case_root / "streams" / f"{case_id}.stdout.json"
            stderr_path = case_root / "streams" / f"{case_id}.stderr.log"
            _write_json(stdout_path, summary)
            stderr_path.parent.mkdir(parents=True, exist_ok=True)
            stderr_path.write_bytes(b"validator stderr is retained\n")
            inventory = [
                {"path": f"{case_id}.case", "bytes": 19, "sha256": input_digest},
                {"path": f"{case_id}.result", "bytes": 20, "sha256": output_digest},
            ]
            common = {
                "schema": "cellerator.factorized_interactions.bounded_case.v1",
                "status": "pass",
                "case_id": case_id,
                "workload": workload,
                "mode": mode,
                "summary": summary,
                "wire_inventory": inventory,
                "input_case_sha256": input_digest,
                "output_result_sha256": output_digest,
                "build": self.build.copy(),
                "validator_sha256": self.identity["validator_sha256"],
                "bounded_validation_sha256": recorded.BOUNDED_VALIDATION_SHA256,
                "cache_owner_relative_path": "reference-cache/owner.json",
                "cache_owner_sha256": "7" * 64,
                "context": context.copy(),
                "evidence_path": str(evidence_path),
                "stdout_relative_path": stdout_path.relative_to(report_path.parent).as_posix(),
                "stderr_relative_path": stderr_path.relative_to(report_path.parent).as_posix(),
                "validated_relative_path": validated_path.relative_to(report_path.parent).as_posix(),
                "stdout_sha256": _sha(stdout_path.read_bytes()),
                "stderr_sha256": _sha(stderr_path.read_bytes()),
                "peak_scratch_bytes": 1024,
                "scratch_limit_bytes": 128 * 1024 * 1024,
                "peak_measurement": "50ms sampled whole packet tree plus post-write evidence allowance",
            }
            prior = common | {"cleanup_pending": True, "success_wires_removed": False}
            _write_json(validated_path, prior)
            final = common | {
                "cleanup_pending": False,
                "success_wires_removed": True,
                "validated_record_sha256": _sha(validated_path.read_bytes()),
            }
            _write_json(evidence_path, final)
            records.append({
                "case_id": case_id,
                "workload": workload,
                "mode": mode,
                "activation": activation,
                "parameter": parameter,
                "output": output,
                "status": "pass",
                "selected_backend": implementation[0],
                "actual_implementation": implementation[1],
                "structural_layout": implementation[2],
                "input_case_sha256": input_digest,
                "output_result_sha256": output_digest,
                "evidence_relative_path": evidence_path.relative_to(report_path.parent).as_posix(),
                "evidence_sha256": _sha(evidence_path.read_bytes()),
                "peak_scratch_bytes": 1024,
            })
        report = {
            "schema": "cellerator.factorized_interactions.bounded_qualification.v1",
            "status": "pass",
            "role": role,
            "kind": "aux",
            "partition": partition,
            "full_auxiliary_corpus_count": 12,
            "full_auxiliary_corpus_qualified_by_this_report": False,
            "partition_workloads": [item[0] for item in plan],
            "expected_case_count": count,
            "passed_case_count": count,
            "failures": [],
            "scratch_limit_bytes": 128 * 1024 * 1024,
            "peak_scratch_bytes": 2048,
            "validator_sha256": self.identity["validator_sha256"],
            "successful_wire_policy": "digested and removed only after durable fixed-validator evidence",
            "harness": {"qualification_sha256": recorded.QUALIFICATION_DRIVER_SHA256,
                        "bounded_validation_sha256": recorded.BOUNDED_VALIDATION_SHA256},
            "scheduler_context": context,
            "build": self.build.copy(),
            "records": records,
        }
        return report, report_path

    def _attempt_context(self, index: int) -> dict[str, str]:
        job = f"scheduler-job-{index:02d}"
        attempt = f"scheduler-attempt-{index:02d}"
        autobench_attempt = f"autobench-attempt-{index:02d}"
        unit = "device-scheduler-attempt-" + hashlib.sha256(attempt.encode()).hexdigest()[:32] + ".service"
        return {
            "schema": "cellerator.factorized_interactions.outer_execution_context.v1",
            "status": "observed",
            "scheduler_job_id": job,
            "scheduler_attempt_id": attempt,
            "autobench_attempt_id": autobench_attempt,
            "device_uuid": self.uuid,
            "cgroup": f"/user.slice/user-1000.slice/app.slice/{unit}",
            "unit_name": unit,
        }

    def _receipt_pair(self, index: int, context: dict[str, str], report_path: Path,
                      *, wrong_report_hash: bool = False) -> tuple[Path, Path]:
        job_id = context["scheduler_job_id"]
        attempt_id = context["scheduler_attempt_id"]
        autobench_id = context["autobench_attempt_id"]
        resource_id = f"gpu:test-host:{self.uuid}"
        device = [{"resource_id": resource_id, "kind": "gpu", "uuid": self.uuid}]
        job = {
            "id": job_id,
            "state": "succeeded",
            "cleanup_verified": True,
            "attempt": {
                "attempt_id": attempt_id,
                "job_id": job_id,
                "state": "succeeded",
                "cleanup_verified": True,
                "execution_context": {
                    "allocations": [{"resource_id": resource_id, "quantities": {"memory_bytes": 1}}],
                    "resources": [{"resource_id": resource_id, "kind": "gpu",
                                   "attributes": {"uuid": self.uuid}}],
                },
                "execution_handle": {
                    "external_id": context["unit_name"],
                    "metadata": {"control_group": context["cgroup"]},
                },
            },
            "spec": {"resources": [{"kind": "gpu", "selectors": {"resource_id": resource_id}}]},
        }
        worker = {"status": "succeeded", "returncode": 0,
                  "scheduler_job_id": job_id, "scheduler_attempt_id": attempt_id}
        result = {
            "job": job,
            "worker": worker,
            "identity": {"source_fingerprint": "synthetic-source-fingerprint",
                         "build_fingerprint": "synthetic-build-fingerprint",
                         "device_identity": device},
        }
        pointer = {
            "fingerprint": f"fingerprint-{index:02d}",
            "attempt": autobench_id,
            "result": worker.copy(),
            "source_fingerprint": "synthetic-source-fingerprint",
            "build_fingerprint": "synthetic-build-fingerprint",
            "build_output_identity": f"synthetic-build-output-{index:02d}",
            "device_identity": device,
            "scheduler_job_id": job_id,
            "cleanup_verified": True,
        }
        result_row = {
            "id": autobench_id,
            "job_id": job_id,
            "fingerprint": pointer["fingerprint"],
            "state": "succeeded",
            "result_json": json.dumps(result, sort_keys=True),
        }
        report_digest = recorded.sha256_file(report_path)
        receipt = {"schema": "cellerator.factorized_interactions.finished_evidence.v1",
                   "autobench_results_row": result_row,
                   "report_sha256": "0" * 64 if wrong_report_hash else report_digest}
        receipt_path = report_path.parent / "finished-evidence.json"
        pointer_path = report_path.parent / "current.json"
        _write_json(receipt_path, receipt)
        _write_json(pointer_path, pointer)
        return receipt_path, pointer_path

    def _assembly_inputs(self, *, missing_partition: bool = False,
                         wrong_receipt_report_hash: str | None = None):
        inputs = {}
        root = self._fixture_dir("assembly")
        report_root = root / "reports"
        report_root.mkdir()
        for index, kind in enumerate(recorded.KINDS, start=1):
            context = self._attempt_context(index)
            if kind == "dtype":
                report, path = self._bounded_dtype_report(context, report_root)
            elif kind.startswith("aux_packet_"):
                partition = "regular" if kind.endswith("_regular") else "high-fanin-1m"
                report, path = self._aux_report("packet", partition, context=context,
                                                fixture_parent=report_root)
            elif kind.startswith("aux_strongest_direct_"):
                partition = "regular" if kind.endswith("_regular") else "high-fanin-1m"
                report, path = self._aux_report("strongest-direct", partition, context=context,
                                                fixture_parent=report_root)
            else:
                path = self._fixture_dir("sanitizer", report_root) / "partition.json"
                report = {"schema": "synthetic-sanitizer-report.v1",
                          "scheduler_context": context}
            if missing_partition and kind == "aux_packet_regular":
                report.pop("partition", None)
            _write_json(path, report)
            wrong_digest = kind == wrong_receipt_report_hash
            receipt_path, pointer_path = self._receipt_pair(index, context, path,
                                                            wrong_report_hash=wrong_digest)
            inputs[kind] = (path, receipt_path, pointer_path)

        source = root / "source"
        source.mkdir()
        candidate_manifest = root / "candidate-manifest.json"
        baseline_manifest = root / "baseline-manifest.json"
        _write_json(candidate_manifest, {"synthetic": "candidate"})
        _write_json(baseline_manifest, {"synthetic": "baseline"})
        wrapper = root / "fixed-sanitizer-wrapper.py"
        wrapper.write_text("# synthetic wrapper boundary\n", encoding="utf-8")
        return inputs, source, candidate_manifest, baseline_manifest, wrapper

    def _patched_assembler(self, wrapper: Path):
        return (
            mock.patch.object(recorded, "_binary_identity", return_value=self.identity.copy()),
            mock.patch.object(recorded, "_sanitizer_report", return_value={
                "compute_sanitizer_home": "/synthetic/cuda",
                "compute_sanitizer_path": "/synthetic/cuda/compute-sanitizer",
                "compute_sanitizer_binary_sha256": "1" * 64,
                "compute_sanitizer_support_tree_sha256": "2" * 64,
                "compute_sanitizer_version": "synthetic-test-only",
                "suite_sha256": "3" * 64,
                "wrapper_sha256": recorded.sha256_file(wrapper),
            }),
            mock.patch.object(recorded, "SANITIZER_WRAPPER_SHA256",
                              recorded.sha256_file(wrapper)),
        )

    def test_assembler_and_bundle_validator_round_trip_nine_distinct_attempts(self) -> None:
        inputs, source, candidate, baseline, wrapper = self._assembly_inputs()
        patches = self._patched_assembler(wrapper)
        with patches[0], patches[1], patches[2]:
            bundle = recorded.assemble_qualification(inputs, source, candidate, baseline,
                                                     self.uuid, wrapper,
                                                     evidence_root=source.parent)
            self.assertEqual(bundle["status"], "validated_pending_independent_review")
            self.assertEqual(set(row["kind"] for row in bundle["reports"]), set(recorded.KINDS))
            self.assertEqual(len({row["scheduler_job_id"] for row in bundle["reports"]}), 9)
            self.assertEqual(len({row["scheduler_attempt_id"] for row in bundle["reports"]}), 9)
            self.assertEqual(bundle["coverage"]["dtype"], 32)
            self.assertEqual(bundle["coverage"]["aux_packet"], 12)
            self.assertEqual(bundle["coverage"]["aux_strongest_direct"], 12)

            bundle_path = source.parent / "qualification-bundle.json"
            _write_json(bundle_path, bundle)
            reconstructed = recorded.validate_existing_bundle(bundle_path, source, candidate,
                                                              baseline, self.uuid)
            self.assertEqual(reconstructed, bundle)

    def test_assembler_rejects_missing_partition_and_report_receipt_hash_mismatch(self) -> None:
        inputs, source, candidate, baseline, wrapper = self._assembly_inputs(missing_partition=True)
        patches = self._patched_assembler(wrapper)
        with patches[0], patches[1], patches[2]:
            with self.assertRaisesRegex(recorded.QualificationError, "partial-corpus boundary"):
                recorded.assemble_qualification(inputs, source, candidate, baseline,
                                                self.uuid, wrapper,
                                                evidence_root=source.parent)

        inputs, source, candidate, baseline, wrapper = self._assembly_inputs(
            wrong_receipt_report_hash="dtype")
        patches = self._patched_assembler(wrapper)
        with patches[0], patches[1], patches[2]:
            with self.assertRaisesRegex(recorded.QualificationError, "report digest"):
                recorded.assemble_qualification(inputs, source, candidate, baseline,
                                                self.uuid, wrapper,
                                                evidence_root=source.parent)

    def test_assembler_requires_all_nine_partitions_and_bundle_rejects_tampered_sources(self) -> None:
        inputs, source, candidate, baseline, wrapper = self._assembly_inputs()
        incomplete = dict(inputs)
        incomplete.pop("aux_packet_high_fanin_1m")
        patches = self._patched_assembler(wrapper)
        with patches[0], patches[1], patches[2]:
            with self.assertRaisesRegex(recorded.QualificationError, "exactly nine"):
                recorded.assemble_qualification(incomplete, source, candidate, baseline,
                                                self.uuid, wrapper,
                                                evidence_root=source.parent)
            bundle = recorded.assemble_qualification(inputs, source, candidate, baseline,
                                                     self.uuid, wrapper,
                                                     evidence_root=source.parent)
            bundle_path = source.parent / "qualification-bundle.json"
            _write_json(bundle_path, bundle)

            report_path, receipt_path, _ = inputs["dtype"]
            original_report = report_path.read_bytes()
            report_path.write_bytes(original_report + b" ")
            with self.assertRaisesRegex(recorded.QualificationError, "report changed"):
                recorded.validate_existing_bundle(bundle_path, source, candidate,
                                                  baseline, self.uuid)
            report_path.write_bytes(original_report)

            original_receipt = receipt_path.read_bytes()
            receipt_path.write_bytes(original_receipt + b" ")
            with self.assertRaisesRegex(recorded.QualificationError, "receipt changed"):
                recorded.validate_existing_bundle(bundle_path, source, candidate,
                                                  baseline, self.uuid)
            receipt_path.write_bytes(original_receipt)

            tampered_bundle = dict(bundle)
            tampered_bundle["reports"] = [dict(row) for row in bundle["reports"]]
            tampered_bundle["reports"][0]["report_sha256"] = "0" * 64
            tampered_path = source.parent / "tampered-bundle.json"
            _write_json(tampered_path, tampered_bundle)
            with self.assertRaisesRegex(recorded.QualificationError, "report changed"):
                recorded.validate_existing_bundle(tampered_path, source, candidate,
                                                  baseline, self.uuid)

    def test_portable_bundle_relocates_and_rejects_copy_tampering_escape_and_symlink(self) -> None:
        inputs, source, candidate, baseline, wrapper = self._assembly_inputs()
        evidence_root = source.parent
        patches = self._patched_assembler(wrapper)
        with patches[0], patches[1], patches[2]:
            bundle = recorded.assemble_qualification(
                inputs, source, candidate, baseline, self.uuid, wrapper,
                evidence_root=evidence_root,
            )
            self.assertEqual(bundle["evidence_root"], ".")
            for row in bundle["reports"]:
                for field in ("report_path", "receipt_path", "pointer_path"):
                    self.assertFalse(Path(row[field]).is_absolute())
                    self.assertNotIn("..", Path(row[field]).parts)
            self.assertFalse(Path(bundle["sanitizer_wrapper_path"]).is_absolute())

            bundle_path = evidence_root / "qualification-bundle.json"
            _write_json(bundle_path, bundle)
            original_report = inputs["dtype"][0].read_bytes()

            relocated = evidence_root.parent / "relocated-evidence"
            shutil.copytree(evidence_root, relocated)
            reconstructed = recorded.validate_existing_bundle(
                relocated / "qualification-bundle.json",
                relocated / "source", relocated / "candidate-manifest.json",
                relocated / "baseline-manifest.json", self.uuid,
            )
            self.assertEqual(reconstructed, bundle)

            dtype_row = next(row for row in bundle["reports"] if row["kind"] == "dtype")
            copied_report = relocated / dtype_row["report_path"]
            copied_report.write_bytes(copied_report.read_bytes() + b" ")
            with self.assertRaisesRegex(recorded.QualificationError, "report changed"):
                recorded.validate_existing_bundle(
                    relocated / "qualification-bundle.json",
                    relocated / "source", relocated / "candidate-manifest.json",
                    relocated / "baseline-manifest.json", self.uuid,
                )
            self.assertEqual(inputs["dtype"][0].read_bytes(), original_report)

            traversal_root = evidence_root.parent / "traversal-evidence"
            shutil.copytree(evidence_root, traversal_root)
            traversal_bundle_path = traversal_root / "qualification-bundle.json"
            traversal_bundle = copy.deepcopy(bundle)
            traversal_row = next(row for row in traversal_bundle["reports"] if row["kind"] == "dtype")
            traversal_row["report_path"] = "../outside-qualification.json"
            _write_json(traversal_bundle_path, traversal_bundle)
            with self.assertRaisesRegex(recorded.QualificationError, "escapes"):
                recorded.validate_existing_bundle(
                    traversal_bundle_path, traversal_root / "source",
                    traversal_root / "candidate-manifest.json",
                    traversal_root / "baseline-manifest.json", self.uuid,
                )

            symlink_root = evidence_root.parent / "symlink-evidence"
            shutil.copytree(evidence_root, symlink_root)
            symlink_bundle_path = symlink_root / "qualification-bundle.json"
            symlink_row = next(row for row in bundle["reports"] if row["kind"] == "dtype")
            symlink_target = symlink_root / symlink_row["report_path"]
            external = self.root / "external-report.json"
            external.write_bytes(original_report)
            symlink_target.unlink()
            symlink_target.symlink_to(external)
            with self.assertRaisesRegex(recorded.QualificationError, "symbolic link"):
                recorded.validate_existing_bundle(
                    symlink_bundle_path, symlink_root / "source",
                    symlink_root / "candidate-manifest.json",
                    symlink_root / "baseline-manifest.json", self.uuid,
                )

    def test_legacy_dtype_accepts_full_eight_triple_four_mode_matrix(self) -> None:
        report, path = self._dtype_report()
        checked = recorded._legacy_dtype(report, path, self.identity, self.context)
        self.assertEqual(len(checked), 32)
        self.assertEqual({(row["activation"], row["parameter"], row["output"])
                          for row in checked}, set(itertools.product(("fp16", "fp32"), repeat=3)))
        self.assertEqual({row["mode"] for row in checked}, set(recorded.MODES))

    def test_legacy_dtype_rejects_missing_and_duplicate_dtype_rows(self) -> None:
        missing, path = self._dtype_report()
        missing["records"].pop()
        with self.assertRaisesRegex(recorded.QualificationError, "incomplete records"):
            recorded._legacy_dtype(missing, path, self.identity, self.context)

        duplicate, duplicate_path = self._dtype_report()
        duplicate["records"][-1] = duplicate["records"][0].copy()
        with self.assertRaisesRegex(recorded.QualificationError, "duplicate or unexpected"):
            recorded._legacy_dtype(duplicate, duplicate_path, self.identity, self.context)

    def test_legacy_dtype_rejects_backend_mismatch_changed_wire_and_failed_ratio(self) -> None:
        report, path = self._dtype_report()
        report["records"][0]["selected_backend"] = "direct"
        with self.assertRaisesRegex(recorded.QualificationError, "declared implementation"):
            recorded._legacy_dtype(report, path, self.identity, self.context)

        report, path = self._dtype_report()
        first = report["records"][0]
        case = path.parent / "cases" / f"{first['mode']}.a{first['activation']}.p{first['parameter']}.o{first['output']}"
        (case / "input.case").write_bytes(b"altered after validation")
        with self.assertRaisesRegex(recorded.QualificationError, "wire differs"):
            recorded._legacy_dtype(report, path, self.identity, self.context)

        report, path = self._dtype_report(bad_ratio=True)
        with self.assertRaisesRegex(recorded.QualificationError, "tolerance failed"):
            recorded._legacy_dtype(report, path, self.identity, self.context)

    def test_bounded_auxiliary_accepts_all_role_partition_combinations(self) -> None:
        for role in ("packet", "strongest-direct"):
            for partition in ("regular", "high-fanin-1m"):
                with self.subTest(role=role, partition=partition):
                    report, path = self._aux_report(role, partition)
                    kind = f"aux_{role.replace('-', '_')}_{partition.replace('-', '_')}"
                    checked = recorded._bounded_qualification(report, path, kind,
                                                              self.identity, self.context)
                    expected_count = 11 if partition == "regular" else 1
                    self.assertEqual(len(checked), expected_count)
                    self.assertEqual({row["workload"] for row in checked},
                                     {row["workload"] for row in report["records"]})

    def test_bounded_auxiliary_rejects_missing_or_mixed_partition_cases(self) -> None:
        report, path = self._aux_report("packet", "regular")
        report["records"].pop()
        with self.assertRaisesRegex(recorded.QualificationError, "incomplete"):
            recorded._bounded_qualification(report, path, "aux_packet_regular",
                                            self.identity, self.context)

        report, path = self._aux_report("strongest-direct", "regular")
        report["records"][0]["workload"] = "correctness_packet_high_fanin_1m"
        with self.assertRaisesRegex(recorded.QualificationError, "out-of-scope"):
            recorded._bounded_qualification(report, path, "aux_strongest_direct_regular",
                                            self.identity, self.context)

    def test_bounded_auxiliary_rejects_stale_evidence_and_strongest_mode_mismatch(self) -> None:
        report, path = self._aux_report("packet", "high-fanin-1m")
        evidence = path.parent / report["records"][0]["evidence_relative_path"]
        evidence.write_text(evidence.read_text() + " ", encoding="utf-8")
        with self.assertRaisesRegex(recorded.QualificationError, "evidence digest changed"):
            recorded._bounded_qualification(report, path, "aux_packet_high_fanin_1m",
                                            self.identity, self.context)

        report, path = self._aux_report("strongest-direct", "high-fanin-1m",
                                        direct_mode="direct_unique_output")
        with self.assertRaisesRegex(recorded.QualificationError, "strongest direct control"):
            recorded._bounded_qualification(report, path,
                                            "aux_strongest_direct_high_fanin_1m",
                                            self.identity, self.context)

    def test_bounded_auxiliary_requires_output_digest_and_contract_pass(self) -> None:
        report, path = self._aux_report("packet", "high-fanin-1m")
        del report["records"][0]["output_result_sha256"]
        with self.assertRaisesRegex(recorded.QualificationError, "output_result_sha256"):
            recorded._bounded_qualification(report, path, "aux_packet_high_fanin_1m",
                                            self.identity, self.context)

        report, path = self._aux_report("packet", "regular", bad_ratio=True)
        with self.assertRaisesRegex(recorded.QualificationError, "tolerance failed"):
            recorded._bounded_qualification(report, path, "aux_packet_regular",
                                            self.identity, self.context)

    def test_bounded_auxiliary_rejects_changed_driver_or_case_helper_hash(self) -> None:
        report, path = self._aux_report("packet", "regular")
        report["harness"]["qualification_sha256"] = "0" * 64
        with self.assertRaisesRegex(recorded.QualificationError, "unapproved collector"):
            recorded._bounded_qualification(report, path, "aux_packet_regular",
                                            self.identity, self.context)

        report, path = self._aux_report("strongest-direct", "regular")
        row = report["records"][0]
        evidence_path = path.parent / row["evidence_relative_path"]
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        validated_path = path.parent / evidence["validated_relative_path"]
        validated = json.loads(validated_path.read_text(encoding="utf-8"))
        validated["bounded_validation_sha256"] = "0" * 64
        _write_json(validated_path, validated)
        evidence["bounded_validation_sha256"] = "0" * 64
        evidence["validated_record_sha256"] = _sha(validated_path.read_bytes())
        _write_json(evidence_path, evidence)
        row["evidence_sha256"] = _sha(evidence_path.read_bytes())
        with self.assertRaisesRegex(recorded.QualificationError, "different bounded validation helper"):
            recorded._bounded_qualification(report, path, "aux_strongest_direct_regular",
                                            self.identity, self.context)


if __name__ == "__main__":
    unittest.main()
