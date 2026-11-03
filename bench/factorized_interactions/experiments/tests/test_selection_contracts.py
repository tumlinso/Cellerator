from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from adapter import (  # noqa: E402
    DESCRIPTIVE_DIRECT_ID,
    _candidate_argument_for_case,
    _candidate_mode_identity,
    alternating_pair_order,
    get_or_measure_pair,
)
import adapter as adapter_module  # noqa: E402
from common import AdapterError, digest_file  # noqa: E402
from in_job_kda import CONTEXT_FILE, SCHEMA as OUTER_CONTEXT_SCHEMA, scheduler_unit  # noqa: E402
from prepare_task import _direct_control_facts, make_task  # noqa: E402
from selection_guard import _guard, _validate_auxiliary_corpus  # noqa: E402


def native_identity(facts: dict, candidate: str) -> tuple[dict, dict]:
    if candidate == "direct_unique_output":
        selected, implementation, layout = "direct", "direct_unique_output", "canonical_direct"
    elif candidate == "minimal_direct":
        selected = facts["native_direct_backend_from_record_stats"]
        implementation = selected
        layout = "canonical_direct" if selected == "direct" else "full_packet"
    elif candidate == "native_direct":
        selected = facts["native_direct_backend_from_record_stats"]
        implementation, layout = selected, "full_packet"
    else:
        selected, implementation, layout = "packet", "packet", "full_packet"
    native = {
        "selected_backend": selected,
        "actual_implementation": implementation,
        "structural_layout": layout,
        "candidate": candidate,
        "cells": facts["cells"], "channels": facts["channels"],
        "mechanisms": facts["mechanisms"], "outputs": facts["outputs"],
        "structural_maxima": {
            "output_mechanisms": facts["max_output_mechanisms"],
            "input_roles": facts["max_input_roles"],
            "context_terms": facts["max_context_terms"],
            "alpha_mechanisms": facts["max_alpha_mechanisms"],
            "bias_mechanisms": facts["max_bias_mechanisms"],
            "gamma_terms": facts["max_gamma_terms"],
            "mechanism_terms": facts["max_mechanism_terms"],
        },
    }
    metrics = {
        "selected_backend": selected,
        "actual_implementation": implementation,
        "structural_layout": layout,
    }
    return native, metrics


class DirectControlMappingTests(unittest.TestCase):
    def test_exact_unique_output_record_selects_unique_control(self) -> None:
        facts = _direct_control_facts("packet_a4_m32_u16")
        self.assertEqual(facts["mechanisms"], facts["outputs"])
        self.assertEqual(facts["max_output_mechanisms"], 1)
        self.assertTrue(facts["unique_output_direct_applicable"])
        self.assertEqual(facts["strongest_direct_candidate_argument"], "direct_unique_output")

        native, metrics = native_identity(facts, "direct_unique_output")
        variant = {
            "allowed_backends": ["direct"],
            "allowed_implementations": ["direct_unique_output"],
            "allowed_layouts": ["canonical_direct"],
        }
        self.assertEqual(
            _candidate_mode_identity(
                native, metrics, variant, candidate_argument="direct_unique_output",
                direct_control_facts=facts,
            ),
            ("direct_unique_output", "canonical_direct"),
        )

    def test_nonunique_and_high_fanin_records_use_minimal_direct_mode(self) -> None:
        for workload_id in (
            "correctness_ordered_tied_zero",
            "correctness_packet_high_fanin_fp32_100k",
        ):
            with self.subTest(workload_id=workload_id):
                facts = _direct_control_facts(workload_id)
                self.assertFalse(facts["unique_output_direct_applicable"])
                self.assertEqual(facts["strongest_direct_candidate_argument"], "minimal_direct")
                native, metrics = native_identity(facts, "minimal_direct")
                variant = {
                    "allowed_backends": ["direct", "stable"],
                    "allowed_implementations": ["direct", "stable"],
                    "allowed_layouts": ["canonical_direct", "full_packet"],
                }
                self.assertIn(
                    _candidate_mode_identity(
                        native, metrics, variant, candidate_argument="minimal_direct",
                        direct_control_facts=facts,
                    )[0],
                    {"direct", "stable"},
                )

    def test_case_argument_uses_record_facts_not_variant_default(self) -> None:
        facts = _direct_control_facts("packet_a4_m32_u16")
        variant_map = {
            "comparison": {"incumbent": "direct", "challenger": "packet"},
            "variants": {
                "direct": {"candidate_argument": "native_direct"},
                "packet": {"candidate_argument": "packet"},
            },
        }
        case = {"parameters": {"direct_control": facts}}
        self.assertEqual(
            _candidate_argument_for_case(variant_map, "direct", case)[0],
            "direct_unique_output",
        )
        self.assertEqual(
            _candidate_argument_for_case(variant_map, "packet", case)[0],
            "packet",
        )


class PairedBlockTests(unittest.TestCase):
    def test_three_arm_order_rotates_and_cache_reuses_exact_block(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            block_path = Path(temporary) / "pair.json"
            calls: list[str] = []

            def measure(candidate_id: str) -> dict:
                calls.append(candidate_id)
                return {"candidate_id": candidate_id, "sample": 2.0}

            orders = []
            for round_index in range(3):
                order = alternating_pair_order("direct", "packet", "exploration", round_index)
                position = round_index % 3
                block = get_or_measure_pair(
                    block_path, pair_key=f"pair-{round_index}",
                    identity={"round": round_index}, order=order,
                    measure=measure, descriptive_position=position,
                )
                self.assertEqual(block["execution_order"][position], DESCRIPTIVE_DIRECT_ID)
                orders.append(block["execution_order"])
                block_path.unlink()
            self.assertEqual(len(calls), 9)
            self.assertEqual(len({row.index(DESCRIPTIVE_DIRECT_ID) for row in orders}), 3)

            before = len(calls)
            order = ["direct", "packet"]
            block_path = Path(temporary) / "cached.json"
            first = get_or_measure_pair(
                block_path, pair_key="cached", identity={"case": "x"}, order=order,
                measure=measure, descriptive_position=1,
            )
            second = get_or_measure_pair(
                block_path, pair_key="cached", identity={"case": "x"}, order=order,
                measure=lambda _: self.fail("cached measurements must not be rerun"),
                descriptive_position=1,
            )
            self.assertEqual(first, second)
            self.assertEqual(len(calls) - before, 3)


class AuxiliaryValidationTests(unittest.TestCase):
    def test_auxiliary_validation_rejects_wrong_direct_mode(self) -> None:
        facts_unique = {
            "workload_id": "unique", "strongest_direct_candidate_argument": "direct_unique_output",
        }
        facts_minimal = {
            "workload_id": "fanin", "strongest_direct_candidate_argument": "minimal_direct",
        }
        variants = {
            "comparison": {"incumbent": "direct", "challenger": "packet"},
            "variants": {
                "direct": {"route": "direct", "candidate_argument": "native_direct"},
                "packet": {"route": "packet", "candidate_argument": "packet"},
            },
        }
        identities = {
            "direct": {"runner_binary_sha256": "a" * 64},
            "packet": {"runner_binary_sha256": "b" * 64},
        }
        rows_direct = [
            {"workload_id": "unique", "status": "pass", "runner_binary_sha256": "a" * 64,
             "candidate_argument": "direct_unique_output", "direct_control_facts": facts_unique},
            {"workload_id": "fanin", "status": "pass", "runner_binary_sha256": "a" * 64,
             "candidate_argument": "minimal_direct", "direct_control_facts": facts_minimal},
        ]
        rows_packet = [
            {"workload_id": case, "status": "pass", "runner_binary_sha256": "b" * 64,
             "candidate_argument": "packet"}
            for case in ("unique", "fanin")
        ]
        summary = {"validation": {
            "direct": {"status": "pass", "response": {"auxiliary_correctness": {
                "status": "pass", "cases": rows_direct,
            }}},
            "packet": {"status": "pass", "response": {"auxiliary_correctness": {
                "status": "pass", "cases": rows_packet,
            }}},
        }}
        task = {"numerical_requirements": {
            "auxiliary_correctness_workload_ids": ["unique", "fanin"],
        }}
        self.assertEqual(
            _validate_auxiliary_corpus(summary, variants, identities, task)["status"], "pass",
        )
        rows_direct[0]["candidate_argument"] = "minimal_direct"
        self.assertEqual(
            _validate_auxiliary_corpus(summary, variants, identities, task)["status"], "fail",
        )


class ExplicitProbeTests(unittest.TestCase):
    UUID = "GPU-12345678-1234-5678-9abc-def012345678"
    OTHER_UUID = "GPU-87654321-4321-8765-cba9-876543210fed"

    @staticmethod
    def result(stdout: str):
        class Result:
            returncode = 0
            stderr = ""
            def __init__(self, value: str):
                self.stdout = value
        return Result(stdout)

    def test_probe_queries_only_the_verified_uuid(self) -> None:
        request = {
            "request_id": "probe-1",
            "toolchain": {"nvcc": "nvcc"},
            "declared_target": {"expected_device_uuid": self.UUID},
        }
        context = {
            "scheduler_job_id": "job-1", "scheduler_attempt_id": "attempt-1",
            "device_uuid": self.UUID, "context_sha256": "c" * 64,
            "cgroup": "/device-scheduler/job/attempt.service", "unit_name": "attempt.service",
        }
        stdout = f"{self.UUID}, 7.0, driver, Tesla V100\n"
        with patch.object(adapter_module, "_require_admitted", return_value=context), \
                patch.object(adapter_module.subprocess, "run", side_effect=[
                    self.result(stdout),
                    self.result("Cuda compilation tools, release 12.9, V12.9.0\n"),
                ]) as run:
            response = adapter_module._probe(request, Path("unused-request.json"))
        self.assertEqual(run.call_args_list[0].args[0][1:3], ["-i", self.UUID])
        self.assertEqual(response["environment"]["device_uuid"], self.UUID)
        self.assertEqual(response["environment"]["scheduler_attempt_id"], "attempt-1")

    def test_probe_rejects_a_different_physical_uuid(self) -> None:
        request = {
            "request_id": "probe-2",
            "toolchain": {"nvcc": "nvcc"},
            "declared_target": {"expected_device_uuid": self.UUID},
        }
        context = {
            "scheduler_job_id": "job-1", "scheduler_attempt_id": "attempt-1",
            "device_uuid": self.UUID, "context_sha256": "c" * 64,
            "cgroup": "/device-scheduler/job/attempt.service", "unit_name": "attempt.service",
        }
        with patch.object(adapter_module, "_require_admitted", return_value=context), \
                patch.object(adapter_module.subprocess, "run", return_value=self.result(
                    f"{self.OTHER_UUID}, 7.0, driver, Other GPU\n"
                )):
            with self.assertRaises(AdapterError):
                adapter_module._probe(request, Path("unused-request.json"))


class ReferenceCacheWiringTests(unittest.TestCase):
    UUID = "GPU-12345678-1234-5678-9abc-def012345678"

    def test_timing_and_candidate_validation_share_one_cache_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempt = root / "attempt"
            attempt.mkdir()
            cache_dir = adapter_module._reference_cache_directory(attempt)
            workload = {
                "id": "case-1", "workload_sha256": "a" * 64,
                "input": {"workload_id": "fixed-case"},
                "parameters": {
                    "activation_dtype": "fp16", "parameter_dtype": "fp16",
                    "output_dtype": "fp16", "gradient_mode": "overwrite",
                },
            }
            timing_request = {"case": workload, "timing": {"metric": "primary"}}
            captured: list[list[str]] = []

            class StopAfterArgv(Exception):
                pass

            def capture_timing(argv, **_kwargs):
                captured.append(argv)
                raise StopAfterArgv

            with patch.object(adapter_module, "_run_json", side_effect=capture_timing):
                with self.assertRaises(StopAfterArgv):
                    adapter_module._measurement_command(
                        timing_request, candidate_id="packet-candidate",
                        variant={"route": "packet"}, runner_path=root / "runner",
                        manifest_path=root / "manifest.json", runner_hash="b" * 64,
                        snapshot_hash="c" * 64, artifact_dir=root / "timing-output",
                        iterations=1, device_uuid=self.UUID,
                        reference_cache_dir=cache_dir,
                        candidate_argument_override="packet",
                    )

            request_path = root / "probe-request.json"
            request_path.write_text("{}\n", encoding="utf-8")
            variant = {
                "route": "packet", "candidate_argument": "packet",
                "allowed_backends": ["packet"],
                "allowed_implementations": ["packet"],
                "allowed_layouts": ["full_packet"],
            }
            native = {
                "candidate": "packet", "selected_backend": "packet",
                "actual_implementation": "packet", "structural_layout": "full_packet",
                "requested_route": "packet",
            }
            validation_request = {
                "request_id": "validate-1",
                "candidate": {"id": "packet-candidate", "snapshot_sha256": "d" * 64},
                "cases": [workload],
                "numerical_requirements": {"auxiliary_correctness_workload_ids": ["correctness"]},
            }
            summary = {"validated": True, "native_execution": native}
            fake_workloads = types.ModuleType("workloads")
            fake_workloads.correctness_workload_ids = lambda: ["correctness"]
            borrowed_context = {
                "schema": OUTER_CONTEXT_SCHEMA, "status": "observed",
                "scheduler_job_id": "job-test", "scheduler_attempt_id": "attempt-test",
                "device_uuid": self.UUID, "cgroup": "/system.slice/test.service",
                "unit_name": "test.service", "task_sha256": "e" * 64,
                "entrypoint_sha256": "f" * 64, "pid": 42, "context_sha256": "a" * 64,
            }
            with patch.dict(sys.modules, {"workloads": fake_workloads}), \
                    patch.object(adapter_module, "_require_admitted",
                                  return_value=borrowed_context), \
                    patch.object(adapter_module, "attempt_dir_for_request", return_value=attempt), \
                    patch.object(adapter_module, "_workspace_from_attempt", return_value=root), \
                    patch.object(adapter_module, "read_json", return_value={}), \
                    patch.object(adapter_module, "_candidate_record",
                                 return_value={"snapshot_sha256": "d" * 64}), \
                    patch.object(adapter_module, "checked_build",
                                 return_value=(root / "runner", root / "manifest.json", "b" * 64)), \
                    patch.object(adapter_module, "_candidate_argument_for_case",
                                 return_value=("packet", None)), \
                    patch.object(adapter_module, "_run_json", return_value=summary) as run_json:
                result = adapter_module._validate_candidate(
                    validation_request, request_path,
                    {"comparison": {"incumbent": "direct-control"},
                     "variants": {"packet-candidate": variant}},
                )

            self.assertEqual(result["status"], "ok")
            self.assertEqual(run_json.call_count, 2)
            all_argv = [captured[0], *[call.args[0] for call in run_json.call_args_list]]
            for argv in all_argv:
                cache_index = argv.index("--reference-cache-dir")
                self.assertEqual(argv[cache_index + 1], str(cache_dir))
            self.assertEqual(cache_dir.parent, attempt.resolve())


class PreparedTaskExecutionTests(unittest.TestCase):
    UUID = "GPU-12345678-1234-5678-9abc-def012345678"

    def test_task_binds_direct_mode_uuid_and_trusted_in_job_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            helper = root / "build_helper.py"
            helper.write_text("raise SystemExit(0)\n", encoding="utf-8")
            site = root / "site.json"
            site.write_text("{}\n", encoding="utf-8")
            nvcc = root / "nvcc"
            nvcc.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            variant_path = root / "variants.json"
            variant_path.write_text(__import__("json").dumps({
                "schema_version": 1,
                "comparison": {
                    "incumbent": "direct-control", "challenger": "packet-candidate",
                    "strongest_direct_control": "direct-control",
                    "same_source_runner_required": True,
                },
                "runner_iterations": 2,
                "builder": {
                    "trusted_script": str(helper),
                    "snapshot_script_relative_path": "bench/factorized_interactions/build.py",
                    "argv": ["python3", "{source}", "--build", "{build}"],
                },
                "variants": {
                    "direct-control": {
                        "enabled": True, "route": "direct", "candidate_argument": "native_direct",
                        "allowed_backends": ["direct", "stable"],
                    },
                    "packet-candidate": {
                        "enabled": True, "route": "packet", "candidate_argument": "packet",
                        "allowed_backends": ["packet", "stable"],
                        "allowed_implementations": ["packet"],
                        "allowed_layouts": ["full_packet"],
                        "win_backend": "packet", "win_layout": "full_packet",
                    },
                },
            }), encoding="utf-8")
            task_path = root / "task.json"
            task = make_task(
                output=task_path, variant_map_path=variant_path,
                site_policy_path=site, nvcc=nvcc,
                expected_device_uuid=self.UUID,
                candidate_win_cases=["packet_a4_m32_u16"],
                max_seconds=30, task_id="cpu-task-contract",
            )
            self.assertEqual(task["execution"], {"mode": "direct"})
            self.assertEqual(task["target"]["expected_device_uuid"], self.UUID)
            self.assertEqual(
                task["factorized_interaction"]["expected_device_uuid"], self.UUID,
            )
            self.assertEqual(
                task["factorized_interaction"]["outer_execution_context_schema"],
                "cellerator.factorized_interactions.outer_execution_context.v1",
            )
            self.assertTrue(any(
                item["id"] == "in_job_entrypoint" and
                item["path"] == str((HERE / "in_job_kda.py").resolve())
                for item in task["trusted_inputs"]
            ))
            reference_cache = HERE.parents[2] / "tests" / "factorized_interactions" / "reference_cache.py"
            self.assertTrue(any(
                item["id"] == "reference_cache" and
                item["path"] == str(reference_cache.resolve())
                for item in task["trusted_inputs"]
            ))
            qualification_inputs = {
                "strict_sanitizer_suite": HERE.parent / "run_sanitizer_suite.py",
                "dtype_qualification_driver": HERE / "run_dtype_qualification.py",
                "performance_packet_driver": HERE / "run_performance_packet.py",
            }
            by_id = {item["id"]: item["path"] for item in task["trusted_inputs"]}
            for identifier, path in qualification_inputs.items():
                self.assertEqual(by_id[identifier], str(path.resolve()))
            self.assertNotIn("launcher_argv", task["execution"])


class ConditionalSelectionTests(unittest.TestCase):
    UUID = "GPU-12345678-1234-5678-9abc-def012345678"

    def test_passing_root_gates_remain_conditional_pending_outer_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            attempt = workspace / "attempt-1"
            attempt.mkdir()
            task_path = root / "task.json"
            variant_path = root / "variant-map.json"
            variant_path.write_text("{}\n", encoding="utf-8")
            summary_path = attempt / "summary.json"
            summary_path.write_text("{}\n", encoding="utf-8")

            job_id = "job-cpu-test"
            attempt_id = "attempt-cpu-test"
            unit_name = scheduler_unit(attempt_id)
            cgroup = f"/system.slice/{unit_name}"
            expected_uuid = self.UUID
            entrypoint = HERE / "in_job_kda.py"
            context = {
                "schema": OUTER_CONTEXT_SCHEMA,
                "status": "observed",
                "scheduler_job_id": job_id,
                "scheduler_attempt_id": attempt_id,
                "device_uuid": expected_uuid,
                "cgroup": cgroup,
                "unit_name": unit_name,
                "task_sha256": "pending",
                "entrypoint_sha256": digest_file(entrypoint),
                "pid": 123,
            }
            trusted = [{"id": "in_job_entrypoint", "path": str(entrypoint.resolve())}]
            task = {
                "task_id": "conditional-cpu-test",
                "execution": {"mode": "direct"},
                "target": {"expected_device_uuid": expected_uuid},
                "trusted_inputs": trusted,
                "numerical_requirements": {
                    "root_selection_policy": {
                        "experiment_only": True,
                        "minimum_paired_blocks_per_case": 7,
                        "minimum_confirmation_blocks_per_case": 7,
                        "required_candidate_win_point_speedup": 1.10,
                        "required_candidate_win_ci_low_exclusive": 1.0,
                        "maximum_supported_case_time_regression": 0.05,
                        "kda_max_regression_benefit_units": 1.0 - 1.0 / 1.05,
                    },
                    "auxiliary_correctness_workload_ids": ["correctness"],
                },
                "promotion": {
                    "min_improvement": 0.0,
                    "max_regression": 1.0 - 1.0 / 1.05,
                    "confidence": 0.95,
                },
                "timing": {
                    "rounds": 7,
                    "confirmation_rounds": 7,
                    "metric": "warm_forward_vjp_batch_host_wall_mean",
                },
                "cases": [{
                    "id": "perf",
                    "parameters": {
                        "candidate_win_required": True,
                        "direct_control": {
                            "strongest_direct_candidate_argument": "minimal_direct",
                        },
                    },
                }],
                "factorized_interaction": {
                    "expected_device_uuid": expected_uuid,
                    "outer_execution_context_schema": OUTER_CONTEXT_SCHEMA,
                    "outer_execution_context_file": CONTEXT_FILE,
                    "in_job_entrypoint_path": str(entrypoint.resolve()),
                    "in_job_entrypoint_sha256": digest_file(entrypoint),
                    "variant_map_path": str(variant_path),
                    "all_fixed_workload_ids": ["perf", "correctness"],
                    "timed_performance_case_ids": ["perf"],
                    "correctness_only_case_ids_direct_unmeasured": ["correctness"],
                    "unmeasured_performance_case_ids_direct_fallback": [],
                    "direct_control_facts_by_performance_case": {},
                },
            }
            task_path.write_text("{}\n", encoding="utf-8")
            context["task_sha256"] = digest_file(task_path)
            context_path = workspace / CONTEXT_FILE
            context_path.write_text(__import__("json").dumps(context), encoding="utf-8")

            summary = {
                "status": "complete",
                "run_id": attempt.name,
                "task_sha256": digest_file(task_path),
                "trusted_inputs_unchanged": True,
                "tools_unchanged": True,
                "execution_receipt": {"status": "ok", "mode": "direct"},
                "environment": {
                    "scheduler_job_id": job_id,
                    "scheduler_attempt_id": attempt_id,
                    "device_uuid": expected_uuid,
                    "outer_execution_context_sha256": digest_file(context_path),
                    "outer_cgroup": cgroup,
                    "outer_unit_name": unit_name,
                },
                "decision": {
                    "eligible": True,
                    "candidate_id": "packet",
                    "ci_low": 1.1,
                    "per_case": {"perf": {"score": 1.25, "ci_low": 1.1, "ci_high": 1.4}},
                },
                "preferred": {"candidate_id": "packet"},
            }
            task_path.write_text(__import__("json").dumps(task), encoding="utf-8")
            task["factorized_interaction"]["variant_map_sha256"] = digest_file(variant_path)
            task_path.write_text(__import__("json").dumps(task), encoding="utf-8")
            # The context is task-bound, so refresh its digest after the final task bytes are written.
            context["task_sha256"] = digest_file(task_path)
            context_path.write_text(__import__("json").dumps(context), encoding="utf-8")
            summary["task_sha256"] = digest_file(task_path)
            summary["environment"]["outer_execution_context_sha256"] = digest_file(context_path)

            variant_map = {
                "comparison": {"incumbent": "direct", "challenger": "packet"},
                "variants": {
                    "direct": {
                        "route": "direct", "candidate_argument": "minimal_direct",
                        "allowed_backends": ["direct", "stable"],
                        "accepted_route": "direct",
                    },
                    "packet": {
                        "route": "packet", "candidate_argument": "packet",
                        "allowed_backends": ["packet", "stable"],
                        "accepted_route": "packet", "win_backend": "packet",
                        "win_layout": "full_packet",
                    },
                },
            }
            auxiliary = {
                "status": "pass", "failures": [],
                "candidate_results": {
                    "direct": {"cases": [{"workload_id": "correctness",
                                             "direct_control_facts": {
                                                 "strongest_direct_candidate_argument": "minimal_direct",
                                             }}]},
                },
            }
            grouped = {}
            for phase, phase_offset in (("exploration", 0), ("confirmation", 7)):
                for round_index in range(7):
                    order = ["direct", "packet"] if (round_index + phase_offset) % 2 == 0 \
                        else ["packet", "direct"]
                    position = (round_index + phase_offset) % 3
                    execution_order = list(order)
                    execution_order.insert(position, DESCRIPTIVE_DIRECT_ID)
                    pair_key = f"{phase}-perf-{round_index}"
                    rows = {}
                    for candidate_id, sample, implementation, layout in (
                            ("direct", 1.25, "direct", "canonical_direct"),
                            ("packet", 1.0, "packet", "full_packet")):
                        rows[candidate_id] = {
                            "sample": sample,
                            "paired_block": {
                                "pair_key": pair_key, "order": order,
                                "execution_order": execution_order,
                                "descriptive_control": {"position": position},
                                "backend_pair_distinct": True,
                            },
                            "measurement": {
                                "actual_implementation": implementation,
                                "selected_backend": implementation,
                                "structural_layout": layout,
                                "metrics": {},
                            },
                            "response": {
                                "descriptive_control_measurement": {
                                    "sample": 1.3, "actual_implementation": "direct",
                                    "selected_backend": "direct", "structural_layout": "full_packet",
                                    "metrics": {},
                                },
                            },
                        }
                    grouped[(phase, "perf", round_index)] = rows

            fake_workloads = types.ModuleType("workloads")
            fake_workloads.correctness_workload_ids = lambda: ["correctness"]
            fake_workloads.performance_workload_ids = lambda: ["perf"]
            fake_workloads.workload_ids = lambda: ["perf", "correctness"]
            with patch.dict(sys.modules, {"workloads": fake_workloads}), \
                    patch("selection_guard._candidate_identities", return_value={
                        "direct": {"candidate_id": "direct", "runner_binary_sha256": "a" * 64},
                        "packet": {"candidate_id": "packet", "runner_binary_sha256": "b" * 64},
                    }), \
                    patch("selection_guard._validate_auxiliary_corpus", return_value=auxiliary), \
                    patch("selection_guard._responses_for_run", return_value=grouped):
                record = _guard(
                    task=task, task_path=task_path, workspace=workspace, attempt=attempt,
                    summary=summary, variant_map=variant_map, summary_path=summary_path,
                )

            self.assertEqual(record["status"], "conditional")
            self.assertEqual(record["gate_status"], "pass")
            self.assertEqual(record["outer_receipt_verification"], "pending")
            self.assertEqual(
                record["outer_execution_context"]["verification_status"],
                "pending_external_scheduler_receipt_and_cleanup",
            )
            self.assertEqual(record["routes_by_case"]["perf"]["route"], "packet")
            self.assertFalse(record["production_route_changed"])


if __name__ == "__main__":
    unittest.main()
