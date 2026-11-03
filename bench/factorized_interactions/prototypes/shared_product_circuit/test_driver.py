"""CPU-only fixtures and validator checks; these do not claim CUDA execution."""

from __future__ import annotations

import math
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import driver  # noqa: E402
import fixtures  # noqa: E402
from reference import evaluate  # noqa: E402


class SharedProductFixtureTests(unittest.TestCase):
    def test_frozen_wire_corpus_and_oracle_cache_are_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            a = fixtures.freeze_cases(first)
            b = fixtures.freeze_cases(second)
            self.assertEqual(a["sha256"], b["sha256"])
            self.assertEqual([row["sha256"] for row in a["cases"]],
                             [row["sha256"] for row in b["cases"]])
            cases = fixtures.verify_frozen_cases(Path(a["path"]))
            expected = fixtures.load_oracle_cache(Path(a["path"]))
            self.assertEqual(set(cases), set(expected))
            self.assertEqual(len(cases), 9)
            for case_id, values in expected.items():
                self.assertEqual(set(values), {
                    "y", "scale_y", "dx", "dz", "dalpha", "dbias", "dgamma",
                    "scale_dx", "scale_dz", "scale_dalpha", "scale_dbias", "scale_dgamma",
                }, case_id)
                self.assertTrue(all(array.dtype == np.float64 for array in values.values()))

    def test_fixed_shapes_dtypes_and_two_distinct_ordinary_seeds(self):
        cases = fixtures.build_cases()
        for case_id, (problem, dy) in cases.items():
            with self.subTest(case=case_id):
                self.assertEqual(problem.x.shape[0], 10)
                self.assertEqual(problem.x.shape[1], 32)
                self.assertEqual(problem.x.dtype, np.float16)
                self.assertEqual(problem.z.dtype, np.float16)
                self.assertEqual(problem.alpha.shape, (4, 32))
                self.assertEqual(problem.bias.shape, (3,))
                self.assertEqual(problem.gamma.shape, (4,))
                self.assertEqual(problem.alpha.dtype, np.float16)
                self.assertEqual(problem.bias.dtype, np.float16)
                self.assertEqual(problem.gamma.dtype, np.float16)
                self.assertEqual(dy.dtype, np.float32)
                self.assertEqual(dy.shape, (1, 32, problem.x.shape[2]))
                self.assertEqual(len(problem.mechanisms), 8)
                self.assertEqual(sum(len(m.context_ids) for m in problem.mechanisms), 16)
        first = cases["ordinary_seed_0"][0]
        second = cases["ordinary_seed_1"][0]
        timed = cases["timed_seed_0"][0]
        self.assertEqual(first.x.shape[2], 64)
        self.assertEqual(second.x.shape[2], 64)
        self.assertEqual(timed.x.shape[2], 1024)
        self.assertFalse(np.array_equal(first.x, second.x))
        for mechanism_id, mechanism in enumerate(first.mechanisms):
            self.assertEqual(mechanism.target, 0)
            self.assertEqual(mechanism.participants, (0, 1, 2 + mechanism_id))
            self.assertEqual(mechanism.context_ids, (0, 1))
            self.assertEqual(mechanism.gamma_ids,
                             (mechanism_id % 4, (mechanism_id + 1) % 4))
            self.assertEqual(mechanism.alpha_id, mechanism_id % 4)
            self.assertEqual(mechanism.bias_id, mechanism_id % 3)

    def test_edge_fixtures_expose_zero_cancellation_saturation_and_repeated_roles(self):
        cases = fixtures.build_cases()
        zero, dy_zero = cases["zero_participants"]
        self.assertEqual(float(zero.x[0, 0, 0]), 0.0)
        zero_eval = evaluate(zero, dy_zero, output_dtype=np.float32)
        self.assertTrue(np.isfinite(zero_eval.vjp.dx).all())
        self.assertNotEqual(float(zero_eval.vjp.dx[0, 0, 0]), 0.0)

        cancellation, dy_cancel = cases["signed_cancellation"]
        cancellation_eval = evaluate(cancellation, dy_cancel, output_dtype=np.float32)
        self.assertTrue(np.all(cancellation_eval.y_float64 == 0.0))
        self.assertTrue(np.all(cancellation_eval.sum_abs_y_contributions > 0.0))

        saturated, _ = cases["saturated_gates"]
        gates = []
        for mechanism in saturated.mechanisms:
            preactivation = float(saturated.bias[mechanism.bias_id])
            for context_id, gamma_id in zip(mechanism.context_ids, mechanism.gamma_ids):
                preactivation += (float(saturated.gamma[gamma_id])
                                  * float(saturated.z[context_id, 0]))
            gates.append(1.0 / (1.0 + math.exp(-preactivation)))
        self.assertLess(min(gates), 2.0e-5)
        self.assertGreater(max(gates), 1.0 - 2.0e-5)

        repeated, _ = cases["repeated_tail_role"]
        self.assertEqual(repeated.mechanisms[0].participants, (0, 1, 0))
        self.assertEqual(repeated.mechanisms[1].participants, (0, 1, 3))

    def test_parameter_refresh_keeps_structure_and_changes_only_runtime_values(self):
        cases = fixtures.build_cases()
        before, before_dy = cases["runtime_refresh_before"]
        after, after_dy = cases["runtime_refresh_after"]
        self.assertEqual(fixtures.structure_sha256(before),
                         fixtures.structure_sha256(after))
        self.assertTrue(np.array_equal(before.x, after.x))
        self.assertTrue(np.array_equal(before.z, after.z))
        self.assertTrue(np.array_equal(before_dy, after_dy))
        self.assertFalse(np.array_equal(before.alpha, after.alpha))
        self.assertFalse(np.array_equal(before.bias, after.bias))
        self.assertFalse(np.array_equal(before.gamma, after.gamma))

    def test_tensor_validator_checks_six_outputs_and_seeded_repeated_vjp(self):
        problem, dy = fixtures.build_cases()["repeated_tail_role"]
        evaluation = evaluate(problem, dy=dy, output_dtype=np.float32)
        actual = {"y": evaluation.y_float64.astype(np.float32)}
        for name in driver.GRADIENTS:
            actual[name] = getattr(evaluation.vjp, name).astype(np.float32)
            actual[f"{name}_repeat"] = actual[name].copy()
        evidence = driver._validate_result_tensors(
            actual, fixtures._evaluation_arrays(evaluation),
            gradient_mode="overwrite", repeat_vjp=True)
        self.assertEqual(set(evidence), {
            "y", *driver.GRADIENTS,
            *(f"{name}_repeat" for name in driver.GRADIENTS),
        })

        accumulated = {key: value.copy() for key, value in actual.items()}
        for name in driver.GRADIENTS:
            accumulated[name] += np.float32(driver.GRADIENT_SEED)
            accumulated[f"{name}_repeat"] += np.float32(driver.GRADIENT_SEED)
        driver._validate_result_tensors(
            accumulated, fixtures._evaluation_arrays(evaluation),
            gradient_mode="accumulate", repeat_vjp=True)

        incomplete = dict(actual)
        incomplete.pop("dgamma")
        with self.assertRaisesRegex(driver.DriverError, "six required tensors"):
            driver._validate_result_tensors(
                incomplete, fixtures._evaluation_arrays(evaluation),
                gradient_mode="overwrite", repeat_vjp=True)
        bad = dict(actual)
        bad["dx"] = bad["dx"].copy()
        bad["dx"].flat[0] += np.float32(1.0)
        with self.assertRaisesRegex(driver.DriverError, "outside FP32 contract"):
            driver._validate_result_tensors(
                bad, fixtures._evaluation_arrays(evaluation),
                gradient_mode="overwrite", repeat_vjp=True)

    def test_runner_metadata_memory_and_setup_breakdowns_match_native_json(self):
        problem = fixtures.build_cases()["repeated_tail_role"][0]
        case = {"repeated_tail": True}
        context = {"device_uuid": "GPU-0123456789abcdef",
                   "scheduler_job_id": "job-test", "scheduler_attempt_id": "attempt-test"}
        memory = {
            "persistent_layout": 0,
            "resident_values_and_cotangent": 512 * 1024 * 1024,
            "output": 8192,
            "gradient_destinations": 1000,
            "tape": 200,
            "forward_scratch": 0,
            "backward_scratch": 300,
            "repeat_gradient_and_scratch": 1300,
        }
        setup = {
            "host_preparation": 0.1, "device_layout_upload": 0.2,
            "arm_allocation_and_seed": 0.3, "shared_value_allocation": 0.4,
            "shared_activation_and_cotangent_upload": 0.5,
            "shared_parameter_upload": 0.6,
        }
        metadata = {
            "schema": driver.RUNNER_SCHEMA,
            "requested_candidate": "circuit",
            "selected_backend": "shared_product_circuit",
            "gradient_mode": "accumulate",
            "gradient_seed": 0.125,
            "repeat_vjp": True,
            "device_uuid": context["device_uuid"],
            "scheduler_job_id": context["scheduler_job_id"],
            "scheduler_attempt_id": context["scheduler_attempt_id"],
            "cells": 64, "channels": 32, "mechanisms": 8, "terms": 16,
            "repeated_tail": True,
            "memory_bytes": memory,
            "setup_ms": setup,
        }
        evidence = driver._validate_metadata(
            metadata, candidate="circuit", gradient_mode="accumulate",
            problem=problem, case=case, context=context, repeat_vjp=True)
        self.assertEqual(evidence["scratch_bytes"], 800)
        self.assertGreater(evidence["total_allocation_bytes"], driver.MAX_SCRATCH_BYTES)
        over_budget = dict(metadata)
        over_budget["memory_bytes"] = dict(memory)
        over_budget["memory_bytes"]["backward_scratch"] = driver.MAX_SCRATCH_BYTES
        over_budget["memory_bytes"]["repeat_gradient_and_scratch"] = (
            memory["gradient_destinations"] + driver.MAX_SCRATCH_BYTES)
        with self.assertRaisesRegex(driver.DriverError, "shared scratch"):
            driver._validate_metadata(
                over_budget, candidate="circuit", gradient_mode="accumulate",
                problem=problem, case=case, context=context, repeat_vjp=True)

    def test_paired_metrics_parser_accepts_runner_schedule_and_rejects_bad_step_counts(self):
        context = {"device_uuid": "GPU-0123456789abcdef",
                   "scheduler_job_id": "job-test", "scheduler_attempt_id": "attempt-test"}
        metrics = {
            "schema": driver.PAIRED_SCHEMA,
            **context,
            "cells": 1024, "channels": 32, "mechanisms": 8,
            "terms": 16, "gradient_mode": "overwrite",
            "warmup_iterations_per_arm": 10,
            "block_count": 7, "iterations_per_block": 30,
            "measurements": [], "actual_order": [],
            "diagnostics": [], "arms": [],
        }
        for block in range(7):
            native_position = block % 3
            pair = ["unfactored", "circuit"] if block % 2 == 0 else ["circuit", "unfactored"]
            order = []
            pair_index = 0
            for position in range(3):
                candidate = "native_direct" if position == native_position else pair[pair_index]
                if position != native_position:
                    pair_index += 1
                order.append(candidate)
                metrics["measurements"].append({
                    "block": block, "position": position, "candidate": candidate,
                    "selected_backend": {
                        "unfactored": "literal_unfactored",
                        "circuit": "shared_product_circuit",
                        "native_direct": "stable",
                    }[candidate],
                    "host_block_ms": 3.0, "iterations": 30, "host_mean_ms": 0.1,
                })
            metrics["actual_order"].append(order)
        backends = {
            "unfactored": "literal_unfactored",
            "circuit": "shared_product_circuit",
            "native_direct": "stable",
        }
        for candidate, backend in backends.items():
            metrics["diagnostics"].append({
                "candidate": candidate, "selected_backend": backend,
                "forward_ms": [0.1] * 5, "backward_ms": [0.1] * 5,
                "full_ms": [0.2] * 5,
            })
            metrics["arms"].append({
                "candidate": candidate, "selected_backend": backend,
                "memory_bytes": {
                    "persistent_layout": 1,
                    "resident_values_and_cotangent": 2,
                    "output": 3, "gradient_destinations": 4,
                    "tape": 5, "forward_scratch": 6,
                    "backward_scratch": 7,
                    "repeat_gradient_and_scratch": 0,
                },
                "setup_ms": {
                    "host_preparation": 0.1, "device_layout_upload": 0.2,
                    "arm_allocation_and_seed": 0.3, "shared_value_allocation": 0.4,
                    "shared_activation_and_cotangent_upload": 0.5,
                    "shared_parameter_upload": 0.6,
                },
            })
        accepted = driver._validate_metrics(
            metrics, context=context,
            timed_case={"cells": 1024, "channels": 32, "mechanisms": 8})
        self.assertEqual(len(accepted["raw_measurements"]), 21)
        self.assertEqual(accepted["memory_summary"]["shared_values_counted_once_bytes"], 2)
        self.assertEqual(accepted["memory_summary"]["simultaneous_scratch_bytes"], 54)
        broken = {**metrics, "measurements": [dict(row) for row in metrics["measurements"]]}
        broken["measurements"][0]["iterations"] = 29
        with self.assertRaisesRegex(driver.DriverError, "30 full"):
            driver._validate_metrics(
                broken, context=context,
                timed_case={"cells": 1024, "channels": 32, "mechanisms": 8})

    def test_content_identity_allows_attempt_local_runner_copy_but_not_new_binary(self):
        original = {
            "manifest_sha256": "manifest", "build_schema": driver.BUILD_SCHEMA,
            "source_closure_sha256": "source", "source_file_count": 19,
            "runner_binary_sha256": "binary", "architecture": "sm_70",
            "fast_math": False, "runner_path": "/attempt/one/runner",
        }
        copied = dict(original, runner_path="/attempt/two/runner")
        self.assertTrue(driver._same_build_identity(original, copied))
        changed = dict(copied, runner_binary_sha256="different-binary")
        self.assertFalse(driver._same_build_identity(original, changed))

    def test_measurement_gate_uses_content_identity_for_attempt_local_copy(self):
        # This deliberately incomplete parser fixture is rejected later for
        # lacking runs; it is never accepted or used as GPU evidence.
        original = {
            "manifest_sha256": "manifest", "build_schema": driver.BUILD_SCHEMA,
            "source_closure_sha256": "source", "source_file_count": 19,
            "runner_binary_sha256": "binary", "architecture": "sm_70",
            "fast_math": False, "runner_path": "/attempt/one/runner",
        }
        copied = dict(original, runner_path="/attempt/two/runner")
        summary = {
            "schema": driver.DRIVER_SCHEMA, "kind": "gpu_validation",
            "gpu_executed": True, "validated": True, "timed": False,
            "fixture_manifest_sha256": "fixture", "build_identity": original,
            "python_source_identity_before": driver._python_source_identity(),
            "python_source_identity_after": driver._python_source_identity(),
            "runs": [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "incomplete-parser-only.json"
            path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(driver.DriverError, "lacks its per-run evidence"):
                driver._verify_validation_summary(
                    path, fixture_sha="fixture", build_identity=copied,
                    python_sources=driver._python_source_identity())


if __name__ == "__main__":
    unittest.main()
