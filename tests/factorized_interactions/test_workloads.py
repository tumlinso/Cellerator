"""Checks that the fixed benchmark corpus reconstructs declared semantics."""

import unittest

import numpy as np

from workloads import (_values, MASK64, SEED, _uniform,
                       correctness_workload_ids, make_workload, manifest,
                       performance_workload_ids, workload_ids, workload_spec)
from reference import evaluate


class FixedWorkloadCorpusTests(unittest.TestCase):
    def test_unique_names_and_supported_arities(self):
        ids = workload_ids()
        self.assertEqual(len(ids), len(set(ids)))
        arities = {case["arity"] for case in manifest()["cases"]}
        self.assertEqual(arities, {2, 3, 4})

    def test_each_case_has_exact_participant_cardinality_and_fanin(self):
        for workload_id in performance_workload_ids():
            with self.subTest(workload_id=workload_id):
                spec = workload_spec(workload_id)
                problem, dy = make_workload(workload_id)
                roles = [participant for mechanism in problem.mechanisms
                         for participant in mechanism.participants]
                self.assertEqual(len(roles), spec["mechanisms"] * spec["arity"])
                self.assertEqual(len(set(roles)), spec["unique_participants"])
                self.assertEqual(len(problem.mechanisms), spec["mechanisms"])
                self.assertEqual(problem.target_count * spec["output_fanin"],
                                 spec["mechanisms"])
                self.assertEqual(dy.shape,
                                 (problem.target_count, spec["channels"], spec["cells"]))
                targets = [mechanism.target for mechanism in problem.mechanisms]
                unique_targets, counts = np.unique(targets, return_counts=True)
                self.assertEqual(len(unique_targets), problem.target_count)
                np.testing.assert_array_equal(counts,
                                              np.full(problem.target_count,
                                                      spec["output_fanin"]))

    def test_generator_is_stable_and_values_respect_contract(self):
        for workload_id in performance_workload_ids():
            with self.subTest(workload_id=workload_id):
                first, first_dy = make_workload(workload_id, np.float16)
                second, second_dy = make_workload(workload_id, np.float16)
                for name in ("x", "z", "alpha", "bias", "gamma"):
                    np.testing.assert_array_equal(getattr(first, name),
                                                  getattr(second, name))
                np.testing.assert_array_equal(first_dy, second_dy)
                for name in ("x", "alpha", "bias", "gamma"):
                    self.assertLessEqual(float(np.max(np.abs(getattr(first, name)),
                                                            initial=0.0)), 1.0)
                self.assertLessEqual(float(np.max(np.abs(first.z), initial=0.0)), 1.0)
                self.assertLessEqual(float(np.max(np.abs(first_dy), initial=0.0)), 1.0)

    def test_vector_generator_matches_scalar_splitmix_definition(self):
        seed = (SEED + 0x10000000) & MASK64
        expected = np.array([-0.95 + 1.90 * _uniform(seed, i) for i in range(12)])
        actual = _values((12,), 0x10000000, -0.95, 0.95, np.float64)
        np.testing.assert_array_equal(actual, expected)

    def test_fp32_alternative_preserves_same_structure(self):
        for workload_id in performance_workload_ids():
            half, dy_half = make_workload(workload_id, np.float16)
            single, dy_single = make_workload(workload_id, np.float32)
            self.assertEqual(half.mechanisms, single.mechanisms)
            self.assertEqual(half.target_count, single.target_count)
            self.assertEqual(half.x.shape, single.x.shape)
            np.testing.assert_array_equal(dy_half, dy_single)
            for name in ("x", "z", "alpha", "bias", "gamma"):
                self.assertEqual(getattr(half, name).dtype, np.float16)
                self.assertEqual(getattr(single, name).dtype, np.float32)

        mixed, _ = make_workload("packet_a4_m32_u16", np.float16, np.float32)
        self.assertEqual(mixed.x.dtype, np.float16)
        self.assertEqual(mixed.z.dtype, np.float16)
        for name in ("alpha", "bias", "gamma"):
            self.assertEqual(getattr(mixed, name).dtype, np.float32)

    def test_literal_correctness_fixtures_cover_declared_adversaries(self):
        self.assertEqual(len(correctness_workload_ids()), 12)
        edge, dy = make_workload("correctness_ordered_tied_zero", np.float16, np.float32)
        self.assertEqual([len(m.participants) for m in edge.mechanisms], [2, 3, 4, 4, 3])
        self.assertEqual(edge.mechanisms[1].participants, (1, 1, 2))
        self.assertEqual(edge.mechanisms[1].context_ids, (1, 1))
        self.assertEqual(edge.mechanisms[0].alpha_id, edge.mechanisms[1].alpha_id)
        self.assertEqual(edge.mechanisms[0].bias_id, edge.mechanisms[1].bias_id)
        self.assertEqual(edge.mechanisms[0].gamma_ids[1], edge.mechanisms[1].gamma_ids[0])
        self.assertEqual(edge.mechanisms[0].target, edge.mechanisms[1].target)
        self.assertEqual(edge.x.dtype, np.float16)
        self.assertEqual(edge.alpha.dtype, np.float32)
        self.assertEqual(dy.dtype, np.float32)
        self.assertTrue(np.any(edge.x == 0.0))
        self.assertTrue(np.any(edge.x < 0.0))

        saturated, _ = make_workload("correctness_saturated_gates", np.float32)
        self.assertEqual(saturated.z.shape, (32, 1))
        self.assertEqual(saturated.gamma.size, 64)
        empty_mechs, _ = make_workload("correctness_empty_mechanisms")
        self.assertEqual(len(empty_mechs.mechanisms), 0)
        self.assertEqual(empty_mechs.x.shape[0], 0)
        self.assertEqual(empty_mechs.z.shape[0], 0)
        empty_cells, _ = make_workload("correctness_empty_cells")
        self.assertEqual(empty_cells.x.shape[2], 0)
        empty_channels, _ = make_workload("correctness_empty_channels")
        self.assertEqual(empty_channels.x.shape[1], 0)

        role_order, _ = make_workload("correctness_ordered_role_context_variants")
        self.assertEqual(role_order.mechanisms[0].participants, (0, 0, 1, 2))
        self.assertEqual(role_order.mechanisms[1].participants, (0, 1, 0, 2))
        self.assertEqual(role_order.mechanisms[0].context_ids, (0, 1))
        self.assertEqual(role_order.mechanisms[1].context_ids, (1, 0))

    def test_fma_product_residual_fixture_preserves_exact_fp32_adversary(self):
        problem, dy = make_workload(
            "correctness_fma_product_residual_cancellation", np.float32, np.float32)
        self.assertEqual(len(problem.mechanisms[0].context_ids), 8192)
        self.assertEqual(problem.mechanisms[0].context_ids[:4], (0, 1, 0, 1))
        a, c = problem.gamma
        b, d = problem.z[:, 0]
        rounded_pair = (np.float32(a * b) + np.float32(c * d))
        self.assertEqual(rounded_pair, np.float32(0.0))

        expected_logit = sum(
            float(problem.gamma[gamma_id]) * float(problem.z[context_id, 0])
            for context_id, gamma_id in zip(
                problem.mechanisms[0].context_ids,
                problem.mechanisms[0].gamma_ids))
        result = evaluate(problem, dy, output_dtype=np.float32)
        self.assertAlmostEqual(expected_logit, 2.555961254984e-4, places=15)
        self.assertAlmostEqual(result.y_float64.item(), 0.5000638990310, places=12)
        self.assertAlmostEqual(result.sum_abs_y_contributions.item(),
                               result.y_float64.item(), places=14)
        with self.assertRaisesRegex(AssertionError, "outside FP32 contract"):
            from reference import assert_fp32_contract
            assert_fp32_contract(np.array([0.5]), result.y_float64.reshape(1),
                                 result.sum_abs_y_contributions.reshape(1),
                                 "rounded product counterexample")

        with self.assertRaisesRegex(ValueError, "requires FP32 storage"):
            make_workload("correctness_fma_product_residual_cancellation",
                          np.float16, np.float16)

    def test_long_context_cancellation_and_ordered_role_reduction_reference(self):
        middle, dy = make_workload("correctness_context_cancellation_middle",
                                   np.float32, np.float32)
        tail, _ = make_workload("correctness_context_cancellation_tail",
                                np.float32, np.float32)
        self.assertEqual(len(middle.mechanisms[0].context_ids), 8193)
        expected_middle = evaluate(middle, dy, output_dtype=np.float64)
        expected_tail = evaluate(tail, dy, output_dtype=np.float64)
        self.assertAlmostEqual(expected_middle.y_float64.item(), 0.50005, places=10)
        self.assertAlmostEqual(expected_tail.y_float64.item(), 0.50005, places=10)
        self.assertAlmostEqual(expected_middle.y_float64.item(),
                               expected_tail.y_float64.item(), places=10)

    def test_100k_tied_fp16_high_fanin_reference_value(self):
        problem, dy = make_workload("correctness_high_fanin_100k",
                                    np.float16, np.float16)
        self.assertEqual(len(problem.mechanisms), 100_000)
        self.assertEqual(problem.alpha[0, 0], np.float16(0.0002))
        result = evaluate(problem, dy, output_dtype=np.float32)
        self.assertAlmostEqual(result.vjp.dx[0, 0, 0],
                               10.001659393310547, places=10)
        self.assertAlmostEqual(result.vjp.dx[1, 0, 0],
                               10.001659393310547, places=10)
        self.assertLessEqual(abs(result.vjp.dx[0, 0, 0]
                                 - result.vjp.dx[1, 0, 0]), 1.0e-12)

    def test_100k_packet_fp32_parameter_high_fanin_reference_value(self):
        problem, dy = make_workload("correctness_packet_high_fanin_fp32_100k",
                                    np.float16, np.float32)
        self.assertEqual(len(problem.mechanisms), 100_000)
        self.assertEqual(problem.alpha.dtype, np.float32)
        result = evaluate(problem, dy, output_dtype=np.float32)
        expected = 0.5 * float(problem.alpha[0, 0]) * 100_000
        self.assertAlmostEqual(result.y_float64.item(), expected, places=8)
        self.assertAlmostEqual(result.vjp.dx[0, 0, 0], expected, places=8)
        self.assertAlmostEqual(result.vjp.dx[1, 0, 0], expected, places=8)

    def test_packet_million_mechanism_fixture_is_exactly_sized(self):
        problem, dy = make_workload("correctness_packet_high_fanin_1m",
                                    np.float16, np.float16)
        self.assertEqual(len(problem.mechanisms), 1_000_000)
        self.assertEqual(problem.alpha[0, 0], np.float16(0.0002))
        self.assertEqual(dy.item(), 1.0)


if __name__ == "__main__":
    unittest.main()
