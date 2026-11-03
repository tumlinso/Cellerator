"""CPU-only conformance tests for the independent semantic oracle."""

from dataclasses import replace
import unittest

import numpy as np

from reference import (Mechanism, Problem, assert_fp16_contract,
                       assert_fp32_contract, evaluate)


def smooth_problem() -> Problem:
    # Covers arity 2/3/4, ordered repeats, tied parameter identities, duplicate
    # output targets, multiple context terms, and mixed positive/negative data.
    return Problem(
        x=np.array([
            [[0.25, -0.40], [0.75, 0.20], [-0.30, 0.50]],
            [[-0.50, 0.60], [0.10, -0.80], [0.90, 0.25]],
            [[0.70, 0.30], [-0.20, 0.40], [0.45, -0.65]],
            [[-0.10, 0.80], [0.35, -0.55], [0.60, 0.15]],
        ], dtype=np.float64),  # [occurrence, channel, cell]
        z=np.array([[0.30, -0.25], [-0.60, 0.40]], dtype=np.float64),
        alpha=np.array([[0.50, -0.75, 0.25], [-0.40, 0.65, 0.80]],
                       dtype=np.float64),
        bias=np.array([0.20, -0.35], dtype=np.float64),
        gamma=np.array([0.45, -0.30, 0.20], dtype=np.float64),
        mechanisms=(
            Mechanism(0, (0, 1), (0, 1), (0, 1), 0, 0),
            Mechanism(0, (1, 1, 2), (1,), (2,), 0, 0),
            Mechanism(1, (0, 2, 3, 0), (0, 0), (1, 1), 1, 1),
            Mechanism(0, (3, 2), (), (), 1, 1),
        ),
        target_count=2,
    )


def objective(problem: Problem, dy: np.ndarray) -> float:
    evaluated = evaluate(problem, output_dtype=np.float64)
    return float(np.sum(evaluated.y_float64 * dy, dtype=np.float64))


class FactorizedInteractionReferenceTests(unittest.TestCase):
    def test_channel_gate_adjoint_reduces_all_channels_before_sigmoid(self):
        problem = Problem(
            x=np.array([[[1.0], [2.0], [1.0]],
                        [[1.0], [1.0], [-1.0]]], dtype=np.float64),
            z=np.array([[0.5]], dtype=np.float64),
            alpha=np.array([[1.0, 2.0, -2.0]], dtype=np.float64),
            bias=np.array([-1.0], dtype=np.float64),
            gamma=np.array([2.0], dtype=np.float64),
            mechanisms=(Mechanism(0, (0, 1), (0,), (0,), 0, 0),),
            target_count=1,
        )
        dy = np.array([[[3.0], [5.0], [7.0]]], dtype=np.float64)
        got = evaluate(problem, dy=dy, output_dtype=np.float32)
        self.assertEqual(got.y_float64[:, :, 0].tolist(), [[0.5, 2.0, 1.0]])
        self.assertAlmostEqual(got.vjp.dbias[0], 9.25, places=14)
        self.assertAlmostEqual(got.vjp.dgamma[0], 4.625, places=14)
        self.assertAlmostEqual(got.vjp.dz[0, 0], 18.5, places=14)
        np.testing.assert_allclose(got.vjp.dalpha[0], [1.5, 5.0, -3.5], rtol=0, atol=1e-14)
        np.testing.assert_allclose(got.vjp.dx[:, :, 0],
                                   [[1.5, 5.0, 7.0], [1.5, 10.0, -7.0]],
                                   rtol=0, atol=1e-14)

    def test_zero_safe_product_derivative_and_repeated_roles(self):
        one_zero = Problem(
            x=np.array([[[0.0]], [[2.0]], [[3.0]]], dtype=np.float64),
            z=np.zeros((0, 1), dtype=np.float64),
            alpha=np.array([[4.0]], dtype=np.float64),
            bias=np.array([0.0], dtype=np.float64),
            gamma=np.zeros((0,), dtype=np.float64),
            mechanisms=(Mechanism(0, (0, 1, 2), (), (), 0, 0),),
            target_count=1,
        )
        dy = np.ones((1, 1, 1), dtype=np.float64)
        result = evaluate(one_zero, dy=dy, output_dtype=np.float64)
        self.assertEqual(result.y_float64.item(), 0.0)
        np.testing.assert_array_equal(result.vjp.dx[:, 0, 0], [12.0, 0.0, 0.0])
        self.assertTrue(np.isfinite(result.vjp.dx).all())

        repeated_zero = replace(
            one_zero,
            mechanisms=(Mechanism(0, (0, 0, 1), (), (), 0, 0),),
        )
        repeated = evaluate(repeated_zero, dy=dy, output_dtype=np.float64)
        self.assertTrue(np.isfinite(repeated.vjp.dx).all())
        np.testing.assert_array_equal(repeated.vjp.dx[:, 0, 0], [0.0, 0.0, 0.0])

        repeated_nonzero = replace(
            one_zero,
            x=np.array([[[2.0]], [[3.0]], [[0.5]]], dtype=np.float64),
            alpha=np.array([[2.0]], dtype=np.float64),
            mechanisms=(Mechanism(0, (0, 0, 2), (), (), 0, 0),),
        )
        repeated_nonzero_result = evaluate(repeated_nonzero, dy=dy,
                                           output_dtype=np.float64)
        # d/dx0 of x0*x0*x2 is two ordered-role contributions.
        self.assertAlmostEqual(repeated_nonzero_result.vjp.dx[0, 0, 0], 2.0,
                               places=14)

    def test_empty_mechanisms_and_empty_context_terms(self):
        empty = Problem(
            x=np.ones((2, 3, 1), dtype=np.float32),
            z=np.zeros((0, 1), dtype=np.float32),
            alpha=np.zeros((0, 3), dtype=np.float32),
            bias=np.zeros((0,), dtype=np.float32),
            gamma=np.zeros((0,), dtype=np.float32),
            mechanisms=(), target_count=2,
        )
        empty_result = evaluate(empty, dy=np.ones((2, 3, 1), dtype=np.float32),
                                output_dtype=np.float32)
        np.testing.assert_array_equal(empty_result.y, np.zeros((2, 3, 1), dtype=np.float32))
        for array in vars(empty_result.vjp).values():
            self.assertEqual(float(np.sum(np.abs(array))), 0.0)

        no_context = Problem(
            x=np.array([[[0.5]], [[-0.25]]], dtype=np.float32),
            z=np.zeros((0, 1), dtype=np.float32),
            alpha=np.array([[0.75]], dtype=np.float32),
            bias=np.array([0.4], dtype=np.float32),
            gamma=np.zeros((0,), dtype=np.float32),
            mechanisms=(Mechanism(0, (0, 1), (), (), 0, 0),),
            target_count=1,
        )
        no_context_result = evaluate(no_context, dy=np.ones((1, 1, 1), dtype=np.float32),
                                     output_dtype=np.float64)
        stored_bias = float(np.float32(0.4))
        gate = 1.0 / (1.0 + np.exp(-stored_bias))
        self.assertAlmostEqual(no_context_result.y_float64.item(),
                               gate * 0.75 * 0.5 * -0.25, places=14)
        self.assertEqual(no_context_result.vjp.dz.shape, (0, 1))

    def test_tied_parameter_ids_accumulate_across_mechanisms(self):
        problem = Problem(
            x=np.array([[[0.5]], [[-0.25]]], dtype=np.float64),
            z=np.array([[0.2]], dtype=np.float64),
            alpha=np.array([[0.75]], dtype=np.float64),
            bias=np.array([0.1], dtype=np.float64),
            gamma=np.array([0.4], dtype=np.float64),
            mechanisms=(
                Mechanism(0, (0, 1), (0,), (0,), 0, 0),
                Mechanism(0, (0, 1), (0,), (0,), 0, 0),
            ),
            target_count=1,
        )
        result = evaluate(problem, dy=np.ones((1, 1, 1), dtype=np.float64),
                          output_dtype=np.float64)
        single = replace(problem, mechanisms=(problem.mechanisms[0],))
        one = evaluate(single, dy=np.ones((1, 1, 1), dtype=np.float64),
                       output_dtype=np.float64)
        self.assertAlmostEqual(result.y_float64.item(), 2.0 * one.y_float64.item(), places=14)
        self.assertAlmostEqual(result.vjp.dbias[0], 2.0 * one.vjp.dbias[0], places=14)
        self.assertAlmostEqual(result.vjp.dgamma[0], 2.0 * one.vjp.dgamma[0], places=14)
        self.assertAlmostEqual(result.vjp.dalpha[0, 0], 2.0 * one.vjp.dalpha[0, 0], places=14)

    def test_all_vjps_match_centered_differences(self):
        problem = smooth_problem()
        dy = np.array([
            [[0.7, -0.2], [0.4, 0.9], [-0.5, 0.3]],
            [[-0.6, 0.8], [0.2, -0.7], [0.5, 0.1]],
        ], dtype=np.float64)
        got = evaluate(problem, dy=dy, output_dtype=np.float64)
        step = 1.0e-6
        arrays = ("x", "z", "alpha", "bias", "gamma")
        grads = (got.vjp.dx, got.vjp.dz, got.vjp.dalpha,
                 got.vjp.dbias, got.vjp.dgamma)
        for name, gradient in zip(arrays, grads):
            base = getattr(problem, name)
            for index in np.ndindex(base.shape):
                plus, minus = base.copy(), base.copy()
                plus[index] += step
                minus[index] -= step
                numeric = (objective(replace(problem, **{name: plus}), dy)
                           - objective(replace(problem, **{name: minus}), dy)) / (2 * step)
                self.assertAlmostEqual(float(gradient[index]), numeric,
                                       delta=3.0e-8 + 3.0e-7 * abs(numeric),
                                       msg=f"{name}{index}")

    def test_stable_gate_reference_covers_saturated_range(self):
        contexts = tuple(range(32))
        problem = Problem(
            x=np.ones((2, 1, 1), dtype=np.float32),
            z=np.ones((32, 1), dtype=np.float32),
            alpha=np.ones((1, 1), dtype=np.float32),
            bias=np.array([4.0], dtype=np.float32),
            gamma=np.ones((32,), dtype=np.float32),
            mechanisms=(Mechanism(0, (0, 1), contexts, contexts, 0, 0),),
            target_count=1,
        )
        result = evaluate(problem, dy=np.ones((1, 1, 1), dtype=np.float32),
                          output_dtype=np.float32)
        self.assertGreater(result.y_float64.item(), 0.999999999999)
        self.assertGreater(result.vjp.dbias[0], 0.0)
        self.assertTrue(np.isfinite(result.vjp.dbias).all())

    def test_storage_quantization_is_applied_before_reference_arithmetic(self):
        problem = smooth_problem()
        fp16 = replace(
            problem,
            x=problem.x.astype(np.float16),
            z=problem.z.astype(np.float16),
            alpha=problem.alpha.astype(np.float16),
            bias=problem.bias.astype(np.float16),
            gamma=problem.gamma.astype(np.float16),
        )
        result = evaluate(fp16, dy=np.ones((2, 3, 2), dtype=np.float32),
                          output_dtype=np.float16)
        quantized_again = evaluate(
            replace(fp16, x=fp16.x.astype(np.float64), z=fp16.z.astype(np.float64),
                    alpha=fp16.alpha.astype(np.float64), bias=fp16.bias.astype(np.float64),
                    gamma=fp16.gamma.astype(np.float64)),
            dy=np.ones((2, 3, 2), dtype=np.float32), output_dtype=np.float64)
        np.testing.assert_array_equal(result.y_float64, quantized_again.y_float64)
        self.assertEqual(result.y.dtype, np.float16)

    def test_frozen_fp32_and_fp16_acceptance_bounds(self):
        ref = np.array([0.0, 1.0, -2.0], dtype=np.float64)
        scales = np.array([0.0, 10.0, 3.0], dtype=np.float64)
        allowance = 2.0e-6 + 2.0e-5 * scales
        assert_fp32_contract(ref + allowance * (1.0 - 1.0e-10), ref, scales,
                             "bound interior")
        with self.assertRaisesRegex(AssertionError, "outside FP32 contract"):
            assert_fp32_contract(ref + allowance * 1.01, ref, scales, "bound overrun")

        ref16 = np.array([1.0], dtype=np.float16)
        one_step = np.nextafter(ref16, np.array([np.inf], dtype=np.float16))
        assert_fp16_contract(one_step, ref16, np.zeros((1,), dtype=np.float64),
                             "one fp16 ulp")
        two_steps = np.nextafter(one_step, np.array([np.inf], dtype=np.float16))
        with self.assertRaisesRegex(AssertionError, "outside FP16 output contract"):
            assert_fp16_contract(two_steps, ref16, np.zeros((1,), dtype=np.float64),
                                 "two fp16 ulps")

    def test_invalid_numerical_and_structural_inputs_fail_clearly(self):
        problem = smooth_problem()
        bad_x = problem.x.copy()
        bad_x[0, 0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            evaluate(replace(problem, x=bad_x))
        bad_arity = replace(
            problem, mechanisms=(Mechanism(0, (0,), (), (), 0, 0),)
        )
        with self.assertRaisesRegex(ValueError, "unsupported arity"):
            evaluate(bad_arity)


if __name__ == "__main__":
    unittest.main()
