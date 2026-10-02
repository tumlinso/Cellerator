"""Run directly: CUDA_VISIBLE_DEVICES='' python test_polynomial.py."""

import unittest

import torch

from polynomial import polynomial, polynomial_jvp, polynomial_vjp


class PolynomialTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(20261002)
        self.values = tuple(torch.randn(3, 3, generator=generator, dtype=torch.float64)
                            for _ in range(4))
        self.directions = tuple(torch.randn(3, 3, generator=generator, dtype=torch.float64)
                                for _ in range(4))
        self.G = torch.randn(3, 3, generator=generator, dtype=torch.float64)

    def test_forward_and_input_preservation(self):
        X, L, R, M = self.values
        snapshots = tuple(value.clone() for value in self.values)
        expected = torch.empty_like(X)
        # Independent indexed expression catches matrix-order mistakes.
        for i in range(3):
            for j in range(3):
                expected[i, j] = sum(L[i, k] * X[k, j] + X[i, k] * R[k, j]
                                     for k in range(3)) + sum(
                    X[i, k] * M[k, q] * X[q, j] for k in range(3) for q in range(3))
        torch.testing.assert_close(polynomial(*self.values), expected)
        polynomial_jvp(*self.values, *self.directions)
        polynomial_vjp(*self.values, self.G)
        for original, snapshot in zip(self.values, snapshots):
            self.assertTrue(torch.equal(original, snapshot))

    def test_jvp_finite_difference_and_autograd(self):
        for active in (None, 0, 1, 2, 3):
            tangents = tuple(direction if active is None or active == index
                             else torch.zeros_like(direction)
                             for index, direction in enumerate(self.directions))
            with self.subTest(active=active):
                explicit = polynomial_jvp(*self.values, *tangents)
                epsilon = 1e-6
                plus = polynomial(*(value + epsilon * direction
                                    for value, direction in zip(self.values, tangents)))
                minus = polynomial(*(value - epsilon * direction
                                     for value, direction in zip(self.values, tangents)))
                torch.testing.assert_close(explicit, (plus - minus) / (2 * epsilon),
                                           rtol=1e-8, atol=1e-8)
                _, automatic = torch.autograd.functional.jvp(polynomial, self.values, tangents)
                torch.testing.assert_close(explicit, automatic, rtol=1e-12, atol=1e-12)

    def test_vjp_autograd_and_adjoint(self):
        values = tuple(value.clone().requires_grad_() for value in self.values)
        automatic = torch.autograd.grad(polynomial(*values), values, self.G)
        explicit = polynomial_vjp(*values, self.G)
        for index, (actual, expected) in enumerate(zip(explicit, automatic)):
            with self.subTest(gradient=index):
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        lhs = (polynomial_jvp(*values, *self.directions) * self.G).sum()
        rhs = sum((gradient * direction).sum()
                  for gradient, direction in zip(explicit, self.directions))
        torch.testing.assert_close(lhs, rhs, rtol=1e-12, atol=1e-12)

    def test_all_apis_remain_autograd_differentiable(self):
        values = tuple(value.clone().requires_grad_() for value in self.values)
        self.assertTrue(torch.autograd.gradcheck(polynomial, values))
        self.assertTrue(torch.autograd.gradgradcheck(polynomial, values))
        self.assertTrue(torch.autograd.gradcheck(polynomial_jvp, values + self.directions))
        self.assertTrue(torch.autograd.gradcheck(polynomial_vjp, values + (self.G,)))

    def test_float32_and_noncontiguous(self):
        values = tuple(value.float().T for value in self.values)
        directions = tuple(value.float().T for value in self.directions)
        G = self.G.float().T
        self.assertFalse(values[0].is_contiguous())
        lhs = (polynomial_jvp(*values, *directions) * G).sum()
        rhs = sum((gradient * direction).sum()
                  for gradient, direction in zip(polynomial_vjp(*values, G), directions))
        torch.testing.assert_close(lhs, rhs, rtol=2e-5, atol=2e-5)
        self.assertEqual(polynomial(*values).dtype, torch.float32)

    def test_zero_state_preserves_state_response(self):
        X, L, R, M = self.values
        X = torch.zeros_like(X)
        grad_X, grad_L, grad_R, grad_M = polynomial_vjp(X, L, R, M, self.G)
        torch.testing.assert_close(grad_X, L.T @ self.G + self.G @ R.T)
        for value in (polynomial(X, L, R, M), grad_L, grad_R, grad_M):
            self.assertEqual(torch.count_nonzero(value).item(), 0)

    def test_admission_for_every_operand(self):
        calls = ((polynomial, self.values),
                 (polynomial_jvp, self.values + self.directions),
                 (polynomial_vjp, self.values + (self.G,)))
        for function, operands in calls:
            for index in range(len(operands)):
                invalids = (None, torch.ones(2, 3, dtype=torch.float64),
                            torch.ones(3, 3, dtype=torch.float32),
                            torch.ones(3, 3, dtype=torch.int64),
                            torch.ones(3, 3, dtype=torch.float64, device="meta"),
                            torch.full((3, 3), float("nan"), dtype=torch.float64),
                            torch.full((3, 3), float("inf"), dtype=torch.float64))
                for invalid in invalids:
                    args = list(operands)
                    args[index] = invalid
                    with self.subTest(api=function.__name__, operand=index, invalid=str(invalid)):
                        with self.assertRaises((TypeError, ValueError)):
                            function(*args)
        for shape in ((0, 0), (3,), (1, 3, 3)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                polynomial(*(torch.ones(shape, dtype=torch.float64) for _ in range(4)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
