import dataclasses
import ctypes
import unittest
import numpy as np
from product.ops import forward, vjp, jvp, _library


class ProductChecks(unittest.TestCase):
    def setUp(self):
        self.x = np.array([0., 1.25, -.75, .4], np.float32)
        self.k = np.array([.5, -.25, 1.5, 0., .75, -.3], np.float32)
        self.a = np.array([0, 1, 1, 2, 3, 2], np.int64)
        self.b = np.array([1, 1, 2, 0, 3, 1], np.int64)

    def test_forward_and_zero_repeated(self):
        y, tape = forward(self.x, self.k, self.a, self.b)
        np.testing.assert_array_equal(y, (self.k * self.x[self.a]) * self.x[self.b])
        dx, dk = vjp(tape, np.ones_like(self.k))
        expected = np.zeros_like(self.x)
        np.add.at(expected, self.a, self.k * self.x[self.b])
        np.add.at(expected, self.b, self.k * self.x[self.a])
        np.testing.assert_allclose(dx, expected, atol=1e-7)
        np.testing.assert_allclose(dk, self.x[self.a] * self.x[self.b], atol=1e-7)
        # A zero primal retains its nonzero slope without any division.
        self.assertAlmostEqual(float(dx[0]), .625)

    def test_finite_difference_and_adjoint(self):
        _, tape = forward(self.x, self.k, self.a, self.b)
        rng = np.random.default_rng(419)
        vx = rng.normal(size=self.x.size).astype(np.float32)
        vk = rng.normal(size=self.k.size).astype(np.float32)
        g = rng.normal(size=self.k.size).astype(np.float32)
        tangent = jvp(tape, vx, vk)
        h = np.float32(.001)
        plus, _ = forward(self.x+h*vx, self.k+h*vk, self.a, self.b)
        minus, _ = forward(self.x-h*vx, self.k-h*vk, self.a, self.b)
        np.testing.assert_allclose(tangent, (plus-minus)/(2*h), rtol=4e-4, atol=1e-4)
        gx, gk = vjp(tape, g)
        self.assertAlmostEqual(float(g@tangent), float(gx@vx+gk@vk), places=5)
        for base, gradient in [(self.x, gx), (self.k, gk)]:
            for i in range(base.size):
                delta = np.zeros_like(base)
                delta[i] = h
                xp, xm = (self.x+delta, self.x-delta) if base is self.x else (self.x, self.x)
                kp, km = (self.k+delta, self.k-delta) if base is self.k else (self.k, self.k)
                yp, _ = forward(xp, kp, self.a, self.b)
                ym, _ = forward(xm, km, self.a, self.b)
                np.testing.assert_allclose(gradient[i], g@(yp-ym)/(2*h), rtol=4e-4, atol=1e-4)

    def test_saved_primal_generations_and_nonmutation(self):
        _, tape = forward(self.x, self.k, self.a, self.b, (2, 3, 4, 5))
        saved = tuple(v.copy() for v in (tape.x, tape.k, tape.a, tape.b))
        self.x[:] = 99
        self.k[:] = 88
        self.a[:] = 0
        vjp(tape, np.ones_like(self.k), (2, 3, 4, 5))
        jvp(tape, np.ones_like(self.x), np.ones_like(self.k), (2, 3, 4, 5))
        for actual, original in zip((tape.x, tape.k, tape.a, tape.b), saved):
            np.testing.assert_array_equal(actual, original)
            with self.assertRaises(ValueError):
                actual.flags.writeable = True
        with self.assertRaises(dataclasses.FrozenInstanceError):
            tape.generations = (0, 0, 0, 0)
        with self.assertRaises(ValueError):
            vjp(tape, np.ones_like(self.k), (2, 4, 4, 5))
        with self.assertRaises(ValueError):
            jvp(tape, np.ones_like(self.x), np.ones_like(self.k), (2, 3, 4, 6))

    def test_empty_and_admission(self):
        empty = np.array([], np.float32)
        indices = np.array([], np.int64)
        y, tape = forward(empty, empty, indices, indices)
        self.assertEqual(y.size, 0)
        self.assertEqual(vjp(tape, empty)[0].size, 0)
        self.assertEqual(jvp(tape, empty, empty).size, 0)
        _, tape = forward(self.x, empty, indices, indices)
        np.testing.assert_array_equal(vjp(tape, empty)[0], np.zeros_like(self.x))
        for a in [np.array([-1]*6, np.int64), np.array([4]*6, np.int64)]:
            with self.assertRaises(ValueError):
                forward(self.x, self.k, a, self.b)
        for x, k, a, b in [(self.x.astype(np.float64), self.k, self.a, self.b),
                            (self.x, self.k, self.a.astype(np.int32), self.b),
                            (self.x, self.k, self.a[:-1], self.b)]:
            with self.assertRaises(ValueError):
                forward(x, k, a, b)
        _, tape = forward(self.x, self.k, self.a, self.b)
        with self.assertRaises(ValueError):
            vjp(tape, empty)
        with self.assertRaises(ValueError):
            jvp(tape, empty, self.k)

    def test_native_admission_before_output_write(self):
        fp = ctypes.POINTER(ctypes.c_float)
        ip = ctypes.POINTER(ctypes.c_int64)
        bad = np.array([-1] * self.k.size, np.int64)
        output = np.full_like(self.k, 912.)
        common = [self.x.size, self.k.size, self.x.ctypes.data_as(fp), self.k.ctypes.data_as(fp),
                  bad.ctypes.data_as(ip), self.b.ctypes.data_as(ip)]
        self.assertEqual(_library().product_forward(*common, output.ctypes.data_as(fp)), 2)
        np.testing.assert_array_equal(output, np.full_like(output, 912.))
        self.assertEqual(_library().product_forward(-1, 0, None, None, None, None, None), 1)
        self.assertEqual(_library().product_forward(1, 0, None, None, None, None, None), 1)


if __name__ == "__main__":
    unittest.main()
