"""Run directly with a CPU Torch Python interpreter."""

import json
import unittest
from unittest.mock import patch

import torch

from flow import SylvesterFlow


class FlowTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(311)
        self.X = torch.randn(2, 3, generator=generator, dtype=torch.float64)
        self.L = (0.1 * torch.randn(2, 2, generator=generator,
                                  dtype=torch.float64)).requires_grad_()
        self.R = (0.1 * torch.randn(3, 3, generator=generator,
                                  dtype=torch.float64)).requires_grad_()
        self.flow = SylvesterFlow(self.L, self.R, definition_id="test-M03",
                                  structure_generation=2, parameter_generation=7)

    def test_exact_diagonal_solution_and_composition(self):
        L = torch.diag(torch.tensor([0.2, -0.3], dtype=torch.float64))
        R = torch.diag(torch.tensor([0.1, -0.2, 0.4], dtype=torch.float64))
        flow = SylvesterFlow(L, R, definition_id="diagonal")
        expected = self.X * torch.exp(0.7 * (L.diag()[:, None] + R.diag()[None, :]))
        torch.testing.assert_close(flow(self.X, 0.7), expected)
        torch.testing.assert_close(flow(flow(self.X, 0.2), 0.5), expected)

    def test_gradcheck_X_L_R_dt(self):
        X = self.X.clone().requires_grad_()
        dt = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda x, l, r, t: self.flow(x, t, L=l, R=r, training=True),
            (X, self.L, self.R, dt), eps=1e-6, atol=1e-5, rtol=1e-4))
        self.assertEqual(self.flow.cache_info()["entries"], 0)

    def test_two_optimizer_steps_fresh_graphs(self):
        optimizer = torch.optim.SGD([self.L, self.R], lr=0.03)
        original = self.L.detach().clone()
        for _ in range(2):
            optimizer.zero_grad()
            output = self.flow(self.X, 0.4)
            loss = output.square().sum()
            loss.backward()
            for parameter in (self.L, self.R):
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.norm().item(), 0)
            optimizer.step()
        self.assertFalse(torch.equal(original, self.L))
        self.assertEqual(self.flow.cache_info()["entries"], 0)
        # Repeated backward on independent calls also works without mutation.
        self.flow(self.X, 0.4).sum().backward()
        self.flow(self.X, 0.4).sum().backward()

    def test_inference_reuse_mutation_and_replacement(self):
        with patch("torch.matrix_exp", wraps=torch.matrix_exp) as exp:
            first = self.flow(self.X, 0.3, training=False)
            second = self.flow(self.X, 0.3, training=False)
            self.assertEqual(exp.call_count, 2)
            torch.testing.assert_close(first, second)
            self.assertFalse(second.requires_grad)
            with torch.no_grad():
                self.L.add_(0.05)
            changed = self.flow(self.X, 0.3, training=False)
            self.assertEqual(exp.call_count, 4)
            self.assertFalse(torch.equal(first, changed))
            self.flow.R = (self.R.detach() + 0.07).requires_grad_()
            self.flow(self.X, 0.3, training=False)
            self.assertEqual(exp.call_count, 6)
            # A bypass of the version counter still changes the value snapshot.
            self.flow.R.data.add_(0.01)
            self.flow(self.X, 0.3, training=False)
            self.assertEqual(exp.call_count, 8)
            self.assertEqual(self.flow.parameter_generation, 7)

    def test_cache_metadata_dt_and_precision(self):
        self.flow(self.X, 0.3, training=False)
        self.flow(self.X, 0.4, training=False)
        self.flow.parameter_generation += 1
        self.flow(self.X, 0.4, training=False)
        self.flow.structure_generation += 1
        self.flow(self.X, 0.4, training=False)
        self.flow.definition_id = "other-definition"
        self.flow(self.X, 0.4, training=False)
        self.flow(self.X.float(), 0.4, L=self.L.float(), R=self.R.float(), training=False)
        self.assertEqual(self.flow.cache_info()["misses"], 6)

    def test_dt_inference_guard_and_grad_mode(self):
        dt = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
        with self.assertRaisesRegex(ValueError, "differentiable dt"):
            self.flow(self.X, dt, training=False)
        self.flow(self.X, dt).sum().backward()
        self.assertIsNotNone(dt.grad)
        with torch.no_grad():
            self.flow(self.X, 0.2)
            self.flow(self.X, 0.2)
        self.assertEqual(self.flow.cache_info()["hits"], 1)

    def test_serialized_checkpoint_restores_values_gradients_empty_cache(self):
        self.flow(self.X, 0.3, training=False)
        payload = self.flow.checkpoint()
        restored = SylvesterFlow.from_checkpoint(json.loads(json.dumps(payload)))
        self.assertEqual(restored.cache_info(), {"entries": 0, "hits": 0, "misses": 0})
        self.assertEqual(restored.definition_id, self.flow.definition_id)
        self.assertEqual(restored.structure_generation, 2)
        self.assertEqual(restored.parameter_generation, 7)
        for flow in (self.flow, restored):
            X = self.X.clone().requires_grad_()
            dt = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
            output = flow(X, dt)
            gradients = torch.autograd.grad(output.square().sum(), (X, flow.L, flow.R, dt))
            if flow is self.flow:
                expected_output, expected_gradients = output, gradients
            else:
                torch.testing.assert_close(output, expected_output)
                for actual, expected in zip(gradients, expected_gradients):
                    torch.testing.assert_close(actual, expected)
        with torch.no_grad():
            restored.L.add_(1)
        torch.testing.assert_close(torch.tensor(payload["L"], dtype=self.L.dtype), self.L)

    def test_checkpoint_admission(self):
        payload = self.flow.checkpoint()
        for key, value in (("dtype", "float16"), ("L", [[float("nan")]]),
                           ("L", [[1.0, 2.0]]), ("R", []),
                           ("definition_id", ""), ("structure_generation", -1),
                           ("parameter_generation", True), ("L_requires_grad", 1)):
            with self.assertRaises(ValueError):
                SylvesterFlow.from_checkpoint({**payload, key: value})

    def test_checkpoint_exact_schema(self):
        payload = self.flow.checkpoint()
        missing = dict(payload)
        del missing["L"]
        for malformed in (None, [], {**payload, "schema_version": True},
                          {**payload, "extra": 1}, missing):
            with self.subTest(payload=malformed):
                with self.assertRaises(ValueError):
                    SylvesterFlow.from_checkpoint(malformed)

    def test_admission_errors(self):
        for X, L, R, dt in ((self.X, self.L.float(), self.R, 0.1),
                            (self.X, self.L, self.R, float("nan")),
                            (self.X, self.L, self.R, torch.ones(1, dtype=torch.float64)),
                            (self.X * float("inf"), self.L, self.R, 0.1),
                            (self.X, torch.zeros(3, 3, dtype=torch.float64), self.R, 0.1)):
            with self.assertRaises(ValueError):
                self.flow(X, dt, L=L, R=R)


if __name__ == "__main__":
    unittest.main(verbosity=2)
