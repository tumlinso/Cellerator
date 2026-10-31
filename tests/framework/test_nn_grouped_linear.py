"""NN composition and native CUDA autograd contracts for grouped linear.

CUDA tests run only with CELLERATOR_TEST_CUDA=1 inside an admitted scheduler
job. CPU discovery never probes CUDA availability.
"""
from __future__ import annotations

import copy
import os
import unittest

import torch
from torch import nn

from cellerator.torch.grouped_linear import GroupedLinear, grouped_linear


def oracle(inputs, weights, offsets):
    # Independent row-wise interpretation, including a zero dependency on all
    # experts so unused weights have an explicit zero derivative.
    rows = [
        torch.stack([torch.dot(row, output_weight) for output_weight in weights[e]])
        for e, (start, end) in enumerate(zip(offsets, offsets[1:]))
        for row in inputs[start:end]
    ]
    if rows:
        return torch.stack(rows) + weights.sum() * 0
    return inputs.new_empty((0, weights.shape[1])) + inputs.sum() * 0 + weights.sum() * 0


class GroupedLinearReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(318)

    def test_ragged_rows_and_both_gradients_match_row_oracle(self):
        x = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
        w = torch.randn(4, 3, 5, dtype=torch.float64, requires_grad=True)
        offsets = (0, 2, 2, 6, 7)
        y = grouped_linear(x, w, offsets)
        expected = oracle(x, w, offsets)
        upstream = torch.randn_like(y)
        got = torch.autograd.grad(y, (x, w), upstream)
        want = torch.autograd.grad(expected, (x, w), upstream)
        torch.testing.assert_close(y, expected)
        for actual, reference in zip(got, want):
            torch.testing.assert_close(actual, reference)
        self.assertTrue(torch.equal(got[1][1], torch.zeros_like(w[1])))

    def test_zero_rows_keep_zero_input_and_parameter_gradients(self):
        x = torch.empty(0, 5, requires_grad=True)
        w = torch.randn(3, 4, 5, requires_grad=True)
        y = grouped_linear(x, w, (0, 0, 0, 0))
        y.sum().backward()
        self.assertEqual(tuple(y.shape), (0, 4))
        self.assertEqual(tuple(x.grad.shape), (0, 5))
        self.assertTrue(torch.equal(w.grad, torch.zeros_like(w)))

    def test_reference_gradcheck_and_higher_order(self):
        x = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
        w = torch.randn(2, 2, 2, dtype=torch.float64, requires_grad=True)
        function = lambda a, b: grouped_linear(a, b, (0, 1, 3), implementation="reference")
        self.assertTrue(torch.autograd.gradcheck(function, (x, w)))
        self.assertTrue(torch.autograd.gradgradcheck(function, (x, w)))

    def test_noncontiguous_inputs_and_weights_preserve_gradients(self):
        source = torch.randn(5, 7, dtype=torch.float64, requires_grad=True)
        parameters = torch.randn(3, 5, 2, dtype=torch.float64, requires_grad=True)
        x, w = source.t(), parameters.transpose(1, 2)
        self.assertFalse(x.is_contiguous())
        self.assertFalse(w.is_contiguous())
        y = grouped_linear(x, w, torch.tensor([0, 2, 2, 7]))
        expected = oracle(x, w, (0, 2, 2, 7))
        got = torch.autograd.grad(y.square().sum(), (source, parameters))
        want = torch.autograd.grad(expected.square().sum(), (source, parameters))
        for actual, reference in zip(got, want):
            torch.testing.assert_close(actual, reference)

    def test_shared_module_accumulates_gradients_once_and_uses_standard_optimizer(self):
        module = GroupedLinear(2, 3, 2, dtype=torch.float64)
        first, second = torch.randn(4, 3, dtype=torch.float64), torch.randn(5, 3, dtype=torch.float64)
        weights = module.weight.detach().clone().requires_grad_()
        expected_loss = oracle(first, weights, (0, 1, 4)).square().sum()
        expected_loss = expected_loss + oracle(second, weights, (0, 3, 5)).square().sum()
        expected_gradient, = torch.autograd.grad(expected_loss, (weights,))
        loss = module(first, (0, 1, 4)).square().sum() + module(second, (0, 3, 5)).square().sum()
        loss.backward()
        self.assertEqual(list(module.parameters()), [module.weight])
        torch.testing.assert_close(module.weight.grad, expected_gradient)
        before = module.weight.detach().clone()
        torch.optim.SGD(module.parameters(), lr=0.01).step()
        torch.testing.assert_close(module.weight, before - 0.01 * expected_gradient)

    def test_standard_state_dict_and_to(self):
        first = GroupedLinear(2, 3, 4)
        second = GroupedLinear(2, 3, 4)
        second.load_state_dict(copy.deepcopy(first.state_dict()))
        second.to(dtype=torch.float64)
        self.assertEqual(second.weight.dtype, torch.float64)
        torch.testing.assert_close(second.weight, first.weight.double())
        self.assertEqual(set(first.state_dict()), {"weight"})

    def test_malformed_offsets_and_mismatched_operands_fail(self):
        x, w = torch.randn(5, 3), torch.randn(3, 2, 3)
        for offsets in [(0, 5), (1, 2, 3, 5), (0, 3, 2, 5), (0, 2, 4, 6)]:
            with self.subTest(offsets=offsets), self.assertRaises(ValueError):
                grouped_linear(x, w, offsets)
        with self.assertRaises(TypeError):
            grouped_linear(x, w, (0, True, 4, 5))
        with self.assertRaises(TypeError):
            grouped_linear(x, w, torch.tensor([0., 2., 4., 5.]))
        with self.assertRaises(ValueError):
            grouped_linear(x.double(), w, (0, 2, 4, 5))
        with self.assertRaises(ValueError):
            grouped_linear(x, w, (0, 2, 4, 5), implementation="unknown")
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            grouped_linear(x, w, (0, 2, 4, 5), implementation="native")


@unittest.skipUnless(os.environ.get("CELLERATOR_TEST_CUDA") == "1", "requires admitted CUDA execution")
class GroupedLinearNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # An explicitly requested CUDA qualification must fail rather than skip
        # if its admitted environment lacks the required backend.
        if not torch.cuda.is_available():
            raise RuntimeError("admitted CUDA test environment has no usable device")
        from cellerator.torch.grouped_linear import _native_module
        _native_module()
        cls.device = torch.device("cuda", int(os.environ.get("CELLERATOR_TEST_CUDA_DEVICE", "0")))

    def setUp(self):
        torch.manual_seed(914)

    def assertNumericallyClose(self, actual, expected, dtype):
        if dtype == torch.float16:
            torch.testing.assert_close(actual, expected, rtol=6e-3, atol=4e-3)
        else:
            torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)

    def test_fp16_fp32_ragged_forward_and_both_vjps(self):
        offsets = (0, 2, 2, 6, 7)
        for dtype in (torch.float32, torch.float16):
            with self.subTest(dtype=dtype):
                x = torch.randn(7, 5, device=self.device, dtype=dtype, requires_grad=True)
                w = torch.randn(4, 3, 5, device=self.device, dtype=dtype, requires_grad=True)
                gradient = torch.randn(7, 3, device=self.device, dtype=dtype)
                expected_x = x.detach().float().requires_grad_()
                expected_w = w.detach().float().requires_grad_()
                expected = oracle(expected_x, expected_w, offsets)
                want = torch.autograd.grad(expected, (expected_x, expected_w), gradient.float())
                actual = grouped_linear(x, w, offsets, implementation="native")
                got = torch.autograd.grad(actual, (x, w), gradient)
                self.assertNumericallyClose(actual, expected.to(dtype), dtype)
                for a, b in zip(got, want):
                    self.assertNumericallyClose(a, b.to(dtype), dtype)
                self.assertTrue(torch.equal(got[1][1], torch.zeros_like(w[1])))

    def test_zero_rows_and_all_empty_experts(self):
        x = torch.empty(0, 5, device=self.device, requires_grad=True)
        w = torch.randn(3, 4, 5, device=self.device, requires_grad=True)
        grouped_linear(x, w, (0, 0, 0, 0)).sum().backward()
        self.assertEqual(tuple(x.grad.shape), (0, 5))
        self.assertTrue(torch.equal(w.grad, torch.zeros_like(w)))

    def test_fp32_master_to_half_compute_gradients_reach_original_leaf(self):
        x = torch.randn(6, 5, device=self.device, requires_grad=True)
        w = nn.Parameter(torch.randn(3, 4, 5, device=self.device))
        upstream = torch.randn(6, 4, device=self.device, dtype=torch.float16)
        y = grouped_linear(x.half(), w.half(), (0, 2, 2, 6))
        got = torch.autograd.grad(y, (x, w), upstream)
        xx = x.detach().half().float().requires_grad_()
        ww = w.detach().half().float().requires_grad_()
        want = torch.autograd.grad(oracle(xx, ww, (0, 2, 2, 6)), (xx, ww), upstream.float())
        self.assertEqual(y.dtype, torch.float16)
        for a, b in zip(got, want):
            self.assertEqual(a.dtype, torch.float32)
            self.assertNumericallyClose(a, b.half().float(), torch.float16)

    def test_noncontiguous_operands_and_nondefault_stream(self):
        stream = torch.cuda.Stream(device=self.device)
        with torch.cuda.stream(stream):
            source = torch.randn(5, 7, device=self.device, requires_grad=True)
            parameters = torch.randn(3, 5, 2, device=self.device, requires_grad=True)
            x, w = source.t(), parameters.transpose(1, 2)
            y = grouped_linear(x, w, (0, 2, 2, 7))
            expected = oracle(x, w, (0, 2, 2, 7))
            got = torch.autograd.grad(y.square().sum(), (source, parameters))
            want = torch.autograd.grad(expected.square().sum(), (source, parameters))
            stream.synchronize()
        for actual, reference in zip(got, want):
            self.assertNumericallyClose(actual, reference, torch.float32)

    def test_saved_parameter_mutation_and_higher_order_are_rejected(self):
        x = torch.randn(4, 3, device=self.device, requires_grad=True)
        w = torch.randn(2, 2, 3, device=self.device, requires_grad=True)
        y = grouped_linear(x, w, (0, 2, 4))
        with torch.no_grad():
            w.add_(1)
        with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
            y.sum().backward()
        y = grouped_linear(x, w, (0, 2, 4))
        with self.assertRaisesRegex(RuntimeError, "first-order gradients only"):
            torch.autograd.grad(y.sum(), (x, w), create_graph=True)

    def test_composed_training_shared_parameters_and_standard_optimizer(self):
        before = nn.Linear(5, 4, device=self.device)
        grouped = GroupedLinear(3, 4, 3, device=self.device)
        after = nn.Linear(3, 2, device=self.device)
        reference = copy.deepcopy(nn.ModuleList([before, grouped, after]))
        modules = nn.ModuleList([before, grouped, after])
        optimizer = torch.optim.AdamW(modules.parameters(), lr=1e-3)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3)
        first = torch.randn(7, 5, device=self.device)
        second = torch.randn(5, 5, device=self.device)
        for _ in range(3):
            optimizer.zero_grad(); reference_optimizer.zero_grad()
            loss = after(grouped(before(first), (0, 2, 2, 7))).square().mean()
            loss = loss + after(grouped(before(second), (0, 1, 4, 5))).square().mean()
            expected = reference[2](reference[1](reference[0](first), (0, 2, 2, 7), implementation="reference")).square().mean()
            expected = expected + reference[2](reference[1](reference[0](second), (0, 1, 4, 5), implementation="reference")).square().mean()
            loss.backward(); expected.backward()
            for actual, want in zip(modules.parameters(), reference.parameters()):
                self.assertNumericallyClose(actual.grad, want.grad, torch.float32)
            optimizer.step(); reference_optimizer.step()
        for actual, want in zip(modules.parameters(), reference.parameters()):
            self.assertNumericallyClose(actual, want, torch.float32)


if __name__ == "__main__":
    unittest.main()
