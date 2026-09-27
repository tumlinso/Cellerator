#!/usr/bin/env python3
"""CPU algebra witness for the proposed shared-support learning construction.

Requires Python 3.10+ and PyTorch. No CelleraTorch or Cellerator implementation
is exercised. No biological data or performance claims. This is a mathematical
reference for later native tests, not a production sparse backend.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import unittest
import torch

DTYPE = torch.float64

def edges() -> tuple[torch.Tensor, torch.Tensor]:
    # Duplicate logical edges are intentional and retain separate parameters.
    return (torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 1, 0]),
            torch.tensor([0, 2, 1, 2, 0, 4, 3, 4, 1, 0]))

def compose(w, x, s, a, dst, src):
    q = s * x
    v = q.new_zeros((q.shape[0], a.shape[1])).index_add(
        1, dst, q[:, src] * w)
    return a * v

def explicit(w, x, s, a, dst, src):
    per_instance_edges = w * s[:, src] * a[:, dst]
    return x.new_zeros((x.shape[0], a.shape[1])).index_add(
        1, dst, per_instance_edges * x[:, src])

def dense(w, x, s, a, dst, src):
    m, n = a.shape[1], x.shape[1]
    matrix = w.new_zeros(m * n).index_add(0, dst * n + src, w).view(m, n)
    return a * ((s * x) @ matrix.T)

def analytic(w, x, s, a, cotangent, dst, src):
    q = s * x
    v = q.new_zeros((q.shape[0], a.shape[1])).index_add(1, dst, q[:, src] * w)
    u = a * cotangent
    transposed = q.new_zeros(q.shape).index_add(1, src, u[:, dst] * w)
    return ((u[:, dst] * q[:, src]).sum(0),
            s * transposed, x * transposed, cotangent * v)

class AlgebraTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(47)
        self.dst, self.src = edges()
        self.args = tuple(torch.randn(shape, generator=generator, dtype=DTYPE,
                                      requires_grad=True)
                          for shape in [(10,), (3, 5), (3, 5), (3, 4)])
        self.cotangent = torch.randn((3, 4), generator=generator, dtype=DTYPE)

    def close(self, a, b):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-11)

    def test_forward_three_realizations(self):
        y = compose(*self.args, self.dst, self.src)
        self.close(y, explicit(*self.args, self.dst, self.src))
        self.close(y, dense(*self.args, self.dst, self.src))

    def test_analytic_vjp(self):
        loss = (compose(*self.args, self.dst, self.src) * self.cotangent).sum()
        observed = torch.autograd.grad(loss, self.args)
        expected = analytic(*self.args, self.cotangent, self.dst, self.src)
        for actual, reference in zip(observed, expected): self.close(actual, reference)

    def test_finite_difference_gradcheck(self):
        self.assertTrue(torch.autograd.gradcheck(
            lambda *args: compose(*args, self.dst, self.src), self.args,
            eps=1e-6, atol=1e-5, rtol=1e-4))

    def test_repeated_shared_parameter_accumulates_without_mutation(self):
        w, x, s, a = self.args
        before = w.detach().clone()
        contexts = [(x, s, a, self.cotangent),
                    (0.7*x + 0.1, 1.2*s, -0.8*a, 0.3*self.cotangent)]
        expected = torch.zeros_like(w)
        for xb, sb, ab, lb in contexts:
            (compose(w, xb, sb, ab, self.dst, self.src) * lb).sum().backward()
            expected = expected + analytic(w, xb, sb, ab, lb, self.dst, self.src)[0]
        self.close(w.grad, expected)
        self.close(w, before)

    def test_zero_weight_can_have_nonzero_gradient(self):
        w = torch.zeros(10, dtype=DTYPE, requires_grad=True)
        x, s, a = (torch.ones((2, n), dtype=DTYPE) for n in [5, 5, 4])
        grad, = torch.autograd.grad(compose(w, x, s, a, self.dst, self.src).sum(), w)
        self.close(grad, torch.full_like(w, 2.0))

    def test_zero_activity_can_have_nonzero_gradient(self):
        w = torch.ones(10, dtype=DTYPE)
        x = torch.ones((2, 5), dtype=DTYPE)
        s = torch.zeros((2, 5), dtype=DTYPE, requires_grad=True)
        a = torch.ones((2, 4), dtype=DTYPE)
        grad, = torch.autograd.grad(compose(w, x, s, a, self.dst, self.src).sum(), s)
        self.assertTrue(bool(torch.all(grad > 0)))

    def test_duplicate_edge_parameters_remain_distinct(self):
        w, x, s, a = self.args
        grad, = torch.autograd.grad(compose(w, x, s, a, self.dst, self.src).sum(), w)
        self.assertEqual(grad.numel(), 10)
        self.close(grad[0], grad[9])
        self.close(grad[2], grad[8])

    def test_physical_edge_permutation_preserves_identity(self):
        w, x, s, a = self.args
        order = torch.tensor([9, 4, 0, 6, 2, 8, 3, 1, 7, 5])
        y = compose(w, x, s, a, self.dst, self.src)
        packed = compose(w[order], x, s, a, self.dst[order], self.src[order])
        self.close(y, packed)
        grad, = torch.autograd.grad((y*self.cotangent).sum(), w, retain_graph=True)
        grad_p, = torch.autograd.grad((packed*self.cotangent).sum(), w)
        self.close(grad, grad_p)

    def test_domain_permutation_requires_corresponding_index_remap(self):
        w, x, s, a = self.args
        source_order = torch.tensor([2, 4, 0, 3, 1])
        target_order = torch.tensor([2, 0, 3, 1])
        inverse_s, inverse_t = source_order.argsort(), target_order.argsort()
        transformed = compose(w, x[:,source_order], s[:,source_order],
                              a[:,target_order], inverse_t[self.dst], inverse_s[self.src])
        original = compose(w, x, s, a, self.dst, self.src)
        self.close(original[:,target_order], transformed)

    def test_general_context_edge_changes_not_always_gate_factorizable(self):
        # For a nonzero 2x2 base, multiplicative row/column gates preserve the
        # cross-ratio. These two matrices violate that necessary identity.
        base = torch.ones((2, 2), dtype=DTYPE)
        alternative = torch.tensor([[1., 1.], [1., 2.]], dtype=DTYPE)
        ratio = alternative / base
        self.assertNotEqual(float(ratio[0,0]*ratio[1,1]),
                            float(ratio[0,1]*ratio[1,0]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', type=Path, help='Optional local report path')
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(AlgebraTests))
    report = {'ok': result.wasSuccessful(), 'tests_run': result.testsRun,
              'failures': len(result.failures), 'errors': len(result.errors),
              'torch_version': torch.__version__, 'device': 'cpu', 'dtype': 'float64',
              'scope': 'Original mathematical reference only; no Cellerator/CelleraTorch/native/GPU or biological validation.',
              'assumptions': 'Finite real-valued inputs, sum assembly, distinct logical edge parameters; no numerical zero-pruning.'}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return 0 if result.wasSuccessful() else 1
if __name__ == '__main__':
    raise SystemExit(main())
