"""First-order Torch qualification through the installed Cellerator package."""
from importlib import metadata
from pathlib import Path
import unittest

import torch
import cellerator
import cellerator.torch as cellerator_torch
from cellerator import _native
from cellerator.torch import Product2Module, product2, product2_jvp


def inputs():
    return (torch.tensor([0., 1.3, -.7, 2.], requires_grad=True),
            torch.tensor([.4, -.2, 1.1, .3, -.8], requires_grad=True),
            torch.tensor([0, 1, 1, 2, 3]), torch.tensor([0, 1, 2, 1, 0]))


class ProductAdapterTests(unittest.TestCase):
    def test_calls_resolve_from_installed_native_package(self):
        distribution = metadata.distribution('cellerator')
        installed_files = {Path(distribution.locate_file(entry)).resolve()
                           for entry in distribution.files or ()}
        for module_path in (cellerator.__file__, _native.__file__, cellerator_torch.__file__):
            self.assertIn(Path(module_path).resolve(), installed_files)
            self.assertTrue(Path(module_path).is_file())
        self.assertTrue(Path(_native.__file__).name.startswith('_native'))
        self.assertEqual(product2(*inputs()).numel(), 5)

    def test_native_forward_vjp_repeated_zeros(self):
        x, k, a, b = inputs()
        originals = [v.detach().clone() for v in (x, k, a, b)]
        g = torch.tensor([.3, -.5, .4, 1., .6])
        y = product2(x, k, a, b)
        grads = torch.autograd.grad(y, (x, k), g)
        xr, kr = x.detach().clone().requires_grad_(), k.detach().clone().requires_grad_()
        expected = kr * xr[a] * xr[b]
        reference = torch.autograd.grad(expected, (xr, kr), g)
        torch.testing.assert_close(y, expected)
        for actual, target in zip(grads, reference):
            torch.testing.assert_close(actual, target)
        for actual, prior in zip((x, k, a, b), originals):
            torch.testing.assert_close(actual, prior)

    def test_finite_difference_all_inputs_parameters(self):
        x, k, a, b = inputs()
        g = torch.tensor([.7, -.3, .2, .4, -.6])
        grads = torch.autograd.grad(product2(x, k, a, b), (x, k), g)
        for variable, analytic in zip((x, k), grads):
            for index in range(variable.numel()):
                plus, minus = variable.detach().clone(), variable.detach().clone()
                plus[index] += .001
                minus[index] -= .001
                if variable is x:
                    diff = product2(plus, k.detach(), a, b) - product2(minus, k.detach(), a, b)
                else:
                    diff = product2(x.detach(), plus, a, b) - product2(x.detach(), minus, a, b)
                numerical = (g * diff).sum() / .002
                torch.testing.assert_close(analytic[index], numerical, atol=1e-4, rtol=3e-4)

    def test_jvp_adjoint_and_finite_difference(self):
        x, k, a, b = inputs()
        dx, dk = torch.tensor([.2, -.3, .4, .5]), torch.tensor([.1, .2, -.4, .7, -.6])
        g = torch.tensor([.7, -.3, .2, .4, -.6])
        primal, dy = product2_jvp(x, k, a, b, dx, dk)
        grads = torch.autograd.grad(product2(x, k, a, b), (x, k), g)
        torch.testing.assert_close((g * dy).sum(), (grads[0] * dx).sum() + (grads[1] * dk).sum())
        fd = (product2(x.detach() + .001 * dx, k.detach() + .001 * dk, a, b) -
              product2(x.detach() - .001 * dx, k.detach() - .001 * dk, a, b)) / .002
        torch.testing.assert_close(dy, fd, atol=2e-4, rtol=4e-4)
        self.assertFalse(primal.requires_grad)
        self.assertFalse(dy.requires_grad)

    def test_empty_packets_and_empty_state(self):
        x = torch.empty(0, requires_grad=True)
        k = torch.empty(0, requires_grad=True)
        ids = torch.empty(0, dtype=torch.int64)
        y = product2(x, k, ids, ids)
        gx, gk = torch.autograd.grad(y.sum(), (x, k))
        self.assertEqual((y.numel(), gx.numel(), gk.numel()), (0, 0, 0))

    def test_noncontiguous_and_shared_storage(self):
        data = torch.tensor([.2, .7, -.4, .9, 1.2, -.2], requires_grad=True)
        x, k = data[::2], data[1::2]
        a, b = torch.tensor([0, 1, 2]), torch.tensor([1, 1, 0])
        actual = product2(x, k, a, b)
        expected = k * x[a] * x[b]
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual.sum(), data, retain_graph=True)[0],
                                   torch.autograd.grad(expected.sum(), data)[0])
        shared = torch.tensor([.2, .4, .8], requires_grad=True)
        y = product2(shared, shared, a, b)
        reference = shared * shared[a] * shared[b]
        torch.testing.assert_close(torch.autograd.grad(y.sum(), shared)[0],
                                   torch.autograd.grad(reference.sum(), shared)[0])

    def test_stale_all_saved_tensors(self):
        for changed in range(4):
            values = inputs()
            y = product2(*values)
            with torch.no_grad():
                values[changed][0] += 1
            with self.assertRaisesRegex(RuntimeError, 'modified by an inplace operation'):
                y.sum().backward()

    def test_composition_optimizer_and_checkpoint(self):
        x, k, a, b = inputs()
        model = Product2Module(k.detach(), a, b)
        optimizer = torch.optim.SGD(model.parameters(), lr=.02)
        initial = model.coefficients.detach().clone()
        target = torch.zeros(k.numel())
        first_loss = None
        for _ in range(12):
            optimizer.zero_grad()
            loss = (torch.tanh(model(x)) - target).square().sum()
            first_loss = loss.item() if first_loss is None else first_loss
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), first_loss)
        self.assertFalse(torch.equal(initial, model.coefficients))
        checkpoint = {name: value.clone() for name, value in model.state_dict().items()}
        restored = Product2Module(initial, a, b)
        restored.load_state_dict(checkpoint)
        torch.testing.assert_close(restored(x), model(x))
        self.assertEqual(set(checkpoint), {'coefficients', 'a', 'b'})

    def test_admission_types_shapes_extents_indices(self):
        x, k, a, b = inputs()
        for invalid in (x.double(), x.reshape(2, 2), x.to('meta')):
            with self.assertRaises(ValueError):
                product2(invalid, k, a, b)
        with self.assertRaises(ValueError):
            product2(x, k, a.int(), b)
        with self.assertRaises(ValueError):
            product2(x, k, a[:-1], b)
        for bad in (-1, x.numel()):
            invalid = a.clone(); invalid[0] = bad
            with self.assertRaises(ValueError):
                product2(x, k, invalid, b)
        with self.assertRaises(ValueError):
            product2_jvp(x, k, a, b, x[:-1], k)
        with self.assertRaises(TypeError):
            product2([1.], k, a, b)

    def test_double_backward_rejected(self):
        x, k, a, b = inputs()
        gx = torch.autograd.grad(product2(x, k, a, b).square().sum(), x, create_graph=True)[0]
        with self.assertRaises(RuntimeError):
            torch.autograd.grad(gx.sum(), x)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
