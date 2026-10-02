"""Run with unittest on the assigned CPU Torch interpreter."""

import copy
import io
import unittest

import torch

from regrowth import ResidualRegrowth, reset_adam_moments


class RegrowthTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.x = torch.tensor([[1., -2.], [-0.4, 0.7], [2., 1.]], dtype=torch.float64)
        self.target = torch.tensor([[0.8], [-0.6], [1.4]], dtype=torch.float64)

    def branch(self):
        return ResidualRegrowth(2, 1, 4, seed=17)

    def train_step(self, branch, optimizer):
        optimizer.zero_grad()
        loss = (branch(self.x) - self.target).square().mean()
        loss.backward()
        optimizer.step()
        return loss.item()

    def test_seed_is_nonzero_deterministic_and_does_not_change_global_rng(self):
        state = torch.random.get_rng_state().clone()
        branch = self.branch()
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(branch.U, self.branch().U))
        self.assertEqual(torch.count_nonzero(branch.U).item(), branch.U.numel())
        self.assertFalse(torch.equal(branch.U, ResidualRegrowth(2, 1, 4, seed=18).U))

    def test_initial_function_preservation_and_no_hidden_updates(self):
        branch = self.branch()
        optimizer = torch.optim.Adam(branch.parameters(), lr=0.05)
        existing = self.target.clone().requires_grad_()
        before = copy.deepcopy(branch.state_dict())
        output = branch(self.x, existing)
        self.assertTrue(torch.equal(output, existing))
        output.square().sum().backward()
        for key, value in before.items():
            self.assertTrue(torch.equal(value, branch.state_dict()[key]))
        self.assertEqual(len(optimizer.state), 0)
        self.assertTrue(torch.equal(existing.grad, 2 * existing))

    def test_outgoing_gradient_then_incoming_after_external_step(self):
        branch = self.branch()
        optimizer = torch.optim.Adam(branch.parameters(), lr=0.05)
        loss = (branch(self.x) - self.target).square().mean()
        loss.backward()
        self.assertGreater(branch.V.grad.abs().sum().item(), 0)
        self.assertEqual(branch.U.grad.abs().sum().item(), 0)
        before = branch.U.detach().clone()
        optimizer.step()
        self.assertTrue(torch.equal(before, branch.U))
        optimizer.zero_grad()
        (branch(self.x) - self.target).square().mean().backward()
        self.assertGreater(branch.U.grad.abs().sum().item(), 0)

    def test_fitting_reduces_loss(self):
        branch = self.branch()
        optimizer = torch.optim.Adam(branch.parameters(), lr=0.04)
        first = self.train_step(branch, optimizer)
        for _ in range(150):
            self.train_step(branch, optimizer)
        final = (branch(self.x) - self.target).square().mean().item()
        self.assertLess(final, first * 0.01)

    def test_partial_and_whole_moment_reset_preserve_unrelated_state(self):
        branch = self.branch()
        optimizer = torch.optim.Adam(branch.parameters(), lr=0.02, amsgrad=True)
        for _ in range(3):
            self.train_step(branch, optimizer)
        before_u = copy.deepcopy(optimizer.state[branch.U])
        before_v = copy.deepcopy(optimizer.state[branch.V])
        reset_adam_moments(optimizer, branch.U, [1])
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            self.assertEqual(torch.count_nonzero(optimizer.state[branch.U][name][1]).item(), 0)
            self.assertTrue(torch.equal(optimizer.state[branch.U][name][[0, 2, 3]], before_u[name][[0, 2, 3]]))
            self.assertTrue(torch.equal(optimizer.state[branch.V][name], before_v[name]))
        self.assertTrue(torch.equal(optimizer.state[branch.U]["step"], before_u["step"]))
        reset_adam_moments(optimizer, branch.U)
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            self.assertEqual(torch.count_nonzero(optimizer.state[branch.U][name]).item(), 0)
            self.assertTrue(torch.equal(optimizer.state[branch.V][name], before_v[name]))
        self.assertTrue(torch.equal(optimizer.state[branch.U]["step"], before_u["step"]))

    def test_recycle_selected_slots_and_gradients(self):
        branch = self.branch()
        optimizer = torch.optim.Adam(branch.parameters(), lr=0.02, amsgrad=True)
        for _ in range(3):
            self.train_step(branch, optimizer)
        before_u, before_v = branch.U.detach().clone(), branch.V.detach().clone()
        moments = [copy.deepcopy(optimizer.state[p]) for p in (branch.U, branch.V)]
        grads = [p.grad.clone() for p in (branch.U, branch.V)]
        branch.recycle([1, 3], seed=29, optimizer=optimizer, discard_live=True)
        self.assertTrue(torch.equal(branch.U[[0, 2]], before_u[[0, 2]]))
        self.assertTrue(torch.equal(branch.V[:, [0, 2]], before_v[:, [0, 2]]))
        self.assertFalse(torch.equal(branch.U[[1, 3]], before_u[[1, 3]]))
        self.assertEqual(torch.count_nonzero(branch.V[:, [1, 3]]).item(), 0)
        for p, axis, saved, grad in zip((branch.U, branch.V), (0, 1), moments, grads):
            for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                self.assertEqual(torch.count_nonzero(optimizer.state[p][name].index_select(axis, torch.tensor([1, 3]))).item(), 0)
                self.assertTrue(torch.equal(optimizer.state[p][name].index_select(axis, torch.tensor([0, 2])), saved[name].index_select(axis, torch.tensor([0, 2]))))
            self.assertTrue(torch.equal(optimizer.state[p]["step"], saved["step"]))
            self.assertEqual(torch.count_nonzero(p.grad.index_select(axis, torch.tensor([1, 3]))).item(), 0)
            self.assertTrue(torch.equal(p.grad.index_select(axis, torch.tensor([0, 2])), grad.index_select(axis, torch.tensor([0, 2]))))

    def test_zero_capacity_recycling_preserves_output(self):
        branch = self.branch()
        with torch.no_grad():
            branch.V[:, 0] = 0.3
        before = branch(self.x).detach().clone()
        branch.recycle([1, 2], seed=27)
        self.assertTrue(torch.equal(before, branch(self.x)))
        with self.assertRaises(ValueError):
            branch.recycle([0], seed=28)

    def test_bad_moment_metadata_recycle_fails_atomically(self):
        branch = self.branch()
        optimizer = torch.optim.Adam(branch.parameters(), lr=0.02)
        self.train_step(branch, optimizer)
        optimizer.state[branch.V]["exp_avg_sq"] = torch.ones(1, 3)
        before = copy.deepcopy(branch.state_dict())
        before_optimizer = copy.deepcopy(optimizer.state_dict())
        with self.assertRaises(ValueError):
            branch.recycle([1], seed=29, optimizer=optimizer, discard_live=True)
        for key, value in before.items():
            self.assertTrue(torch.equal(value, branch.state_dict()[key]))
        for key, state in before_optimizer["state"].items():
            for name, value in state.items():
                self.assertTrue(torch.equal(value, optimizer.state_dict()["state"][key][name]))

    def test_state_dict_serialization_resumes_optimizer_exactly(self):
        branch = self.branch()
        optimizer = torch.optim.Adam(branch.parameters(), lr=0.02, amsgrad=True)
        for _ in range(4):
            self.train_step(branch, optimizer)
        branch.recycle([2], seed=43, optimizer=optimizer, discard_live=True)
        stream = io.BytesIO()
        torch.save({"model": branch.state_dict(), "optimizer": optimizer.state_dict()}, stream)
        stream.seek(0)
        saved = torch.load(stream, weights_only=True)
        restored = self.branch()
        restored.load_state_dict(saved["model"])
        restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.02, amsgrad=True)
        restored_optimizer.load_state_dict(saved["optimizer"])
        self.assertTrue(torch.equal(branch(self.x), restored(self.x)))
        self.train_step(branch, optimizer)
        self.train_step(restored, restored_optimizer)
        for key, value in branch.state_dict().items():
            self.assertTrue(torch.equal(value, restored.state_dict()[key]))

    def test_invalid_coordinates_and_unsupported_optimizer_rejected(self):
        branch = self.branch()
        for rows in ([4], [-1], [1, 1], [True], [1.5]):
            with self.assertRaises(ValueError):
                branch.recycle(rows, seed=1)
        with self.assertRaises(TypeError):
            reset_adam_moments(torch.optim.SGD(branch.parameters(), lr=0.1), branch.U)


if __name__ == "__main__":
    unittest.main(verbosity=2)
