from copy import deepcopy
import io
import unittest
import torch

from checkpoint import (capture_training_checkpoint, restore_training_checkpoint,
                        refactor_training_transform, refactor_training_quotient)
from snapshot import Coordinate, Snapshot, SnapshotController
from regrowth import ResidualRegrowth


class CheckpointTests(unittest.TestCase):
    def make_session(self):
        dtype = torch.float64
        coordinates = (Coordinate("x", 1), Coordinate("y", 1))
        controller = SnapshotController(Snapshot(coordinates,
            torch.tensor([0.3, -0.6], dtype=dtype),
            torch.tensor([[0.9, 0.1], [-0.2, 0.7]], dtype=dtype),
            torch.tensor([[0.5, -0.4]], dtype=dtype), structure_epoch=3, value_generation=7,
            replica_maps=((0, 1, 0),)))
        branch = ResidualRegrowth(2, 1, 3, seed=8, dtype=dtype)
        optimizer = torch.optim.Adam(branch.parameters(), lr=0.03, amsgrad=True)
        x = torch.tensor([[0.3, -0.6], [0.8, 0.4]], dtype=dtype)
        target = torch.tensor([[0.4], [-0.1]], dtype=dtype)
        for _ in range(8):
            optimizer.zero_grad()
            loss = (branch(x) - target).square().mean()
            loss.backward()
            optimizer.step()
        return controller, branch, optimizer

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def response(self, controller, branch):
        branch.zero_grad()
        snapshot = controller.snapshot
        x = snapshot.state.detach().clone().requires_grad_()
        y = branch(x)
        y.sum().backward()
        residual = (y.detach(), x.grad.clone(), branch.U.grad.clone(), branch.V.grad.clone())
        with controller.acquire_tape(steps=3) as tape:
            output = tape.output.detach().clone()
            gradients = tape.backward()
        return output, gradients, residual

    def test_serialized_fresh_restore_matches_state_outputs_gradients_and_next_adam_step(self):
        controller, branch, optimizer = self.make_session()
        checkpoint = capture_training_checkpoint(controller, branch, optimizer)
        stream = io.BytesIO()
        torch.save(checkpoint, stream)
        stream.seek(0)
        restored = restore_training_checkpoint(torch.load(stream, weights_only=True))
        self.assertIsNot(restored[0], controller)
        self.assert_nested_equal(checkpoint, capture_training_checkpoint(*restored))
        self.assert_nested_equal(self.response(controller, branch), self.response(restored[0], restored[1]))
        x = torch.tensor([0.2, -0.3], dtype=torch.float64)
        for current_branch, current_optimizer in ((branch, optimizer), (restored[1], restored[2])):
            current_optimizer.zero_grad()
            current_branch(x).square().sum().backward()
            current_optimizer.step()
        self.assert_nested_equal(branch.state_dict(), restored[1].state_dict())
        self.assert_nested_equal(optimizer.state_dict(), restored[2].state_dict())

    def test_rejected_restore_or_transform_keeps_original_session_unchanged(self):
        controller, branch, optimizer = self.make_session()
        before = capture_training_checkpoint(controller, branch, optimizer)
        tape = controller.acquire_tape()
        with self.assertRaises(RuntimeError):
            restore_training_checkpoint(before, previous_controller=controller)
        tape.release()
        malformed = deepcopy(before)
        incoming_id = malformed["optimizer"]["param_groups"][0]["params"][0]
        malformed["optimizer"]["state"][incoming_id]["exp_avg"] = torch.zeros(11)
        with self.assertRaises(ValueError):
            restore_training_checkpoint(malformed, previous_controller=controller)
        with self.assertRaises((ValueError, RuntimeError)):
            refactor_training_transform(controller, branch, optimizer, torch.zeros((2, 2)),
                                        (Coordinate("a", 2), Coordinate("b", 2)))
        self.assert_nested_equal(before, capture_training_checkpoint(controller, branch, optimizer))

    def test_reversed_equal_shape_optimizer_is_explicitly_rejected(self):
        controller, _, _ = self.make_session()
        snapshot = controller.snapshot
        controller = SnapshotController(Snapshot(snapshot.coordinates, snapshot.state, snapshot.law,
                                                 torch.eye(2, dtype=torch.float64)))
        branch = ResidualRegrowth(2, 2, 2, seed=8, dtype=torch.float64)
        optimizer = torch.optim.Adam([branch.V, branch.U], lr=0.03)
        for parameter, value in ((branch.U, 2.), (branch.V, 5.)):
            optimizer.state[parameter] = {"step": torch.tensor(3.),
                "exp_avg": torch.full_like(parameter, value),
                "exp_avg_sq": torch.full_like(parameter, value * value)}
        with self.assertRaisesRegex(ValueError, "declared U,V order"):
            capture_training_checkpoint(controller, branch, optimizer)

    def test_invalid_initialized_adam_states_rejected_without_mutation(self):
        controller, branch, optimizer = self.make_session()
        before = capture_training_checkpoint(controller, branch, optimizer)
        incoming_id = before["optimizer"]["param_groups"][0]["params"][0]
        for mutation in ("negative_second", "negative_maximum", "missing_step", "negative_lr", "dtype_config", "dtype_parameter"):
            malformed = deepcopy(before)
            state = malformed["optimizer"]["state"][incoming_id]
            if mutation == "negative_second":
                state["exp_avg_sq"][0, 0] = -1.
            elif mutation == "negative_maximum":
                state["max_exp_avg_sq"][0, 0] = -1.
            elif mutation == "missing_step":
                del state["step"]
            elif mutation == "negative_lr":
                malformed["optimizer"]["param_groups"][0]["lr"] = -1.
            elif mutation == "dtype_config":
                malformed["branch_config"]["dtype"] = "torch.float32"
            else:
                malformed["branch"]["U"] = malformed["branch"]["U"].float()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                restore_training_checkpoint(malformed, previous_controller=controller)
            self.assert_nested_equal(before, capture_training_checkpoint(controller, branch, optimizer))

    def test_dense_transform_preserves_full_model_and_resets_only_incoming_moments(self):
        controller, branch, optimizer = self.make_session()
        transform = torch.tensor([[1., 0.5], [-0.3, 1.]], dtype=torch.float64)
        transformed = refactor_training_transform(controller, branch, optimizer, transform,
                         (Coordinate("a", 2), Coordinate("b", 2)), replica_maps=((0, 1, 1),))
        old = controller.snapshot
        new = transformed[0].snapshot
        torch.testing.assert_close(old.readout @ old.state + branch(old.state),
                                   new.readout @ new.state + transformed[1](new.state))
        old_x = old.state.clone().requires_grad_()
        new_x = (transform @ old_x.detach()).requires_grad_()
        old_y = old.readout @ torch.linalg.matrix_power(old.law, 3) @ old_x + branch(old_x)
        new_y = new.readout @ torch.linalg.matrix_power(new.law, 3) @ new_x + transformed[1](new_x)
        torch.testing.assert_close(old_y, new_y)
        old_y.sum().backward(); new_y.sum().backward()
        torch.testing.assert_close(old_x.grad, transform.T @ new_x.grad)
        self.assert_nested_equal(optimizer.state[branch.V], transformed[2].state[transformed[1].V])
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            self.assertEqual(torch.count_nonzero(transformed[2].state[transformed[1].U][name]).item(), 0)
        self.assertEqual(new.structure_epoch, old.structure_epoch + 1)

    def test_invariant_quotient_migrates_residual_and_parameter_shapes(self):
        controller, branch, optimizer = self.make_session()
        previous = controller.snapshot
        controller = SnapshotController(Snapshot(previous.coordinates,
            torch.tensor([0.4, 0.4], dtype=torch.float64), torch.eye(2, dtype=torch.float64) * 0.8,
            previous.readout, structure_epoch=3, value_generation=7))
        embedding = torch.tensor([[1.], [1.]], dtype=torch.float64)
        projection = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
        quotient = refactor_training_quotient(controller, branch, optimizer, projection, embedding,
                                             (Coordinate("shared", 2),))
        old = controller.snapshot; new = quotient[0].snapshot
        torch.testing.assert_close(old.readout @ old.state + branch(old.state),
                                   new.readout @ new.state + quotient[1](new.state))
        self.assertEqual(quotient[1].U.shape, (3, 1))
        self.assert_nested_equal(optimizer.state[branch.V], quotient[2].state[quotient[1].V])
        self.assertEqual(torch.count_nonzero(quotient[2].state[quotient[1].U]["exp_avg"]).item(), 0)


if __name__ == "__main__":
    unittest.main()
