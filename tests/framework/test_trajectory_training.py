"""CPU contracts for shared-parameter, privately stateful training windows."""
from dataclasses import dataclass
import unittest

import torch
from torch import nn

from cellerator.adapters.torch import TorchModel
from cellerator.adapters.training import (
    TrainingStep, TrainingTrajectory, TrajectoryWindowRunner, detach_state,
)
from cellerator.modeling import DynamicalCell, StepResult


class Growth(nn.Module):
    def __init__(self, initial=2.0):
        super().__init__()
        self.rate = nn.Parameter(torch.tensor(initial, dtype=torch.float64))
        self.seen = []

    def forward(self, state, context):
        self.seen.append((context.time, context.dt, float(self.rate.detach())))
        return {"x": state["x"] * self.rate, "tags": state["tags"]}


def trajectory(identity, value):
    return TrainingTrajectory(identity, {"x": torch.tensor(value, dtype=torch.float64),
                                          "tags": [identity]})


def squared(prediction, target, context):
    return (prediction["x"] - target).square().sum()


class TrajectoryTrainingTests(unittest.TestCase):
    def test_ragged_cells_share_gradient_keep_states_and_update_after_all_steps(self):
        dynamics = Growth()
        model = TorchModel(dynamics)
        runner = TrajectoryWindowRunner(model, max_steps=2, reduction="sum")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        first, second = trajectory("a", 1.), trajectory("b", [3., 4.])
        result = runner.run_window([first, second], {
            "a": [TrainingStep(0.25), TrainingStep(0.75, target=0.)],
            "b": [TrainingStep(2., target=0.)],
        }, squared, optimizer=optimizer)
        # a: (w^2)^2, b: (3w)^2+(4w)^2 -> d/dw = 4w^3+50w=132.
        self.assertAlmostEqual(dynamics.rate.grad.item(), 132.)
        self.assertAlmostEqual(dynamics.rate.item(), 0.68)
        self.assertEqual([entry[2] for entry in dynamics.seen], [2., 2., 2.])
        self.assertEqual([item.time for item in result.trajectories], [1., 2.])
        self.assertEqual([item.step_index for item in result.trajectories], [2, 1])
        self.assertEqual([len(item.history) for item in result.trajectories], [2, 1])
        self.assertEqual(result.trajectories[0].state["x"].item(), 4.)
        self.assertTrue(torch.equal(result.trajectories[1].state["x"], torch.tensor([6., 8.])))
        self.assertEqual(first.state["x"].item(), 1.)
        self.assertEqual(second.state["tags"], ["b"])
        self.assertIsNone(result.trajectories[0].state["x"].grad_fn)
        self.assertEqual((result.loss, result.observations, result.transitions), (116., 2, 3))

    def test_explicit_truncation_differentiates_only_current_window(self):
        dynamics = Growth()
        model = TorchModel(dynamics)
        runner = TrajectoryWindowRunner(model, max_steps=1, reduction="sum")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.)
        state = [trajectory("a", 1.)]
        first = runner.run_window(state, {"a": [TrainingStep(target=0.)]}, squared,
                                  optimizer=optimizer)
        self.assertEqual(dynamics.rate.grad.item(), 4.)
        second = runner.run_window(first.trajectories, {"a": [TrainingStep(target=0.)]}, squared,
                                   optimizer=optimizer)
        # x0 of second window is detached 2, so derivative (2*w)^2 is 16, not 32.
        self.assertEqual(dynamics.rate.grad.item(), 16.)
        self.assertEqual(second.trajectories[0].state["x"].item(), 4.)
        self.assertEqual(second.trajectories[0].time, 2.)
        self.assertEqual(len(second.trajectories[0].history), 2)

    def test_accumulate_completed_windows_then_step_with_explicit_normalization(self):
        dynamics = Growth()
        model = TorchModel(dynamics)
        runner = TrajectoryWindowRunner(model, max_steps=1, reduction="sum")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        runner.run_window([trajectory("a", 1.)], {"a": [TrainingStep(target=0.)]}, squared,
                          optimizer=optimizer, step_optimizer=False, loss_scale=.5)
        result = runner.run_window([trajectory("b", 3.)], {"b": [TrainingStep(target=0.)]}, squared,
                                   optimizer=optimizer, zero_grad=False, loss_scale=.5)
        self.assertEqual(dynamics.rate.grad.item(), 20.)
        self.assertAlmostEqual(dynamics.rate.item(), 0.)
        self.assertTrue(result.optimizer_stepped)

    def test_weighted_reductions_do_not_silently_weight_longer_cells_equally(self):
        expected = {"sum": 28., "mean": 7., "mean_trajectories": 10.}
        # a: losses 4,4 weights 1,2 -> cell mean4; b: loss16 weight1 -> mean16.
        for reduction, loss in expected.items():
            with self.subTest(reduction=reduction):
                dynamics = Growth()
                model = TorchModel(dynamics)
                runner = TrajectoryWindowRunner(model, max_steps=2, reduction=reduction)
                out = runner.run_window([trajectory("a", 1.), trajectory("b", 2.)], {
                    "a": [TrainingStep(target=0.), TrainingStep(dt=0., advance=False, target=0., weight=2.)],
                    "b": [TrainingStep(target=0.)],
                }, squared, step_optimizer=False)
                self.assertEqual(out.loss, loss)
                self.assertEqual(out.observations, 3)

    def test_inactive_empty_and_zero_loss_windows_do_not_clear_or_step(self):
        dynamics = Growth()
        model = TorchModel(dynamics)
        runner = TrajectoryWindowRunner(model, max_steps=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1, weight_decay=1.)
        dynamics.rate.grad = torch.tensor(9., dtype=torch.float64)
        original = trajectory("a", 1.)
        out = runner.run_window([original], {}, squared, optimizer=optimizer)
        self.assertFalse(out.optimizer_stepped)
        self.assertIsNone(out.loss)
        self.assertEqual(dynamics.rate.item(), 2.)
        self.assertEqual(dynamics.rate.grad.item(), 9.)
        self.assertEqual(out.trajectories[0].time, 0.)
        self.assertEqual(out.trajectories[0].history, ())
        self.assertEqual(runner.run_window([], {}, squared, optimizer=optimizer).trajectories, ())
        out = runner.run_window([original], {"a": [TrainingStep()]}, squared, optimizer=optimizer)
        self.assertEqual(out.transitions, 1)
        self.assertFalse(out.optimizer_stepped)
        self.assertEqual(dynamics.rate.grad.item(), 9.)

    def test_time_aware_observation_events_and_inputs(self):
        class Irregular(nn.Module):
            def __init__(self):
                super().__init__()
                self.rate = nn.Parameter(torch.tensor(2.))
            def forward(self, state, context):
                return StepResult(state + self.rate * context.inputs, time=context.time + context.dt / 2,
                                  events=({"value": self.rate * 2},), metadata={"source": "test"})
        module = Irregular()
        seen = []
        model = TorchModel(module, lambda state, ctx: (state, ctx.time))
        runner = TrajectoryWindowRunner(model, max_steps=2)
        def loss(prediction, target, context):
            seen.append((prediction[1], context.trajectory_id, context.step_index, context.transition))
            return (prediction[0] - target).square()
        result = runner.run_window([TrainingTrajectory("a", torch.tensor(1.), time=3.)], {
            "a": [TrainingStep(dt=2., inputs=3., target=0.),
                  TrainingStep(dt=0., advance=False, target=0.)]}, loss, step_optimizer=False)
        self.assertEqual([entry[:3] for entry in seen], [(4., "a", 0), (4., "a", 1)])
        self.assertIsNone(seen[1][3])
        self.assertEqual(result.trajectories[0].time, 4.)
        event_value = result.trajectories[0].history[0].events[0]["value"]
        self.assertIsNone(event_value.grad_fn)
        self.assertEqual(event_value.item(), 4.)

    def test_nested_dataclass_tuple_list_and_sparse_state_detachment(self):
        @dataclass(frozen=True)
        class State:
            fields: dict
        leaf = torch.tensor(2., requires_grad=True)
        sparse = torch.sparse_coo_tensor(torch.tensor([[0]]), leaf.reshape(1), (3,))
        value = State({"x": [leaf * 3, (sparse, "identity")]})
        result = detach_state(value)
        self.assertIsInstance(result, State)
        self.assertIsNone(result.fields["x"][0].grad_fn)
        self.assertTrue(result.fields["x"][1][0].is_sparse)
        self.assertTrue(value.fields["x"][0].requires_grad)
        self.assertEqual(result.fields["x"][0].data_ptr(), value.fields["x"][0].data_ptr())

    def test_custom_state_policy_and_native_style_autograd_callable(self):
        class Handle:
            def __init__(self, value):
                self.value = value
        class NativeStyle(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, weight):
                ctx.save_for_backward(x, weight)
                return x * weight
            @staticmethod
            def backward(ctx, gradient):
                x, weight = ctx.saved_tensors
                return gradient * weight, (gradient * x).sum()
        class Dynamics(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(2.))
            def forward(self, state, context):
                return Handle(NativeStyle.apply(state.value, self.weight))
        def policy(value):
            return Handle(value.value.detach()) if isinstance(value, Handle) else detach_state(value)
        module = TorchModel(Dynamics())
        runner = TrajectoryWindowRunner(module, max_steps=1, detach=policy)
        leaf = torch.tensor(3., requires_grad=True)
        out = runner.run_window([TrainingTrajectory("a", Handle(leaf))], {
            "a": [TrainingStep(target=0.)]},
            lambda prediction, target, context: prediction.value.square(), step_optimizer=False)
        self.assertEqual(module.dynamics.weight.grad.item(), 36.)
        self.assertEqual(leaf.grad.item(), 24.)
        self.assertIsNone(out.trajectories[0].state.value.grad_fn)
        with self.assertRaisesRegex(TypeError, "detachment policy"):
            detach_state(Handle(leaf))

    def test_cell_capture_preserves_identity_and_leaves_cell_unchanged(self):
        model = TorchModel(Growth())
        cell = DynamicalCell(model.as_model(), trajectory("a", 3.).state, id="cell", time=4.)
        item = TrainingTrajectory.from_cell(cell)
        runner = TrajectoryWindowRunner(model, max_steps=1)
        out = runner.run_window([item], {"cell": [TrainingStep(target=0.)]}, squared,
                                step_optimizer=False)
        self.assertEqual((out.trajectories[0].id, out.trajectories[0].time), ("cell", 5.))
        self.assertEqual(cell.time, 4.)
        self.assertEqual(cell.state["x"].item(), 3.)

    def test_rejects_parameter_mutation_before_backward_and_optimizer_step(self):
        model = TorchModel(Growth())
        runner = TrajectoryWindowRunner(model, max_steps=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        def bad_loss(prediction, target, context):
            with torch.no_grad():
                model.dynamics.rate.add_(1.)
            return prediction["x"].square()
        with self.assertRaisesRegex(RuntimeError, "parameters changed"):
            runner.run_window([trajectory("a", 1.)], {"a": [TrainingStep(target=0.)]},
                              bad_loss, optimizer=optimizer)
        self.assertIsNone(model.dynamics.rate.grad)
        self.assertEqual(model.dynamics.rate.item(), 3.)  # callback mutation cannot be rolled back

    def test_callback_failure_keeps_input_versions_and_preexisting_gradients(self):
        model = TorchModel(Growth())
        runner = TrajectoryWindowRunner(model, max_steps=2)
        model.dynamics.rate.grad = torch.tensor(7., dtype=torch.float64)
        initial = trajectory("a", 1.)
        def failure(*args):
            raise RuntimeError("user callback")
        with self.assertRaisesRegex(RuntimeError, "user callback"):
            runner.run_window([initial], {"a": [TrainingStep(), TrainingStep(target=0.)]},
                              failure, step_optimizer=False)
        self.assertEqual(initial.state["x"].item(), 1.)
        self.assertEqual(initial.time, 0.)
        self.assertEqual(model.dynamics.rate.grad.item(), 7.)
        # Failure must release the reentrancy guard.
        self.assertIsNone(runner.run_window([], {}, failure, step_optimizer=False).loss)

    def test_validation_prevents_overlong_unknown_duplicate_schedules(self):
        model = TorchModel(Growth())
        runner = TrajectoryWindowRunner(model, max_steps=1)
        one = trajectory("a", 1.)
        for states, schedule in [([one], {"a": [TrainingStep(), TrainingStep()]}),
                                 ([one], {"b": [TrainingStep()]}), ([one, one], {})]:
            with self.subTest(schedule=schedule), self.assertRaises(ValueError):
                runner.run_window(states, schedule, squared, step_optimizer=False)
        self.assertEqual(model.dynamics.seen, [])

    def test_rejects_non_scalar_nonfinite_or_nondifferentiable_loss(self):
        for loss in [torch.tensor([1.]), torch.tensor(float("nan")), torch.tensor(1.)]:
            model = TorchModel(Growth())
            runner = TrajectoryWindowRunner(model, max_steps=1)
            with self.subTest(loss=loss), self.assertRaises((TypeError, ValueError)):
                runner.run_window([trajectory("a", 1.)], {"a": [TrainingStep(target=0.)]},
                                  lambda *args: loss, step_optimizer=False)
            self.assertIsNone(model.dynamics.rate.grad)

    def test_zero_weight_and_none_target_are_explicit(self):
        model = TorchModel(Growth())
        runner = TrajectoryWindowRunner(model, max_steps=2)
        targets = []
        def loss(prediction, target, context):
            targets.append(target)
            return prediction["x"].square()
        result = runner.run_window([trajectory("a", 1.)], {
            "a": [TrainingStep(target="skip", weight=0.), TrainingStep(target=None)]},
            loss, step_optimizer=False)
        self.assertEqual(targets, [None])
        self.assertEqual(result.observations, 1)


if __name__ == "__main__":
    unittest.main()
