"""Focused experiment trajectory and timing checks."""
import random
import unittest

from cellerator.events import Event
from cellerator.modeling import Cell, DynamicalCell, Experiment, DynamicalModel, StepResult, simulate


class SimulationTests(unittest.TestCase):
    def test_experiment_rejects_a_generic_cell(self):
        with self.assertRaisesRegex(TypeError, "DynamicalCell"):
            Experiment(Cell({"rna": [1, 2]}), (0.0,))

    def test_schedule_boundaries_observations_and_state_snapshots(self):
        def dynamics(state, context):
            rate = context.inputs["rate"]
            value = state["value"] + rate * context.dt
            event = Event("advance", time=context.time + context.dt, payload=value)
            return StepResult({"value": value}, events=(event,))

        model = DynamicalModel(
            dynamics,
            observation=lambda state, context: (state["value"], context.inputs["rate"]),
            parameters={"shared": True},
        )
        cell = DynamicalCell(model, {"value": 0.0}, id="c1")
        result = Experiment(
            cell,
            (0.5, 1.5),
            input_schedule={0.0: {"rate": 1.0}, 1.0: {"rate": 3.0}},
            max_step=2.0,
        ).run()

        self.assertEqual(result.times, (0.5, 1.5))
        self.assertEqual(result.observations, ((0.5, 1.0), (2.5, 3.0)))
        self.assertEqual(cell.time, 1.5)
        self.assertEqual(cell.state, {"value": 2.5})
        self.assertEqual([point.time for point in result.trajectory], [0.0, 0.5, 1.5])
        self.assertEqual([event.payload for event in result.events], [0.5, 1.0, 2.5])
        self.assertEqual(result.trajectory[0].state, {"value": 0.0})
        self.assertEqual(result.trajectory[1].state, {"value": 0.5})

    def test_callback_inputs_and_seeded_stochastic_dynamics(self):
        def dynamics(state, context):
            return state + context.rng.random() * context.inputs

        def run(seed):
            return simulate(
                DynamicalCell(DynamicalModel(dynamics), 0.0),
                (1.0, 2.0),
                input_schedule=lambda time, state: 1.0 + state * 0,
                rng=random.Random(seed),
                max_step=0.25,
            )

        first = run(7)
        second = run(7)
        self.assertEqual(first.final_state, second.final_state)
        self.assertEqual(first.times, second.times)

    def test_overshooting_dynamics_does_not_get_labeled_as_requested_time(self):
        cell = DynamicalCell(DynamicalModel(lambda state, context: StepResult(
            state + 1, time=context.time + context.dt + 0.1
        )), 0)
        with self.assertRaisesRegex(ValueError, "past the requested time bound"):
            simulate(cell, (1.0,))
        self.assertEqual(cell.state, 0)
        self.assertEqual(cell.time, 0.0)

    def test_opaque_state_can_supply_its_own_trajectory_clone_policy(self):
        class OpaqueState:
            def __init__(self, value):
                self.value = value

            def __deepcopy__(self, _memo):
                raise TypeError("state has no generic clone operation")

        cell = DynamicalCell(
            DynamicalModel(lambda state, context: OpaqueState(state.value + context.dt)),
            OpaqueState(0.0),
        )
        result = simulate(
            cell,
            (1.0,),
            clone_policy=lambda state: OpaqueState(state.value),
        )
        self.assertEqual(result.trajectory[0].state.value, 0.0)
        self.assertEqual(result.trajectory.final.state.value, 1.0)
        self.assertEqual(result.final_state.value, 1.0)


if __name__ == "__main__":
    unittest.main()
