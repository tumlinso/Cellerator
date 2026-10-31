import unittest

from cellerator.dynamics import (
    JumpChannel, MappingAlgebra, additive, apply_events, euler, gillespie, rk4, sequence,
)
from cellerator.modeling import DynamicalCell, DynamicalModel, StepContext, StepResult


class DeterministicRandom:
    def __init__(self, values):
        self.values = iter(values)

    def random(self):
        return next(self.values)


class DynamicsTests(unittest.TestCase):
    def test_euler_and_rk4_advance_model_native_scalar_state(self):
        context = StepContext(time=2.0, dt=0.1)
        self.assertAlmostEqual(euler(lambda x, c: x)(1.0, context).state, 1.1)
        result = rk4(lambda x, c: x)(1.0, context)
        self.assertAlmostEqual(result.state, 1.1051708333333332)
        self.assertAlmostEqual(result.time, 2.1)

    def test_rk4_uses_time_dependent_input_context(self):
        seen = []
        def field(x, ctx):
            seen.append((ctx.time, ctx.inputs))
            return ctx.time
        rk4(field)(0, StepContext(time=2, dt=2, inputs="cue"))
        self.assertEqual(seen, [(2, "cue"), (3.0, "cue"), (3.0, "cue"), (4, "cue")])

    def test_cell_uses_continuous_dynamics(self):
        cell = DynamicalCell(DynamicalModel(euler(lambda x, c: 2.0)), 3.0)
        out = cell.advance(0.5)
        self.assertEqual((cell.state, cell.time, out.time), (4.0, 0.5, 0.5))

    def test_stochastic_jump_records_occurrence_and_context_time(self):
        seen = []
        channel = JumpChannel(
            "pulse", lambda x, ctx: 1.0,
            lambda x, ctx: seen.append(ctx.time) or x + 2,
        )
        result = gillespie([channel])(0, StepContext(dt=2, rng=DeterministicRandom([0.2, 0.5, 0.99])))
        self.assertEqual(result.state, 2)
        self.assertEqual(result.metadata["jump_count"], 1)
        self.assertAlmostEqual(seen[0], 0.2231435513)
        self.assertEqual(result.events[0].channel, "pulse")

    def test_additive_and_sequence_keep_distinct_meanings(self):
        context = StepContext(dt=0.5)
        deriv = additive(lambda x, c: 2*x, lambda x, c: 1.0)
        self.assertEqual(euler(deriv)(2, context).state, 4.5)
        combined = sequence(
            lambda x, c: StepResult(x+1, events=("a",)),
            lambda x, c: StepResult(x*2, events=("b",)),
        )(2, context)
        self.assertEqual((combined.state, combined.events), (6, ("a", "b")))

    def test_apply_events_preserves_supplied_order(self):
        result = apply_events(lambda x, e, c: x + e)(0, StepContext(inputs=[1, 4, -2]))
        self.assertEqual((result.state, result.events), (3, (1, 4, -2)))

    def test_mapping_algebra_integrates_named_fields_without_dense_coercion(self):
        transition = rk4(lambda state, ctx: {"rna": state["rna"], "signal": -state["signal"]}, algebra=MappingAlgebra())
        result = transition({"rna": 1.0, "signal": 1.0}, StepContext(dt=0.1))
        self.assertAlmostEqual(result.state["rna"], 1.1051708333333332)
        self.assertAlmostEqual(result.state["signal"], 0.9048375)

    def test_reaction_rate_validation(self):
        with self.assertRaises(ValueError):
            gillespie([JumpChannel("bad", lambda x, c: -1, lambda x, c: x)])(0, StepContext(dt=1))


if __name__ == "__main__":
    unittest.main()
