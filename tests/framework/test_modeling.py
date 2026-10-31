"""Focused checks for backend-neutral model and cell contracts."""
from dataclasses import FrozenInstanceError
import warnings
import unittest

from cellerator.modeling import (
    Cell,
    DynamicalCell,
    DynamicalModel,
    Model,
    StepContext,
    StepResult,
    sequential_updates,
)


class ModelingTests(unittest.TestCase):
    def test_generic_cell_is_an_identity_and_data_reference(self):
        source = {"counts": [1, 2]}
        cell = Cell(source, id="cell-a")
        self.assertEqual(cell.id, "cell-a")
        self.assertIs(cell.data, source)
        self.assertFalse(hasattr(cell, "time"))
        self.assertFalse(hasattr(cell, "advance"))
        self.assertIs(Model, DynamicalModel)

    def test_binding_shares_program_identity_and_live_data(self):
        class Readouts:
            def __call__(self, value):
                return value["rna"]

            def predict(self, value, *, view):
                return value[view]

        cell = Cell({"rna": 3, "atac": 7}, id="cell-a")
        program = Readouts()
        bound = cell.bind(program)

        self.assertIsNot(bound, cell)
        self.assertEqual(bound.id, cell.id)
        self.assertIs(bound.program, program)
        self.assertEqual(bound.predict(), 3)
        self.assertEqual(bound.predict("atac"), 7)
        cell.data = {"rna": 5, "atac": 11}
        self.assertEqual(bound.predict(), 5)

    def test_dynamical_cell_is_specialized_cell_with_live_data_view(self):
        cell = DynamicalCell(DynamicalModel(lambda state, _: state + 2), 1)
        self.assertIsInstance(cell, Cell)
        bound = cell.bind(lambda value: value * 3)
        self.assertEqual(bound.predict(), 3)
        cell.advance()
        self.assertEqual(cell.data, 3)
        self.assertEqual(bound.data, 3)
        self.assertEqual(bound.predict(), 9)

    def test_legacy_constructor_returns_dynamic_cell_with_warning(self):
        model = DynamicalModel(lambda state, _: state + 1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cell = Cell(model, 4, id="legacy", time=2.5)
        self.assertIs(type(cell), DynamicalCell)
        self.assertEqual((cell.state, cell.time, cell.id), (4, 2.5, "legacy"))
        self.assertTrue(any(item.category is DeprecationWarning for item in caught))
        with warnings.catch_warnings(record=True) as caught_keywords:
            warnings.simplefilter("always")
            keyword_cell = Cell(model=model, state=7, id="legacy-keyword")
        self.assertIs(type(keyword_cell), DynamicalCell)
        self.assertEqual((keyword_cell.state, keyword_cell.id), (7, "legacy-keyword"))
        self.assertTrue(any(item.category is DeprecationWarning for item in caught_keywords))

    def test_legacy_reference_factory_returns_dynamic_cell_with_warning(self):
        class ExternalState:
            identity = "external-cell"

            def __init__(self):
                self.value = {"x": 1}

            def get(self):
                return self.value

            def set(self, value):
                self.value = value

        reference = ExternalState()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cell = Cell.from_reference(
                DynamicalModel(lambda state, _: {"x": state["x"] + 1}), reference
            )
        self.assertIs(type(cell), DynamicalCell)
        self.assertTrue(any(item.category is DeprecationWarning for item in caught))
        self.assertEqual(cell.data, {"x": 1})
        reference.value = {"x": 8}
        self.assertEqual(cell.data, {"x": 8})
        cell.advance()
        self.assertEqual(cell.data, {"x": 9})

    def test_cells_have_independent_state_and_share_model_parameters(self):
        params = {"rate": 2.0}
        model = DynamicalModel(lambda state, ctx: state + ctx.parameters["rate"], parameters=params)
        a = DynamicalCell(model, 1.0, id="a")
        b = DynamicalCell(model, 10.0, id="b")

        self.assertEqual(a.advance(dt=0.5).state, 3.0)
        self.assertEqual(a.time, 0.5)
        self.assertEqual(b.state, 10.0)
        self.assertIs(a.model, b.model)
        self.assertIs(a.model, model)
        self.assertNotEqual(a.id, b.id)

    def test_explicit_parameter_override_is_preserved(self):
        seen = []
        model = DynamicalModel(lambda state, ctx: (seen.append(ctx.parameters), state)[1], parameters=3)
        context = StepContext(parameters=5)
        model.step(0, context)
        self.assertEqual(model.step(0, StepContext()).state, 0)
        self.assertEqual(seen, [5, 3])
        model.step(0, StepContext(parameters=None))
        self.assertIsNone(seen[-1])

    def test_model_preserves_custom_sparse_state_identity(self):
        state = {"gene-a": 3, "gene-z": 1}
        model = DynamicalModel(lambda current, _: current)
        cell = DynamicalCell(model, state)

        result = cell.advance()

        self.assertIs(result.state, state)
        self.assertIs(cell.state, state)

    def test_transition_events_effects_and_observation_are_preserved(self):
        model = DynamicalModel(
            lambda state, ctx: StepResult(
                state + 1,
                events=("activation",),
                effects=("signal",),
                time=ctx.time + ctx.dt,
            ),
            observation=lambda state, _: {"reported": state},
        )
        cell = DynamicalCell(model, 3)

        result = cell.advance()

        self.assertEqual(
            (result.state, result.events, result.effects, result.time),
            (4, ("activation",), ("signal",), 1.0),
        )
        self.assertEqual(cell.observe(), {"reported": 4})

    def test_failed_transition_does_not_replace_cell_state_or_time(self):
        state = object()

        def fail(_, __):
            raise RuntimeError("transition failed")

        cell = DynamicalCell(DynamicalModel(fail), state, time=4.0)
        with self.assertRaisesRegex(RuntimeError, "transition failed"):
            cell.advance(dt=2.0)
        self.assertIs(cell.state, state)
        self.assertEqual(cell.time, 4.0)

    def test_external_state_write_failure_does_not_advance_cell_time(self):
        class ReadOnlyState:
            def __init__(self):
                self.value = {"x": 2}

            def get(self):
                return self.value

            def set(self, _):
                raise TypeError("this state reference is read-only")

        reference = ReadOnlyState()
        cell = DynamicalCell.from_reference(DynamicalModel(lambda state, _: {"x": 3}), reference)
        with self.assertRaisesRegex(TypeError, "read-only"):
            cell.advance()
        self.assertEqual(reference.get(), {"x": 2})
        self.assertEqual(cell.time, 0.0)

    def test_unbounded_result_time_may_follow_model_rewind_convention(self):
        model = DynamicalModel(lambda state, ctx: StepResult(state - 1, time=ctx.time - ctx.dt))
        cell = DynamicalCell(model, 3, time=2.0)
        self.assertEqual(cell.advance(dt=1.0).time, 1.0)
        with self.assertRaisesRegex(ValueError, "backwards"):
            cell.advance(dt=1.0, until=2.0)

    def test_failed_time_bound_does_not_commit_state_or_time(self):
        cell = DynamicalCell(DynamicalModel(lambda state, ctx: StepResult(state + 1, time=ctx.time + 2)), 0)
        with self.assertRaisesRegex(ValueError, "past the requested time bound"):
            cell.advance(dt=2.0, until=1.0)
        self.assertEqual((cell.state, cell.time), (0, 0.0))

    def test_context_validates_time_and_keeps_metadata_mapping_read_only(self):
        context = StepContext(time=1.0, dt=0.25, metadata={"regime": "pulse"})
        self.assertEqual(context.time, 1.0)
        with self.assertRaises(TypeError):
            context.metadata["regime"] = "steady"
        with self.assertRaisesRegex(ValueError, "finite"):
            StepContext(time=float("nan"))
        with self.assertRaises(FrozenInstanceError):
            context.dt = 2.0

    def test_cell_can_bind_to_externally_owned_state(self):
        class ExternalState:
            value = {"count": 1}
            identity = "cell-1"

            def get(self):
                return self.value

            def set(self, value):
                self.value = value

        owner = ExternalState()
        cell = DynamicalCell.from_reference(
            DynamicalModel(lambda state, _: {"count": state["count"] + 1}), owner
        )
        cell.advance()
        self.assertEqual(owner.get(), {"count": 2})
        self.assertIs(cell.state, owner.get())
        self.assertEqual(cell.id, "cell-1")

    def test_sequential_updates_make_order_and_joint_effects_explicit(self):
        dynamics = sequential_updates(
            lambda state, _: StepResult(state + 1, events=("first",)),
            lambda state, _: StepResult(state * 2, effects=("second",)),
        )
        result = DynamicalModel(dynamics).step(2, StepContext())
        self.assertEqual(result.state, 6)
        self.assertEqual(result.events, ("first",))
        self.assertEqual(result.effects, ("second",))


if __name__ == "__main__":
    unittest.main()
