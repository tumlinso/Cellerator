"""Small reference scenarios for identity, communication and lifecycle semantics."""
from copy import deepcopy
import unittest

from cellerator.events import Event, EventHistory
from cellerator.modeling import Cell, DynamicalCell, DynamicalModel, StepResult
from cellerator.execution import (
    Broadcast, CausalityError, ContextCapacityError, Create, Eject,
    ExecutionFailure, Fork, LocalExecutor, Population, Remove, Resume, Send,
    Suspend,
)


class ExecutionTests(unittest.TestCase):
    def test_population_rejects_a_generic_cell(self):
        with self.assertRaisesRegex(TypeError, "DynamicalCell"):
            Population([Cell({"events": []})])

    def test_shared_models_keep_arbitrary_state_and_external_inputs_separate(self):
        model = DynamicalModel(lambda state, context: {
            **state, "value": state["value"] + context.parameters["rate"],
            "external": context.inputs.external,
        }, parameters={"rate": 2})
        population = Population([DynamicalCell(model, {"value": 1}, id="a"),
                                 DynamicalCell(model, {"value": 10}, id="b")])
        result = population.advance(inputs={"a": {"opaque": "input"}})
        self.assertEqual(result.order, ("a", "b"))
        self.assertEqual(population["a"].state["value"], 3)
        self.assertEqual(population["b"].state["value"], 12)
        self.assertIs(population["a"].model, population["b"].model)
        self.assertEqual(population["a"].state["external"], {"opaque": "input"})
        self.assertIsNone(population["b"].state["external"])
        self.assertEqual(len(population.context("a")), 0)

    def test_bounded_context_ejection_is_independent_of_history(self):
        history = EventHistory()
        admitted = []

        def update(state, context):
            admitted.extend(record.value for record in context.inputs.incoming)
            return StepResult(tuple(context.inputs), events=(Event("observed"),))

        model = DynamicalModel(update)
        population = Population([DynamicalCell(model, (), id="a")], context_capacity=2, history=history)
        for value in ("first", "second", "third"):
            population.send("a", value)
        report = population.advance()
        self.assertEqual(population["a"].state, ("second", "third"))
        self.assertEqual(tuple(record.value for record in report.ejections[0].records), ("first",))
        self.assertEqual(report.ejections[0].reason, "capacity")
        self.assertEqual(admitted, ["second", "third"])
        population.eject("a", predicate=lambda record: record.value == "second")
        self.assertEqual(tuple(population.context("a")), ("third",))
        self.assertEqual(len(history), 1)

    def test_reject_overflow_keeps_pending_until_explicit_ejection(self):
        population = Population([DynamicalCell(DynamicalModel(lambda state, _: state), 0, id="a")],
                                context_capacity=1, overflow="reject")
        population.send("a", "first")
        population.advance()
        population.send("a", "second")
        with self.assertRaises(ExecutionFailure) as caught:
            population.advance()
        self.assertIsInstance(caught.exception.__cause__, ContextCapacityError)
        self.assertEqual(population["a"].time, 1)
        self.assertEqual(tuple(population.context("a")), ("first",))
        self.assertEqual(len(population.pending("a")), 1)
        population.eject("a")
        population.advance()
        self.assertEqual(tuple(population.context("a")), ("second",))

    def test_oversized_rejecting_inbox_recovers_by_explicit_pending_discard(self):
        population = Population([DynamicalCell(DynamicalModel(lambda state, _: state), 0, id="a")],
                                context_capacity=1, overflow="reject")
        population.send("a", "keep")
        population.send("a", "discard")
        with self.assertRaises(ExecutionFailure):
            population.advance()
        removed = population.discard_pending("a", predicate=lambda record: record.value == "discard")
        self.assertEqual(tuple(record.value for record in removed), ("discard",))
        population.advance()
        self.assertEqual(tuple(population.context("a")), ("keep",))

    def test_suspension_keeps_state_time_and_pending_inputs(self):
        model = DynamicalModel(lambda state, context: state + len(context.inputs.incoming))
        population = Population([DynamicalCell(model, 0, id="a")])
        population.suspend("a", reason="caller policy")
        population.send("a", "pulse", time=0.5)
        self.assertEqual(population.advance().order, ())
        self.assertEqual(population["a"].time, 0)
        self.assertEqual(len(population.pending("a")), 1)
        population.resume("a")
        population.advance(dt=2)
        self.assertEqual(population["a"].time, 0.5)
        self.assertEqual(population["a"].state, 0)
        population.advance(dt=0.5)
        self.assertEqual(population["a"].state, 1)
        population.advance(dt=0.5)
        self.assertEqual(population["a"].state, 1)  # retained context is not a new occurrence

    def test_deferred_event_and_effect_routing_has_stable_tie_order(self):
        sender = DynamicalModel(lambda state, context: StepResult(state,
            events=(Event("event", target="receiver"),),
            effects=(Send("receiver", "effect"),)))
        receiver = DynamicalModel(lambda state, context: state + tuple(record.value for record in context.inputs.incoming))
        population = Population([DynamicalCell(sender, 0, id="sender"), DynamicalCell(receiver, (), id="receiver")])
        population.advance()
        self.assertEqual(population["receiver"].state, ())
        self.assertEqual(len(population.pending("receiver")), 2)
        population.advance(ids=("receiver",))
        values = population["receiver"].state
        self.assertIsInstance(values[0], Event)
        self.assertEqual(values[1], "effect")
        self.assertEqual([record.sequence for record in population.context("receiver").records], [0, 1])

    def test_lifecycle_effects_apply_after_selection_and_fork_is_caller_owned(self):
        state = {"memory": ["parent"]}
        child_model = DynamicalModel(lambda current, _: current)
        parent_model = DynamicalModel(lambda current, _: StepResult(current, effects=(
            Fork(deepcopy, id="child", inherit_context=True),
            Create(child_model, {"other": 1}, id="created"),
            Suspend(reason="quiescent"),
        )))
        population = Population([DynamicalCell(parent_model, state, id="parent")])
        population.send("parent", "retained")
        report = population.advance()
        self.assertEqual(report.order, ("parent",))
        self.assertEqual(population.ids, ("parent", "child", "created"))
        self.assertIs(population["child"].model, parent_model)
        self.assertEqual(population["child"].time, 1)
        population["child"].state["memory"].append("child-only")
        self.assertEqual(state, {"memory": ["parent"]})
        self.assertEqual(tuple(population.context("child")), ("retained",))
        self.assertTrue(population.is_suspended("parent"))
        self.assertEqual(population.pending("child"), ())

    def test_synchronized_peers_stop_at_shared_pending_message_boundary(self):
        sender = DynamicalModel(lambda state, context: StepResult(state, effects=(Send("target", "pulse"),)))
        target = DynamicalModel(lambda state, context: state + len(context.inputs.incoming))
        population = Population([DynamicalCell(sender, 0, id="sender"), DynamicalCell(target, 0, id="target")])
        population.send("sender", "boundary", time=0.5)
        population.advance(dt=1)
        self.assertEqual(population["sender"].time, 0.5)
        self.assertEqual(population["target"].time, 0.5)
        self.assertEqual(population.pending("target")[0].time, 0.5)
        population.advance(dt=0.5)
        self.assertEqual(population["target"].state, 1)

    def test_effect_fork_rejects_unsynchronized_parent(self):
        parent = DynamicalCell(DynamicalModel(lambda state, _: state), {}, id="parent")
        emitter = DynamicalCell(DynamicalModel(lambda state, _: StepResult(state, effects=(
            Fork(deepcopy, parent="parent", id="child"),))), {}, id="emitter")
        population = Population([parent, emitter])
        with self.assertRaises(ExecutionFailure) as caught:
            population.advance(ids=("emitter",))
        self.assertIsInstance(caught.exception.__cause__, CausalityError)
        self.assertNotIn("child", population.ids)

    def test_removed_identity_is_never_reused_and_discarded_pending_is_reported(self):
        records = []
        model = DynamicalModel(lambda state, _: state)
        population = Population([DynamicalCell(model, object(), id="a")], recorder=records.append)
        population.send("a", "future", time=5)
        removed = population.remove("a")
        self.assertEqual(len(records[-1].value["pending"]), 1)
        self.assertIs(removed.model, model)
        with self.assertRaises(ValueError):
            population.create(model, object(), id="a")
        with self.assertRaises(KeyError):
            population.send("a", "late")

    def test_resource_grouping_uses_model_identity_and_explicit_work_keys(self):
        model = DynamicalModel(lambda state, _: state)
        other = DynamicalModel(lambda state, _: state)
        population = Population([DynamicalCell(model, 1, id="a"), DynamicalCell(other, 2, id="b"),
                                 DynamicalCell(model, 3, id="c")])
        self.assertEqual([batch.ids for batch in population.plan()], [("a", "c"), ("b",)])
        self.assertEqual([batch.ids for batch in population.plan(group_by=lambda cell: cell.state % 2)],
                         [("a", "c"), ("b",)])
        report = LocalExecutor(population, batch_size=1).step()
        self.assertEqual(report.order, ("a", "c", "b"))
        self.assertEqual(set(report.results), {"a", "b", "c"})

    def test_causal_violations_reject_before_transition_commit(self):
        population = Population([DynamicalCell(DynamicalModel(lambda state, _: state + 1), 0, id="a", time=2)])
        with self.assertRaises(CausalityError):
            population.send("a", "past", time=1)
        invalid = DynamicalCell(DynamicalModel(lambda state, _: StepResult(state + 1, time=4)), 0, id="b")
        population.add(invalid)
        population.send("b", "boundary", time=0.5)
        with self.assertRaises(ExecutionFailure) as caught:
            population.advance(ids=("b",))
        self.assertEqual(caught.exception.phase, "evaluation")
        self.assertEqual(invalid.state, 0)
        self.assertEqual(invalid.time, 0)

    def test_partial_progress_and_unhandled_domain_effects_are_observable(self):
        def fail(state, context):
            raise RuntimeError("caller model failed")

        population = Population([DynamicalCell(DynamicalModel(lambda state, _: StepResult(state + 1, effects=("domain",))), 0, id="a"),
                                 DynamicalCell(DynamicalModel(fail), 0, id="b")])
        with self.assertRaises(ExecutionFailure) as caught:
            population.advance()
        self.assertEqual(population["a"].state, 1)
        self.assertEqual(population["b"].time, 0)
        self.assertEqual(tuple(caught.exception.report.results), ("a",))
        report = population.advance(ids=("a",))
        self.assertEqual(report.unhandled_effects[0].value, "domain")

    def test_broadcast_resume_ejection_and_removal_are_explicit_effects(self):
        target_model = DynamicalModel(lambda state, _: state)
        sender_model = DynamicalModel(lambda state, _: StepResult(state, effects=(
            Broadcast("pulse", targets=("target",)), Resume("target"),
            Eject(count=1), Remove(),
        )))
        population = Population([DynamicalCell(sender_model, 0, id="sender"), DynamicalCell(target_model, 0, id="target")])
        population.send("sender", "old")
        population.suspend("target")
        report = population.advance()
        self.assertEqual(population.ids, ("target",))
        self.assertFalse(population.is_suspended("target"))
        self.assertEqual(report.ejections[0].records[0].value, "old")
        self.assertEqual(population.pending("target")[0].time, 1)


if __name__ == "__main__":
    unittest.main()
