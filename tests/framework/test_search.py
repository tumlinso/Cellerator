"""Small bounded event-history reconstruction example."""
import unittest

from cellerator.inference import (
    EventPath,
    Observation,
    TimedEvent,
    search_event_histories,
)
from cellerator.modeling import DynamicalModel


class EventSearchTests(unittest.TestCase):
    def test_observation_preserves_ambiguous_event_histories(self):
        model = DynamicalModel(lambda state, _: state, observation=lambda state, _: state)

        def proposals(state, context, observation):
            return (
                EventPath((TimedEvent("activate-a", 0.5, "proposal-a"),), provenance="a"),
                EventPath((TimedEvent("activate-b", 0.5, "proposal-b"),), provenance="b"),
            )

        def apply_event(state, event, context):
            return state + (1 if event == "activate-a" else 2)

        result = search_event_histories(
            model,
            [0],
            [Observation(1.0, 1, provenance="measurement-1")],
            proposals,
            apply_event,
            lambda predicted, observed: 0.0 if predicted == observed else -2.0,
            beam_width=2,
        )

        self.assertTrue(result.complete)
        self.assertEqual(result.expanded, 2)
        self.assertEqual(len(result.hypotheses), 2)
        self.assertEqual([h.events[0].event for h in result.hypotheses],
                         ["activate-a", "activate-b"])
        self.assertEqual([h.state for h in result.hypotheses], [1, 2])
        self.assertGreater(result.hypotheses[0].compatibility_weight,
                           result.hypotheses[1].compatibility_weight)
        self.assertIn("proposal-b", repr(result.hypotheses[1].provenance))

    def test_beam_pruning_is_stable_and_declared(self):
        result = search_event_histories(
            DynamicalModel(lambda state, _: state), [0], [Observation(1.0, None)],
            lambda *_: (EventPath((), provenance="first"), EventPath((), provenance="second")),
            lambda state, event, _: state,
            lambda *_: 0.0,
            beam_width=1,
        )

        self.assertFalse(result.complete)
        self.assertEqual(result.truncation_reasons, ("beam_width",))
        self.assertEqual(result.hypotheses[0].provenance[0][1], "first")

    def test_model_cannot_end_before_event_time(self):
        from cellerator.modeling import StepResult

        model = DynamicalModel(
            lambda state, ctx: StepResult(state, time=ctx.time + ctx.dt / 2),
            observation=lambda state, _: state,
        )
        with self.assertRaisesRegex(ValueError, "expected 0.5"):
            search_event_histories(
                model, [0], [Observation(1.0, 0)],
                lambda *_: (EventPath((TimedEvent("x", 0.5),)),),
                lambda state, event, _: state,
                lambda *_: 0.0,
            )

    def test_expansion_limit_is_reported(self):
        result = search_event_histories(
            DynamicalModel(lambda state, _: state), [0], [Observation(1.0, None)],
            lambda *_: tuple(EventPath((), provenance=f"path-{i}") for i in range(3)),
            lambda state, event, _: state,
            lambda *_: 0.0,
            beam_width=4,
            max_expansions=1,
        )

        self.assertFalse(result.complete)
        self.assertEqual(result.expanded, 1)
        self.assertEqual(result.truncation_reasons, ("max_expansions",))


if __name__ == "__main__":
    unittest.main()
