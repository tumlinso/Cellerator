import math
import random
import unittest

from cellerator.dynamics import JumpOccurrence, gillespie
from cellerator.dynamics.reactions import Reaction, ReactionNetwork
from cellerator.modeling import Cell, Model, StepContext


class Draws:
    def __init__(self, *values):
        self.values = iter(values)

    def random(self):
        return next(self.values)


class ReactionTests(unittest.TestCase):
    def network(self):
        return ReactionNetwork(["B", "A", "other"], [
            Reaction.mass_action("forward", {"A": 1}, {"B": 1}, 2),
            Reaction.mass_action("reverse", {"B": 1}, {"A": 1}, 1),
            Reaction.mass_action("source", {}, {"other": 1}, 0.25),
        ])

    def test_identity_and_sparse_worklist(self):
        network = self.network()
        initial = network.state({"A": 5})
        self.assertEqual(initial.counts, (0, 5, 0))
        work = network.workspace(initial)
        self.assertEqual(work.refresh(StepContext()), (0, 1, 2))
        self.assertEqual(work.rates, (10, 0, 0.25))
        work.fire("forward")
        self.assertEqual(work.pending, (0, 1))
        with self.assertRaises(RuntimeError):
            _ = work.rates
        self.assertEqual(work.refresh(StepContext()), (0, 1))
        self.assertEqual(work.rates, (8, 1, 0.25))
        self.assertEqual(initial["A"], 5)
        self.assertEqual(work.snapshot()["A"], 4)

    def test_transactional_negative_rejection(self):
        network = self.network()
        initial = network.state({})
        work = network.workspace(initial)
        with self.assertRaises(ValueError):
            work.fire("forward")
        self.assertEqual(dict(work.snapshot()), dict(initial))

    def test_conservation_is_structural(self):
        network = self.network()
        self.assertEqual(network.conservation_residuals({"A": 1, "B": 1}),
                         {"forward": 0, "reverse": 0, "source": 0})
        self.assertEqual(network.conservation_residuals({"other": 1})["source"], 1)

    def test_combinatorial_mass_action_with_catalyst(self):
        network = ReactionNetwork(["A", "E", "B"], [
            Reaction.mass_action("dimer", {"A": 2, "E": 1}, {"E": 1, "B": 1}, 0.5)
        ])
        work = network.workspace(network.state({"A": 5, "E": 3}))
        work.refresh(StepContext())
        self.assertEqual(work.rates, (15.0,))
        self.assertEqual(network.affected(["E"]), (0,))
        work.fire(0)
        self.assertEqual(dict(work.snapshot()), {"A": 3, "E": 3, "B": 1})

    def test_horizon_does_not_fire_future_event(self):
        network = ReactionNetwork(["A"], [Reaction.mass_action("birth", {}, {"A": 1}, 1)])
        result = network.simulate(network.state({}), StepContext(time=2, dt=0.1, rng=Draws(0.99)))
        self.assertEqual(result.time, 2.1)
        self.assertEqual((result.state["A"], result.events), (0, ()))

    def test_sparse_ssa_matches_generic_reference_random_stream(self):
        network = self.network()
        initial = network.state({"A": 4})
        sparse = network.simulate(initial, StepContext(dt=4, rng=random.Random(17)))
        reference = gillespie(network.channels())(initial, StepContext(dt=4, rng=random.Random(17)))
        self.assertEqual(dict(sparse.state), dict(reference.state))
        self.assertEqual(sparse.events, reference.events)
        self.assertLess(sparse.metadata["propensity_evaluations"], 3 * (len(sparse.events) + 1))
        self.assertEqual(sparse.state["A"] + sparse.state["B"], 4)

    def test_invalid_hazards_and_declared_time_dependence(self):
        for invalid in [-1.0, float("nan"), float("inf")]:
            network = ReactionNetwork(["A"], [Reaction("bad", {"A": 1}, lambda s, c: invalid)])
            with self.assertRaises(ValueError):
                network.simulate(network.state({}), StepContext())
        network = ReactionNetwork(["A"], [Reaction("clock", {"A": 1}, lambda s, c: c.time,
                                                  time_dependent=True)])
        with self.assertRaises(ValueError):
            network.simulate(network.state({}), StepContext())

    def test_failed_refresh_remains_dirty(self):
        network = ReactionNetwork(["A"], [Reaction("bad", {}, lambda s, c: -1)])
        work = network.workspace(network.state({}))
        with self.assertRaises(ValueError):
            work.refresh(StepContext())
        self.assertEqual(work.pending, (0,))
        with self.assertRaises(RuntimeError):
            _ = work.rates

    def test_complete_default_dependencies_and_consumption_dependencies(self):
        network = ReactionNetwork(["A", "B"], [
            Reaction("unknown", {"B": 1}, lambda s, c: s["A"]),
            Reaction("constant_death", {"A": -1}, lambda s, c: 1, ()),
        ])
        self.assertEqual(network.affected(["A"]), (0, 1))
        self.assertEqual(network.affected(["B"]), (0,))
        work = network.workspace(network.state({}))
        work.refresh(StepContext())
        self.assertEqual(work.rates, (0, 0))

    def test_large_counts_remain_exact(self):
        network = ReactionNetwork(["A"], [Reaction.mass_action("birth", {}, {"A": 1}, 1)])
        state = network.state({"A": 2**60})
        self.assertEqual(network.apply(state, 0)["A"], 2**60 + 1)

    def test_model_observation_and_empty_network(self):
        network = ReactionNetwork(["A"], [])
        cell = Cell(Model(network, lambda state, ctx: state["A"] * 2), network.state({"A": 7}))
        cell.advance(0.5)
        self.assertEqual(cell.time, 0.5)
        self.assertEqual(cell.observe(), 14)

    def test_invalid_counts_and_layout_rejected(self):
        network = self.network()
        for counts in [{"A": -1}, {"A": 0.5}, {"A": True}]:
            with self.assertRaises((ValueError, TypeError)):
                network.state(counts)
        with self.assertRaises(ValueError):
            network.workspace(self.network().state({}))
        with self.assertRaises(KeyError):
            network.state({"unregistered": 1})

    def test_max_events_only_errors_when_another_event_is_due(self):
        network = ReactionNetwork(["A"], [Reaction.mass_action("birth", {}, {"A": 1}, 1)])
        result = network.simulate(network.state({}), StepContext(dt=1, rng=Draws(0.1, 0.5, 0.99)),
                                  max_events=1)
        self.assertEqual(len(result.events), 1)
        with self.assertRaises(RuntimeError):
            network.simulate(network.state({}), StepContext(dt=1, rng=Draws(0.1, 0.5, 0.1)), max_events=1)

    def test_complete_history_density_has_survival_tail(self):
        network = ReactionNetwork(["A"], [Reaction.mass_action("birth", {}, {"A": 1}, 2)])
        score = network.score_history(network.state({}),
                                      [JumpOccurrence("birth", 0.2), JumpOccurrence("birth", 0.6)],
                                      StepContext(dt=1))
        self.assertAlmostEqual(score.log_likelihood, 2 * math.log(2) - 2)
        self.assertEqual(score.integrated_hazard, 2)
        self.assertEqual(score.terminal_state["A"], 2)
        empty = network.score_history(network.state({}), [], StepContext(dt=1))
        self.assertEqual(empty.log_likelihood, -2)
        with self.assertRaises(ValueError):
            network.score_history(network.state({}), [JumpOccurrence("birth", 1)], StepContext(dt=1))

    def test_time_resolution_and_zero_wait_are_explicit_failures(self):
        network = ReactionNetwork(["A"], [Reaction.mass_action("birth", {}, {"A": 1}, 1)])
        with self.assertRaises(ArithmeticError):
            network.simulate(network.state({}), StepContext(time=1e30, dt=1))
        with self.assertRaises(ArithmeticError):
            network.simulate(network.state({}), StepContext(dt=1, rng=Draws(0)))


if __name__ == "__main__":
    unittest.main()
