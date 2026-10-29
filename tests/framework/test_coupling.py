"""Exchange accounting, snapshot semantics and numerical contract checks."""
import math
import unittest

import numpy as np

from cellerator.spatial.coupling import (
    ExchangeLink, ExchangePlan, ExchangeState, concentrations,
)


class CouplingTests(unittest.TestCase):
    def setUp(self):
        self.links = (ExchangeLink("a", "sender", "bath"),
                      ExchangeLink("b", 42, "bath"))
        self.plan = ExchangePlan(("sender", 42), ("bath",), ("signal", "nutrient"), self.links)
        self.kwargs = dict(cell_volumes={"sender": 2., 42: 1.},
                           compartment_volumes={"bath": 4.}, permeability=[1., .5], dt=.25)
        self.state = ExchangeState({"sender": [4., 0.], 42: [0., 2.]}, {"bath": [0., 4.]})

    @staticmethod
    def total(state):
        return np.sum(list(state.cells.values()) + list(state.compartments.values()), axis=0)

    def test_amount_conservation_with_unequal_volumes(self):
        result = self.plan.step(self.state, **self.kwargs)
        np.testing.assert_allclose(self.total(result.state), self.total(self.state), atol=1e-14)
        np.testing.assert_allclose(result.state.cells["sender"], [3.5, .125])
        self.assertEqual(result.state.cells[42][0], 0.)
        self.assertEqual(result.passive_dt_limit, 1.)
        self.assertFalse(result.limited)
        self.assertEqual(self.state.cells["sender"], [4., 0.])
        result.state.cells["sender"][0] = 10
        self.assertEqual(self.state.cells["sender"][0], 4.)

    def test_shared_reservoir_limits_uptake_proportionally(self):
        state = ExchangeState({"sender": [0., 0.], 42: [0., 0.]}, {"bath": [3., 0.]})
        params = {**self.kwargs, "dt": 1., "permeability": 0.,
                  "active_rates": {"a": [-4., 0.], "b": [-8., 0.]}}
        with self.assertRaisesRegex(ValueError, "donor"):
            self.plan.step(state, **params)
        result = self.plan.step(state, donor_policy="limit", **params)
        np.testing.assert_allclose(result.state.cells["sender"], [1., 0.])
        np.testing.assert_allclose(result.state.cells[42], [2., 0.])
        np.testing.assert_allclose(self.total(result.state), [3., 0.], atol=1e-14)
        self.assertGreaterEqual(result.state.compartments["bath"][0], 0.)
        self.assertTrue(result.limited)

    def test_link_and_identity_order_do_not_change_physics(self):
        reversed_plan = ExchangePlan((42, "sender"), ("bath",), self.plan.species,
                                     reversed(self.links))
        left = self.plan.step(self.state, **self.kwargs)
        right = reversed_plan.step(self.state, **self.kwargs)
        for identity in self.plan.cells:
            np.testing.assert_allclose(left.state.cells[identity], right.state.cells[identity])
        for identity in self.plan.compartments:
            np.testing.assert_allclose(left.state.compartments[identity], right.state.compartments[identity])

    def test_incoming_material_cannot_be_spent_in_same_step(self):
        state = ExchangeState({"sender": [5., 0.], 42: [0., 0.]}, {"bath": [0., 0.]})
        result = self.plan.step(state, **{**self.kwargs, "permeability": 0., "dt": 1.},
                                active_rates={"a": [4., 0.], "b": [-3., 0.]}, donor_policy="limit")
        self.assertEqual(result.state.cells[42][0], 0.)
        self.assertEqual(result.state.compartments["bath"][0], 4.)

    def test_explicit_bound_and_limiter_have_distinct_meanings(self):
        with self.assertRaisesRegex(ValueError, "positivity bound"):
            self.plan.step(self.state, **{**self.kwargs, "dt": 2.}, donor_policy="limit")
        result = self.plan.step(self.state, **{**self.kwargs, "dt": 100.},
                                donor_policy="limit", enforce_stability=False)
        self.assertTrue(result.limited)
        np.testing.assert_allclose(self.total(result.state), self.total(self.state), atol=1e-13)

    def test_equal_concentrations_are_stationary(self):
        state = ExchangeState({"sender": [6., 4.], 42: [3., 2.]}, {"bath": [12., 8.]})
        result = self.plan.step(state, **self.kwargs)
        for value in result.transfers.values():
            np.testing.assert_array_equal(value, [0., 0.])
        np.testing.assert_allclose(concentrations(state.compartments, {"bath": 4.})["bath"], [3., 2.])

    def test_invalid_inputs_fail_without_mutation(self):
        for change in ({"dt": math.nan}, {"dt": -1}, {"permeability": [-1., 0.]},
                       {"cell_volumes": {"sender": 0., 42: 1.}},
                       {"active_rates": {"missing": [0., 0.]}},
                       {"donor_policy": "clip"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.plan.step(self.state, **{**self.kwargs, **change})
        self.assertEqual(self.state.cells["sender"], [4., 0.])
        with self.assertRaises(ValueError):
            ExchangePlan([1], [2], [3], [ExchangeLink(0, 1, 99)])
        with self.assertRaises(ValueError):
            ExchangePlan([1], [2], [3], [ExchangeLink(0, 1, 2), ExchangeLink(0, 1, 2)])

    def test_high_fanout_donor_budget(self):
        count = 1000
        plan = ExchangePlan(["cell"], range(count), ["x"],
                            [ExchangeLink(i, "cell", i) for i in range(count)])
        state = ExchangeState({"cell": [1.]}, {i: [0.] for i in range(count)})
        result = plan.step(state, cell_volumes={"cell": 1.},
                           compartment_volumes={i: 1. for i in range(count)},
                           dt=10., permeability=1., donor_policy="limit", enforce_stability=False)
        self.assertGreaterEqual(result.state.cells["cell"][0], 0.)
        np.testing.assert_allclose(self.total(result.state), [1.], atol=1e-13)

    def test_empty_incidence_and_zero_weight_are_inert(self):
        for links in ([], [ExchangeLink("a", "sender", "bath", 0.)]):
            plan = ExchangePlan(self.plan.cells, self.plan.compartments, self.plan.species, links)
            result = plan.step(self.state, **self.kwargs)
            self.assertEqual(result.passive_dt_limit, math.inf)
            np.testing.assert_array_equal(self.total(result.state), self.total(self.state))


if __name__ == "__main__":
    unittest.main()
