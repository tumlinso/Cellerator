import itertools
import unittest
import numpy as np
from state_reference import (Coordinate, Epoch, Slot, StateOwner, StateView,
                             Footprint, pack, local_port_transport)


class StateChecks(unittest.TestCase):
    def setUp(self):
        self.a, self.b, self.reserve = Coordinate(11, 0, 7), Coordinate(12, 0, 3), Coordinate(11, 1, 0)
        self.values = np.array([0., 2., 0.])
        self.owner = StateOwner(self.values, [self.a, self.b, self.reserve], Epoch(2, 3, 4, 5))
        self.slots = [Slot(self.b, 1, activity=False), Slot(self.a, 0, activity=False),
                      Slot(self.a, 0, activity=False, residual=True),
                      Slot(self.reserve, 2, support=False, activity=False, residual=True)]

    def test_nonowning_reordered_alias_and_pullback(self):
        view = StateView(self.owner, self.slots)
        self.assertIs(self.owner.values, self.values)
        np.testing.assert_array_equal(view.gather(), [2, 0, 0, 0])
        np.testing.assert_array_equal(view.pullback([5, 2, 3, 0]), [5, 5, 0])
        # Adjoint identity of the physical replication map, including zero primal.
        rng = np.random.default_rng(4)
        direction = rng.normal(size=3)
        cotangent = rng.normal(size=4)
        physical = direction[[1, 0, 0, 2]]
        self.assertAlmostEqual(physical @ cotangent, direction @ view.pullback(cotangent))
        with self.assertRaises(ValueError):
            view.pullback([1])

    def test_all_saved_generations_rejected(self):
        saved_primal = StateView(self.owner, self.slots).gather()
        for field in ("structure", "values", "activity", "parameters"):
            view = StateView(self.owner, self.slots)
            self.owner.touch(field)
            with self.assertRaisesRegex(ValueError, "stale"):
                view.gather()
        self.values[0] = 9
        self.owner.touch("values")
        self.assertEqual(StateView(self.owner, self.slots).gather()[1], 9)
        np.testing.assert_array_equal(saved_primal, [2, 0, 0, 0])
        self.assertFalse(np.shares_memory(saved_primal, self.values))

    def test_incarnation_and_identity_cannot_alias(self):
        for bad in (Slot(Coordinate(11, 0, 8), 0), Slot(self.b, 0), Slot(self.a, 9)):
            with self.assertRaises(ValueError):
                StateView(self.owner, [bad])
        with self.assertRaises(ValueError):
            StateOwner(np.zeros(2), [self.a, Coordinate(11, 0, 8)])
        with self.assertRaises(ValueError):
            Slot(self.a, -1)

    def test_support_activity_capacity_are_distinct(self):
        view = StateView(self.owner, self.slots)
        self.assertTrue(view.slots[1].support)  # zero/inactive retains support
        self.assertTrue(view.slots[3].capacity)
        self.assertFalse(view.slots[3].support)
        for bad in (Slot(self.a, 0, capacity=False), Slot(self.a, 0, support=False, activity=True)):
            with self.assertRaises(ValueError):
                StateView(self.owner, [bad])
        with self.assertRaises(ValueError):
            StateView(self.owner, [Slot(self.a, 0), Slot(self.a, 0, activity=True)])

    def test_exact_output_ownership_and_snapshot(self):
        view = StateView(self.owner, self.slots)
        saved = self.values.copy()
        out = view.assemble([("p0", self.a, 3), ("p1", self.a, 4), ("p2", self.b, 8)], [self.b, self.a])
        np.testing.assert_array_equal(out, [8, 7])
        np.testing.assert_array_equal(self.values, saved)
        self.assertFalse(np.shares_memory(out, self.values))
        for contributions, outputs in [([("p0", self.a, 1), ("p0", self.a, 2)], [self.a]),
                                       ([("p0", self.b, 1)], [self.a]),
                                       ([], [self.a, self.a]), ([], [self.reserve])]:
            with self.assertRaises(ValueError):
                view.assemble(contributions, outputs)


class PackingChecks(unittest.TestCase):
    def test_permutation_determinism_and_no_identity_merge(self):
        fs = [Footprint(Coordinate(i, 0, 0), frozenset({0, 1}), frozenset({i})) for i in range(4)]
        expected = pack(fs, width=2)
        self.assertEqual(len(expected), 2)
        self.assertEqual(sorted(c for group in expected for c in group), [f.coordinate for f in fs])
        for perm in itertools.permutations(fs):
            self.assertEqual(pack(perm, width=2), expected)
        with self.assertRaises(ValueError):
            pack([fs[0], fs[0]])
        with self.assertRaises(ValueError):
            pack(fs, width=0)

    def test_tagged_support_empty_and_unknown_program(self):
        a = Footprint(Coordinate(0, 0, 0), frozenset({5}), frozenset())
        b = Footprint(Coordinate(1, 0, 0), frozenset(), frozenset({5}))
        c = Footprint(Coordinate(2, 0, 0), frozenset(), frozenset(), program="other")
        self.assertEqual(pack([c, b, a]), [(a.coordinate,), (b.coordinate,), (c.coordinate,)])
        with self.assertRaises(ValueError):
            Footprint(a.coordinate, frozenset({-1}), frozenset())


class PortChecks(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(6)
        self.hs = [rng.normal(size=w) for w in (2, 3, 1)]
        self.es = [rng.normal(size=(2, len(h))) for h in self.hs]
        self.ds = [rng.normal(size=(len(h), 2)) for h in self.hs]
        self.a = np.array([[0., 2., -1.], [3., 0., 4.], [1., -2., 0.]])

    def test_heterogeneous_distinct_maps_against_scalar_oracle(self):
        fields = [h ** 2 for h in self.hs]
        got = local_port_transport(self.hs, self.es, self.ds, self.a, fields)
        for i, (h, d) in enumerate(zip(self.hs, self.ds)):
            expected = h ** 2 + sum(self.a[i, j] * (d @ (self.es[j] @ self.hs[j])) for j in range(3))
            np.testing.assert_allclose(got[i], expected, atol=1e-14)
        self.assertEqual([x.shape for x in got], [(2,), (3,), (1,)])

    def test_independent_private_basis_changes(self):
        qs = [np.array([[2., 1.], [0., 3.]]),
              np.array([[1., .2, 0.], [0., 2., .1], [.3, 0., 1.]]), np.array([[4.]])]
        got = local_port_transport(self.hs, self.es, self.ds, self.a)
        hp = [q @ h for q, h in zip(qs, self.hs)]
        ep = [e @ np.linalg.inv(q) for e, q in zip(self.es, qs)]
        dp = [q @ d for q, d in zip(qs, self.ds)]
        transformed = local_port_transport(hp, ep, dp, self.a)
        for q, old, new in zip(qs, got, transformed):
            np.testing.assert_allclose(new, q @ old, atol=1e-13)

    def test_torch_cpu_oracle_and_independent_parameter_gradients(self):
        import torch
        hs = [torch.tensor(h, dtype=torch.float64, requires_grad=True) for h in self.hs]
        es = [torch.tensor(e, dtype=torch.float64, requires_grad=True) for e in self.es]
        ds = [torch.tensor(d, dtype=torch.float64, requires_grad=True) for d in self.ds]
        a = torch.tensor(self.a, dtype=torch.float64, requires_grad=True)
        msg = torch.stack([e @ h for e, h in zip(es, hs)])
        incoming = a @ msg
        out = [d @ incoming[i] for i, d in enumerate(ds)]
        expected = local_port_transport(self.hs, self.es, self.ds, self.a)
        for actual, wanted in zip(out, expected):
            np.testing.assert_allclose(actual.detach().numpy(), wanted, atol=1e-14)
        loss = sum((x ** 2).sum() for x in out)
        loss.backward()
        for tensor in hs + es + ds + [a]:
            self.assertEqual(tensor.device.type, "cpu")
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(float(tensor.grad.abs().sum()), 0)
        # Verify one independent encoder parameter by finite differences.
        step = 1e-6
        def score(shift):
            changed = [e.copy() for e in self.es]
            changed[1][0, 2] += shift
            return sum((x ** 2).sum() for x in local_port_transport(self.hs, changed, self.ds, self.a))
        finite = (score(step) - score(-step)) / (2 * step)
        self.assertAlmostEqual(float(es[1].grad[0, 2]), float(finite), places=5)

    def test_port_admission(self):
        for a in (np.ones((2, 2)), np.full((3, 3), np.nan)):
            with self.assertRaises(ValueError):
                local_port_transport(self.hs, self.es, self.ds, a)
        with self.assertRaises(ValueError):
            local_port_transport(self.hs, self.es[:-1], self.ds, self.a)


if __name__ == "__main__":
    unittest.main(verbosity=2)
