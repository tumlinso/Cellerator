"""Focused CPU delta-ledger qualification; run directly with Python."""
import copy
import json
import unittest
import numpy as np

from delta import DeltaLedger


class DeltaTests(unittest.TestCase):
    def test_gradual_subthreshold_steps_accumulate_from_last_transmission(self):
        ledger = DeltaLedger([[2., -3.]], [0., 0.], tau=0.25)
        for x in ([0.1, 0.], [0.2, 0.]):
            y, bound = ledger.update(x)
            np.testing.assert_array_equal(y, [0.])
            np.testing.assert_allclose(bound, [2 * x[0]])
            np.testing.assert_array_equal(ledger.x_sent, [0., 0.])
        y, bound = ledger.update([0.3, 0.1])
        np.testing.assert_allclose(y, [0.6])
        np.testing.assert_allclose(bound, [0.3])
        np.testing.assert_array_equal(ledger.x_sent, [0.3, 0.])

    def test_threshold_equality_is_not_transmitted(self):
        ledger = DeltaLedger([[1.]], [0.], tau=0.25)
        y, bound = ledger.update([0.25])
        np.testing.assert_array_equal(y, [0.])
        np.testing.assert_array_equal(bound, [0.25])

    def test_tau_zero_agrees_with_direct_with_rounding_tolerance(self):
        rng = np.random.default_rng(22)
        W = rng.normal(size=(4, 7))
        ledger = DeltaLedger(W, np.zeros(7))
        for x in rng.normal(size=(100, 7)):
            y, bound = ledger.update(x)
            np.testing.assert_allclose(y, W @ x, rtol=1e-12, atol=1e-12)
            np.testing.assert_array_equal(bound, np.zeros(4))

    def test_bound_covers_instantaneous_omission(self):
        W = np.array([[2., -4., 1.], [-1., 3., -2.]])
        ledger = DeltaLedger(W, [0., 0., 0.], tau=0.5)
        x = np.array([0.3, 0.4, 0.7])
        y, bound = ledger.update(x)
        expected = np.abs(W) @ np.abs(x - ledger.x_sent)
        np.testing.assert_array_equal(bound, expected)
        np.testing.assert_array_equal(ledger.discrepancy_bound(x), expected)
        self.assertTrue(np.all(np.abs(W @ x - y) <= bound + 1e-14))

    def test_generations_and_actual_weights_invalidate(self):
        for generation in ("parameter_generation", "structure_generation"):
            ledger = DeltaLedger([[2.]], [0.], tau=1.)
            ledger.update([0.2])
            y, bound = ledger.update([0.3], **{generation: 1})
            np.testing.assert_allclose(y, [0.6])
            np.testing.assert_array_equal(bound, [0.])
            np.testing.assert_array_equal(ledger.x_sent, [0.3])
        W = np.array([[2.]])
        ledger = DeltaLedger(W, [0.], tau=1.)
        W[0, 0] = 3.
        y, bound = ledger.update([0.2], W=W)
        np.testing.assert_allclose(y, [0.6])
        np.testing.assert_array_equal(bound, [0.])
        self.assertEqual(ledger.parameter_generation, 0)
        ledger.update([1., 2.], W=[[1., 2.], [3., 4.]], structure_generation=1)
        np.testing.assert_array_equal(ledger.y, [5., 11.])

    def test_owned_snapshots_and_return_values(self):
        W, x = np.array([[2.]]), np.array([1.])
        ledger = DeltaLedger(W, x)
        W[:] = 90
        x[:] = 90
        for array in (ledger.W, ledger.x_sent, ledger.y):
            array[:] = 99
        np.testing.assert_array_equal(ledger.W, [[2.]])
        np.testing.assert_array_equal(ledger.x_sent, [1.])
        np.testing.assert_array_equal(ledger.y, [2.])
        y, bound = ledger.update([2.])
        y[:] = 90
        bound[:] = 90
        np.testing.assert_array_equal(ledger.y, [4.])
        payload = ledger.checkpoint()
        payload["W"][0][0] = 90
        np.testing.assert_array_equal(ledger.W, [[2.]])

    def test_failed_updates_leave_all_state_unchanged(self):
        ledger = DeltaLedger([[2.]], [1.], tau=0.1, parameter_generation=2)
        failures = [dict(x=[np.nan]), dict(x=[1., 2.]),
                    dict(x=[2.], W=[[np.inf]], parameter_generation=3),
                    dict(x=[1., 2.], W=[[1., 2.]]),
                    dict(x=[2.], parameter_generation=1),
                    dict(x=[2.], structure_generation=True),
                    dict(x=[2.], parameter_generation=2.5),
                    dict(x=[1e308], W=[[1e308]], parameter_generation=3)]
        for kwargs in failures:
            with self.subTest(kwargs=kwargs):
                before = ledger.checkpoint()
                with self.assertRaises((ValueError, FloatingPointError)):
                    ledger.update(**kwargs)
                self.assertEqual(before, ledger.checkpoint())
        # Overflow in delta arithmetic must also preserve the old sent vector.
        ledger = DeltaLedger([[0.]], [-1e308])
        before = ledger.checkpoint()
        with self.assertRaises(FloatingPointError):
            ledger.update([1e308])
        self.assertEqual(before, ledger.checkpoint())

    def test_checkpoint_continuation_preserves_recurrence_and_transmitted_state(self):
        ledger = DeltaLedger([[0.1, 0.3]], [0., 0.], tau=0.2)
        for x in ([0.15, 0.05], [0.25, 0.1], [0.3, 0.15]):
            ledger.update(x)
        payload = json.loads(json.dumps(ledger.checkpoint()))
        restored = DeltaLedger.from_checkpoint(payload)
        self.assertEqual(restored.checkpoint(), ledger.checkpoint())
        payload["y"][0] = 999
        for x in ([0.35, 0.2], [0.6, 0.35], [-0.2, 0.45]):
            expected = ledger.update(x)
            actual = restored.update(x)
            for a, b in zip(expected, actual):
                np.testing.assert_array_equal(a, b)
            self.assertEqual(restored.checkpoint(), ledger.checkpoint())
        # A float64 recurrence can differ from direct product. Restoration
        # must retain that stored rounding result, not silently replace it.
        ledger = DeltaLedger([[1.]], [1e16])
        ledger.update([1.])
        self.assertNotEqual(ledger.y[0], (ledger.W @ ledger.x_sent)[0])
        restored = DeltaLedger.from_checkpoint(ledger.checkpoint())
        np.testing.assert_array_equal(restored.y, ledger.y)

    def test_checkpoint_rejects_malformed_payload(self):
        good = DeltaLedger([[1.]], [0.]).checkpoint()
        changes = [("dtype", "float32"), ("schema_version", True),
                   ("W", [[np.nan]]), ("W", [[1., 2.]]),
                   ("x_sent", [0., 0.]), ("y", [0., 1.]),
                   ("y", [np.inf]), ("parameter_generation", -1),
                   ("structure_generation", 0.5), ("tau", -1.),
                   ("W", [["1"]]), ("tau", np.nan)]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                bad = copy.deepcopy(good)
                bad[key] = value
                with self.assertRaises(ValueError):
                    DeltaLedger.from_checkpoint(bad)
        with self.assertRaises(ValueError):
            DeltaLedger.from_checkpoint({})

    def test_constructor_admission(self):
        for W, x, tau in [([], [], 0), ([[1.]], [True], 0),
                          ([[1j]], [0.], 0), ([[1.]], [0.], True),
                          ([[1.]], [0.], np.inf), ([[1.]], [0.], -1)]:
            with self.subTest(W=W, x=x, tau=tau):
                with self.assertRaises(ValueError):
                    DeltaLedger(W, x, tau)


if __name__ == "__main__":
    unittest.main(verbosity=2)
