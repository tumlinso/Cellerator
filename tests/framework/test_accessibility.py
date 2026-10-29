import unittest

from cellerator.experimental.accessibility import (
    AccessibilityRule,
    AccessibilityState,
    HysteresisThreshold,
)


class AccessibilityTests(unittest.TestCase):
    def test_required_forbidden_open_bits_and_revision_provenance(self):
        state = AccessibilityState([
            AccessibilityRule("growth", ("open_chromatin", "signal"), ("repressor",), revision=3),
            AccessibilityRule("repair", ("open_chromatin",), revision=2, initially_open=False),
        ], structure_revision=8)
        self.assertFalse(state.eligible("growth"))
        self.assertEqual(state.set_inputs((("open_chromatin", True), ("signal", True))), 2)
        self.assertTrue(state.eligible("growth"))
        self.assertFalse(state.eligible("repair"))
        self.assertTrue(state.set_input("repressor", True, provenance="induced"))
        self.assertFalse(state.eligible("growth"))
        self.assertEqual(state.eligibility("growth").structure_revision, 8)
        self.assertEqual(state.eligibility("growth").program_revision, 3)
        self.assertEqual(state.changes[-1].provenance, "induced")

    def test_forks_share_structure_but_changes_are_branch_local(self):
        founder = AccessibilityState([
            AccessibilityRule("program", ("competence",), initially_open=True)
        ], structure_revision=11)
        founder.set_input("competence", True)
        sister_a, sister_b = founder.fork(), founder.fork()
        self.assertTrue(founder.shares_structure_with(sister_a))
        self.assertTrue(sister_a.shares_structure_with(sister_b))
        self.assertTrue(sister_a.eligible("program"))
        sister_a.set_input("competence", False, provenance="asymmetric inheritance")
        sister_a.set_open("program", False, provenance="differentiation")
        self.assertFalse(sister_a.eligible("program"))
        self.assertTrue(sister_b.eligible("program"))
        self.assertTrue(founder.eligible("program"))
        self.assertEqual(sister_a.changes[-1].provenance, "differentiation")
        self.assertEqual(sister_a.generation, 3)

    def test_batch_update_is_transactional_and_gate_accepts_context_inputs(self):
        state = AccessibilityState([
            AccessibilityRule("edge", ("signal", "competence"), ("blocked",))
        ])
        gate = state.gate("edge")
        self.assertFalse(gate({}, {"accessibility_inputs": ("signal",)}))
        self.assertTrue(gate({}, {"accessibility_inputs": ("signal", "competence")}))
        before = state.generation
        with self.assertRaises(KeyError):
            state.set_inputs((("signal", True), ("unknown", True)))
        self.assertEqual(state.generation, before)
        self.assertTrue(state.set_inputs((("signal", True), ("competence", True)), provenance="cue"))
        self.assertTrue(state.eligible("edge"))
        state.set_input("blocked", True)
        self.assertFalse(state.eligible("edge"))

    def test_threshold_hysteresis_dwell_and_time_validation(self):
        gate = HysteresisThreshold(0.8, 0.2, minimum_dwell=2.0)
        self.assertFalse(gate.update(False, 0.79, 1.0))
        self.assertTrue(gate.update(False, 0.81, 2.0))
        self.assertTrue(gate.update(True, 0.1, 3.0))
        self.assertFalse(gate.update(True, 0.1, 4.1))
        with self.assertRaises(ValueError):
            gate.update(False, 0.9, 3.0)


if __name__ == "__main__":
    unittest.main()
