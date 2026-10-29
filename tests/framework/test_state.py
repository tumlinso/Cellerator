"""Focused contract tests for representation-neutral state machinery."""
import unittest

from cellerator.state import LayoutMap, RaggedState, SparseState, StateReference, StructuredState


class StateTests(unittest.TestCase):
    def test_structured_fields_keep_arbitrary_payloads_and_separate_generations(self):
        opaque = object()
        state = StructuredState({"counts": [1, 2], "memory": opaque})
        self.assertIs(state.get_field("memory"), opaque)
        initial = state.snapshot()
        state["counts"] = [2, 2]
        self.assertEqual((state.structure_generation, state.value_generation), (0, 1))
        state["chromatin"] = {"open": True}
        self.assertEqual((state.structure_generation, state.value_generation), (1, 1))
        self.assertEqual(initial["counts"], [1, 2])
        with self.assertRaises(TypeError):
            initial["new"] = 3

    def test_sparse_updates_use_stable_identity_and_generation_classes(self):
        values = SparseState({"cell-a": [1], "cell-z": [2]})
        values["cell-z"] = [3]
        self.assertEqual((values.structure_generation, values.value_generation), (0, 1))
        values.update_values({"cell-b": [4], "cell-a": [5]})
        self.assertEqual((values.structure_generation, values.value_generation), (1, 2))
        self.assertEqual(values["cell-a"], [5])
        del values["cell-z"]
        self.assertEqual(values.structure_generation, 2)

    def test_layout_round_trips_logical_values_through_physical_reorder(self):
        layout = LayoutMap(("a", "b", "c"), ("c", "a", "b"))
        packed = layout.to_physical({"a": 1, "b": 2, "c": 3})
        self.assertEqual(packed, (3, 1, 2))
        self.assertEqual(layout.to_logical(packed), {"a": 1, "b": 2, "c": 3})
        with self.assertRaises(ValueError):
            LayoutMap(("a", "b"), ("a", "a"))

    def test_ragged_state_keeps_variable_length_segments_without_padding(self):
        state = RaggedState(("cell-1", "cell-2", "cell-3"), (0, 2, 2, 5),
                            ["a", "b", "c", "d", "e"])
        self.assertEqual(state.segment("cell-1"), ["a", "b"])
        self.assertEqual(state.segment("cell-2"), [])
        self.assertEqual(state.segment("cell-3"), ["c", "d", "e"])
        self.assertEqual(state.lengths(), {"cell-1": 2, "cell-2": 0, "cell-3": 3})
        with self.assertRaises(ValueError):
            RaggedState(("x",), (0, 4), [1, 2])

    def test_state_reference_delegates_ownership_and_can_be_read_only(self):
        owner = {"state": [1, 2]}
        ref = StateReference(lambda: owner["state"], lambda value: owner.__setitem__("state", value),
                             identity="cell-9")
        ref.set([3])
        self.assertEqual(ref.get(), [3])
        self.assertEqual(ref.identity, "cell-9")
        with self.assertRaises(TypeError):
            StateReference(lambda: owner["state"]).set([])


if __name__ == "__main__":
    unittest.main()
