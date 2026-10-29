"""Focused reference checks for sparse state-dependent interactions."""
import unittest

from cellerator.interactions import (
    ActiveWorklist, Interaction, InteractionGraph, NativeInteractionIndex,
    native_available,
)


class InteractionTests(unittest.TestCase):
    def test_overlapping_hyperedges_use_only_sparse_candidates_and_gates(self):
        graph = InteractionGraph.from_hyperedges([
            ("module-a", ("tf", "signal"), ("gene-a",)),
            ("module-b", ("tf", "chromatin"), ("gene-b",)),
            ("unrelated", ("other", "signal"), ("gene-c",)),
        ], lambda values, ctx: sum(values),
            active_if=lambda state, ctx: state.get("ready", False))
        self.assertEqual(tuple(r.id for r in graph.incident("tf")),
                         ("module-a", "module-b"))
        work = graph.worklist(changed=("tf",), state={"ready": True})
        self.assertEqual(tuple(work), ("module-a", "module-b"))
        result = graph.evaluate({"tf": 2, "signal": 3, "chromatin": 5,
                                 "ready": True}, worklist=work)
        self.assertEqual(result.writes, {"gene-a": 5, "gene-b": 7})
        self.assertEqual(result.state["gene-b"], 7)
        self.assertEqual(result.evaluated, ("module-a", "module-b"))

    def test_state_gates_filter_active_work_without_removing_structure(self):
        interaction = Interaction.directed("regulation", "tf", "target",
                                           lambda values, _: values[0] * 2,
                                           active_if=lambda state, _: state["tf"] > 0)
        graph = InteractionGraph((interaction,))
        self.assertEqual(len(graph), 1)
        self.assertEqual(graph.worklist(changed=("tf",), state={"tf": 0}).drain(), ())
        result = graph.evaluate({"tf": 0}, changed=("tf",))
        self.assertEqual(result.evaluated, ())
        self.assertEqual(result.skipped, ())

    def test_bipartite_relation_preserves_source_order_and_reduces_writers(self):
        bipartite = InteractionGraph.from_bipartite(
            [("a", "x"), ("b", "x"), ("b", "y")],
            lambda values, _: sum(values))
        graph = InteractionGraph((*bipartite.interactions,
            Interaction.directed("extra-writer", "c", "x", lambda values, _: values[0])))
        self.assertEqual(graph[("bipartite", "x")].inputs, ("a", "b"))
        with self.assertRaisesRegex(ValueError, "multiple active interactions"):
            graph.evaluate({"a": 1, "b": 2, "c": 4})
        result = graph.evaluate({"a": 1, "b": 2, "c": 4}, reducer="sum")
        self.assertEqual(result.writes, {"x": 7, "y": 2})

    def test_worklist_is_stable_and_deduplicates(self):
        work = ActiveWorklist(("b", "a", "b"))
        self.assertEqual(work.drain(), ("b", "a"))
        self.assertEqual(len(work), 0)

    def test_structural_errors_are_explicit(self):
        with self.assertRaisesRegex(ValueError, "duplicate interaction"):
            InteractionGraph((
                Interaction.directed("x", "a", "b", lambda x, _: x),
                Interaction.directed("x", "b", "c", lambda x, _: x),
            ))
        with self.assertRaisesRegex(ValueError, "two input participants"):
            Interaction.hyperedge("h", ("a",), ("b",), lambda x, _: x)
        with self.assertRaisesRegex(ValueError, "duplicate bipartite edge"):
            InteractionGraph.from_bipartite((("a", "b"), ("a", "b")), lambda x, _: x)

    @unittest.skipUnless(native_available(), "optional native modeling component is not built")
    def test_native_sparse_index_and_incidence_plan_bridge(self):
        graph = InteractionGraph.from_hyperedges([
            ("joint", ("tf/a", "signal/b"), ("gene/c",)),
            ("other", ("tf/a", "chromatin/d"), ("gene/e",)),
        ], lambda values, _: values)
        index = NativeInteractionIndex(graph, structure_epoch=4)
        work = index.query(changed=("signal/b",), state={})
        self.assertEqual(work.drain(), (("joint",)))
        self.assertEqual(index.structure_epoch, 4)
        index.set_active(("other",), False)
        self.assertEqual(index.query(changed=("tf/a",), state={}).drain(), (("joint",)))

        import numpy as np
        plan = index.plan(("joint",),
                          ("tf/a", "signal/b", "gene/c", "chromatin/d", "gene/e"))
        values = np.asarray([2.0, 3.0, 0.0, 5.0, 0.0], dtype=np.float32)
        np.testing.assert_array_equal(plan.gather(values), np.asarray([2.0, 3.0, 0.0], dtype=np.float32))
        np.testing.assert_array_equal(plan.joint_reduce(values), np.asarray([5.0], dtype=np.float32))
        np.testing.assert_array_equal(
            plan.joint_reduce_scatter(values, input_role=NativeInteractionIndex.INPUT_ROLE,
                                      output_role=NativeInteractionIndex.OUTPUT_ROLE),
            np.asarray([0.0, 0.0, 5.0, 0.0, 0.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
