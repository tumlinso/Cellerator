"""Focused checks for dynamic geometry and local state coupling."""
import unittest

from cellerator.spatial import SpatialIndex, diffuse


class SpatialIndexTests(unittest.TestCase):
    def test_radius_queries_follow_incremental_cell_moves(self):
        index = SpatialIndex({"a": (0.0, 0.0), 8: (0.5, 0.0), "c": (3.0, 0.0)},
                             search_radius=1.0)
        self.assertEqual([neighbor.identity for neighbor in index.query("a")], [8])
        old_structure = index.structure_generation
        index.move(8, (2.5, 0.0))
        self.assertEqual(index.structure_generation, old_structure)
        self.assertEqual(index.value_generation, 1)
        self.assertEqual(index.query("a"), ())
        self.assertEqual([neighbor.identity for neighbor in index.query("c")], [8])
        removed = index.remove(8)
        self.assertEqual(tuple(removed), (2.5, 0.0))
        self.assertNotIn(8, index.identities)

    def test_spatial_graph_is_sparse_state_gated_and_snapshot_bound(self):
        index = SpatialIndex({"left": (0.0, 0.0), "middle": (0.2, 0.0),
                              "far": (8.0, 0.0)}, search_radius=0.5)
        graph = index.interaction_graph(
            lambda source, target, distance, context: source * context["gain"],
            active_if=lambda state, _: state.get("active", False),
        )
        self.assertEqual(len(graph), 2)
        state = {"left": 1.0, "middle": 2.0, "far": 7.0, "active": True}
        result = graph.evaluate(state, context={"gain": 3.0})
        self.assertEqual(result.writes, {"middle": 3.0, "left": 6.0})
        index.move("middle", (5.0, 0.0))
        refreshed = index.interaction_graph(lambda source, target, distance, ctx: source)
        self.assertEqual(len(refreshed), 0)
        self.assertEqual(len(graph), 2)  # explicit snapshot, never silently stale-update

    def test_graph_diffusion_conserves_total_and_checks_population(self):
        index = SpatialIndex({0: (0.0,), 1: (0.5,), 2: (2.0,)}, search_radius=1.0)
        before = {0: 3.0, 1: 0.0, 2: 5.0}
        after = diffuse(index, before, dt=0.1, diffusivity=1.0)
        self.assertAlmostEqual(sum(after.values()), sum(before.values()), places=12)
        self.assertGreater(after[1], before[1])
        self.assertEqual(after[2], before[2])
        with self.assertRaises(ValueError):
            diffuse(index, {0: 1.0}, dt=1.0, diffusivity=1.0)


if __name__ == "__main__":
    unittest.main()
