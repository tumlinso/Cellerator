import unittest

from cellerator.execution import (
    DeviceSpec, ExpertDemand, ExpertPlacement, ExpertSpec, ExpertWorklist,
    ExecutionStrategy, HardwareTopology, LinkSpec, MemoryCategory, ParameterKey,
    PlacementPlanner, PlanningError, RollingDemandPredictor, SelectedWork,
    UnreachableDevice,
)


class TopologyPlanningTests(unittest.TestCase):
    def topology(self, capacity=100000):
        return HardwareTopology([
            DeviceSpec("host", kind="host", memory_bytes=capacity),
            DeviceSpec("a", memory_bytes=capacity),
            DeviceSpec("b", memory_bytes=capacity),
        ], [LinkSpec("host", "a", 100, bidirectional=True),
            LinkSpec("host", "b", 100, bidirectional=True),
            LinkSpec("a", "b", 100, bidirectional=True)])

    def test_route_accounts_for_direction_and_size_dependent_path(self):
        topology = HardwareTopology([DeviceSpec(name) for name in ("a", "b", "c")], [
            LinkSpec("a", "b", 1000, latency_seconds=1),
            LinkSpec("a", "c", 10), LinkSpec("c", "b", 10),
        ])
        self.assertEqual(topology.route("a", "b", 1).path, ("a", "c", "b"))
        self.assertEqual(topology.route("a", "b", 100).path, ("a", "b"))
        with self.assertRaises(UnreachableDevice):
            topology.route("b", "a", 1)
        self.assertEqual(topology.transfer_time("a", "a", 100), 0)

    def test_compact_worklist_preserves_occurrences_versions_and_inverse(self):
        first, second = ParameterKey("shared", 1), ParameterKey("shared", 2)
        selected = (SelectedWork(first, "a", "r0", 0.25),
                    SelectedWork(second, "b", "r1", 1),
                    SelectedWork(first, "a", "r0", 0.75))
        work = ExpertWorklist.from_selections(selected)
        self.assertEqual(work.expert_keys, (first, second))
        self.assertEqual(work.expert_offsets, (0, 2, 3))
        self.assertEqual(work.dispatch_to_selection, (0, 2, 1))
        for original, dispatch in enumerate(work.selection_to_dispatch):
            self.assertEqual(work.dispatch_to_selection[dispatch], original)
        self.assertEqual(work.selections, selected)
        demands = work.demands(activation_bytes_per_row=4, output_bytes_per_row=8)
        self.assertEqual(demands[0].work_ids, ("r0", "r0"))
        self.assertEqual((demands[0].activation_bytes, demands[0].uses), (8, 2))
        with self.assertRaises(ValueError):
            ExpertWorklist(selected, (first, second), (0, 2, 3), (0, 0, 1))
        with self.assertRaises(ValueError):
            ExpertWorklist(selected, (first, second), (0, 2, 3), (0, 1, 2))

    def test_small_activations_move_to_resident_expert_without_weight_copy(self):
        key = ParameterKey("expert", 1)
        demand = ExpertDemand(key, "a", 10, 10, work_ids=("row",))
        planner = PlacementPlanner(self.topology(), [ExpertSpec(key, 1000)], [ExpertPlacement(key, "b")])
        plan = planner.plan([demand])
        chosen = plan.decisions[0]
        self.assertEqual(chosen.strategy, ExecutionStrategy.REMOTE_ACTIVATIONS)
        self.assertEqual(chosen.execution_device, "b")
        self.assertIs(chosen.demand, demand)
        self.assertAlmostEqual(chosen.estimated_seconds, 0.2)
        self.assertEqual([transfer.category for transfer in chosen.chosen.transfers],
                         [MemoryCategory.ACTIVATION, MemoryCategory.ACTIVATION])

    def test_large_activation_movement_prefers_weight_transfer(self):
        key = ParameterKey("expert", 1)
        planner = PlacementPlanner(self.topology(), [ExpertSpec(key, 1000)], [ExpertPlacement(key, "b")])
        plan = planner.plan([ExpertDemand(key, "a", 10000, 10000)])
        chosen = plan.decisions[0]
        self.assertEqual(chosen.strategy, ExecutionStrategy.MOVE_WEIGHTS)
        self.assertAlmostEqual(chosen.estimated_seconds, 10)
        self.assertEqual(plan.projected_memory["a"][MemoryCategory.PARAMETER], 0)
        self.assertEqual(plan.peak_memory["a"][MemoryCategory.PARAMETER], 1000)

    def test_reuse_can_propose_replication_with_explicit_dependency(self):
        key = ParameterKey("expert", 7)
        demand = ExpertDemand(key, "a", 10, 10)
        planner = PlacementPlanner(self.topology(), [ExpertSpec(key, 1000)],
                                   [ExpertPlacement(key, "host", executable=False)])
        plan = planner.plan([demand, demand], repeats={(key, "a"): 5})
        self.assertEqual(plan.decisions[0].strategy, ExecutionStrategy.REPLICATE)
        self.assertEqual(plan.decisions[1].strategy, ExecutionStrategy.LOCAL)
        self.assertEqual(plan.decisions[1].chosen.dependencies, (0,))
        self.assertEqual(plan.proposed_replicas[0].key, key)
        self.assertFalse(plan.proposed_replicas[0].ready)
        self.assertEqual(plan.projected_memory["a"][MemoryCategory.PARAMETER], 1000)
        # Planning did not rewrite the actual placement snapshot.
        self.assertEqual(len(planner.placements), 1)

    def test_missing_or_pending_exact_revision_never_substitutes_expert(self):
        old, requested = ParameterKey("expert", 1), ParameterKey("expert", 2)
        planner = PlacementPlanner(self.topology(), [ExpertSpec(old, 10), ExpertSpec(requested, 10)],
                                   [ExpertPlacement(old, "b"), ExpertPlacement(requested, "a", ready=False)])
        demand = ExpertDemand(requested, "a", 1, 1)
        with self.assertRaises(PlanningError) as caught:
            planner.plan([demand])
        self.assertIs(caught.exception.demand, demand)

    def test_optimizer_and_cell_state_bytes_can_prevent_parameter_materialization(self):
        key = ParameterKey("expert", 1)
        usage = {"a": {MemoryCategory.OPTIMIZER: 50, MemoryCategory.CELL_STATE: 40},
                 "host": {MemoryCategory.PARAMETER: 20}}
        planner = PlacementPlanner(self.topology(capacity=100), [ExpertSpec(key, 20)],
            [ExpertPlacement(key, "host", executable=False)], memory_usage=usage)
        with self.assertRaises(PlanningError) as caught:
            planner.plan([ExpertDemand(key, "a", 1, 1)])
        self.assertTrue(all(candidate.reason == "insufficient memory" for candidate in caught.exception.candidates))

    def test_existing_local_copy_and_compute_scale_affect_cost(self):
        key = ParameterKey("expert", 1)
        topology = HardwareTopology([DeviceSpec("a", memory_bytes=1000),
                                    DeviceSpec("b", memory_bytes=1000, compute_scale=100)],
                                   [LinkSpec("a", "b", 100, bidirectional=True)])
        planner = PlacementPlanner(topology, [ExpertSpec(key, 10, compute_seconds=10)],
                                   [ExpertPlacement(key, "a"), ExpertPlacement(key, "b")])
        plan = planner.plan([ExpertDemand(key, "a", 1, 1)])
        self.assertEqual(plan.decisions[0].strategy, ExecutionStrategy.REMOTE_ACTIVATIONS)
        self.assertAlmostEqual(plan.decisions[0].estimated_seconds, 0.12)

    def test_remote_return_output_requires_origin_memory(self):
        key = ParameterKey("expert", 1)
        usage = {"a": {MemoryCategory.CELL_STATE: 95},
                 "b": {MemoryCategory.PARAMETER: 10}}
        planner = PlacementPlanner(self.topology(capacity=100), [ExpertSpec(key, 10)],
                                   [ExpertPlacement(key, "b")], memory_usage=usage)
        with self.assertRaises(PlanningError) as caught:
            planner.plan([ExpertDemand(key, "a", 1, 10)])
        remote = next(candidate for candidate in caught.exception.candidates
                      if candidate.strategy is ExecutionStrategy.REMOTE_ACTIVATIONS)
        self.assertFalse(remote.feasible)
        self.assertEqual(remote.activation_reservations, (("b", 11), ("a", 10)))

    def test_demand_forecasts_are_bounded_advisory_and_do_not_gate_work(self):
        first, second = ParameterKey("a", 1), ParameterKey("b", 1)
        predictor = RollingDemandPredictor(decay=0.5, max_entries=1)
        demand = ExpertDemand(first, "origin", 1, 1, uses=4)
        predictor.observe([demand])
        self.assertEqual(predictor.forecast()[0].expected_uses, 2)
        predictor.observe([ExpertDemand(second, "origin", 1, 1)])
        self.assertEqual([item.key for item in predictor.forecast()], [second])
        self.assertEqual(demand.key, first)
        self.assertEqual(demand.uses, 4)


if __name__ == "__main__":
    unittest.main()
