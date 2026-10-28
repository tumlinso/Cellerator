"""GPU-gated integration coverage for SparseSupportModule's native route cache."""
from __future__ import annotations

import os
import unittest

import torch

from cellerator.torch.mechanism import guarded_step
from cellerator.torch.routing import RankedGeneSlice, prepare_sparse_slice
from cellerator.torch.structure import SupportSpec


@unittest.skipUnless(os.environ.get("CELLERATOR_RUN_CUDA_TESTS") == "1",
                     "requires the scheduler-admitted sparse support CUDA run")
class SparseModulePreparedRouteIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = __import__("cellerator.torch._torch", fromlist=["*"])
        cls.device = torch.cuda.current_device()

    def _module(self):
        return prepare_sparse_slice(
            RankedGeneSlice,
            SupportSpec((10, 11), (0, 1), (2, 2), 1),
            entities=16, mechanisms=2, rank_count=4, device=self.device,
            parameter_axis_words=(231, 232, 241, 242, 251, 252, 261, 262),
            seed=29, transition_mode="shared")

    def _query(self, module, entity_ids, strategy):
        with torch.no_grad(), module._pinned_forward():
            offsets, mechanisms = module.affected_mechanism_workset(
                entity_ids, strategy=strategy)
            return offsets.detach().clone(), mechanisms.detach().clone()

    def test_prepared_queries_match_python_reuse_and_rebuild_after_publication(self):
        module = self._module()
        owner_leaf = module.parameters_flat
        optimizer = torch.optim.Adam([owner_leaf], lr=1e-2)
        device = owner_leaf.device
        first_ids = torch.tensor(
            [[0, 1, 1, -1], [2, 2, 3, -1], [-1, -1, -1, -1]],
            dtype=torch.int64, device=device)
        second_ids = torch.tensor(
            [[4, 5, -1], [6, 6, 7], [8, -1, 9]],
            dtype=torch.int32, device=device)

        expected_first = self._query(module, first_ids, "python")
        actual_first = self._query(module, first_ids, "prepared")
        actual_second = self._query(module, second_ids, "prepared")
        expected_second = self._query(module, second_ids, "python")
        self.assertTrue(torch.equal(actual_first[0], expected_first[0]))
        self.assertTrue(torch.equal(actual_first[1], expected_first[1]))
        self.assertTrue(torch.equal(actual_second[0], expected_second[0]))
        self.assertTrue(torch.equal(actual_second[1], expected_second[1]))
        self.assertEqual(module.application_state["prepared_route_prepare_count"], 1)
        self.assertEqual(module.application_state["prepared_route_query_count"], 2)
        self.assertEqual(module.application_state["prepared_route_count_syncs"], 2)
        old_cache = module._prepared_mechanism_route_cache
        old_router = old_cache._router
        old_identity = old_cache.identity
        self.assertIsNotNone(old_router)
        self.assertEqual(old_identity.route_generation, module._mechanism_route_generation)

        generation_before = self.native.sparse_generation(module._native)
        gradient = torch.zeros_like(owner_leaf)
        gradient[module._blocks["transition_B"].offset] = 1.0
        owner_leaf.grad = gradient
        self.assertTrue(guarded_step(module, optimizer))
        self.assertIs(module.parameters_flat, owner_leaf)
        generation_after = self.native.sparse_generation(module._native)
        self.assertGreater(generation_after, generation_before)

        actual_after_update = self._query(module, first_ids, "prepared")
        expected_after_update = self._query(module, first_ids, "python")
        self.assertTrue(torch.equal(actual_after_update[0], expected_after_update[0]))
        self.assertTrue(torch.equal(actual_after_update[1], expected_after_update[1]))
        new_cache = module._prepared_mechanism_route_cache
        self.assertIsNot(new_cache._router, old_router)
        self.assertEqual(new_cache.identity.route_generation, generation_after)
        self.assertEqual(module.application_state["prepared_route_prepare_count"], 2)
        self._query(module, second_ids, "prepared")
        self.assertEqual(module.application_state["prepared_route_prepare_count"], 2)
        generation_router = new_cache._router

        module.stage_promotion(
            12, 0, 1, discovery_round_id=9,
            scorer_generation=generation_after, proposal_score=0.5)
        module.publish_support(2)
        self.assertEqual(module.structure_epoch, 2)
        actual_after_structure = self._query(module, first_ids, "prepared")
        expected_after_structure = self._query(module, first_ids, "python")
        self.assertTrue(torch.equal(actual_after_structure[0], expected_after_structure[0]))
        self.assertTrue(torch.equal(actual_after_structure[1], expected_after_structure[1]))
        structure_cache = module._prepared_mechanism_route_cache
        self.assertIsNot(structure_cache._router, generation_router)
        self.assertEqual(structure_cache.identity.structure_epoch, module.structure_epoch)
        self.assertEqual(module.application_state["prepared_route_prepare_count"], 3)


if __name__ == "__main__":
    unittest.main()
