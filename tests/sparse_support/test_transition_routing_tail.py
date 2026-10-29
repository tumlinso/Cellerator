"""Adversarial check for the legacy all-B routing role in shared mode."""
from __future__ import annotations

import os
import unittest

import torch

from cellerator.torch.mechanism import guarded_step
from cellerator.torch.routing import (
    RankedGeneSlice,
    prepare_sparse_slice,
)
from cellerator.torch.ops import sparse_latent_transition
from cellerator.torch.structure import SupportSpec


@unittest.skipUnless(os.environ.get("CELLERATOR_RUN_CUDA_TESTS") == "1",
                     "requires the scheduler-admitted sparse support CUDA run")
class TransitionRoutingTailTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = __import__("cellerator.torch._torch", fromlist=["*"])
        cls.device = torch.cuda.current_device()

    def test_shared_mode_keeps_B_tail_in_routing_but_out_of_transition(self):
        module = prepare_sparse_slice(
            RankedGeneSlice,
            SupportSpec((10, 11), (0, 1), (2, 2), 1),
            entities=16, mechanisms=2, rank_count=4, device=self.device,
            parameter_axis_words=(131, 132, 141, 142, 151, 152, 161, 162),
            seed=29, transition_mode="shared")
        owner_leaf = module.parameters_flat
        optimizer = torch.optim.Adam(
            [owner_leaf], lr=10.0, betas=(0.0, 0.0), eps=1e-8)

        entity_ids = torch.arange(16, device=owner_leaf.device, dtype=torch.int64)[:, None]

        def postings_for_all_entities():
            with torch.no_grad(), module._pinned_forward():
                module.affected_mechanism_workset(entity_ids)
                return module._mechanism_route_postings

        before_postings = postings_for_all_entities()
        candidates = [entity for entity, posting in enumerate(before_postings)
                      if 1 not in posting]
        self.assertTrue(candidates, "fixture must include an entity routed without B tail block 1")
        entity = candidates[0]

        transition_a = module._blocks["transition_A"]
        transition_b = module._blocks["transition_B"]
        state = torch.zeros((1, 2, transition_a.shape[1]),
                            device=owner_leaf.device, dtype=torch.float16)
        inputs = torch.ones((1, transition_b.shape[2]),
                            device=owner_leaf.device, dtype=torch.float16)
        offsets = torch.tensor([0, 1], device=owner_leaf.device, dtype=torch.int64)
        affected = torch.tensor([1], device=owner_leaf.device, dtype=torch.int32)

        with torch.no_grad(), module._pinned_forward():
            transition_before = sparse_latent_transition(
                module, state, inputs, offsets, affected).detach().clone()
            entity_vector = module.block("entity_embedding")[entity].detach().float()
        self.assertGreater(float(entity_vector.abs().sum().item()), 0.0)

        # Publish a stock-Adam update to only B[1, 0, :]. Its direction raises
        # the selected entity's routing response while preserving block zero.
        gradient = torch.zeros_like(owner_leaf)
        block_shape = transition_b.shape
        row_begin = transition_b.offset + (1 * block_shape[1]) * block_shape[2]
        row_end = row_begin + block_shape[2]
        gradient[row_begin:row_end] = -entity_vector.sign()
        before_values = owner_leaf.detach().clone()
        owner_leaf.grad = gradient
        self.assertTrue(guarded_step(module, optimizer))
        self.assertIs(module.parameters_flat, owner_leaf)

        changed = owner_leaf.detach() != before_values
        expected_changed = torch.zeros_like(changed)
        expected_changed[row_begin:row_end] = True
        self.assertTrue(torch.equal(changed, expected_changed),
                        "optimizer publication must change only the selected B-tail row")

        after_postings = postings_for_all_entities()
        self.assertNotIn(1, before_postings[entity])
        self.assertIn(1, after_postings[entity])

        with torch.no_grad(), module._pinned_forward():
            transition_after = sparse_latent_transition(
                module, state, inputs, offsets, affected).detach()
        self.assertTrue(torch.equal(transition_before, transition_after),
                        "shared transition math must continue to use B block zero")


if __name__ == "__main__":
    unittest.main()
