"""Independent CUDA checks for sparse owner and support lifecycle boundaries.

Run only in the finite Device Scheduler admission used for sparse support CUDA
tests. This module never selects a GPU or treats CUDA visibility as authority.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import torch

from cellerator.torch.mechanism import guarded_step, load_checkpoint, save_checkpoint
from cellerator.torch.sparse_support import (
    RankedGeneSlice,
    SupportSpec,
    compile_sparse_support_ce_geo,
    prepare_sparse_slice,
)


@unittest.skipUnless(os.environ.get("CELLERATOR_RUN_CUDA_TESTS") == "1",
                     "requires the scheduler-admitted sparse support CUDA run")
class SparseSupportCudaLifecycleExtraTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = __import__("cellerator.torch._torch", fromlist=["*"])
        cls.device = torch.cuda.current_device()

    def _module(self, *, prepared_projection=False):
        support = SupportSpec((10, 11), (0, 1), (2, 2), 1)
        projection = {}
        if prepared_projection:
            compiled = compile_sparse_support_ce_geo(
                support, source_count=3, destination_count=3, projection_id=19)
            projection = {
                "ce_geo_image": compiled["image"],
                "source_group_members": compiled["source_group_members"],
                "destination_group_members": compiled["destination_group_members"],
                "geometry_edge_ids": compiled["geometry_edge_ids_in_support_order"],
                "projection_id": 19,
                "max_cells": 1,
                "max_primal_items": 3,
            }
        module = prepare_sparse_slice(
            RankedGeneSlice,
            support,
            entities=3,
            mechanisms=2,
            rank_count=4,
            device=self.device,
            parameter_axis_words=(31, 32, 41, 42, 51, 52, 61, 62),
            seed=29,
            **projection,
        )
        if prepared_projection:
            module.application_state["prepared_capacity"] = {
                "max_cells": 1,
                "max_primal_items": 3,
                "support_growth_budget": 1,
                "initial_support_edges": 2,
            }
        return module

    @staticmethod
    def _stage_promotion(module, native, *, edge_id=12, source=0, destination=1):
        module.stage_promotion(
            edge_id, source, destination, discovery_round_id=9,
            scorer_generation=native.sparse_generation(module._native),
            proposal_score=0.5,
        )

    def test_live_tape_freezes_structure_but_not_parameter_generation(self):
        module = self._module()
        owner_leaf = module.parameters_flat
        generation_before = self.native.sparse_generation(module._native)
        with module._pinned_forward():
            output = module.block("entity_embedding").sum()
        self.assertEqual(len(module._read_tickets), 1)
        self._stage_promotion(module, self.native)

        with self.assertRaisesRegex(RuntimeError, "read tapes to retire"):
            module.publish_support(2)
        self.assertEqual(module.structure_epoch, 1)
        self.assertEqual(self.native.sparse_generation(module._native), generation_before)
        self.assertIs(module.parameters_flat, owner_leaf)

        output.backward()
        self.assertEqual(len(module._read_tickets), 0)
        module.publish_support(2)
        self.assertEqual(module.structure_epoch, 2)
        self.assertEqual(self.native.sparse_generation(module._native), generation_before)
        self.assertIs(module.parameters_flat, owner_leaf)

    def test_parameter_update_published_on_one_stream_is_visible_on_another(self):
        module = self._module()
        parameter = module.parameters_flat
        optimizer = torch.optim.Adam([parameter], lr=0.125)
        before = parameter.detach().clone()
        generation_before = self.native.sparse_generation(module._native)
        writer = torch.cuda.Stream(device=parameter.device)
        reader = torch.cuda.Stream(device=parameter.device)

        with torch.cuda.stream(writer):
            parameter.grad = torch.ones_like(parameter)
            self.assertTrue(guarded_step(module, optimizer))
        expected = before - 0.125

        with torch.cuda.stream(reader):
            with module._pinned_forward():
                observed = module.parameters_flat.sum()
            observed.backward()

        self.assertTrue(torch.allclose(parameter, expected, atol=1e-7, rtol=0))
        self.assertTrue(torch.allclose(observed, expected.sum(), atol=1e-5, rtol=1e-6))
        self.assertEqual(self.native.sparse_generation(module._native), generation_before + 1)
        self.assertEqual(len(module._read_tickets), 0)

    def test_checkpoint_continues_after_promotion_and_revocation(self):
        source = self._module(prepared_projection=True)
        source_optimizer = torch.optim.Adam([source.parameters_flat], lr=0.01)
        source.parameters_flat.square().sum().backward()
        self.assertTrue(guarded_step(source, source_optimizer))

        self._stage_promotion(source, self.native)
        source.publish_support(2)
        source.stage_revocation(11)
        source.publish_support(3)
        source.application_state["window"] = 7

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sparse-revised-support.pt"
            save_checkpoint(path, source, source_optimizer)

            target = self._module(prepared_projection=True)
            target_optimizer = torch.optim.Adam([target.parameters_flat], lr=0.5)
            target_leaf = target.parameters_flat
            load_checkpoint(path, target, target_optimizer)

        self.assertIs(target.parameters_flat, target_leaf)
        self.assertEqual(target.support.edge_ids, (10, 12))
        self.assertEqual(target.support.sources, (0, 0))
        self.assertEqual(target.support.destinations, (2, 1))
        self.assertEqual(target.structure_epoch, 3)
        self.assertEqual(target.application_state["window"], 7)
        self.assertEqual(target_optimizer.param_groups[0]["lr"], 0.01)
        self.assertTrue(torch.equal(target.parameters_flat, source.parameters_flat))
        self.assertTrue(torch.equal(
            target_optimizer.state[target_leaf]["exp_avg"],
            source_optimizer.state[source.parameters_flat]["exp_avg"],
        ))

        # Continue both copies with the same next gradient. Matching the full
        # state after Adam's next update checks optimizer identity and restored
        # logical support together, rather than only comparing checkpoint bytes.
        next_gradient = torch.linspace(
            -0.25, 0.25, source.parameters_flat.numel(),
            device=source.parameters_flat.device,
        )
        source_optimizer.zero_grad(set_to_none=True)
        target_optimizer.zero_grad(set_to_none=True)
        source.parameters_flat.grad = next_gradient.clone()
        target.parameters_flat.grad = next_gradient.clone()
        self.assertTrue(guarded_step(source, source_optimizer))
        self.assertTrue(guarded_step(target, target_optimizer))

        self.assertTrue(torch.equal(target.parameters_flat, source.parameters_flat))
        self.assertTrue(torch.equal(
            target_optimizer.state[target_leaf]["exp_avg"],
            source_optimizer.state[source.parameters_flat]["exp_avg"],
        ))
        self.assertTrue(torch.equal(
            target_optimizer.state[target_leaf]["exp_avg_sq"],
            source_optimizer.state[source.parameters_flat]["exp_avg_sq"],
        ))


if __name__ == "__main__":
    unittest.main()
