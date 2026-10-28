"""CUDA numerical checks; run only inside the admitted finite GPU job.

Set CELLERATOR_RUN_CUDA_TESTS=1 in the scheduler-launched test environment.
This module never selects a GPU or treats CUDA visibility as authorization.
"""
from __future__ import annotations

import os
import tempfile
import unittest
import gc
from pathlib import Path

import torch

from cellerator.torch.mechanism import guarded_step, load_checkpoint, save_checkpoint
from cellerator.torch.sparse_support import (
    GateValues,
    PerCellWorkset,
    RankedGeneSlice,
    SupportSpec,
    learned_gates,
    prepare_sparse_slice,
)


@unittest.skipUnless(os.environ.get("CELLERATOR_RUN_CUDA_TESTS") == "1",
                     "requires the scheduler-admitted sparse support CUDA run")
class SparseSupportCudaTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = __import__("cellerator.torch._torch", fromlist=["*"])
        cls.device = torch.cuda.current_device()

    def _handle(self):
        import numpy as np

        native = self.native
        edge_ids = np.asarray([10, 11], dtype=np.int64)
        offsets = np.asarray([0, 2, 4], dtype=np.int64)
        roles = np.asarray([0, 1, 0, 1], dtype=np.int32)
        participants = np.asarray([1, 3, 2, 3], dtype=np.int64)
        support = native.NativeSupportHandle(
            edge_ids, offsets, roles, participants, 71, 81, 1, 72)
        block_count = 4
        initial = torch.zeros(block_count, dtype=torch.float32)
        blocks = [{"name": "shared", "offset": 0, "shape": [block_count],
                   "numel": block_count}]
        ids = [(100 + index, 200 + index) for index in range(block_count)]
        return native.sparse_create_handle(
            [10, 11], [0, 1], [2, 2], 2, 3, 1,
            [1, 2, 3, 4, 5, 6, 7, 8], ids, initial, blocks,
            self.device, 8, support)

    def _module(self, support=None, transition_mode="shared"):
        support = support or SupportSpec((10, 11), (0, 1), (2, 2), 1)
        return prepare_sparse_slice(
            RankedGeneSlice, support, entities=3, mechanisms=2, rank_count=4,
            device=self.device, parameter_axis_words=(31, 32, 41, 42, 51, 52, 61, 62),
            seed=29, transition_mode=transition_mode)

    def _publish_route(self, module):
        module.stage_promotion(12, 0, 1, discovery_round_id=9,
                               scorer_generation=self.native.sparse_generation(module._native),
                               proposal_score=0.5)
        module.publish_support(2)

    def test_no_grad_scoring_completes_parameter_reader_before_publication(self):
        module = self._module()
        with torch.no_grad():
            scores = module.score_pair_proposals([0], [1])
        self.assertEqual(scores.numel(), 1)
        self.assertEqual(len(module._read_tickets), 0)
        self._publish_route(module)
        self.assertEqual(module.structure_epoch, 2)

    def test_dropped_training_graph_retires_parameter_reader(self):
        module = self._module()
        with module._pinned_forward():
            abandoned = module.block("entity_embedding").sum()
        self.assertEqual(len(module._read_tickets), 1)
        del abandoned
        gc.collect()
        self.assertEqual(len(module._read_tickets), 0)
        self._publish_route(module)
        self.assertEqual(module.structure_epoch, 2)

    def test_exception_in_pinned_forward_retires_parameter_reader(self):
        module = self._module()
        with self.assertRaisesRegex(RuntimeError, "synthetic forward failure"):
            with module._pinned_forward():
                raise RuntimeError("synthetic forward failure")
        self.assertEqual(len(module._read_tickets), 0)
        self._publish_route(module)
        self.assertEqual(module.structure_epoch, 2)

    def test_relation_forward_and_ste_vjp_match_torch_reference(self):
        handle = self._handle()
        device = torch.device("cuda", self.device)
        x = torch.randn(1, 2, 64, device=device, dtype=torch.float16, requires_grad=True)
        weights = torch.tensor([0.7, -0.3], device=device, dtype=torch.float32,
                               requires_grad=True)
        scores = torch.tensor([0.4, -0.2], device=device, dtype=torch.float32,
                              requires_grad=True)
        probabilities = torch.sigmoid(scores)
        admitted = torch.tensor([1, 0], device=device, dtype=torch.uint8)
        offsets = torch.tensor([0, 2], device=device, dtype=torch.int64)
        ordinals = torch.tensor([0, 1], device=device, dtype=torch.int64)
        ids = torch.tensor([10, 11], device=device, dtype=torch.int64)
        gate_indices = torch.tensor([0, 1], device=device, dtype=torch.int64)
        output = self.native.sparse_relation_apply(
            x, weights, probabilities, admitted,
            offsets, ordinals, ids, gate_indices,
            offsets, ordinals, ids, gate_indices,
            ids, ordinals, 1, 2, 1, 2, 1, handle)

        x_ref = x.detach().clone().requires_grad_()
        weights_ref = weights.detach().clone().requires_grad_()
        scores_ref = scores.detach().clone().requires_grad_()
        p_ref = torch.sigmoid(scores_ref)
        q = p_ref + (p_ref * admitted.float() - p_ref).detach()
        expected = torch.zeros((1, 3, 64), device=device, dtype=torch.float32)
        for edge, source in enumerate((0, 1)):
            expected[:, 2] += x_ref[:, source].float() * weights_ref[edge].half().float() * q[edge]
        self.assertTrue(torch.allclose(output, expected, atol=2e-3, rtol=2e-3))
        upstream = torch.randn_like(output)
        (output * upstream).sum().backward()
        (expected * upstream).sum().backward()
        self.assertTrue(torch.allclose(x.grad, x_ref.grad, atol=3e-3, rtol=3e-3))
        self.assertTrue(torch.allclose(weights.grad, weights_ref.grad, atol=3e-3, rtol=3e-3))
        self.assertTrue(torch.allclose(scores.grad, scores_ref.grad, atol=3e-3, rtol=3e-3))
        # Native relation backward returns derivative-candidate gate gradients,
        # including edge 11 whose primal forward gate is closed.
        self.assertNotEqual(float(scores.grad[1].abs()), 0.0)
        self.assertTrue(torch.isfinite(weights.grad).all())

    def test_prepared_ce_geo_forward_and_simt_vjp_match_same_torch_reference(self):
        handle = self._handle()
        compiled = self.native.compile_sparse_support_ce_geo(
            [10, 11], [0, 1], [2, 2], 2, 3, 1, 19)
        self.native.sparse_prepare_ce_geo_n64(
            handle, compiled["image"], compiled["source_group_members"],
            compiled["destination_group_members"], 19, 1, 2,
            compiled["geometry_edge_ids_in_support_order"])
        device = torch.device("cuda", self.device)
        x = torch.randn(1, 2, 64, device=device, dtype=torch.float16, requires_grad=True)
        weights = torch.tensor([0.7, -0.3], device=device, dtype=torch.float32,
                               requires_grad=True)
        scores = torch.tensor([0.4, -0.2], device=device, dtype=torch.float32,
                              requires_grad=True)
        probabilities = torch.sigmoid(scores)
        admitted = torch.tensor([1, 0], device=device, dtype=torch.uint8)
        offsets = torch.tensor([0, 2], device=device, dtype=torch.int64)
        ordinals = torch.tensor([0, 1], device=device, dtype=torch.int64)
        ids = torch.tensor([10, 11], device=device, dtype=torch.int64)
        gate_indices = torch.tensor([0, 1], device=device, dtype=torch.int64)
        output = self.native.sparse_relation_apply(
            x, weights, probabilities, admitted,
            offsets, ordinals, ids, gate_indices,
            offsets, ordinals, ids, gate_indices,
            ids, ordinals, 1, 2, 1, 2, 1, handle)
        x_ref = x.detach().clone().requires_grad_()
        weights_ref = weights.detach().clone().requires_grad_()
        scores_ref = scores.detach().clone().requires_grad_()
        p_ref = torch.sigmoid(scores_ref)
        q = p_ref + (p_ref * admitted.float() - p_ref).detach()
        expected = torch.zeros((1, 3, 64), device=device, dtype=torch.float32)
        for edge, source in enumerate((0, 1)):
            expected[:, 2] += x_ref[:, source].float() * weights_ref[edge].half().float() * q[edge]
        self.assertTrue(torch.allclose(output, expected, atol=3e-3, rtol=3e-3))
        upstream = torch.randn_like(output)
        (output * upstream).sum().backward()
        (expected * upstream).sum().backward()
        self.assertTrue(torch.allclose(x.grad, x_ref.grad, atol=4e-3, rtol=4e-3))
        self.assertTrue(torch.allclose(weights.grad, weights_ref.grad,
                                        atol=4e-3, rtol=4e-3))
        self.assertTrue(torch.allclose(scores.grad, scores_ref.grad,
                                        atol=4e-3, rtol=4e-3))

    def test_latent_transition_and_recurrent_vjp_match_torch_reference(self):
        handle = self._handle()
        device = torch.device("cuda", self.device)
        generator = torch.Generator(device=device).manual_seed(821)
        state = (torch.randn(1, 2, 64, generator=generator, device=device) * 0.1)
        state = state.to(torch.float16).requires_grad_()
        event = (torch.randn(1, 64, generator=generator, device=device) * 0.1)
        event = event.to(torch.float16).requires_grad_()
        a = (torch.randn(2, 64, 64, generator=generator, device=device) * 0.01).requires_grad_()
        b = (torch.randn(2, 64, 64, generator=generator, device=device) * 0.01).requires_grad_()
        bias = (torch.randn(2, 64, generator=generator, device=device) * 0.01).requires_grad_()
        offsets = torch.tensor([0, 1], device=device, dtype=torch.int64)
        affected = torch.tensor([0], device=device, dtype=torch.int32)
        actual = self.native.sparse_latent_transition(
            state, event, offsets, affected, a, b, bias, handle)
        state_ref = state.detach().clone().requires_grad_()
        event_ref = event.detach().clone().requires_grad_()
        a_ref = a.detach().clone().requires_grad_()
        b_ref = b.detach().clone().requires_grad_()
        bias_ref = bias.detach().clone().requires_grad_()
        z = state_ref.float()
        u = event_ref.float()
        ref = z.clone()
        ref[:, 0] = torch.tanh(
            torch.nn.functional.linear(z[:, 0].half().float(), a_ref[0].half().float())
            + torch.nn.functional.linear(u.half().float(), b_ref[0].half().float())
            + bias_ref[0].half().float())
        self.assertTrue(torch.allclose(actual, ref, atol=5e-3, rtol=5e-3))
        upstream = torch.randn_like(actual)
        (actual * upstream).sum().backward()
        (ref * upstream).sum().backward()
        self.assertTrue(torch.allclose(state.grad, state_ref.grad, atol=5e-3, rtol=5e-3))
        self.assertTrue(torch.allclose(event.grad, event_ref.grad, atol=5e-3, rtol=5e-3))
        self.assertTrue(torch.allclose(a.grad, a_ref.grad, atol=5e-3, rtol=5e-3))
        self.assertTrue(torch.allclose(b.grad, b_ref.grad, atol=5e-3, rtol=5e-3))
        self.assertTrue(torch.allclose(bias.grad, bias_ref.grad, atol=5e-3, rtol=5e-3))

    def test_transition_modes_full_forward_and_vjp_match_independent_torch_oracle(self):
        handle = self._handle()
        device = torch.device("cuda", self.device)
        generator = torch.Generator(device=device).manual_seed(1271)
        state = (torch.randn(2, 3, 4, generator=generator, device=device) * 0.2)
        state = state.to(torch.float16).requires_grad_()
        inputs = (torch.randn(2, 2, generator=generator, device=device) * 0.3)
        inputs = inputs.to(torch.float16).requires_grad_()
        matrix_a = (torch.randn(3, 4, 4, generator=generator, device=device) * 0.12)
        matrix_a.requires_grad_()
        matrix_b = (torch.randn(3, 4, 2, generator=generator, device=device) * 0.16)
        matrix_b.requires_grad_()
        bias = (torch.randn(3, 4, generator=generator, device=device) * 0.08)
        bias.requires_grad_()
        offsets = torch.tensor([0, 2, 4], device=device, dtype=torch.int64)
        affected = torch.tensor([0, 2, 1, 2], device=device, dtype=torch.int32)
        upstream = torch.randn(2, 3, 4, generator=generator, device=device)

        for mode in ("shared", "per_mechanism"):
            for value in (state, inputs, matrix_a, matrix_b, bias):
                value.grad = None
            transition_a, transition_b, transition_bias = (
                (matrix_a[0], matrix_b[0], bias[0]) if mode == "shared"
                else (matrix_a, matrix_b, bias))
            actual = self.native.sparse_latent_transition_v2(
                mode, state, inputs, offsets, affected,
                transition_a, transition_b, transition_bias, handle)

            state_ref = state.detach().clone().requires_grad_()
            input_ref = inputs.detach().clone().requires_grad_()
            a_ref = matrix_a.detach().clone().requires_grad_()
            b_ref = matrix_b.detach().clone().requires_grad_()
            bias_ref = bias.detach().clone().requires_grad_()
            expected = state_ref.float().clone()
            affected_rows = ((0, 0), (0, 2), (1, 1), (1, 2))
            for cell, mechanism in affected_rows:
                block = 0 if mode == "shared" else mechanism
                z = state_ref[cell, mechanism].half().float()
                u = input_ref[cell].half().float()
                expected[cell, mechanism] = torch.tanh(
                    torch.mv(a_ref[block].half().float(), z)
                    + torch.mv(b_ref[block].half().float(), u)
                    + bias_ref[block].half().float())
            self.assertTrue(torch.allclose(actual, expected, atol=7e-3, rtol=7e-3))
            (actual * upstream).sum().backward()
            (expected * upstream).sum().backward()
            self.assertTrue(torch.allclose(state.grad, state_ref.grad, atol=8e-3, rtol=8e-3))
            self.assertTrue(torch.allclose(inputs.grad, input_ref.grad, atol=8e-3, rtol=8e-3))
            self.assertTrue(torch.allclose(matrix_a.grad, a_ref.grad, atol=8e-3, rtol=8e-3))
            self.assertTrue(torch.allclose(matrix_b.grad, b_ref.grad, atol=8e-3, rtol=8e-3))
            self.assertTrue(torch.allclose(bias.grad, bias_ref.grad, atol=8e-3, rtol=8e-3))
            self.assertTrue(torch.equal(state.grad[0, 1], upstream[0, 1].half()))
            self.assertTrue(torch.equal(state.grad[1, 0], upstream[1, 0].half()))
            if mode == "shared":
                self.assertTrue(torch.count_nonzero(matrix_a.grad[1:]) == 0)
                self.assertTrue(torch.count_nonzero(matrix_b.grad[1:]) == 0)
                self.assertTrue(torch.count_nonzero(bias.grad[1:]) == 0)

    def test_transition_mode_shape_and_name_validation(self):
        device = torch.device("cuda", self.device)
        state = torch.zeros((1, 2, 4), device=device, dtype=torch.float16)
        inputs = torch.zeros((1, 3), device=device, dtype=torch.float16)
        offsets = torch.tensor([0, 0], device=device, dtype=torch.int64)
        affected = torch.empty((0,), device=device, dtype=torch.int32)
        a = torch.zeros((2, 4, 4), device=device, dtype=torch.float32)
        b = torch.zeros((2, 4, 3), device=device, dtype=torch.float32)
        bias = torch.zeros((2, 4), device=device, dtype=torch.float32)
        handle = self._handle()
        with self.assertRaisesRegex(RuntimeError, "transition_mode"):
            self.native.sparse_latent_transition_v2(
                "implicit", state, inputs, offsets, affected, a, b, bias, handle)
        shared_a_wrong_shape = torch.zeros((4, 3), device=device, dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "shared transition expects"):
            self.native.sparse_latent_transition_v2(
                "shared", state, inputs, offsets, affected,
                shared_a_wrong_shape, b[0], bias[0], handle)
        per_mechanism_a_wrong_shape = torch.zeros((1, 4, 4), device=device,
                                                   dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "per_mechanism transition expects"):
            self.native.sparse_latent_transition_v2(
                "per_mechanism", state, inputs, offsets, affected,
                per_mechanism_a_wrong_shape, b, bias, handle)
        with self.assertRaisesRegex(RuntimeError, "latent state has invalid dtype"):
            self.native.sparse_latent_transition_v2(
                "shared", state.float(), inputs, offsets, affected,
                a[0], b[0], bias[0], handle)

    def test_gate_sampler_keeps_all_positive_and_masks_unsampled_closed_gradients(self):
        module = self._module(SupportSpec(
            (10, 11, 12, 13), (0, 1, 0, 1), (2, 2, 1, 0), 1))
        device = torch.device("cuda", self.device)
        candidate_ids = torch.tensor([10, 11, 12, 13], device=device, dtype=torch.int64)
        candidate_ordinals = torch.arange(4, device=device, dtype=torch.int64)
        candidates = PerCellWorkset(
            torch.tensor([0, 4], device=device, dtype=torch.int64),
            candidate_ordinals,
            candidate_ids,
            candidate_ordinals.clone(),
            structure_epoch=1,
            state_generation=23,
            capacity=8,
        )
        scores = torch.tensor([2.0, -2.0, 1.5, -1.5], device=device,
                              dtype=torch.float32, requires_grad=True)
        probabilities, admitted, train_values = learned_gates(
            scores, temperature=1.0, threshold=0.5, prune=True)
        gates = GateValues(probabilities, admitted, train_values, candidate_ids,
                           candidate_ordinals, structure_epoch=1,
                           state_generation=23)

        primal, derivative = module.select_gate_worksets(
            candidates, gates, seed=41, window_counter=7, dormant_budget=1)
        positive_ids = set(primal.logical_edge_ids.cpu().tolist())
        derivative_ids = derivative.logical_edge_ids.cpu().tolist()
        self.assertEqual(positive_ids, {10, 12})
        self.assertEqual(derivative_ids.count(10), 1)
        self.assertEqual(derivative_ids.count(12), 1)
        selected_closed = set(derivative_ids) & {11, 13}
        self.assertEqual(len(selected_closed), 1)
        self.assertTrue(all(
            int(edge_id) == int(candidate_ids[int(gate_index)])
            for edge_id, gate_index in zip(
                derivative.logical_edge_ids.cpu().tolist(),
                derivative.gate_value_indices.cpu().tolist())))

        inputs = torch.ones((1, 3, 64), device=device, dtype=torch.float16)
        edge_values = torch.ones(4, device=device, dtype=torch.float32,
                                 requires_grad=True)
        with module._pinned_forward():
            output = module.relation(inputs, edge_values, gates, primal, derivative)
        output.sum().backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        sampled_index = 1 if 11 in selected_closed else 3
        omitted_index = 3 if sampled_index == 1 else 1
        # The native backward returns dL/dq in derivative-item order.  Build
        # its independent Torch VJP oracle over every candidate slot: admitted
        # positives and the sampled closed gate receive 64, while the omitted
        # closed gate receives zero.  The STE applies sigmoid'(score) to this
        # complete candidate-space vector.
        scores_ref = scores.detach().clone().requires_grad_()
        probabilities_ref = torch.sigmoid(scores_ref)
        q_ref = probabilities_ref + (
            probabilities_ref * admitted.to(probabilities_ref.dtype)
            - probabilities_ref).detach()
        derivative_gate_indices = derivative.gate_value_indices
        (q_ref.index_select(0, derivative_gate_indices) * 64.0).sum().backward()
        torch.testing.assert_close(scores.grad, scores_ref.grad, rtol=1e-6, atol=1e-7)
        self.assertNotEqual(float(scores.grad[0].abs()), 0.0)
        self.assertNotEqual(float(scores.grad[2].abs()), 0.0)
        self.assertNotEqual(float(scores.grad[sampled_index].abs()), 0.0)
        self.assertEqual(float(scores.grad[omitted_index].abs()), 0.0)

        # The native workset contract permits a candidate gate index to be
        # reused by multiple cells.  Verify scatter accumulation for repeated
        # indices across cells against the same full-vector Torch oracle.
        repeated_primal = PerCellWorkset(
            torch.tensor([0, 2, 4], device=device, dtype=torch.int64),
            torch.tensor([0, 2, 0, 2], device=device, dtype=torch.int64),
            torch.tensor([10, 12, 10, 12], device=device, dtype=torch.int64),
            torch.tensor([0, 2, 0, 2], device=device, dtype=torch.int64),
            structure_epoch=1, state_generation=23, capacity=8)
        sampled_ordinal = sampled_index
        sampled_id = 11 if sampled_index == 1 else 13
        repeated_derivative = PerCellWorkset(
            torch.tensor([0, 3, 6], device=device, dtype=torch.int64),
            torch.tensor([0, sampled_ordinal, 2, 0, sampled_ordinal, 2],
                         device=device, dtype=torch.int64),
            torch.tensor([10, sampled_id, 12, 10, sampled_id, 12],
                         device=device, dtype=torch.int64),
            torch.tensor([0, sampled_index, 2, 0, sampled_index, 2],
                         device=device, dtype=torch.int64),
            structure_epoch=1, state_generation=23, capacity=8)
        repeated_scores = scores.detach().clone().requires_grad_()
        repeated_probabilities, repeated_admitted, repeated_train_values = learned_gates(
            repeated_scores, temperature=1.0, threshold=0.5, prune=True)
        repeated_gates = GateValues(
            repeated_probabilities, repeated_admitted, repeated_train_values,
            candidate_ids, candidate_ordinals, structure_epoch=1,
            state_generation=23)
        repeated_inputs = torch.ones((2, 3, 64), device=device, dtype=torch.float16)
        repeated_edges = torch.ones(4, device=device, dtype=torch.float32,
                                    requires_grad=True)
        with module._pinned_forward():
            repeated_output = module.relation(
                repeated_inputs, repeated_edges, repeated_gates,
                repeated_primal, repeated_derivative)
        repeated_output.sum().backward()
        repeated_scores_ref = repeated_scores.detach().clone().requires_grad_()
        repeated_p_ref = torch.sigmoid(repeated_scores_ref)
        repeated_q_ref = repeated_p_ref + (
            repeated_p_ref * repeated_admitted.to(repeated_p_ref.dtype)
            - repeated_p_ref).detach()
        (repeated_q_ref.index_select(
            0, repeated_derivative.gate_value_indices) * 64.0).sum().backward()
        torch.testing.assert_close(
            repeated_scores.grad, repeated_scores_ref.grad, rtol=1e-6, atol=1e-7)

    def test_adaptive_publication_rebinds_support_without_replacing_owner_leaf(self):
        module = self._module()
        owner_parameter = module.parameters_flat
        optimizer = torch.optim.Adam([owner_parameter], lr=0.01)
        module.stage_promotion(12, 0, 1, discovery_round_id=3,
                               scorer_generation=0, proposal_score=0.75)
        module.publish_support(2)
        self.assertIs(module.parameters_flat, owner_parameter)
        self.assertEqual(module.support.edge_ids, (10, 11, 12))
        self.assertEqual(module.support.sources, (0, 1, 0))
        self.assertEqual(module.support.destinations, (2, 2, 1))
        self.assertEqual(module.structure_epoch, 2)
        self.assertIs(optimizer.param_groups[0]["params"][0], owner_parameter)

    def test_checkpoint_continuation_restores_native_owner_and_adam_state(self):
        source = self._module()
        source_optimizer = torch.optim.Adam([source.parameters_flat], lr=0.01)
        source.parameters_flat.square().sum().backward()
        guarded_step(source, source_optimizer)
        source.stage_promotion(12, 0, 1, discovery_round_id=4,
                              scorer_generation=self.native.sparse_generation(source._native),
                              proposal_score=0.625)
        source.publish_support(2)
        source.application_state["window"] = 5
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sparse-native.pt"
            save_checkpoint(path, source, source_optimizer)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            saved_sparse = next(record for record in saved["native_owners"]
                                if record["kind"] == "sparse_support")
            self.assertEqual(saved_sparse["module_state"]["transition_mode"], "shared")
            self.assertEqual(saved_sparse["module_state"]["transition_storage_layout"],
                             "stacked_fp32_blocks_v1")
            self.assertEqual(saved_sparse["module_state"]["transition_routing_policy"],
                             "all_transition_B_blocks_v1")
            target = self._module()
            target_optimizer = torch.optim.Adam([target.parameters_flat], lr=0.5)
            target_leaf = target.parameters_flat
            load_checkpoint(path, target, target_optimizer)
        self.assertIs(target.parameters_flat, target_leaf)
        self.assertTrue(torch.equal(target.parameters_flat, source.parameters_flat))
        self.assertEqual(target.application_state["window"], 5)
        self.assertEqual(target.support.edge_ids, (10, 11, 12))
        self.assertEqual(target.support.sources, (0, 1, 0))
        self.assertEqual(target.support.destinations, (2, 2, 1))
        self.assertEqual(target.structure_epoch, 2)
        self.assertEqual(target_optimizer.param_groups[0]["lr"], 0.01)
        self.assertTrue(torch.equal(
            target_optimizer.state[target.parameters_flat]["exp_avg"],
            source_optimizer.state[source.parameters_flat]["exp_avg"]))

    def test_per_mechanism_checkpoint_continuation_and_mode_mismatch_preflight(self):
        source = self._module(transition_mode="per_mechanism")
        source_optimizer = torch.optim.Adam([source.parameters_flat], lr=0.01)
        source.parameters_flat.square().sum().backward()
        self.assertTrue(guarded_step(source, source_optimizer))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "per-mechanism.pt"
            save_checkpoint(path, source, source_optimizer)
            target = self._module(transition_mode="per_mechanism")
            target_optimizer = torch.optim.Adam([target.parameters_flat], lr=0.5)
            target_leaf = target.parameters_flat
            load_checkpoint(path, target, target_optimizer)
            self.assertIs(target.parameters_flat, target_leaf)
            self.assertEqual(target.transition_mode, "per_mechanism")
            self.assertTrue(torch.equal(target.parameters_flat, source.parameters_flat))
            self.assertEqual(target_optimizer.param_groups[0]["lr"], 0.01)
            self.assertTrue(torch.equal(
                target_optimizer.state[target_leaf]["exp_avg"],
                source_optimizer.state[source.parameters_flat]["exp_avg"]))

            mismatched = self._module(transition_mode="shared")
            mismatched_optimizer = torch.optim.Adam([mismatched.parameters_flat], lr=0.75)
            before = mismatched.parameters_flat.detach().clone()
            before_epoch = mismatched.structure_epoch
            with self.assertRaisesRegex(ValueError, "transition mode does not match"):
                load_checkpoint(path, mismatched, mismatched_optimizer)
            self.assertTrue(torch.equal(mismatched.parameters_flat, before))
            self.assertEqual(mismatched.structure_epoch, before_epoch)
            self.assertEqual(mismatched_optimizer.param_groups[0]["lr"], 0.75)
            self.assertEqual(mismatched_optimizer.state, {})

        source_optimizer.zero_grad(set_to_none=True)
        target_optimizer.zero_grad(set_to_none=True)
        continuation_gradient = torch.linspace(
            -0.2, 0.2, source.parameters_flat.numel(), device=source.parameters_flat.device)
        source.parameters_flat.grad = continuation_gradient.clone()
        target.parameters_flat.grad = continuation_gradient.clone()
        self.assertTrue(guarded_step(source, source_optimizer))
        self.assertTrue(guarded_step(target, target_optimizer))
        self.assertTrue(torch.equal(source.parameters_flat, target.parameters_flat))
        self.assertTrue(torch.equal(
            source_optimizer.state[source.parameters_flat]["exp_avg"],
            target_optimizer.state[target.parameters_flat]["exp_avg"]))

    def test_untagged_legacy_checkpoint_continues_as_shared_without_storage_migration(self):
        source = self._module(transition_mode="shared")
        optimizer = torch.optim.Adam([source.parameters_flat], lr=0.02)
        source.parameters_flat.square().sum().backward()
        self.assertTrue(guarded_step(source, optimizer))
        with tempfile.TemporaryDirectory() as directory:
            modern = Path(directory) / "modern.pt"
            legacy = Path(directory) / "legacy.pt"
            save_checkpoint(modern, source, optimizer)
            payload = torch.load(modern, map_location="cpu", weights_only=False)
            sparse = next(record for record in payload["native_owners"]
                          if record["kind"] == "sparse_support")
            for key in ("transition_mode", "transition_storage_layout",
                        "transition_routing_policy"):
                del sparse["module_state"][key]
            torch.save(payload, legacy)

            target = self._module()  # default is the legacy shared interpretation
            target_optimizer = torch.optim.Adam([target.parameters_flat], lr=0.5)
            target_leaf = target.parameters_flat
            target_handle_data_ptr = self.native.sparse_parameters(target._native).data_ptr()
            load_checkpoint(legacy, target, target_optimizer)
            self.assertIs(target.parameters_flat, target_leaf)
            self.assertEqual(target.transition_mode, "shared")
            self.assertTrue(torch.equal(target.parameters_flat, source.parameters_flat))
            self.assertEqual(self.native.sparse_parameters(target._native).data_ptr(),
                             target_handle_data_ptr)
            self.assertEqual(target_optimizer.param_groups[0]["lr"], 0.02)
            self.assertTrue(torch.equal(
                target_optimizer.state[target_leaf]["exp_avg"],
                optimizer.state[source.parameters_flat]["exp_avg"]))

        optimizer.zero_grad(set_to_none=True)
        target_optimizer.zero_grad(set_to_none=True)
        continuation_gradient = torch.linspace(
            -0.15, 0.15, source.parameters_flat.numel(), device=source.parameters_flat.device)
        source.parameters_flat.grad = continuation_gradient.clone()
        target.parameters_flat.grad = continuation_gradient.clone()
        self.assertTrue(guarded_step(source, optimizer))
        self.assertTrue(guarded_step(target, target_optimizer))
        self.assertTrue(torch.equal(source.parameters_flat, target.parameters_flat))
        self.assertTrue(torch.equal(
            optimizer.state[source.parameters_flat]["exp_avg"],
            target_optimizer.state[target.parameters_flat]["exp_avg"]))

    def test_checkpoint_restores_promoted_then_revoked_identity_at_exact_epoch(self):
        source = self._module()
        source_optimizer = torch.optim.Adam([source.parameters_flat], lr=0.01)
        source.parameters_flat.square().sum().backward()
        self.assertTrue(guarded_step(source, source_optimizer))
        source.stage_promotion(12, 0, 1, discovery_round_id=9,
                              scorer_generation=self.native.sparse_generation(source._native),
                              proposal_score=0.5)
        source.publish_support(2)
        source.stage_revocation(12)
        source.publish_support(3)
        self.assertEqual(source.support.edge_ids, (10, 11))
        self.assertEqual(source.structure_epoch, 3)
        source_snapshot = source._support_index.snapshot()
        self.assertEqual(source_snapshot["base"]["edge_ids"].tolist(), [10, 11])
        self.assertEqual(source_snapshot["delta"]["edge_ids"].tolist(), [])
        self.assertEqual(source_snapshot["revoked_edge_ids"].tolist(), [12])
        self.assertEqual(source_snapshot["promotion_evidence_edge_ids"].tolist(), [12])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "promoted-then-revoked.pt"
            save_checkpoint(path, source, source_optimizer)
            target = self._module()
            target_optimizer = torch.optim.Adam([target.parameters_flat], lr=0.5)
            target_leaf = target.parameters_flat
            target_handle_owner = self.native.sparse_parameters(target._native).data_ptr()
            load_checkpoint(path, target, target_optimizer)

        self.assertIs(target.parameters_flat, target_leaf)
        self.assertEqual(target.structure_epoch, 3)
        self.assertEqual(target.support.edge_ids, (10, 11))
        self.assertEqual(target.support.sources, (0, 1))
        self.assertEqual(target.support.destinations, (2, 2))
        self.assertEqual(self.native.sparse_parameters(target._native).data_ptr(),
                         target_handle_owner)
        restored = target._support_index.snapshot()
        self.assertEqual(restored["base"]["edge_ids"].tolist(), [10, 11])
        self.assertEqual(restored["delta"]["edge_ids"].tolist(), [])
        self.assertEqual(restored["revoked_edge_ids"].tolist(), [12])
        self.assertEqual(restored["promotion_evidence_edge_ids"].tolist(), [12])
        self.assertEqual(restored["promotion_evidence_round_ids"].tolist(), [9])
        self.assertEqual(target_optimizer.param_groups[0]["lr"], 0.01)
        self.assertTrue(torch.equal(target.parameters_flat, source.parameters_flat))
        self.assertTrue(torch.equal(
            target_optimizer.state[target_leaf]["exp_avg"],
            source_optimizer.state[source.parameters_flat]["exp_avg"]))

        with self.assertRaisesRegex(RuntimeError, "status"):
            target.stage_promotion(12, 0, 1, discovery_round_id=10)

        next_gradient = torch.linspace(
            -0.25, 0.25, source.parameters_flat.numel(),
            device=source.parameters_flat.device)
        source_optimizer.zero_grad(set_to_none=True)
        target_optimizer.zero_grad(set_to_none=True)
        source.parameters_flat.grad = next_gradient.clone()
        target.parameters_flat.grad = next_gradient.clone()
        self.assertTrue(guarded_step(source, source_optimizer))
        self.assertTrue(guarded_step(target, target_optimizer))
        self.assertTrue(torch.equal(target.parameters_flat, source.parameters_flat))
        self.assertTrue(torch.equal(
            target_optimizer.state[target_leaf]["exp_avg"],
            source_optimizer.state[source.parameters_flat]["exp_avg"]))


if __name__ == "__main__":
    unittest.main()
