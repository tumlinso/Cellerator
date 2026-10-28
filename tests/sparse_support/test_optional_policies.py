"""Reference and admitted-GPU checks for optional sparse application policies."""
from __future__ import annotations

import os
import unittest
from unittest import mock

import torch

from cellerator.torch.sparse_support import (
    GateValues,
    KaryoplastEventSlice,
    PerCellWorkset,
    RankedGeneSlice,
    SupportSpec,
    _logical_values_for_ordinals,
    _relation_source_inputs,
    _sum_history_sources,
    _validate_relation_input_policy,
    learned_gates,
    prepare_sparse_slice,
)


class SparseOptionalPolicyReferenceTest(unittest.TestCase):
    def _values(self, ordinals):
        embedding = torch.randn(5, 7, dtype=torch.float32, requires_grad=True)
        scorer = torch.randn(7, dtype=torch.float32, requires_grad=True)
        source = torch.tensor([0, 1, 2, 3, 1, 4], dtype=torch.int64)
        destination = torch.tensor([1, 2, 3, 4, 0, 0], dtype=torch.int64)
        selected = _logical_values_for_ordinals(
            embedding, scorer, source, destination,
            torch.tensor(ordinals, dtype=torch.int64), len(source))
        return embedding, scorer, source, destination, selected

    def test_union_values_and_shared_vjp_match_full_reference_including_dormant_d(self):
        embedding, scorer, source, destination, actual = self._values([0, 2, 5, 2])
        all_values = _logical_values_for_ordinals(
            embedding, scorer, source, destination,
            torch.arange(len(source)), len(source))
        mask = torch.tensor([1, 0, 1, 0, 0, 1], dtype=torch.float32)
        expected = all_values * mask
        self.assertTrue(torch.equal(actual, expected))

        upstream = torch.randn_like(actual)
        (actual * upstream).sum().backward()
        actual_embedding_grad = embedding.grad.clone()
        actual_scorer_grad = scorer.grad.clone()

        reference_embedding = embedding.detach().clone().requires_grad_()
        reference_scorer = scorer.detach().clone().requires_grad_()
        reference_all = _logical_values_for_ordinals(
            reference_embedding, reference_scorer, source, destination,
            torch.arange(len(source)), len(source))
        (reference_all * mask * upstream).sum().backward()
        self.assertTrue(torch.allclose(actual_embedding_grad,
                                       reference_embedding.grad, atol=1e-7, rtol=1e-6))
        self.assertTrue(torch.allclose(actual_scorer_grad,
                                       reference_scorer.grad, atol=1e-7, rtol=1e-6))

    def test_empty_union_returns_zero_vector_and_zero_shared_vjp(self):
        embedding, scorer, source, destination, actual = self._values([])
        self.assertTrue(torch.equal(actual, torch.zeros(6)))
        actual.sum().backward()
        self.assertTrue(torch.equal(embedding.grad, torch.zeros_like(embedding)))
        self.assertTrue(torch.equal(scorer.grad, torch.zeros_like(scorer)))

    def test_history_sum_preserves_repeated_occurrences_and_empty_rows(self):
        embedding = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        actual = _sum_history_sources(embedding, [[1, 2, 1], [], [3]], cells=3)
        expected = torch.zeros(3, 4, 3)
        expected[0, 1] = 2 * embedding[1]
        expected[0, 2] = embedding[2]
        expected[2, 3] = embedding[3]
        self.assertTrue(torch.equal(actual, expected))

    def test_history_sum_rejects_bad_row_count(self):
        embedding = torch.zeros(4, 3)
        with self.assertRaisesRegex(ValueError, "rows must match"):
            _sum_history_sources(embedding, [[1]], cells=2)

    def test_relation_input_policy_is_checkpoint_state_and_locked_after_progress(self):
        state = {"event_position": 4, "relation_input_policy": "eligible_history"}
        _validate_relation_input_policy(state, "eligible_history")
        with self.assertRaisesRegex(ValueError, "cannot change relation_input"):
            _validate_relation_input_policy(state, "current_event")
        self.assertEqual(state["relation_input_policy"], "eligible_history")
        _validate_relation_input_policy({"event_position": 0}, "current_event")
        # Older checkpoints stored no policy and used the current event input.
        _validate_relation_input_policy({"event_position": 4}, "current_event")
        with self.assertRaisesRegex(ValueError, "cannot change relation_input"):
            _validate_relation_input_policy({"event_position": 4}, "eligible_history")

    def test_relation_input_choice_preserves_current_default_and_requires_history(self):
        current = torch.ones(2, 3, 4)
        history = torch.full_like(current, 2)
        self.assertIs(_relation_source_inputs(current, history, "current_event"), current)
        self.assertIs(_relation_source_inputs(current, history, "eligible_history"), history)
        with self.assertRaisesRegex(ValueError, r"requires provider work\[4\]"):
            _relation_source_inputs(current, None, "eligible_history")


@unittest.skipUnless(os.environ.get("CELLERATOR_RUN_CUDA_TESTS") == "1",
                     "requires the scheduler-admitted sparse support CUDA run")
class SparseOptionalPolicyCudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = __import__("cellerator.torch._torch", fromlist=["*"])
        cls.device = torch.cuda.current_device()

    def _module(self, module_type=RankedGeneSlice):
        support = SupportSpec((10, 11, 12), (0, 1, 2), (2, 2, 0), 1)
        return prepare_sparse_slice(
            module_type, support, entities=3, mechanisms=2, rank_count=4,
            device=self.device, parameter_axis_words=(31, 32, 41, 42, 51, 52, 61, 62),
            seed=29)

    def _workset(self, ordinals, ids, *, generation=1, cells=1, gate_indices=None):
        device = torch.device("cuda", self.device)
        ordinals = torch.tensor(ordinals, dtype=torch.int64, device=device)
        ids = torch.tensor(ids, dtype=torch.int64, device=device)
        gate_indices = (torch.zeros_like(ordinals) if gate_indices is None else
                        torch.tensor(gate_indices, dtype=torch.int64, device=device))
        return PerCellWorkset(
            torch.tensor([0, len(ordinals)] + [len(ordinals)] * (cells - 1),
                         dtype=torch.int64, device=device),
            ordinals, ids, gate_indices, 1, generation, 8)

    def test_module_workset_union_matches_full_values_and_vjp(self):
        module = self._module()
        primal = self._workset([0, 0], [10, 10])
        derivative = self._workset([2, 2], [12, 12])  # derivative-only candidate
        with module._pinned_forward():
            full = module.logical_edge_values()
            indexed = module.logical_edge_values(
                policy="workset_union", primal=primal, derivative=derivative, cells=1)
        self.assertTrue(torch.equal(indexed, full * torch.tensor(
            [1, 0, 1], dtype=torch.float32, device=full.device)))
        upstream = torch.tensor([0.2, 0.7, -0.4], device=full.device)
        full_grads = torch.autograd.grad((full * upstream).sum(), module.parameters_flat,
                                         retain_graph=True)[0]
        indexed_grads = torch.autograd.grad((indexed * upstream).sum(),
                                            module.parameters_flat)[0]
        # The reference masks the same shared full parameter graph, including
        # a candidate present only in D and duplicate workset ordinals.
        expected = torch.autograd.grad(
            (full * upstream * torch.tensor([1, 0, 1], device=full.device)).sum(),
            module.parameters_flat)[0]
        self.assertTrue(torch.allclose(indexed_grads, expected, atol=1e-6, rtol=1e-6))
        self.assertGreater(float((full_grads - indexed_grads).abs().sum()), 0.0)

    def test_native_relation_full_and_union_match_forward_and_vjps(self):
        module = self._module()
        device = torch.device("cuda", self.device)
        # P contains admitted edge 10. D adds closed derivative-only edge 11;
        # edge 12 is unused by both worksets.
        primal = self._workset([0], [10])
        derivative = self._workset([0, 1], [10, 11], gate_indices=[0, 1])
        inputs = torch.randn(1, 3, 64, dtype=torch.float16, device=device,
                             generator=torch.Generator(device=device).manual_seed(52),
                             requires_grad=True)
        scores = torch.tensor([2.0, -1.0], dtype=torch.float32, device=device,
                              requires_grad=True)
        probabilities, admitted, train_values = learned_gates(
            scores, temperature=1.0, threshold=0.5, prune=True)
        gates = GateValues(
            probabilities, admitted, train_values,
            torch.tensor([10, 11], dtype=torch.int64, device=device),
            torch.tensor([0, 1], dtype=torch.int64, device=device), 1, 1)
        with module._pinned_forward():
            full_values = module.logical_edge_values()
            union_values = module.logical_edge_values(
                policy="workset_union", primal=primal, derivative=derivative, cells=1)
            full_output = module.relation(inputs, full_values, gates, primal, derivative)
            union_output = module.relation(inputs, union_values, gates, primal, derivative)

        upstream = torch.randn(full_output.shape, dtype=torch.float32, device=device,
                               generator=torch.Generator(device=device).manual_seed(53))
        parameters = module.parameters_flat
        full_vjp = torch.autograd.grad(
            (full_output * upstream).sum(), (inputs, parameters, scores), retain_graph=True)
        union_vjp = torch.autograd.grad(
            (union_output * upstream).sum(), (inputs, parameters, scores))
        self.assertTrue(torch.allclose(union_output, full_output, atol=2e-3, rtol=2e-3))
        for indexed_grad, full_grad in zip(union_vjp, full_vjp):
            self.assertTrue(torch.allclose(indexed_grad, full_grad, atol=3e-3, rtol=3e-3))
        self.assertFalse(bool(admitted[1]))
        self.assertGreater(float(union_vjp[2][1].abs()), 0.0)

    def test_history_relation_input_is_explicit_and_stream_policy_is_retained(self):
        module = self._module(KaryoplastEventSlice)
        device = torch.device("cuda", self.device)
        workset = self._workset([0], [10])
        event_ids = torch.tensor([[1]], dtype=torch.int64, device=device)
        gates = GateValues(
            torch.zeros(3, device=device), torch.zeros(3, dtype=torch.bool, device=device),
            torch.zeros(3, device=device), torch.tensor([10, 11, 12], device=device),
            torch.arange(3, device=device), 1, 1)
        module.score_candidates = lambda *args, **kwargs: gates
        module.select_gate_worksets = lambda derivative, gates, **kwargs: (workset, workset)
        module.relation = lambda inputs, *args: inputs.float()
        with mock.patch.object(self.native, "sparse_latent_transition",
                               side_effect=lambda state, event, offsets, affected,
                               a, b, bias, handle: state.float()):
            history = lambda *args: (workset, workset,
                                     torch.tensor([0, 0], device=device),
                                     torch.tensor([0], device=device), [[0, 1, 1]])
            result = module(event_ids, history, relation_input="eligible_history")
        block = module._blocks["entity_embedding"]
        embeddings = module.parameters_flat.detach().narrow(
            0, block.offset, block.numel).view(block.shape)
        expected = torch.zeros(1, 3, embeddings.shape[-1], device=device)
        expected[0, 0] = embeddings[0]
        expected[0, 1] = 2 * embeddings[1]
        self.assertTrue(torch.allclose(result, expected.half().float()))
        self.assertEqual(module.application_state["relation_input_policy"], "eligible_history")
        checkpoint_state = module._sparse_checkpoint_state()
        self.assertEqual(checkpoint_state["application_state"]["relation_input_policy"],
                         "eligible_history")
        with self.assertRaisesRegex(ValueError, "cannot change relation_input"):
            module(event_ids, history, relation_input="current_event")

        missing_history = self._module(KaryoplastEventSlice)
        missing_history.score_candidates = lambda *args, **kwargs: gates
        missing_history.select_gate_worksets = lambda derivative, gates, **kwargs: (
            workset, workset)
        with mock.patch.object(self.native, "sparse_latent_transition",
                               side_effect=lambda state, event, offsets, affected,
                               a, b, bias, handle: state.float()):
            with self.assertRaisesRegex(ValueError, r"requires provider work\[4\]"):
                missing_history(event_ids, lambda *args: (workset, workset,
                    torch.tensor([0, 0], device=device),
                    torch.tensor([0], device=device)), relation_input="eligible_history")


if __name__ == "__main__":
    unittest.main()
