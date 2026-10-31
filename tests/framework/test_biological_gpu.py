"""Admitted V100-only correctness smoke for biological NN compositions.

Ordinary test discovery skips this module. The scheduler job runner must set
CELLERATOR_TEST_CUDA=1 after admission before any test can allocate CUDA memory.
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import torch

from cellerator.biology import FeatureBinding, PredictionQuery
from cellerator.torch.features import FeatureEncoder
from cellerator.torch.programs import InteractionPrograms
from cellerator.torch.readouts import FeatureReadout
from cellerator.torch.relation_attention import RelationAttention
from cellerator.torch.relations import PreparedRelation


_ADMITTED = os.environ.get("CELLERATOR_TEST_CUDA") == "1"


@unittest.skipUnless(_ADMITTED, "requires an admitted Device Scheduler GPU job")
class BiologicalCUDAIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("scheduler marked this run admitted but CUDA is unavailable")
        cls.device = torch.device("cuda", 0)
        capability = torch.cuda.get_device_capability(cls.device)
        if capability != (7, 0):
            raise RuntimeError(f"expected admitted V100/SM70, received capability {capability}")

    @staticmethod
    def _tolerances(dtype):
        if dtype == torch.float16:
            return {"rtol": 8e-3, "atol": 6e-3}
        return {"rtol": 3e-4, "atol": 3e-5}

    @staticmethod
    def _grads(loss, inputs):
        return torch.autograd.grad(loss, inputs, allow_unused=False)

    def test_prepared_relation_matches_raw_csr_output_and_vjps_fp32_and_fp16(self):
        """Repeated occurrences, duplicate edges, and empty rows survive preparation."""
        torch.manual_seed(20261031)
        source_order = ("gene-a", "gene-b", "gene-b", "gene-c", "gene-d")
        target_order = ("program-x", "program-empty", "program-y")
        row_ptr = (0, 3, 3, 5)
        col_indices = (0, 1, 1, 3, 4)
        prepared = PreparedRelation.from_csr(
            row_ptr, col_indices, source_order=source_order,
            target_order=target_order, device=self.device,
        )
        self.assertEqual(prepared.row_ptr, row_ptr)
        self.assertEqual(prepared.col_indices, col_indices)

        for dtype in (torch.float32, torch.float16):
            with self.subTest(dtype=str(dtype)):
                module = RelationAttention(4, heads=2, dropout=0.0).to(
                    device=self.device, dtype=dtype
                ).eval()
                source = torch.randn(5, 4, device=self.device, dtype=dtype, requires_grad=True)
                target = torch.randn(3, 4, device=self.device, dtype=dtype, requires_grad=True)
                probe = torch.randn(3, 4, device=self.device, dtype=dtype)
                raw = module(source, target, row_ptr=row_ptr, col_indices=col_indices)
                raw_inputs = (source, target, *tuple(module.parameters()))
                raw_gradients = self._grads((raw.float() * probe.float()).sum(), raw_inputs)

                source_prepared = source.detach().clone().requires_grad_(True)
                target_prepared = target.detach().clone().requires_grad_(True)
                prepared_inputs = (source_prepared, target_prepared, *tuple(module.parameters()))
                # A prepared structure is already on-device. Reject tensor-to-host
                # scalar reads while forward and its VJP execute.
                with patch.object(torch.Tensor, "item", side_effect=AssertionError(
                    "prepared relation performed a per-forward scalar host read"
                )):
                    result = module(
                        source_prepared, target_prepared, relation=prepared,
                        source_order=source_order, target_order=target_order,
                    )
                    gradients = self._grads(
                        (result.float() * probe.float()).sum(), prepared_inputs
                    )
                tolerance = self._tolerances(dtype)
                torch.testing.assert_close(result, raw, **tolerance)
                for actual, expected in zip(gradients, raw_gradients):
                    torch.testing.assert_close(actual, expected, **tolerance)
                self.assertTrue(torch.isfinite(result).all().item())

                # The middle target has no incoming edges and therefore exactly
                # zero interaction contribution, even though output projections
                # and surrounding parameters remain trainable.
                torch.testing.assert_close(
                    result[1], torch.zeros_like(result[1]), rtol=0, atol=0
                )

    def test_prepared_relation_matches_raw_and_dense_equivalent_support(self):
        """A deduplicated CSR and Boolean support encode the same relation."""
        torch.manual_seed(110)
        source_order = ("g0", "g1", "g2", "g3")
        target_order = ("p0", "p-empty", "p1")
        offsets_source = (0, 2, 4)
        offsets_target = (0, 2, 3)
        row_ptr, col_indices = (0, 2, 2, 3), (0, 1, 3)
        csr = PreparedRelation.from_csr(
            row_ptr, col_indices, source_order=source_order,
            target_order=target_order, source_offsets=offsets_source,
            target_offsets=offsets_target, structural_generation=5,
            device=self.device,
        )
        dense = PreparedRelation.from_dense(
            [[True, True, False, False], [False] * 4,
             [False, False, False, True]],
            source_order=source_order, target_order=target_order,
            source_offsets=offsets_source, target_offsets=offsets_target,
            structural_generation=5, device=self.device,
        )
        module = RelationAttention(4, heads=2).to(self.device).eval()
        source = torch.randn(4, 4, device=self.device)
        target = torch.randn(3, 4, device=self.device)
        kwargs = dict(
            source_order=source_order, target_order=target_order,
            source_offsets=offsets_source, target_offsets=offsets_target,
            structural_generation=5,
        )
        raw = module(source, target, row_ptr=row_ptr, col_indices=col_indices,
                     source_offsets=offsets_source, target_offsets=offsets_target)
        from_csr = module(source, target, relation=csr, **kwargs)
        from_dense = module(source, target, relation=dense, **kwargs)
        torch.testing.assert_close(from_csr, raw)
        torch.testing.assert_close(from_dense, raw)

        with self.assertRaisesRegex(ValueError, "occurrence order"):
            module(source, target, relation=csr,
                   **{**kwargs, "source_order": tuple(reversed(source_order))})
        with self.assertRaisesRegex(ValueError, "generation"):
            module(source, target, relation=csr,
                   **{**kwargs, "structural_generation": 6})

    def test_named_binding_encoder_and_feature_readout_preserve_requested_identity_order(self):
        """Noncontiguous model coordinates remain distinct from identity positions."""
        binding = FeatureBinding.from_ids(
            ("gene-a", "gene-b", "gene-c"), model_indices=(7, 2, 11),
            model_capacity=12, domain="gene",
        )
        encoder = FeatureEncoder(12, 4, binding=binding).to(self.device)
        values = torch.tensor([[0.2, 0.3, 0.7]], device=self.device, requires_grad=True)
        encoded = encoder.encode_bound(
            (("gene-c", "gene-a", "gene-c"),), values,
            observed=torch.ones((1, 3), dtype=torch.bool, device=self.device),
        )
        self.assertEqual(tuple(encoded.shape), (1, 3, 4))
        self.assertEqual(binding.prepare(
            ("gene-a", "gene-b", "gene-c"), ("gene-c", "gene-a")
        ).model_indices, (11, 7))

        readout = FeatureReadout(binding, 4, output_size=2, heads=2).to(self.device).eval()
        query_ids = ("gene-c", "gene-a")
        alignment = binding.prepare(binding.identities, query_ids)
        query = PredictionQuery(
            "atac", feature_ids=query_ids, sample_ids=("cell-0",)
        )
        model_indices = torch.tensor(
            alignment.model_indices, device=self.device, dtype=torch.long
        )
        selected = readout.query(encoded, model_indices, query=query)
        all_values = readout(encoded)
        full_positions = torch.tensor((2, 0), device=self.device, dtype=torch.long)
        torch.testing.assert_close(selected, all_values.index_select(1, full_positions))

        selected.square().sum().backward()
        self.assertIsNotNone(values.grad)
        self.assertIsNotNone(encoder.feature_embedding.weight.grad)
        self.assertIsNotNone(readout.target_embedding.weight.grad)
        self.assertGreater(float(encoder.feature_embedding.weight.grad.abs().sum().item()), 0.0)

    def test_overlapping_interaction_programs_condition_and_share_gradients(self):
        torch.manual_seed(303)
        feature_order = ("g0", "g1", "g2", "g3")
        program_ids = ("module-a", "module-b", "module-empty")
        relation = PreparedRelation.from_dense(
            [[True, True, False, False],
             [False, True, True, False],
             [False, False, False, False]],
            source_order=feature_order, target_order=program_ids,
            structural_generation=12, device=self.device,
        )
        model = InteractionPrograms(
            4, 3, heads=2, program_ids=program_ids, context_size=2,
        ).to(self.device).eval()
        features = torch.randn(4, 4, device=self.device, requires_grad=True)
        context = torch.tensor([0.3, -0.4], device=self.device, requires_grad=True)
        output = model(
            features, relation=relation, feature_order=feature_order,
            structural_generation=12, context=context,
        )
        self.assertEqual(tuple(output["program_features"].shape), (3, 4))
        self.assertEqual(tuple(output["feature_updates"].shape), (4, 4))
        torch.testing.assert_close(
            output["program_features"][2], torch.zeros_like(output["program_features"][2]),
            rtol=0, atol=0,
        )
        torch.testing.assert_close(
            output["feature_updates"][3], torch.zeros_like(output["feature_updates"][3]),
            rtol=0, atol=0,
        )
        (output["program_features"].float().square().sum()
         + output["feature_updates"].float().square().sum()).backward()
        self.assertIsNotNone(features.grad)
        self.assertIsNotNone(context.grad)
        self.assertIsNotNone(model.program_queries.grad)
        self.assertIsNotNone(model.context_projection.weight.grad)
        self.assertGreater(float(features.grad[1].abs().sum().item()), 0.0)


if __name__ == "__main__":
    unittest.main()
