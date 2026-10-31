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
from cellerator.torch.views import ModelWithViews
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

    @staticmethod
    def _sparse_reference(module, source, target, row_ptr, col_indices):
        """Independent target-row softmax oracle using ordinary Torch operations."""
        heads, head_width = module.heads, module.head_width
        query = module.query(target).reshape(-1, heads, head_width)
        key = module.key(source).reshape(-1, heads, head_width)
        value = module.value(source).reshape(-1, heads, head_width)
        rows = []
        scale = head_width ** -0.5
        for target_index, (left, right) in enumerate(zip(row_ptr, row_ptr[1:])):
            if left == right:
                aggregate = target.new_zeros((heads, head_width))
            else:
                indices = torch.tensor(
                    col_indices[left:right], dtype=torch.long, device=source.device
                )
                keys = key.index_select(0, indices)
                values = value.index_select(0, indices)
                scores = (keys * query[target_index].unsqueeze(0)).sum(dim=-1) * scale
                weights = torch.softmax(scores, dim=0)
                aggregate = (values * weights.unsqueeze(-1)).sum(dim=0)
            rows.append(aggregate)
        combined = torch.stack(rows, dim=0).reshape(target.shape[0], module.width)
        return module.output(combined)

    def test_prepared_relation_matches_raw_csr_output_and_vjps_fp32_and_fp16(self):
        """Repeated occurrences, duplicate edges, and empty rows survive preparation."""
        torch.manual_seed(20261031)
        source_order = ("gene-a", "gene-b", "gene-b", "gene-c", "gene-d")
        target_order = ("program-x", "program-empty", "program-y")
        row_ptr = (0, 3, 3, 5)
        col_indices = (0, 1, 1, 3, 4)
        host_prepared = PreparedRelation.from_csr(
            row_ptr, col_indices, source_order=source_order,
            target_order=target_order,
        )
        prepared = host_prepared.to(self.device)
        self.assertIs(host_prepared.to(self.device), prepared)
        self.assertEqual(prepared.device, self.device)
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
                reference_source = source.detach().clone().requires_grad_(True)
                reference_target = target.detach().clone().requires_grad_(True)
                reference = self._sparse_reference(
                    module, reference_source, reference_target, row_ptr, col_indices
                )
                reference_inputs = (
                    reference_source, reference_target, *tuple(module.parameters())
                )
                reference_gradients = self._grads(
                    (reference.float() * probe.float()).sum(), reference_inputs
                )
                source_raw = source.detach().clone().requires_grad_(True)
                target_raw = target.detach().clone().requires_grad_(True)
                raw = module(
                    source_raw, target_raw, row_ptr=row_ptr, col_indices=col_indices
                )
                raw_inputs = (source_raw, target_raw, *tuple(module.parameters()))
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
                torch.testing.assert_close(raw, reference, **tolerance)
                torch.testing.assert_close(result, raw, **tolerance)
                for actual, expected in zip(raw_gradients, reference_gradients):
                    torch.testing.assert_close(actual, expected, **tolerance)
                for actual, expected in zip(gradients, raw_gradients):
                    torch.testing.assert_close(actual, expected, **tolerance)
                self.assertTrue(torch.isfinite(result).all().item())

                # Value changes use the same prepared structure; they neither
                # rewrite its order nor cause relation reconstruction.
                shifted = module(
                    source_prepared.detach() + 0.17, target_prepared.detach(),
                    relation=prepared, source_order=source_order,
                    target_order=target_order,
                )
                self.assertFalse(torch.allclose(shifted, result.detach()))

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
        model_indices = alignment.model_indices
        selected = readout.query(encoded, model_indices, query=query)
        all_values = readout(encoded)
        full_positions = torch.tensor((2, 0), device=self.device, dtype=torch.long)
        torch.testing.assert_close(selected, all_values.index_select(1, full_positions))

        selected.square().sum().backward()
        self.assertIsNotNone(values.grad)
        self.assertIsNotNone(encoder.feature_embedding.weight.grad)
        self.assertIsNotNone(readout.target_embedding.weight.grad)
        self.assertGreater(float(encoder.feature_embedding.weight.grad.abs().sum().item()), 0.0)

    def test_query_readout_conditioning_reuse_and_frozen_joint_gradients(self):
        class Base(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(4, 4)
                self.calls = 0

            def forward(self, batch):
                self.calls += 1
                return {
                    "latent": self.projection(batch["features"]),
                    "sample_ids": batch["sample_ids"],
                    "row_offsets": batch.get("row_offsets"),
                }

        class Conditioner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(2, 4, bias=False)

            def forward(self, batch, condition):
                return {
                    "features": batch["features"] + self.projection(condition).unsqueeze(1),
                    "sample_ids": batch["sample_ids"],
                }

        torch.manual_seed(404)
        binding = FeatureBinding.from_ids(
            ("gene-a", "gene-b", "gene-c"), model_indices=(7, 2, 11),
            model_capacity=12, domain="gene",
        )
        base, conditioner = Base(), Conditioner()
        rna = FeatureReadout(binding, 4, output_size=2, heads=2)
        atac = FeatureReadout(binding, 4, output_size=2, heads=2)
        model = ModelWithViews(base, conditioner=conditioner).to(self.device)
        model.add_view("rna", rna, source="latent", output_binding=binding, queryable=True)
        model.add_view("atac", atac, source="latent", output_binding=binding, queryable=True)
        sample_ids = ("cell-a", "cell-b")
        features = torch.randn(2, 3, 4, device=self.device, requires_grad=True)
        batch = {"features": features, "sample_ids": sample_ids}
        condition = torch.tensor(
            [[0.3, -0.7], [0.2, 0.9]], device=self.device, requires_grad=True
        )
        first_query = PredictionQuery(
            "rna", feature_ids=("gene-c", "gene-a"), sample_ids=sample_ids,
            conditioning=condition,
        )
        second_query = PredictionQuery(
            "atac", feature_ids=("gene-b", "gene-c"), sample_ids=sample_ids,
            conditioning=condition,
        )
        base.calls = 0
        with patch.object(torch.Tensor, "item", side_effect=AssertionError(
            "bound query selection performed a per-forward scalar host read"
        )):
            first, second = model.query(batch, (first_query, second_query))
        self.assertEqual(base.calls, 1, "same conditioning object should share one base evaluation")
        self.assertEqual(first.feature_ids, ("gene-c", "gene-a"))
        self.assertEqual(second.feature_ids, ("gene-b", "gene-c"))
        self.assertEqual(first.sample_ids, sample_ids)
        self.assertEqual(tuple(first.values.shape), (2, 2, 2))
        (first.values.float().square().sum() + second.values.float().square().sum()).backward()
        self.assertIsNotNone(condition.grad)
        self.assertIsNotNone(conditioner.projection.weight.grad)
        self.assertIsNotNone(base.projection.weight.grad)
        self.assertIsNotNone(rna.target_embedding.weight.grad)
        self.assertIsNotNone(atac.target_embedding.weight.grad)

        # Distinct, equal-valued conditioning objects deliberately make separate
        # base evaluations. Missing a registered conditioner is an error.
        base.calls = 0
        distinct = condition.detach().clone()
        model.query(batch, (
            first_query,
            PredictionQuery("atac", feature_ids=("gene-a",),
                            sample_ids=sample_ids, conditioning=distinct),
        ))
        self.assertEqual(base.calls, 2)

        # The same sample labels cannot silently be attached to a different
        # partition of three packed history-window outputs.
        window_batch = {
            "features": torch.randn(3, 4, device=self.device),
            "sample_ids": sample_ids,
            "row_offsets": (0, 1, 3),
        }
        with self.assertRaisesRegex(ValueError, "offset"):
            model.query(window_batch, PredictionQuery(
                "rna", feature_ids=("gene-a",), sample_ids=sample_ids,
                row_offsets=(0, 2, 3),
            ))

        unsupported = ModelWithViews(Base()).to(self.device)
        unsupported.add_view(
            "rna", FeatureReadout(binding, 4, output_size=2, heads=2),
            source="latent", output_binding=binding, queryable=True,
        )
        with self.assertRaisesRegex(ValueError, "registered conditioner"):
            unsupported.query({"features": features, "sample_ids": sample_ids}, first_query)

        # Frozen base parameters still pass gradients to caller-owned inputs and
        # the selected readout; joint mode also updates the shared base.
        for parameter in base.parameters():
            parameter.grad = None
            parameter.requires_grad_(False)
        model.zero_grad(set_to_none=True)
        frozen_features = features.detach().clone().requires_grad_(True)
        frozen = model.query(
            {"features": frozen_features, "sample_ids": sample_ids},
            PredictionQuery("rna", feature_ids=("gene-c",), sample_ids=sample_ids),
        )
        frozen.values.float().square().sum().backward()
        self.assertIsNotNone(frozen_features.grad)
        self.assertTrue(all(parameter.grad is None for parameter in base.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in rna.parameters()))

        for parameter in base.parameters():
            parameter.requires_grad_(True)
            parameter.grad = None
        model.zero_grad(set_to_none=True)
        joint = model.query(
            {"features": features.detach(), "sample_ids": sample_ids},
            PredictionQuery("rna", feature_ids=("gene-c",), sample_ids=sample_ids),
        )
        joint.values.float().square().sum().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in base.parameters()))

    def test_native_grouped_linear_composes_with_biologically_bound_features(self):
        from cellerator.torch.grouped_linear import grouped_linear

        torch.manual_seed(505)
        binding = FeatureBinding.from_ids(
            ("gene-a", "gene-b", "gene-c"), model_indices=(4, 1, 8),
            model_capacity=9, domain="gene",
        )
        encoder = FeatureEncoder(9, 4, binding=binding).to(self.device).eval()
        weight = torch.nn.Parameter(torch.randn(2, 3, 4, device=self.device))
        feature_ids = ("gene-c", "gene-a", "gene-b")
        reference_features = encoder.encode_bound(feature_ids)
        reference = grouped_linear(
            reference_features, weight, (0, 1, 3), implementation="reference"
        )
        reference_gradients = torch.autograd.grad(
            reference.square().sum(), (encoder.feature_embedding.weight, weight)
        )
        encoder.zero_grad(set_to_none=True)
        try:
            native_features = encoder.encode_bound(feature_ids)
            native = grouped_linear(
                native_features, weight, (0, 1, 3), implementation="native"
            )
        except ImportError as exc:
            self.skipTest(f"isolated optional _modeling_cuda extension unavailable: {exc}")
        native_gradients = torch.autograd.grad(
            native.square().sum(), (encoder.feature_embedding.weight, weight)
        )
        torch.testing.assert_close(native, reference, rtol=3e-4, atol=3e-5)
        for actual, expected in zip(native_gradients, reference_gradients):
            torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-5)

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
        # Prepare the reverse incidence layout before forward execution; both
        # directions then reuse immutable device-resident support.
        reverse_relation = relation.transpose()
        self.assertIs(relation.transpose(), reverse_relation)
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
