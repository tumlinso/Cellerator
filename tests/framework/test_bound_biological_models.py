"""Identity-aware reference-model integrations with explicit biology axes."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from cellerator.biology import FeatureBinding, PredictionQuery
from cellerator.modeling import Cell
from cellerator.torch.manifest import (
    load_biological_manifest,
    save_biological_manifest,
    validate_biological_manifest,
)
from cellerator.torch.supervision import categorical_target_columns
from examples.nn_foundation.reference_models import (
    EventHistoryReferenceModel,
    ExpressionReferenceModel,
    atac_binary_loss,
    attach_atac_view,
    attach_event_view,
    attach_expression_view,
    attach_history_view,
    explicit_dense_anndata_minibatch,
)


def _binding(ids, model_indices, *, domain="gene", capacity=None):
    return FeatureBinding.from_ids(
        ids, model_indices, domain=domain,
        model_capacity=capacity,
        metadata={"reference": "synthetic-v1"},
    )


class BoundReferenceModelTests(unittest.TestCase):
    def test_expression_named_features_align_to_declared_binding_order(self):
        torch.manual_seed(301)
        binding = _binding(("gA", "gB", "gC"), (2, 7, 11), capacity=13)
        model = ExpressionReferenceModel(13, 8, heads=2, binding=binding).eval()
        values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        observed = torch.ones_like(values, dtype=torch.bool)

        def run(ids, order):
            ordered_values = values.index_select(1, torch.tensor(order))
            ordered_observed = observed.index_select(1, torch.tensor(order))
            return model({
                # The legacy integer field remains accepted; named identities
                # are the explicit biological coordinate for this route.
                "feature_ids": torch.zeros_like(ordered_values, dtype=torch.long),
                "feature_identities": ids,
                "values": ordered_values,
                "observed": ordered_observed,
            })

        canonical = run(("gA", "gB", "gC"), (0, 1, 2))
        reordered = run(("gC", "gA", "gB"), (2, 0, 1))
        self.assertEqual(canonical["feature_ids"], binding.identities)
        torch.testing.assert_close(canonical["expression"], reordered["expression"])
        self.assertEqual(model.encoder.feature_embedding.num_embeddings, 13)

    def test_bound_expression_can_use_optional_overlapping_program_composition(self):
        torch.manual_seed(302)
        binding = _binding(("gA", "gB", "gC"), (2, 7, 11), capacity=13)
        model = ExpressionReferenceModel(
            13, 8, heads=2, binding=binding, num_programs=2
        )
        batch = {
            "feature_ids": torch.zeros((2, 3), dtype=torch.long),
            "feature_identities": ("gC", "gA", "gB"),
            "values": torch.rand(2, 3),
            "observed": torch.ones((2, 3), dtype=torch.bool),
        }
        result = model(batch)

        self.assertEqual(tuple(result["program_features"].shape), (2, 2, 8))
        self.assertEqual(tuple(result["features"].shape), (2, 3, 8))
        result["expression"].square().mean().backward()
        self.assertIsNotNone(model.interaction_programs.program_queries.grad)

    def test_event_binding_separates_noncontiguous_embedding_ids_from_output_columns(self):
        torch.manual_seed(303)
        event_binding = _binding(("gene-B", "gene-A", "gene-C"), (4, 9, 13), capacity=15)
        base = EventHistoryReferenceModel(
            3, 8, heads=2, max_history=12, binding=event_binding
        )
        model = attach_event_view(base)
        batch = {
            "event_ids": ("gene-A", "gene-B", "gene-A"),
            "event_offsets": (0, 3),
            "sample_ids": ("cell-1",),
        }
        query = PredictionQuery(
            "events", feature_ids=("gene-A", "gene-C", "gene-B"),
            sample_ids=("cell-1",), row_offsets=(0, 3),
        )
        prediction = model.query(batch, query)

        self.assertEqual(prediction.feature_ids, query.feature_ids)
        self.assertEqual(tuple(prediction.values.shape), (3, 3))
        self.assertEqual(prediction.row_offsets, (0, 3))
        columns = categorical_target_columns(
            prediction, ("gene-A", "gene-B", "gene-A")
        )
        self.assertEqual(columns.tolist(), [0, 2, 0])
        cell = Cell(batch, id="cell-1").bind(model)
        cell_prediction = cell.query(query)
        torch.testing.assert_close(cell_prediction.values, prediction.values)

    def test_bound_query_rejects_same_shaped_incompatible_output_meaning(self):
        binding = _binding(("gene-A", "gene-B"), (2, 7), capacity=9)
        other = _binding(("gene-X", "gene-Y"), (2, 7), capacity=9)
        model = attach_expression_view(ExpressionReferenceModel(9, 8, binding=binding))
        batch = {
            "feature_ids": torch.zeros((1, 2), dtype=torch.long),
            "feature_identities": binding.identities,
            "values": torch.tensor([[0.25, 0.75]]),
            "observed": torch.ones((1, 2), dtype=torch.bool),
            "sample_ids": ("cell-a",),
        }
        with self.assertRaisesRegex(ValueError, "binding|identit|feature"):
            model.query(batch, PredictionQuery("expression", other.identities))

    def test_bounded_sparse_minibatch_aligns_columns_without_densifying_full_data(self):
        try:
            from scipy import sparse
        except ImportError:
            self.skipTest("scipy is not installed")
        from cellerator.adapters.anndata import AnnDataView

        binding = _binding(("gA", "gB", "gC"), (2, 7, 11), capacity=13)
        source = sparse.csr_matrix([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]])
        view = AnnDataView(source, ("cell-1", "cell-2"), ("gC", "gA", "gB"), "X")
        values, feature_ids, cell_ids = explicit_dense_anndata_minibatch(
            view, [1], max_elements=3, binding=binding
        )

        self.assertEqual(feature_ids, ("gA", "gB", "gC"))
        self.assertEqual(cell_ids, ("cell-2",))
        self.assertEqual(values.tolist(), [[4.0, 5.0, 6.0]])
        self.assertTrue(sparse.issparse(view.values))

    def test_history_atac_queries_train_with_frozen_or_joint_base_and_one_forward(self):
        torch.manual_seed(304)
        event_binding = _binding(("gene-A", "gene-B", "gene-C"), (2, 7, 11), capacity=13)
        peak_binding = _binding(("peak-1", "peak-2", "peak-3"), (1, 4, 8),
                                domain="peak", capacity=10)
        base = EventHistoryReferenceModel(
            3, 8, heads=2, max_history=12, binding=event_binding
        )
        model = attach_history_view(
            base, "atac", 3, output_binding=peak_binding
        )
        batch = {
            "event_ids": ("gene-A", "gene-B", "gene-A", "gene-C"),
            "event_offsets": (0, 2, 2, 4),
            "event_id_space": "identity",
            "sample_ids": ("cell-1", "cell-2", "cell-3"),
        }
        query = PredictionQuery(
            "atac", feature_ids=("peak-3", "peak-1"),
            sample_ids=batch["sample_ids"],
        )
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        calls = []
        hook = base.register_forward_hook(lambda *_: calls.append(1))

        base.requires_grad_(False)
        frozen = model.query(batch, query)
        self.assertEqual(tuple(frozen.values.shape), (3, 2))
        self.assertEqual(frozen.feature_ids, query.feature_ids)
        self.assertEqual(len(calls), 1)
        frozen_loss = atac_binary_loss(frozen.values, targets)
        model.zero_grad(set_to_none=True)
        frozen_loss.backward()
        readout = model._views[model._view_indices["atac"]].module
        self.assertTrue(any(parameter.grad is not None for parameter in readout.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in base.parameters()))

        model.zero_grad(set_to_none=True)
        base.requires_grad_(True)
        joint = model.query(batch, query)
        atac_binary_loss(joint.values, targets).backward()
        self.assertEqual(len(calls), 2)
        self.assertTrue(any(parameter.grad is not None for parameter in readout.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in base.parameters()))
        hook.remove()

    def test_semantic_manifest_round_trips_with_reconstructed_model_and_state_dict(self):
        binding = _binding(("gA", "gB"), (3, 8), capacity=10)

        def build():
            return attach_expression_view(ExpressionReferenceModel(10, 8, binding=binding))

        original = build()
        reconstructed = build()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "weights.pt"
            manifest_path = root / "weights.bio.json"
            torch.save(original.state_dict(), weights)
            saved = save_biological_manifest(
                original, manifest_path, weights_path=weights
            )
            loaded = load_biological_manifest(manifest_path, weights_path=weights)
            self.assertEqual(saved, loaded)
            validate_biological_manifest(
                reconstructed, loaded, weights_path=weights
            )
            state = torch.load(weights, map_location="cpu", weights_only=True)
            reconstructed.load_state_dict(state)


if __name__ == "__main__":
    unittest.main()
