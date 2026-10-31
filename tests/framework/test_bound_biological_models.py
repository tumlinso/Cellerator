"""Identity-aware reference-model integrations with explicit biology axes."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from cellerator.biology import FeatureBinding, Prediction, PredictionQuery
from cellerator.modeling import Cell
from cellerator.torch.manifest import (
    load_biological_manifest,
    save_biological_manifest,
    validate_biological_manifest,
)
from cellerator.torch.supervision import categorical_target_columns
from cellerator.torch.supervision import align_prediction_targets, reduce_prediction_loss
from examples.nn_foundation.reference_models import (
    EventHistoryReferenceModel,
    ExpressionReferenceModel,
    atac_binary_loss,
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
                "feature_identities": ids,
                "values": ordered_values,
                "observed": ordered_observed,
            })

        canonical = run(("gA", "gB", "gC"), (0, 1, 2))
        reordered = run(("gC", "gA", "gB"), (2, 0, 1))
        self.assertEqual(canonical["feature_ids"], binding.identities)
        torch.testing.assert_close(canonical["expression"], reordered["expression"])
        self.assertEqual(model.encoder.feature_embedding.num_embeddings, 13)

        hidden_observed = torch.ones_like(values, dtype=torch.bool)
        hidden_observed[:, 1] = False
        hidden_values = values.clone()
        altered_hidden_values = values.clone()
        altered_hidden_values[:, 1] = 1.0e6
        hidden_batch = {
            "feature_identities": binding.identities,
            "values": hidden_values,
            "observed": hidden_observed,
        }
        altered_batch = {
            **hidden_batch,
            "values": altered_hidden_values,
        }
        torch.testing.assert_close(
            model(hidden_batch)["expression"], model(altered_batch)["expression"]
        )

        zero_batch = {
            "feature_identities": binding.identities,
            "values": torch.zeros_like(values),
            "observed": torch.ones_like(values, dtype=torch.bool),
        }
        missing_batch = {
            **zero_batch,
            "observed": torch.zeros_like(values, dtype=torch.bool),
        }
        self.assertFalse(torch.equal(
            model(zero_batch)["expression"], model(missing_batch)["expression"]
        ))

    def test_bound_expression_can_use_optional_overlapping_program_composition(self):
        torch.manual_seed(302)
        binding = _binding(("gA", "gB", "gC"), (2, 7, 11), capacity=13)
        model = ExpressionReferenceModel(
            13, 8, heads=2, binding=binding, num_programs=2
        )
        batch = {
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
        raw_result = base(batch)
        self.assertEqual(raw_result["context_ids"].tolist(), [14, 9, 4])
        # Class positions follow ('gene-B', 'gene-A', 'gene-C'), independently
        # of the noncontiguous model token IDs (4, 9, 13).
        self.assertEqual(raw_result["targets"].tolist(), [1, 0, 1])
        token_result = base({
            "event_ids": torch.tensor([9, 4, 9]),
            "event_offsets": (0, 3),
            "event_id_space": "model",
        })
        self.assertEqual(token_result["target_ids"], ("gene-A", "gene-B", "gene-A"))
        torch.testing.assert_close(token_result["logits"], raw_result["logits"])
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
            "feature_identities": binding.identities,
            "values": torch.tensor([[0.25, 0.75]]),
            "observed": torch.ones((1, 2), dtype=torch.bool),
            "sample_ids": ("cell-a",),
        }
        with self.assertRaisesRegex((ValueError, KeyError), "binding|identit|feature"):
            model.query(batch, PredictionQuery("expression", other.identities))
        raw_batch = {
            "feature_ids": torch.tensor([[7, 2]]),
            "values": batch["values"],
            "observed": batch["observed"],
        }
        with self.assertRaisesRegex(ValueError, "feature_identities"):
            model.query(raw_batch, PredictionQuery("expression"))

    def test_identity_target_alignment_and_explicit_cell_reduction(self):
        torch.manual_seed(305)
        binding = _binding(("gA", "gB", "gC"), (2, 7, 11), capacity=13)
        base = ExpressionReferenceModel(13, 8, binding=binding).eval()
        model = attach_expression_view(base)
        batch = {
            "feature_identities": binding.identities,
            "values": torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            "observed": torch.ones((2, 3), dtype=torch.bool),
            "sample_ids": ("cell-1", "cell-2"),
        }
        query = PredictionQuery(
            "expression", feature_ids=("gC", "gA"),
            sample_ids=batch["sample_ids"],
        )
        prediction = model.query(batch, query)
        target_ids = ("gA", "gB", "gC")
        target_values = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
        alignment = binding.prepare(target_ids, prediction.feature_ids)
        pred_values, aligned = align_prediction_targets(
            prediction, target_values, alignment,
            target_sample_ids=batch["sample_ids"],
        )
        self.assertEqual(aligned.tolist(), [[30.0, 10.0], [60.0, 40.0]])
        losses = torch.nn.functional.mse_loss(pred_values, aligned, reduction="none")
        observation_loss, observation_count = reduce_prediction_loss(
            losses, unit="observation", sample_ids=batch["sample_ids"],
        )
        self.assertEqual(int(observation_count), losses.numel())
        cell_loss, cell_count = reduce_prediction_loss(
            losses, unit="cell", sample_ids=batch["sample_ids"],
            cell_ids=("physical-1", "physical-2"),
        )
        self.assertEqual(int(cell_count), 2)
        torch.testing.assert_close(observation_loss, cell_loss)

        empty, count = reduce_prediction_loss(
            losses, unit="cell", valid_mask=torch.zeros_like(losses, dtype=torch.bool),
            sample_ids=batch["sample_ids"], cell_ids=("physical-1", "physical-2"),
        )
        self.assertEqual(int(count), 0)
        self.assertEqual(float(empty), 0.0)
        self.assertIsNotNone(empty.grad_fn)

    def test_target_alignment_handles_unequal_history_lengths_and_window_order(self):
        binding = _binding(("gA", "gB", "gC"), (2, 7, 11), capacity=13)
        values = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        prediction = Prediction(
            values, feature_ids=("gA", "gB"),
            sample_ids=("cell-1", "cell-2"), row_offsets=(0, 2, 3),
        )
        target_source_ids = ("gB", "gA", "gC")
        targets = torch.tensor([
            [20.0, 10.0, 30.0], [40.0, 30.0, 50.0], [60.0, 50.0, 70.0],
        ])
        alignment = binding.prepare(target_source_ids, prediction.feature_ids)

        pred_values, aligned = align_prediction_targets(
            prediction, targets, alignment,
            target_sample_ids=("cell-2", "cell-1"),
            target_row_offsets=(0, 1, 3),
        )

        torch.testing.assert_close(pred_values, values)
        self.assertEqual(aligned.tolist(), [[30.0, 40.0], [50.0, 60.0], [10.0, 20.0]])

    def test_cell_reduction_weights_each_physical_cell_equally_across_windows(self):
        losses = torch.tensor([1.0, 3.0, 10.0], requires_grad=True)
        cell_loss, cell_count = reduce_prediction_loss(
            losses, unit="cell", sample_ids=("window-a", "window-b", "window-c"),
            row_offsets=(0, 1, 2, 3), cell_ids=("cell-a", "cell-a", "cell-b"),
        )
        self.assertEqual(int(cell_count), 2)
        self.assertEqual(float(cell_loss), 6.0)

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
        event_query = PredictionQuery(
            "events", feature_ids=("gene-C", "gene-A"),
            sample_ids=batch["sample_ids"], row_offsets=batch["event_offsets"],
        )
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        calls = []
        hook = base.register_forward_hook(lambda *_: calls.append(1))

        base.requires_grad_(False)
        event_prediction, frozen = model.query(batch, (event_query, query))
        self.assertEqual(event_prediction.feature_ids, event_query.feature_ids)
        self.assertEqual(event_prediction.row_offsets, batch["event_offsets"])
        self.assertEqual(tuple(event_prediction.values.shape), (4, 2))
        self.assertEqual(tuple(frozen.values.shape), (3, 2))
        self.assertEqual(frozen.feature_ids, query.feature_ids)
        self.assertEqual(len(calls), 1)
        frozen_loss = atac_binary_loss(frozen.values, targets)
        frozen_base_before = [parameter.detach().clone() for parameter in base.parameters()]
        frozen_head_before = [parameter.detach().clone()
                              for parameter in model._views[model._view_indices["atac"]].module.parameters()]
        frozen_optimizer = torch.optim.SGD(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=0.05,
        )
        model.zero_grad(set_to_none=True)
        frozen_loss.backward()
        readout = model._views[model._view_indices["atac"]].module
        self.assertTrue(any(parameter.grad is not None for parameter in readout.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in base.parameters()))
        frozen_optimizer.step()
        self.assertTrue(all(torch.equal(old, new.detach())
                            for old, new in zip(frozen_base_before, base.parameters())))
        self.assertTrue(any(not torch.equal(old, new.detach())
                            for old, new in zip(frozen_head_before, readout.parameters())))

        model.zero_grad(set_to_none=True)
        base.requires_grad_(True)
        joint_base_before = [parameter.detach().clone() for parameter in base.parameters()]
        joint_head_before = [parameter.detach().clone() for parameter in readout.parameters()]
        joint = model.query(batch, query)
        joint_optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        atac_binary_loss(joint.values, targets).backward()
        self.assertEqual(len(calls), 2)
        self.assertTrue(any(parameter.grad is not None for parameter in readout.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in base.parameters()))
        joint_optimizer.step()
        self.assertTrue(any(not torch.equal(old, new.detach())
                            for old, new in zip(joint_head_before, readout.parameters())))
        self.assertTrue(any(not torch.equal(old, new.detach())
                            for old, new in zip(joint_base_before, base.parameters())))
        hook.remove()

    def test_empty_bound_event_query_skips_update_with_differentiable_zero(self):
        event_binding = _binding(("gene-A", "gene-B"), (4, 9), capacity=11)
        model = attach_event_view(EventHistoryReferenceModel(
            2, 8, heads=2, binding=event_binding,
        ))
        sample_ids = ("empty-1", "empty-2")
        batch = {
            "event_ids": (), "event_offsets": (0, 0, 0), "sample_ids": sample_ids,
        }
        query = PredictionQuery(
            "events", feature_ids=event_binding.identities,
            sample_ids=sample_ids, row_offsets=batch["event_offsets"],
        )
        prediction = model.query(batch, query)
        columns = categorical_target_columns(prediction, ())
        terms = torch.nn.functional.cross_entropy(
            prediction.values, columns, reduction="none"
        )
        loss, count = reduce_prediction_loss(
            terms, unit="observation", sample_ids=sample_ids,
            row_offsets=prediction.row_offsets,
        )
        self.assertEqual(tuple(prediction.values.shape), (0, 2))
        self.assertEqual(int(count), 0)
        self.assertEqual(float(loss), 0.0)
        self.assertIsNotNone(loss.grad_fn)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        if int(count):
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        self.assertTrue(all(torch.equal(old, new.detach())
                            for old, new in zip(before, model.parameters())))

    def test_semantic_manifest_round_trips_with_reconstructed_model_and_state_dict(self):
        def build():
            binding = _binding(("gA", "gB"), (3, 8), capacity=10)
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
            incompatible_binding = _binding(("gB", "gA"), (8, 3), capacity=10)
            incompatible = attach_expression_view(ExpressionReferenceModel(
                10, 8, binding=incompatible_binding,
            ))
            with self.assertRaisesRegex(ValueError, "do not match"):
                validate_biological_manifest(
                    incompatible, loaded, weights_path=weights,
                )
            state = torch.load(weights, map_location="cpu", weights_only=True)
            reconstructed.load_state_dict(state)


if __name__ == "__main__":
    unittest.main()
