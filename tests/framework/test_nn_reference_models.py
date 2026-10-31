"""CPU acceptance for the expression, event-history, and auxiliary-view examples."""
import unittest

import torch
from torch import nn

from cellerator.adapters.anndata import AnnDataView
from cellerator.modeling import Cell
from examples.nn_foundation.reference_models import (
    EventHistoryReferenceModel,
    ExpressionReferenceModel,
    atac_binary_loss,
    attach_atac_view,
    attach_history_view,
    event_next_token_loss,
    explicit_dense_anndata_minibatch,
    masked_expression_mse,
)


class ReferenceModelTests(unittest.TestCase):
    def test_expression_feature_reordering_reorders_predictions_without_changing_pool(self):
        torch.manual_seed(11)
        model = ExpressionReferenceModel(12, 8, heads=2).eval()
        ids = torch.tensor([[2, 7, 1, 4], [4, 2, 7, 1]], dtype=torch.long)
        values = torch.tensor([[0.2, 1.3, 0.7, 2.0], [0.8, 0.4, 1.0, 0.5]])
        observed = torch.ones_like(ids, dtype=torch.bool)
        order = torch.tensor([2, 0, 3, 1])

        original = model({"feature_ids": ids, "values": values, "observed": observed})
        reordered = model({
            "feature_ids": ids.index_select(1, order),
            "values": values.index_select(1, order),
            "observed": observed.index_select(1, order),
        })

        torch.testing.assert_close(
            reordered["expression"], original["expression"].index_select(1, order)
        )
        torch.testing.assert_close(reordered["cell_features"], original["cell_features"])

    def test_masked_measurements_cannot_leak_into_expression_encoding(self):
        torch.manual_seed(12)
        model = ExpressionReferenceModel(8, 4).eval()
        ids = torch.tensor([[1, 2, 3]])
        values = torch.tensor([[0.5, float("nan"), 1.7]])
        observed = torch.tensor([[True, False, True]])
        altered = values.clone()
        altered[0, 1] = 100000.0

        first = model({"feature_ids": ids, "values": values, "observed": observed})
        second = model({"feature_ids": ids, "values": altered, "observed": observed})

        torch.testing.assert_close(first["expression"], second["expression"])
        self.assertIsNotNone(first["expression"].grad_fn)

    def test_masked_mse_uses_only_requested_observed_targets_and_empty_mask_is_none(self):
        prediction = torch.tensor([[1.0, 2.0, 8.0]])
        target = torch.tensor([[0.0, 4.0, float("nan")]])
        mask = torch.tensor([[True, True, False]])

        loss = masked_expression_mse(prediction, target, mask)

        self.assertAlmostEqual(float(loss), 2.5)
        self.assertIsNone(masked_expression_mse(prediction, target, torch.zeros_like(mask)))

    def test_ann_data_sparse_rows_are_densified_only_after_the_explicit_bound(self):
        try:
            from scipy import sparse
        except ImportError:
            self.skipTest("scipy is not installed")
        matrix = sparse.csr_matrix([[1.0, 0.0, 2.0], [0.0, 4.0, 0.0]])
        view = AnnDataView(matrix, ("cell-a", "cell-b"), ("g1", "g2", "g3"), "X")

        values, feature_ids, cell_ids = explicit_dense_anndata_minibatch(
            view, [1], max_elements=3
        )

        self.assertEqual(tuple(values.shape), (1, 3))
        self.assertEqual(feature_ids, ("g1", "g2", "g3"))
        self.assertEqual(cell_ids, ("cell-b",))
        with self.assertRaisesRegex(ValueError, "max_elements"):
            explicit_dense_anndata_minibatch(view, [0, 1], max_elements=5)

    def test_event_teacher_forcing_preserves_repeats_start_and_empty_rows(self):
        torch.manual_seed(13)
        model = EventHistoryReferenceModel(4, 8, heads=2, max_history=12)
        events = torch.tensor([1, 1, 3], dtype=torch.long)
        offsets = [0, 3, 3]

        output = model(events, offsets)

        self.assertEqual(output["context_offsets"], (0, 3, 4))
        self.assertEqual(output["context_ids"].tolist(), [4, 1, 1, 4])
        self.assertEqual(output["targets"].tolist(), [1, 1, 3])
        self.assertEqual(tuple(output["logits"].shape), (3, 4))
        loss = event_next_token_loss(output["logits"], output["targets"])
        self.assertIsNotNone(loss)
        loss.backward()
        self.assertIsNotNone(model.encoder.feature_embedding.weight.grad)

        empty = model(torch.empty(0, dtype=torch.long), [0, 0])
        self.assertEqual(empty["context_ids"].tolist(), [4])
        self.assertEqual(tuple(empty["logits"].shape), (0, 4))
        self.assertIsNone(event_next_token_loss(empty["logits"], empty["targets"]))

    def test_leading_and_consecutive_empty_event_rows_do_not_consume_future_events(self):
        torch.manual_seed(132)
        model = EventHistoryReferenceModel(5, 8, heads=2, max_history=12).eval()
        events = torch.tensor([1, 2, 3])
        offsets = [0, 0, 0, 3]

        first = model(events, offsets)
        changed = model(torch.tensor([3, 2, 1]), offsets)

        self.assertEqual(first["context_ids"].tolist(), [5, 5, 5, 1, 2])
        self.assertEqual(first["context_offsets"], (0, 1, 2, 5))
        self.assertEqual(first["targets"].tolist(), [1, 2, 3])
        self.assertEqual(tuple(first["history_features"].shape), (3, 8))
        torch.testing.assert_close(
            first["history_features"][:2], changed["history_features"][:2]
        )

    def test_event_mapping_cell_prediction_and_shared_history_view(self):
        torch.manual_seed(133)
        base = EventHistoryReferenceModel(5, 8, heads=2, max_history=12)
        wrapped = attach_history_view(base, "atac", output_size=3)
        calls = []
        hook = base.register_forward_hook(lambda *_: calls.append(1))
        batch = {
            "event_ids": torch.tensor([1, 1, 3, 2, 4]),
            "event_offsets": [0, 3, 3, 5],
        }

        base_result = wrapped.predict(batch)
        self.assertEqual(tuple(base_result["history_features"].shape), (3, 8))
        self.assertEqual(len(calls), 1)
        auxiliary = wrapped.predict(batch, "atac")
        self.assertEqual(tuple(auxiliary.shape), (3, 3))
        self.assertEqual(len(calls), 2)
        auxiliary.square().mean().backward()
        head = wrapped._views[wrapped._view_indices["atac"]].module
        self.assertIsNotNone(head.weight.grad)
        self.assertTrue(any(parameter.grad is not None for parameter in base.parameters()))

        cell = Cell({
            "event_ids": torch.tensor([2, 2, 4]),
            "event_offsets": [0, 3],
        }, id="history-cell").bind(wrapped)
        self.assertEqual(tuple(cell.predict()["logits"].shape), (3, 5))
        cell_view = cell.predict("atac")
        self.assertEqual(tuple(cell_view.shape), (1, 3))
        self.assertEqual(cell.id, "history-cell")
        self.assertEqual(len(calls), 4)
        hook.remove()

    def test_event_prediction_is_causal_and_isolated_between_packed_rows(self):
        torch.manual_seed(14)
        model = EventHistoryReferenceModel(6, 8, heads=2, max_history=12).eval()
        # Row 0 contains repeated events, row 1 contains two events.
        events = torch.tensor([1, 2, 1, 4, 5])
        offsets = [0, 3, 5]
        original = model(events, offsets)["logits"]

        future_changed = events.clone()
        future_changed[2] = 3
        changed_future_logits = model(future_changed, offsets)["logits"]
        torch.testing.assert_close(changed_future_logits[:2], original[:2])

        other_row_changed = events.clone()
        other_row_changed[3:] = torch.tensor([0, 0])
        changed_other_row_logits = model(other_row_changed, offsets)["logits"]
        torch.testing.assert_close(changed_other_row_logits[:3], original[:3])

    def test_atac_auxiliary_head_trains_frozen_or_joint_base_with_standard_autograd(self):
        torch.manual_seed(15)
        base = ExpressionReferenceModel(10, 6, heads=2)
        wrapped = attach_atac_view(base, peak_count=5)
        batch = {
            "feature_ids": torch.tensor([[1, 2, 3], [4, 2, 1], [3, 5, 1]]),
            "values": torch.rand(3, 3),
            "observed": torch.ones(3, 3, dtype=torch.bool),
        }
        atac_targets = torch.tensor([
            [1.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 1.0],
        ])
        base.requires_grad_(False)
        frozen_loss = atac_binary_loss(wrapped.predict(batch, "atac"), atac_targets)
        frozen_loss.backward()
        head = wrapped._views[wrapped._view_indices["atac"]].module
        self.assertIsNotNone(head.weight.grad)
        self.assertTrue(all(parameter.grad is None for parameter in base.parameters()))

        wrapped.zero_grad(set_to_none=True)
        base.requires_grad_(True)
        joint_loss = atac_binary_loss(wrapped.predict(batch, "atac"), atac_targets)
        joint_loss.backward()
        self.assertIsNotNone(head.weight.grad)
        self.assertTrue(any(parameter.grad is not None for parameter in base.parameters()))

    def test_standard_optimizer_updates_expression_and_event_models_and_skips_empty_targets(self):
        torch.manual_seed(151)
        expression = ExpressionReferenceModel(5, 8, heads=2)
        expression_optimizer = torch.optim.Adam(expression.parameters(), lr=0.02)
        expression_batch = {
            "feature_ids": torch.tensor([[0, 1, 2], [0, 1, 2]]),
            "values": torch.tensor([[0.2, 0.7, 0.3], [0.8, 0.1, 0.6]]),
            "observed": torch.tensor([[True, False, True], [True, False, True]]),
        }
        target_mask = torch.tensor([[False, True, False], [False, True, False]])
        expression_before = expression.encoder.feature_embedding.weight.detach().clone()
        expression_loss = masked_expression_mse(
            expression(expression_batch)["expression"],
            expression_batch["values"],
            target_mask,
        )
        expression_optimizer.zero_grad(set_to_none=True)
        expression_loss.backward()
        expression_optimizer.step()
        self.assertFalse(torch.equal(
            expression_before, expression.encoder.feature_embedding.weight.detach()
        ))

        events = EventHistoryReferenceModel(3, 8, heads=2, max_history=8)
        event_optimizer = torch.optim.Adam(events.parameters(), lr=0.02)
        event_batch = {"event_ids": torch.tensor([0, 1, 0, 1]), "event_offsets": [0, 4]}
        event_before = events.encoder.feature_embedding.weight.detach().clone()
        output = events(event_batch)
        event_loss = event_next_token_loss(output["logits"], output["targets"])
        event_optimizer.zero_grad(set_to_none=True)
        event_loss.backward()
        event_optimizer.step()
        self.assertFalse(torch.equal(
            event_before, events.encoder.feature_embedding.weight.detach()
        ))

        empty_batch = {"event_ids": torch.empty(0, dtype=torch.long), "event_offsets": [0, 0]}
        empty_before = [parameter.detach().clone() for parameter in events.parameters()]
        empty_output = events(empty_batch)
        empty_loss = event_next_token_loss(empty_output["logits"], empty_output["targets"])
        self.assertIsNone(empty_loss)
        if empty_loss is not None:
            event_optimizer.zero_grad(set_to_none=True)
            empty_loss.backward()
            event_optimizer.step()
        self.assertTrue(all(torch.equal(old, new.detach())
                            for old, new in zip(empty_before, events.parameters())))


if __name__ == "__main__":
    unittest.main()
