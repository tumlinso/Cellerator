"""CPU contracts for identity-aligned targets and observation/cell reductions."""
import unittest

import torch

from cellerator.biology import FeatureBinding, Prediction, PreparedAlignment
from cellerator.torch.supervision import (
    align_prediction_targets,
    categorical_target_columns,
    reduce_prediction_loss,
)


class PredictionSupervisionTests(unittest.TestCase):
    def setUp(self):
        self.binding = FeatureBinding.from_ids(
            ("gene-a", "gene-b", "gene-c"),
            model_indices=(8, 3, 5),
            model_capacity=10,
            domain="gene",
        )

    def test_feature_and_sample_identity_alignment_preserves_target_gradients(self):
        alignment = self.binding.prepare(
            ("gene-a", "gene-b", "gene-c"), ("gene-c", "gene-a")
        )
        prediction_values = torch.tensor([[100.0, 200.0], [300.0, 400.0]], requires_grad=True)
        prediction = Prediction(
            prediction_values,
            feature_ids=("gene-c", "gene-a"),
            sample_ids=("cell-2", "cell-1"),
        )
        targets = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)

        aligned_prediction, aligned_targets = align_prediction_targets(
            prediction, targets, alignment,
            target_sample_ids=("cell-1", "cell-2"),
        )

        torch.testing.assert_close(aligned_prediction, prediction_values)
        torch.testing.assert_close(aligned_targets, torch.tensor([[6.0, 4.0], [3.0, 1.0]]))
        (aligned_prediction.sum() + aligned_targets.sum()).backward()
        self.assertIsNotNone(prediction_values.grad)
        self.assertIsNotNone(targets.grad)

    def test_repeated_sample_keys_require_an_explicit_row_map(self):
        alignment = self.binding.prepare(
            ("gene-a", "gene-b", "gene-c"), ("gene-a",)
        )
        prediction = Prediction(
            torch.zeros((2, 1)), feature_ids=("gene-a",), sample_ids=("cell-x", "cell-x")
        )
        targets = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        with self.assertRaisesRegex(ValueError, "ambiguous repeated"):
            align_prediction_targets(
                prediction, targets, alignment, target_sample_ids=("cell-x", "cell-x")
            )
        _, aligned = align_prediction_targets(
            prediction, targets, alignment,
            target_sample_ids=("window-a", "window-b"), row_map=(1, 0),
        )
        torch.testing.assert_close(aligned[:, 0], torch.tensor([4.0, 1.0]))

    def test_packed_sample_windows_align_group_offsets_and_require_matching_extents(self):
        alignment = self.binding.prepare(
            ("gene-a", "gene-b", "gene-c"), ("gene-b",)
        )
        prediction = Prediction(
            torch.zeros((3, 1)), feature_ids=("gene-b",),
            sample_ids=("window-a", "window-b"), row_offsets=(0, 1, 3),
        )
        targets = torch.tensor([
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
        ])
        _, aligned = align_prediction_targets(
            prediction, targets, alignment,
            target_sample_ids=("window-b", "window-a"),
            target_row_offsets=(0, 2, 3),
        )
        torch.testing.assert_close(aligned[:, 0], torch.tensor([30.0, 10.0, 20.0]))

    def test_event_labels_map_to_prediction_columns_not_integer_token_values(self):
        prediction = Prediction(
            torch.zeros((3, 2)), feature_ids=(42, 7)
        )
        columns = categorical_target_columns(prediction, (7, 42, 7))
        torch.testing.assert_close(columns, torch.tensor([1, 0, 1]))
        with self.assertRaisesRegex(KeyError, "no Prediction class column"):
            categorical_target_columns(prediction, (99,))

    def test_observation_reduction_uses_valid_mask_and_empty_mask_returns_grad_zero(self):
        losses = torch.tensor([[1.0, 3.0], [5.0, float("nan")]], requires_grad=True)
        mask = torch.tensor([[True, False], [True, False]])

        mean, count = reduce_prediction_loss(losses, unit="observation", valid_mask=mask)

        self.assertAlmostEqual(float(mean), 3.0)
        self.assertEqual(int(count), 2)
        mean.backward()
        self.assertTrue(torch.isfinite(losses.grad).all())

        all_invalid = torch.zeros_like(mask)
        empty_mean, empty_count = reduce_prediction_loss(
            losses, unit="observation", valid_mask=all_invalid
        )
        self.assertEqual(float(empty_mean), 0.0)
        self.assertEqual(int(empty_count), 0)
        self.assertIsNotNone(empty_mean.grad_fn)

    def test_cell_reduction_weights_cells_equally_across_distinct_windows(self):
        losses = torch.tensor([1.0, 3.0, 5.0, 7.0, 10.0, 14.0], requires_grad=True)
        mean, count = reduce_prediction_loss(
            losses,
            unit="cell",
            row_offsets=(0, 2, 4, 6),
            sample_ids=("cell-a-window-1", "cell-a-window-2", "cell-b-window-1"),
            cell_ids=("cell-a", "cell-a", "cell-b"),
        )
        self.assertAlmostEqual(float(mean), 8.0)
        self.assertEqual(int(count), 2)
        mean.backward()
        self.assertTrue(torch.isfinite(losses.grad).all())

    def test_empty_cell_reduction_returns_differentiable_zero_and_requires_owner_keys(self):
        losses = torch.empty((0,), dtype=torch.float32, requires_grad=True)
        mean, count = reduce_prediction_loss(
            losses, unit="cell", row_offsets=(0, 0),
            sample_ids=("empty-window",), cell_ids=("cell-empty",),
        )
        self.assertEqual(float(mean), 0.0)
        self.assertEqual(int(count), 0)
        self.assertIsNotNone(mean.grad_fn)
        mean.backward()
        self.assertEqual(tuple(losses.grad.shape), (0,))

        with self.assertRaisesRegex(ValueError, "separate sample_ids and physical cell_ids"):
            reduce_prediction_loss(torch.ones(2), unit="cell", cell_ids=("a", "b"))


if __name__ == "__main__":
    unittest.main()
