"""CPU contract checks for biological feature identity and observation encoding."""

import unittest

import torch

from cellerator.torch.features import FeatureEncoder


class FeatureEncoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_identity_alignment_is_equivariant_to_feature_reordering(self):
        encoder = FeatureEncoder(12, 7, value_size=1).cpu()
        ids = torch.tensor([[2, 5, 2], [5, 8, 2]], device="cpu")
        values = torch.tensor([[0.0, 1.5, 2.0], [3.0, 0.5, 4.0]], device="cpu")
        permutation = torch.tensor([2, 0, 1], device="cpu")

        encoded = encoder(ids, values)
        reordered = encoder(ids[:, permutation], values[:, permutation])
        torch.testing.assert_close(reordered, encoded[:, permutation])

    def test_observed_zero_differs_from_missing(self):
        encoder = FeatureEncoder(4, 8, value_size=1).cpu()
        ids = torch.tensor([[2]], device="cpu")
        zero = torch.zeros((1, 1), device="cpu")
        measured = encoder(ids, zero, observed=torch.tensor([[True]], device="cpu"))
        missing = encoder(ids, zero, observed=torch.tensor([[False]], device="cpu"))
        self.assertFalse(torch.equal(measured, missing))

    def test_masked_nan_and_withheld_target_cannot_leak_through_projection(self):
        encoder = FeatureEncoder(9, 6, value_size=1).cpu()
        ids = torch.tensor([[1, 3, 5]], device="cpu")
        observed = torch.tensor([[True, False, True]], device="cpu")
        first = torch.tensor([[2.0, float("nan"), -1.0]], device="cpu")
        second = torch.tensor([[2.0, 1.0e30, -1.0]], device="cpu")
        torch.testing.assert_close(
            encoder(ids, first, observed=observed),
            encoder(ids, second, observed=observed),
        )
        probe = second.clone().requires_grad_()
        encoder(ids, probe, observed=observed).square().sum().backward()
        self.assertEqual(float(probe.grad[0, 1]), 0.0)
        self.assertFalse(torch.isnan(probe.grad).any())

    def test_attributes_and_multi_channel_values_are_aligned(self):
        encoder = FeatureEncoder(6, 5, value_size=2, attribute_size=3).cpu()
        ids = torch.tensor([4, 1], device="cpu")
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cpu")
        attrs = torch.tensor([[0.2, 0.4, 0.6], [0.1, 0.3, 0.5]], device="cpu")
        encoded = encoder(ids, values, attributes=attrs)
        self.assertEqual(tuple(encoded.shape), (2, 5))
        with self.assertRaises(ValueError):
            encoder(ids, values, attributes=attrs.flip(0).unsqueeze(0))

    def test_gradients_flow_to_measurements_and_learned_encodings(self):
        encoder = FeatureEncoder(5, 4, value_size=1, attribute_size=2).cpu()
        ids = torch.tensor([[1, 2]], device="cpu")
        values = torch.tensor([[0.5, -0.5]], requires_grad=True, device="cpu")
        attrs = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True, device="cpu")
        output = encoder(ids, values, attributes=attrs)
        output.square().sum().backward()
        self.assertIsNotNone(values.grad)
        self.assertIsNotNone(attrs.grad)
        self.assertIsNotNone(encoder.feature_embedding.weight.grad)
        self.assertIsNotNone(encoder.value_projection.weight.grad)
        self.assertIsNotNone(encoder.attribute_projection.weight.grad)

    def test_invalid_axes_and_configuration_fail_clearly(self):
        encoder = FeatureEncoder(5, 4, value_size=None).cpu()
        with self.assertRaises(ValueError):
            encoder(torch.tensor([[1]], device="cpu"), torch.ones((1, 1), device="cpu"))
        with self.assertRaises(ValueError):
            encoder(torch.tensor([[1]], device="cpu"), observed=torch.ones((1, 2), dtype=torch.bool, device="cpu"))


if __name__ == "__main__":
    unittest.main()
