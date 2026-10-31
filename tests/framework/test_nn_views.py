"""CPU contracts for named readouts over one shared neural-model result."""
import unittest

import torch
from torch import nn

from cellerator.torch.views import ModelWithViews


class CountingBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 4, bias=False)
        self.calls = 0

    def forward(self, batch):
        self.calls += 1
        features = self.encoder(batch["values"])
        return {"features": features, "batch_marker": batch["marker"]}


class CountingHead(nn.Module):
    def __init__(self, width, output_width):
        super().__init__()
        self.projection = nn.Linear(width, output_width)
        self.calls = 0

    def forward(self, values):
        self.calls += 1
        return self.projection(values)


class ModelWithViewsTests(unittest.TestCase):
    def test_forward_is_a_transparent_base_call(self):
        base = CountingBase()
        model = ModelWithViews(base)
        batch = {"values": torch.ones(2, 3), "marker": object()}
        result = model(batch)
        self.assertEqual(base.calls, 1)
        self.assertEqual(set(result), {"features", "batch_marker"})
        self.assertIs(result["batch_marker"], batch["marker"])
        self.assertTrue(torch.equal(result["features"], base.encoder(batch["values"])))

    def test_predict_without_view_returns_the_base_result_unchanged(self):
        base = CountingBase()
        model = ModelWithViews(base)
        model.add_view("atac", nn.Linear(4, 2), source="features")
        batch = {"values": torch.ones(2, 3), "marker": "same result"}

        result = model.predict(batch, view=None)

        self.assertEqual(base.calls, 1)
        self.assertEqual(set(result), {"features", "batch_marker"})
        self.assertEqual(result["batch_marker"], "same result")

    def test_multiple_views_share_one_base_call_and_only_requested_heads_run(self):
        base = CountingBase()
        model = ModelWithViews(base)
        atac = CountingHead(4, 5)
        rna = CountingHead(4, 3)
        unused = CountingHead(4, 1)
        model.add_view("atac", atac, source="features")
        model.add_view("rna", rna, source=lambda result, _batch: result["features"])
        model.add_view("unused", unused, source="features")
        batch = {"values": torch.randn(7, 3, requires_grad=True), "marker": "paired batch"}

        outputs = model.predict(batch, ["rna", "atac"])

        self.assertEqual(base.calls, 1)
        self.assertEqual(list(outputs), ["rna", "atac"])
        self.assertEqual(outputs["rna"].shape, (7, 3))
        self.assertEqual(outputs["atac"].shape, (7, 5))
        self.assertEqual((rna.calls, atac.calls, unused.calls), (1, 1, 0))
        (outputs["rna"].square().sum() + outputs["atac"].square().sum()).backward()
        self.assertIsNotNone(batch["values"].grad)
        self.assertIsNotNone(base.encoder.weight.grad)
        self.assertIsNotNone(rna.projection.weight.grad)
        self.assertIsNotNone(atac.projection.weight.grad)
        self.assertIsNone(unused.projection.weight.grad)

    def test_single_view_result_and_frozen_base_still_propagates_input_gradient(self):
        base = CountingBase()
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        model = ModelWithViews(base)
        head = CountingHead(4, 2)
        model.add_view("atac", head, source="features")
        values = torch.randn(2, 3, requires_grad=True)

        result = model.predict({"values": values, "marker": None}, "atac")

        self.assertIsInstance(result, torch.Tensor)
        result.sum().backward()
        self.assertIsNotNone(values.grad)
        self.assertIsNone(base.encoder.weight.grad)
        self.assertIsNotNone(head.projection.weight.grad)
        self.assertEqual(base.calls, 1)

    def test_module_source_is_registered_and_receives_base_result_and_original_input(self):
        class Source(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(2.0))

            def forward(self, result, inputs):
                self.marker = inputs["marker"]
                return result["features"] * self.weight

        source = Source()
        base = CountingBase()
        model = ModelWithViews(base)
        model.add_view("atac", nn.Linear(4, 1), source=source)
        values = torch.randn(1, 3, requires_grad=True)
        marker = object()
        model.predict({"values": values, "marker": marker}, "atac").sum().backward()
        self.assertIs(source.marker, marker)
        self.assertTrue(any(parameter is source.weight for parameter in model.parameters()))
        self.assertIsNotNone(source.weight.grad)

    def test_empty_request_does_not_evaluate_base(self):
        base = CountingBase()
        model = ModelWithViews(base)
        self.assertEqual(model.predict(object(), []), {})
        self.assertEqual(base.calls, 0)

    def test_unknown_duplicate_and_invalid_views_fail_before_base_evaluation(self):
        base = CountingBase()
        model = ModelWithViews(base)
        model.add_view("rna", nn.Identity())
        with self.assertRaisesRegex(KeyError, "unknown view"):
            model.predict(None, "atac")
        with self.assertRaisesRegex(ValueError, "unique"):
            model.predict(None, ["rna", "rna"])
        with self.assertRaisesRegex(ValueError, "already registered"):
            model.add_view("rna", nn.Identity())
        with self.assertRaisesRegex(TypeError, "torch.nn.Module"):
            model.add_view("atac", lambda values: values)
        self.assertEqual(base.calls, 0)

    def test_mapping_source_requires_key_present_in_base_result(self):
        model = ModelWithViews(CountingBase())
        model.add_view("atac", nn.Identity(), source="missing")
        with self.assertRaisesRegex(KeyError, "no view source"):
            model.predict({"values": torch.ones(1, 3), "marker": None}, "atac")


if __name__ == "__main__":
    unittest.main()
