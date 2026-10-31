"""CPU contracts for biological queries, readouts, and semantic descriptors."""
import unittest

import torch
from torch import nn

from cellerator.biology import FeatureBinding, PredictionQuery
from cellerator.torch.readouts import FeatureReadout
from cellerator.torch.views import ModelWithViews


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.projection = nn.Linear(3, 4, bias=False)

    def forward(self, batch):
        self.calls += 1
        values = self.projection(batch["values"])
        return {"features": values, "sample_ids": batch["sample_ids"]}


class FixedFeatureHead(nn.Module):
    def __init__(self, columns):
        super().__init__()
        self.register_buffer("columns", torch.tensor(columns, dtype=torch.float32))

    def forward(self, features):
        return features[..., 0:1] * 0.0 + self.columns


class PackedBaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.features = nn.Parameter(torch.randn(3, 4))

    def forward(self, _batch):
        self.calls += 1
        return {
            "features": self.features,
            "sample_ids": ("window-1", "window-2"),
            "row_offsets": (0, 1, 3),
        }


class BiologicalQueryTests(unittest.TestCase):
    def setUp(self):
        self.binding = FeatureBinding.from_ids(
            ("gene-a", "gene-b", "gene-c"),
            model_indices=(5, 2, 4),
            model_capacity=7,
            domain="gene",
        )
        self.batch = {
            "values": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "sample_ids": ("cell-1", "cell-2"),
        }

    def test_ordinary_output_gathers_by_source_order_not_model_indices(self):
        base = BaseModel()
        model = ModelWithViews(base)
        model.add_view(
            "expression",
            FixedFeatureHead((10.0, 20.0, 30.0)),
            source="features",
            output_binding=self.binding,
        )

        prediction = model.query(self.batch, PredictionQuery(
            "expression", feature_ids=("gene-c", "gene-a"), sample_ids=("cell-1", "cell-2")
        ))

        torch.testing.assert_close(
            prediction.values,
            torch.tensor([[30.0, 10.0], [30.0, 10.0]]),
        )
        self.assertEqual(prediction.feature_ids, ("gene-c", "gene-a"))
        self.assertEqual(prediction.sample_ids, ("cell-1", "cell-2"))
        self.assertEqual(base.calls, 1)

    def test_feature_readout_receives_prepared_model_indices_in_request_order(self):
        torch.manual_seed(201)
        base = BaseModel()
        readout = FeatureReadout(self.binding, width=4, heads=2)
        seen = []
        hook = readout.target_embedding.register_forward_pre_hook(
            lambda _module, args: seen.extend(args[0].tolist())
        )
        model = ModelWithViews(base)
        model.add_view(
            "atac", readout, source="features", output_binding=self.binding
        )
        prediction = model.query(self.batch, PredictionQuery(
            "atac", feature_ids=("gene-c", "gene-a"),
            sample_ids=("cell-1", "cell-2"),
        ))

        self.assertEqual(seen, [4, 5])
        self.assertEqual(prediction.feature_ids, ("gene-c", "gene-a"))
        self.assertEqual(tuple(prediction.values.shape), (2, 2))
        self.assertEqual(base.calls, 1)
        hook.remove()

    def test_frozen_base_keeps_input_gradient_through_feature_readout(self):
        torch.manual_seed(202)
        base = BaseModel()
        base.requires_grad_(False)
        model = ModelWithViews(base)
        readout = FeatureReadout(self.binding, width=4, heads=2)
        model.add_view(
            "rna", readout,
            source="features", output_binding=self.binding,
        )
        values = self.batch["values"].clone().requires_grad_()
        batch = {"values": values, "sample_ids": self.batch["sample_ids"]}

        result = model.query(batch, PredictionQuery(
            "rna", feature_ids=("gene-b",), sample_ids=batch["sample_ids"]
        ))
        result.values.sum().backward()

        self.assertIsNotNone(values.grad)
        self.assertTrue(all(parameter.grad is None for parameter in base.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in readout.parameters()))

    def test_packed_output_offsets_label_sample_windows_not_flat_rows(self):
        base = PackedBaseModel()
        model = ModelWithViews(base)
        model.add_view(
            "expression", FixedFeatureHead((10.0, 20.0, 30.0)),
            source="features", output_binding=self.binding,
        )

        prediction = model.query(object(), PredictionQuery(
            "expression", sample_ids=("window-1", "window-2")
        ))

        self.assertEqual(tuple(prediction.values.shape), (3, 3))
        self.assertEqual(prediction.sample_ids, ("window-1", "window-2"))
        self.assertEqual(prediction.row_offsets, (0, 1, 3))
        self.assertEqual(base.calls, 1)

    def test_multiple_compatible_queries_share_one_base_call_and_preserve_order(self):
        base = BaseModel()
        model = ModelWithViews(base)
        model.add_view(
            "expression", FixedFeatureHead((10.0, 20.0, 30.0)),
            source="features", output_binding=self.binding,
        )
        queries = (
            PredictionQuery("expression", feature_ids=("gene-b",), sample_ids=self.batch["sample_ids"]),
            PredictionQuery("expression", feature_ids=("gene-a", "gene-c"), sample_ids=self.batch["sample_ids"]),
        )

        first, second = model.query(self.batch, queries)

        self.assertEqual(first.feature_ids, ("gene-b",))
        self.assertEqual(second.feature_ids, ("gene-a", "gene-c"))
        torch.testing.assert_close(first.values[0], torch.tensor([20.0]))
        torch.testing.assert_close(second.values[0], torch.tensor([10.0, 30.0]))
        self.assertEqual(base.calls, 1)

    def test_conditioning_is_explicit_grouped_by_object_identity(self):
        base = BaseModel()

        def condition_batch(batch, conditioning):
            return {
                "values": batch["values"] * conditioning,
                "sample_ids": batch["sample_ids"],
            }

        model = ModelWithViews(base, conditioner=condition_batch)
        model.add_view("expression", FixedFeatureHead((1.0, 2.0, 3.0)),
                       source="features", output_binding=self.binding)
        shared = torch.tensor(1.0)
        first, second = model.query(self.batch, (
            PredictionQuery("expression", feature_ids=("gene-a",), conditioning=shared),
            PredictionQuery("expression", feature_ids=("gene-c",), conditioning=shared),
        ))
        self.assertEqual(base.calls, 1)
        self.assertEqual(first.feature_ids, ("gene-a",))
        self.assertEqual(second.feature_ids, ("gene-c",))

        distinct_a, distinct_b = torch.tensor(1.0), torch.tensor(1.0)
        model.query(self.batch, (
            PredictionQuery("expression", conditioning=distinct_a),
            PredictionQuery("expression", conditioning=distinct_b),
        ))
        self.assertEqual(base.calls, 3)

    def test_non_none_conditioning_requires_registered_conditioner_before_base_call(self):
        base = BaseModel()
        model = ModelWithViews(base)
        model.add_view("expression", FixedFeatureHead((1.0, 2.0, 3.0)),
                       source="features", output_binding=self.binding)

        with self.assertRaisesRegex(ValueError, "registered conditioner"):
            model.query(self.batch, PredictionQuery("expression", conditioning=object()))
        self.assertEqual(base.calls, 0)

    def test_context_metadata_is_not_conditioning_and_sample_reordering_is_rejected(self):
        base = BaseModel()
        model = ModelWithViews(base)
        model.add_view("expression", FixedFeatureHead((1.0, 2.0, 3.0)),
                       source="features", output_binding=self.binding)

        prediction = model.query(self.batch, PredictionQuery(
            "expression", context_metadata={"condition": "control"}
        ))
        self.assertEqual(base.calls, 1)
        self.assertEqual(prediction.context["condition"], "control")

        with self.assertRaisesRegex(ValueError, "match the declared batch order"):
            model.query(self.batch, PredictionQuery(
                "expression", sample_ids=("cell-2", "cell-1")
            ))
        self.assertEqual(base.calls, 1)

    def test_view_descriptors_capture_binding_and_require_ids_for_callable_selector(self):
        model = ModelWithViews(BaseModel())
        model.add_view("expression", FixedFeatureHead((1.0, 2.0, 3.0)),
                       source="features", output_binding=self.binding)
        descriptor, = model.biological_view_descriptors()
        self.assertEqual(descriptor["name"], "expression")
        self.assertEqual(descriptor["source_kind"], "mapping_key")
        self.assertEqual(descriptor["source_key"], "features")
        self.assertEqual(descriptor["output_binding"]["model_indices"], [5, 2, 4])

        callable_model = ModelWithViews(BaseModel())
        callable_model.add_view(
            "selected", nn.Identity(), source=lambda result, _batch: result["features"],
            output_binding=self.binding,
        )
        with self.assertRaisesRegex(ValueError, "source_id"):
            callable_model.biological_view_descriptors()

        identified_model = ModelWithViews(BaseModel())
        identified_model.add_view(
            "selected", nn.Identity(), source=lambda result, _batch: result["features"],
            output_binding=self.binding, source_id="feature-selector-v1",
        )
        descriptor, = identified_model.biological_view_descriptors()
        self.assertEqual(descriptor["source_id"], "feature-selector-v1")


if __name__ == "__main__":
    unittest.main()
