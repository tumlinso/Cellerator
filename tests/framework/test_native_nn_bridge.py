"""Conformance of existing Python modules with the shared native functions."""
import copy
import json
import unittest
from unittest.mock import patch

import torch

from cellerator.torch._native_nn import native_nn
from cellerator.torch.features import FeatureEncoder
from cellerator.torch.relation_attention import RelationAttention
from cellerator.torch.relations import PreparedRelation
from cellerator import FeatureBinding
from cellerator.torch.views import ModelWithViews
from cellerator.torch.manifest import biological_manifest


class NativeNNBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.native = native_nn("native")
        except ImportError as exc:
            raise unittest.SkipTest(str(exc))

    def setUp(self):
        torch.manual_seed(52)

    def test_feature_values_vjps_and_parameter_identity(self):
        native = FeatureEncoder(9, 8, attribute_size=2, implementation="native")
        reference = copy.deepcopy(native)
        reference.implementation = "reference"
        ids = torch.tensor([[2, 7, 2], [1, 0, 3]])
        observed = torch.tensor([[True, False, True], [False, True, True]])
        values = torch.randn(2, 3)
        values[~observed] = float("nan")
        x = values.clone().requires_grad_()
        y = values.clone().requires_grad_()
        attributes = torch.randn(2, 3, 2)
        owners = [id(parameter) for parameter in native.parameters()]
        with patch.object(self.native, "encode_features", wraps=self.native.encode_features) as call:
            actual = native(ids, x, observed=observed, attributes=attributes)
            self.assertEqual(call.call_count, 1)
        expected = reference(ids, y, observed=observed, attributes=attributes)
        torch.testing.assert_close(actual, expected)
        ag = torch.autograd.grad(actual.square().sum(), (x, *native.parameters()))
        eg = torch.autograd.grad(expected.square().sum(), (y, *reference.parameters()))
        for a, e in zip(ag, eg):
            torch.testing.assert_close(a, e)
        self.assertEqual(owners, [id(parameter) for parameter in native.parameters()])
        self.assertTrue(torch.equal(ag[0][~observed], torch.zeros_like(ag[0][~observed])))

    def test_frozen_feature_parameters_preserve_input_gradient(self):
        model = FeatureEncoder(3, 4, implementation="native").requires_grad_(False)
        values = torch.ones(2, 3, requires_grad=True)
        model(torch.tensor([[0, 1, 2], [2, 1, 0]]), values).sum().backward()
        self.assertIsNotNone(values.grad)
        self.assertGreater(values.grad.abs().sum().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_auto_preserves_custom_child_hooks_and_forward(self):
        model = FeatureEncoder(3, 4)
        ids = torch.tensor([0, 1, 2])
        seen = []
        hook = model.feature_embedding.register_forward_pre_hook(lambda module, args: seen.append(args[0]))
        model(ids)
        hook.remove()
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0], ids)
        attention = RelationAttention(4)
        original = attention.query.forward
        attention.query.forward = lambda values: original(values) * 0.0
        reference = copy.deepcopy(attention)
        reference.implementation = "reference"
        source = torch.randn(1, 3, 4)
        torch.testing.assert_close(attention(source), reference(source))

    def test_dense_and_packed_attention_conformance(self):
        model = RelationAttention(8, heads=2, implementation="native")
        reference = copy.deepcopy(model)
        reference.implementation = "reference"
        for shape, offsets in (((2, 3, 8), None), ((5, 8), (0, 0, 2, 5))):
            x = torch.randn(shape, requires_grad=True)
            y = x.detach().clone().requires_grad_()
            kwargs = {} if offsets is None else {"source_offsets": offsets}
            actual, expected = model(x, **kwargs), reference(y, **kwargs)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
            ag = torch.autograd.grad(actual.square().sum(), (x, *model.parameters()))
            eg = torch.autograd.grad(expected.square().sum(), (y, *reference.parameters()))
            for a, e in zip(ag, eg):
                torch.testing.assert_close(a, e, atol=3e-6, rtol=4e-5)

    def test_prepared_attention_shared_sparse_core(self):
        model = RelationAttention(8, heads=2, implementation="native")
        reference = copy.deepcopy(model)
        reference.implementation = "reference"
        order = ("a", "b", "a", "c")
        relation = PreparedRelation.from_csr(
            [0, 2, 3, 3, 5], [0, 0, 1, 2, 3],
            source_order=order, target_order=order,
            source_offsets=(0, 2, 4), target_offsets=(0, 2, 4),
        )
        x = torch.randn(4, 8, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        kwargs = dict(relation=relation, source_order=order, target_order=order,
                      source_offsets=(0, 2, 4), target_offsets=(0, 2, 4))
        with patch.object(self.native, "sparse_attention", wraps=self.native.sparse_attention) as call:
            actual = model(x, **kwargs)
            self.assertEqual(call.call_count, 1)
        expected = reference(y, **kwargs)
        torch.testing.assert_close(actual, expected)
        for a, e in zip(torch.autograd.grad(actual.square().sum(), (x, *model.parameters())),
                        torch.autograd.grad(expected.square().sum(), (y, *reference.parameters()))):
            torch.testing.assert_close(a, e, atol=3e-6, rtol=3e-5)

    def test_causal_and_batch_isolation(self):
        model = RelationAttention(8, heads=2, implementation="native")
        source = torch.randn(2, 4, 8)
        changed = source.clone()
        changed[0, 3] += 100
        changed[1] -= 50
        torch.testing.assert_close(model(source, causal=True)[0, :3],
                                   model(changed, causal=True)[0, :3])

    def test_native_cpu_product2_trains_with_ordinary_layers(self):
        upstream, downstream = torch.nn.Linear(3, 4), torch.nn.Linear(3, 1)
        coefficient = torch.nn.Parameter(torch.ones(3))
        parameters = [*upstream.parameters(), coefficient, *downstream.parameters()]
        optimizer = torch.optim.SGD(parameters, lr=0.025)
        source = torch.tensor([0.1, 0.2, -0.3])
        lhs, rhs = torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])
        losses = []
        for _ in range(24):
            optimizer.zero_grad()
            products = self.native.product2_cpu(upstream(source), coefficient, lhs, rhs)
            loss = (downstream(products) - 0.7).square().sum()
            loss.backward()
            self.assertTrue(all(parameter.grad is not None for parameter in parameters))
            optimizer.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0])

    def test_actual_python_biological_manifest_roundtrips_native(self):
        genes = FeatureBinding.from_ids(
            ("gene-A", "gene-B"), (2, 7), domain="gene", model_capacity=8,
            metadata={"reference": "fixture-reference"},
        )
        peaks = FeatureBinding.from_ids(
            ("peak-B", "peak-A"), (5, 1), domain="peak", model_capacity=6)
        model = ModelWithViews(FeatureEncoder(8, 4, binding=genes))
        model.add_view("atac", torch.nn.Linear(4, 2), output_binding=peaks)
        order = ("gene-A", "gene-B", "gene-A")
        relation = PreparedRelation.from_csr(
            [0, 1, 2, 3], [0, 1, 2], source_order=order, target_order=order,
            structural_generation=3,
        )
        manifest = biological_manifest(model, relations={"regulatory": relation})
        restored = json.loads(self.native.canonical_biological_manifest(json.dumps(manifest)))
        self.assertEqual(restored, manifest)


class PreparedStorageTests(unittest.TestCase):
    def test_inference_preparation_and_mutation_guard(self):
        with torch.inference_mode():
            relation = PreparedRelation.from_csr(
                [0, 2, 4], [0, 1, 0, 1], source_order=("a", "b"), target_order=("a", "b"))
        causal = relation.causal()
        self.assertFalse(causal.has_future_edges)
        self.assertEqual(causal.row_ptr, (0, 1, 3))
        relation._col_indices.fill_(999)
        model = RelationAttention(4, implementation="reference")
        with self.assertRaisesRegex(ValueError, "mutated"):
            model(torch.randn(2, 4), relation=relation,
                  source_order=("a", "b"), target_order=("a", "b"))


if __name__ == "__main__":
    unittest.main()
