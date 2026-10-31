import unittest
from unittest.mock import patch

import torch

from cellerator._declarations import Axis, Identity
from cellerator.torch.relation_attention import RelationAttention
from cellerator.torch.relations import PreparedRelation


class PreparedRelationTests(unittest.TestCase):
    def test_csr_owns_indices_preserves_duplicate_occurrences_and_empty_rows(self):
        ptr = torch.tensor([0, 2, 2, 4], dtype=torch.int32)
        columns = torch.tensor([0, 0, 1, 2], dtype=torch.int64)
        source_order = ("gene-a", "gene-a", "gene-b")
        target_order = ("program-x", "program-empty", "program-y")
        relation = PreparedRelation.from_csr(
            ptr, columns, source_order=source_order, target_order=target_order
        )
        ptr.fill_(99)
        columns.fill_(99)

        self.assertEqual(relation.row_ptr, (0, 2, 2, 4))
        self.assertEqual(relation.col_indices, (0, 0, 1, 2))
        self.assertEqual(relation.row_indices, (0, 0, 2, 2))
        self.assertEqual(relation.source_order, source_order)
        self.assertEqual(relation.target_order, target_order)
        self.assertEqual(relation.device, torch.device("cpu"))
        self.assertIs(relation.to("cpu"), relation)

    def test_transpose_preserves_repeated_edges_and_swaps_occurrence_orders(self):
        relation = PreparedRelation.from_csr(
            [0, 3, 4], [0, 0, 1, 0],
            source_order=("a", "b"), target_order=("p", "q"),
        )
        transposed = relation.transpose()

        self.assertEqual(transposed.source_order, ("p", "q"))
        self.assertEqual(transposed.target_order, ("a", "b"))
        self.assertEqual(transposed.row_ptr, (0, 3, 4))
        self.assertEqual(transposed.col_indices, (0, 0, 1, 0))
        self.assertEqual(transposed.row_indices, (0, 0, 0, 1))

    def test_dense_support_prepares_same_csr_and_checks_shape(self):
        relation = PreparedRelation.from_dense(
            [[True, False, True], [False, False, False]],
            source_order=(0, 1, 2), target_order=("p", "empty"),
        )
        self.assertEqual(relation.row_ptr, (0, 2, 2))
        self.assertEqual(relation.col_indices, (0, 2))
        with self.assertRaisesRegex(ValueError, "shape"):
            PreparedRelation.from_dense(
                [[True]], source_order=(0, 1), target_order=("p",)
            )

    def test_segmented_cross_group_edges_are_rejected_once_at_preparation(self):
        with self.assertRaisesRegex(ValueError, "group boundary"):
            PreparedRelation.from_csr(
                [0, 1, 1], [1], source_order=("a", "b"),
                target_order=("x", "y"), source_offsets=(0, 1, 2),
                target_offsets=(0, 1, 2),
            )
        # With one global computational group, a selected cross-cell edge is
        # represented directly by its global source occurrence position.
        global_relation = PreparedRelation.from_csr(
            [0, 1, 1], [1], source_order=("cell-a/g", "cell-b/g"),
            target_order=("cell-a/g", "cell-b/g"),
        )
        self.assertEqual(global_relation.col_indices, (1,))

    def test_axis_extent_and_generation_are_part_of_binding(self):
        axis = Axis(Identity(1, 2), Identity(3, 4), Identity(5, 6), Identity(7, 8), 2)
        relation = PreparedRelation.from_csr(
            [0, 1, 2], [0, 1], source_order=("a", "b"), target_order=("x", "y"),
            source_axis=axis, target_axis=axis, structural_generation=4,
        )
        with self.assertRaisesRegex(ValueError, "extent"):
            PreparedRelation.from_csr(
                [0, 0], [], source_order=("a", "b"), target_order=("x",),
                target_axis=axis,
            )
        module = RelationAttention(3).cpu()
        source, target = torch.randn(2, 3), torch.randn(2, 3)
        kwargs = dict(
            relation=relation, source_order=("a", "b"), target_order=("x", "y"),
            source_axis=axis, target_axis=axis, structural_generation=4,
        )
        module(source, target, **kwargs)
        with self.assertRaisesRegex(ValueError, "generation"):
            module(source, target, **{**kwargs, "structural_generation": 5})
        with self.assertRaisesRegex(ValueError, "source occurrence order"):
            module(source, target, **{**kwargs, "source_order": ("b", "a")})
        with self.assertRaisesRegex(ValueError, "target occurrence order"):
            module(source, target, **{**kwargs, "target_order": ("y", "x")})

    def test_prepared_attention_matches_raw_csr_and_gradients_without_item(self):
        torch.manual_seed(41)
        module = RelationAttention(4, heads=2, dropout=0.0).double().cpu().eval()
        source_order = ("a", "a", "b", "c")
        target_order = ("x", "y", "z")
        row_ptr, columns = (0, 2, 2, 4), (0, 0, 1, 3)
        relation = PreparedRelation.from_csr(
            row_ptr, columns, source_order=source_order, target_order=target_order
        )
        source = torch.randn(4, 4, dtype=torch.double, requires_grad=True)
        target = torch.randn(3, 4, dtype=torch.double, requires_grad=True)

        raw = module(source, target, row_ptr=row_ptr, col_indices=columns)
        raw_grads = torch.autograd.grad(raw.square().sum(), (source, target, *module.parameters()))
        source2 = source.detach().clone().requires_grad_(True)
        target2 = target.detach().clone().requires_grad_(True)
        with patch.object(torch.Tensor, "item", side_effect=AssertionError("prepared path synchronized")):
            prepared = module(
                source2, target2, relation=relation,
                source_order=source_order, target_order=target_order,
            )
            prepared_grads = torch.autograd.grad(
                prepared.square().sum(), (source2, target2, *module.parameters())
            )

        torch.testing.assert_close(prepared, raw)
        for actual, expected in zip(prepared_grads, raw_grads):
            torch.testing.assert_close(actual, expected)

    def test_dense_prepared_support_matches_segmented_raw_csr(self):
        torch.manual_seed(42)
        module = RelationAttention(4, heads=2, dropout=0.0).cpu().eval()
        source_order = (("cell-a", 0), ("cell-a", 1), ("cell-a", 2),
                        ("cell-b", 0), ("cell-b", 1), ("cell-b", 2))
        target_order = (("cell-a", "t0"), ("cell-a", "t1"),
                        ("cell-b", "t0"), ("cell-b", "t1"))
        support = [
            [True, True, True, False, False, False],
            [True, True, True, False, False, False],
            [False, False, False, True, True, True],
            [False, False, False, True, True, True],
        ]
        relation = PreparedRelation.from_dense(
            support, source_order=source_order, target_order=target_order,
            source_offsets=(0, 3, 6), target_offsets=(0, 2, 4),
        )
        row_ptr, columns = relation.row_ptr, relation.col_indices
        source, target = torch.randn(2, 3, 4), torch.randn(2, 2, 4)

        raw_output = module(source, target, row_ptr=row_ptr, col_indices=columns)
        prepared_output = module(
            source, target, relation=relation,
            source_order=source_order, target_order=target_order,
        )

        torch.testing.assert_close(prepared_output, raw_output)

    def test_prepared_attention_checks_offsets_extents_and_explicit_device(self):
        relation = PreparedRelation.from_csr(
            [0, 1, 2], [0, 1], source_order=("a", "b"), target_order=("x", "y"),
            source_offsets=(0, 1, 2), target_offsets=(0, 1, 2),
        )
        module = RelationAttention(2).cpu()
        source, target = torch.randn(2, 2), torch.randn(2, 2)
        with self.assertRaisesRegex(ValueError, "offsets"):
            module(source, target, relation=relation, source_order=("a", "b"),
                   target_order=("x", "y"))
        module(source, target, relation=relation, source_order=("a", "b"),
               target_order=("x", "y"), source_offsets=(0, 1, 2),
               target_offsets=(0, 1, 2))
        with self.assertRaisesRegex(ValueError, "extents"):
            module(torch.randn(3, 2), target, relation=relation,
                   source_order=("a", "b"), target_order=("x", "y"),
                   source_offsets=(0, 1, 3), target_offsets=(0, 1, 2))

    def test_manifest_requires_codec_for_custom_ids_and_preserves_occurrences(self):
        identity = Identity(11, 12)
        relation = PreparedRelation.from_csr(
            [0, 2], [0, 1], source_order=(identity, identity), target_order=("p",)
        )
        manifest = relation.to_manifest()
        self.assertEqual(manifest["source_order"], [
            {"kind": "cellerator.identity.v1", "low": 11, "high": 12},
            {"kind": "cellerator.identity.v1", "low": 11, "high": 12},
        ])
        custom = object()
        custom_relation = PreparedRelation.from_csr(
            [0, 1], [0], source_order=(custom,), target_order=("p",)
        )
        with self.assertRaisesRegex(TypeError, "codec"):
            custom_relation.to_manifest()
        self.assertEqual(custom_relation.to_manifest(codec=lambda value: "feature-x")["source_order"], ["feature-x"])


if __name__ == "__main__":
    unittest.main()
