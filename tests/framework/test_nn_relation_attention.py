"""CPU contract checks for dense, packed, and CSR relation attention."""

import unittest

import torch

from cellerator.torch.relation_attention import RelationAttention


class RelationAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)

    def test_dense_and_complete_csr_support_agree(self):
        module = RelationAttention(8, heads=2, dropout=0.0).cpu().eval()
        source = torch.randn((2, 3, 8), device="cpu")
        dense = module(source)
        row_ptr = [0, 3, 6, 9, 12, 15, 18]
        columns = [0, 1, 2, 0, 1, 2, 0, 1, 2,
                   3, 4, 5, 3, 4, 5, 3, 4, 5]
        csr = module(source, row_ptr=row_ptr, col_indices=columns)
        torch.testing.assert_close(csr, dense, rtol=1e-5, atol=1e-6)

    def test_packed_offsets_match_independent_group_calls_and_allow_empty_groups(self):
        module = RelationAttention(6, heads=2, dropout=0.0).cpu().eval()
        source = torch.randn((5, 6), device="cpu")
        target = torch.randn((5, 6), device="cpu")
        offsets = (0, 2, 2, 5)
        target_offsets = (0, 2, 3, 5)
        packed = module(
            source, target, source_offsets=offsets, target_offsets=target_offsets
        )
        first = module(source[0:2], target[0:2])
        empty = module(source[2:2], target[2:3])
        third = module(source[2:5], target[3:5])
        expected = torch.cat((first, empty, third), dim=0)
        torch.testing.assert_close(packed, expected, rtol=1e-5, atol=1e-6)

    def test_csr_preserves_duplicates_and_rejects_cross_group_edges(self):
        module = RelationAttention(4, dropout=0.0).cpu().eval()
        source = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                               [0.0, 2.0, 0.0, 0.0]], device="cpu")
        query = torch.zeros_like(source[:1])
        once = module(source, query, row_ptr=[0, 2], col_indices=[0, 1])
        repeated = module(source, query, row_ptr=[0, 3], col_indices=[0, 0, 1])
        self.assertFalse(torch.allclose(once, repeated))

        with self.assertRaisesRegex(ValueError, "crosses a declared group"):
            module(
                source, source, source_offsets=(0, 1, 2),
                target_offsets=(0, 1, 2), row_ptr=[0, 1, 2], col_indices=[1, 0],
            )

    def test_unsegmented_packed_csr_can_express_cross_cell_relations(self):
        module = RelationAttention(4, dropout=0.0).cpu().eval()
        source = torch.randn((2, 4), device="cpu")
        target = torch.randn((2, 4), device="cpu")
        row_ptr = [0, 1, 2]
        columns = [1, 0]

        # With no offsets all occurrences are in one explicit computational
        # group, allowing the caller to route each target to the other cell.
        global_relation = module(
            source, target, row_ptr=row_ptr, col_indices=columns
        )
        self.assertEqual(tuple(global_relation.shape), (2, 4))
        self.assertFalse(torch.equal(global_relation, torch.zeros_like(global_relation)))

        # Declaring one target/source segment per cell makes those same edges
        # invalid, preserving isolation when segmentation is requested.
        with self.assertRaisesRegex(ValueError, "crosses a declared group"):
            module(
                source, target, source_offsets=(0, 1, 2),
                target_offsets=(0, 1, 2), row_ptr=row_ptr,
                col_indices=columns,
            )

    def test_empty_neighborhoods_are_zero_and_differentiable(self):
        module = RelationAttention(4).cpu()
        source = torch.randn((3, 4), requires_grad=True, device="cpu")
        target = torch.randn((2, 4), requires_grad=True, device="cpu")
        output = module(source, target, row_ptr=[0, 0, 0], col_indices=[])
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))
        output.sum().backward()
        self.assertIsNotNone(source.grad)
        self.assertIsNotNone(target.grad)
        self.assertIsNotNone(module.query.weight.grad)
        self.assertIsNotNone(module.output.weight.grad)

    def test_causal_attention_does_not_observe_future_events(self):
        module = RelationAttention(4, heads=2, dropout=0.0).cpu().eval()
        history = torch.randn((1, 5, 4), device="cpu")
        changed_future = history.clone()
        changed_future[:, 3:, :] += 20.0
        original = module(history, causal=True)
        changed = module(changed_future, causal=True)
        torch.testing.assert_close(original[:, :3], changed[:, :3])
        self.assertFalse(torch.allclose(original[:, 4], changed[:, 4]))

        first_two = history[:, :2]
        in_order = module(first_two, causal=True)
        reversed_order = module(first_two.flip(1), causal=True).flip(1)
        self.assertFalse(torch.allclose(in_order, reversed_order))

    def test_packed_causal_support_uses_within_group_order(self):
        module = RelationAttention(4, heads=2, dropout=0.0).cpu().eval()
        source = torch.randn((4, 4), device="cpu")
        offsets = (0, 2, 4)
        full_causal = module(source, source_offsets=offsets, target_offsets=offsets,
                             causal=True)
        row_ptr = [0, 1, 3, 4, 6]
        columns = [0, 0, 1, 2, 2, 3]
        csr_causal = module(source, source_offsets=offsets, target_offsets=offsets,
                            row_ptr=row_ptr, col_indices=columns, causal=True)
        torch.testing.assert_close(full_causal, csr_causal, rtol=1e-5, atol=1e-6)

    def test_segments_are_isolated_and_feature_permutations_are_equivariant(self):
        module = RelationAttention(6, heads=2, dropout=0.0).cpu().eval()
        source = torch.randn((2, 4, 6), device="cpu")
        changed = source.clone()
        changed[1] += 100
        before = module(source)
        after = module(changed)
        torch.testing.assert_close(before[0], after[0])

        permutation = torch.tensor([2, 0, 3, 1], device="cpu")
        permuted = module(source[:, permutation])
        torch.testing.assert_close(permuted, before[:, permutation])

    def test_attention_parameter_and_input_gradients(self):
        module = RelationAttention(4, heads=2, dropout=0.0).cpu()
        source = torch.randn((2, 3, 4), requires_grad=True, device="cpu")
        target = torch.randn((2, 2, 4), requires_grad=True, device="cpu")
        module(source, target).square().sum().backward()
        self.assertIsNotNone(source.grad)
        self.assertIsNotNone(target.grad)
        for parameter in module.parameters():
            self.assertIsNotNone(parameter.grad)

    def test_csr_input_gradcheck(self):
        module = RelationAttention(2, dropout=0.0).cpu().double().eval()
        source = torch.randn((3, 2), dtype=torch.double, requires_grad=True, device="cpu")
        self.assertTrue(torch.autograd.gradcheck(
            lambda values: module(
                values, row_ptr=(0, 1, 3, 4), col_indices=(0, 0, 1, 2)
            ),
            (source,),
            fast_mode=True,
        ))

    def test_malformed_relations_fail_before_attention(self):
        module = RelationAttention(4).cpu()
        source = torch.randn((3, 4), device="cpu")
        with self.assertRaises(ValueError):
            module(source, source_offsets=(0, 2), target_offsets=(0, 3))
        with self.assertRaises(ValueError):
            module(source, row_ptr=[0, 1], col_indices=[0])
        with self.assertRaises(ValueError):
            module(source, row_ptr=[0, 1, 1, 1], col_indices=[3])


if __name__ == "__main__":
    unittest.main()
