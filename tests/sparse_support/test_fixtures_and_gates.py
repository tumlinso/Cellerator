import unittest
import torch

from cellerator.torch.structure import (  # noqa: E402
    SupportSpec, _stable_scalar_ids, sparse_parameter_layout,
)
from cellerator.torch.gating import learned_gates  # noqa: E402
from cellerator.torch.testing import (  # noqa: E402
    make_masked_gene_fixture, make_next_event_fixture,
)


class SparseFixtureAndGateTest(unittest.TestCase):
    def test_event_fixture_is_deterministic_and_contains_repeated_occurrences(self):
        first = make_next_event_fixture(seed=23, cells=4, steps=9, entities=24)
        second = make_next_event_fixture(seed=23, cells=4, steps=9, entities=24)
        self.assertTrue(torch.equal(first["events"], second["events"]))
        self.assertEqual(tuple(first["events"].shape), (4, 9))
        self.assertEqual(tuple(first["targets"].shape), (4,))
        self.assertTrue(any(len(set(row.tolist())) < row.numel() for row in first["events"]))
        self.assertEqual(tuple(first["validator_support"].shape), (4, 4))
        for row, window, mechanism in zip(first["events"], first["count_windows"],
                                          first["validator_mechanism"]):
            recent = set(row[window[0]:window[1] + 1].tolist())
            required = set(first["validator_support"][mechanism, :3].tolist())
            self.assertTrue(required.issubset(recent))
        # Timestamp eligibility is a separate condition. It excludes the
        # first count-window occurrence while retaining its later repetition.
        self.assertTrue(torch.all(
            first["time_windows"][:, 0]
            > first["timestamps"].gather(1, first["count_windows"][:, :1]).squeeze(1)))


    def test_ranked_fixture_keeps_identity_rank_and_mask_separate(self):
        fixture = make_masked_gene_fixture(seed=19, cells=5, selected=6, entities=24)
        self.assertEqual(tuple(fixture["genes"].shape), (5, 6))
        self.assertEqual(tuple(fixture["ranks"].shape), (5, 6))
        self.assertTrue(torch.all((fixture["genes"] == -1).sum(dim=1) == 1))
        self.assertTrue(torch.equal(fixture["ranks"], torch.arange(6).expand(5, -1)))
        self.assertTrue(torch.all(fixture["targets"] >= 0))
        for genes, target, rank_dependent in zip(
                fixture["genes"], fixture["targets"], fixture["validator_rank_dependent"]):
            module = (int(target) - 16) // 2
            module_target = 16 + 2 * module
            if rank_dependent:
                first, second = fixture["validator_support"][module, :2]
                rank_order = int(torch.nonzero(genes == first)[0]) < int(torch.nonzero(genes == second)[0])
                self.assertEqual(int(target), module_target + int(rank_order))
            else:
                self.assertEqual(int(target), module_target)


    def test_support_spec_preserves_sorted_identity_and_rejects_bad_inputs(self):
        spec = SupportSpec((3, 8), (0, 1), (1, 2), structure_epoch=7)
        self.assertEqual(spec.edge_ids, (3, 8))
        with self.assertRaisesRegex(ValueError, "unique and sorted"):
            SupportSpec((8, 3), (0, 1), (1, 2))
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            SupportSpec((1,), (0, 1), (2,))

    def test_parameter_layout_is_shared_and_scalar_ids_are_stable(self):
        blocks, values = sparse_parameter_layout(entities=24, mechanisms=4,
                                                  rank_count=8)
        self.assertEqual([block["name"] for block in blocks], [
            "entity_embedding", "rank_embedding", "scorer", "transition_A",
            "transition_B", "transition_bias",
        ])
        self.assertEqual(sum(block["numel"] for block in blocks), values.numel())
        axis = tuple(range(8))
        first = _stable_scalar_ids(axis, 12)
        self.assertEqual(first, _stable_scalar_ids(axis, 12))
        self.assertEqual(len(set(first)), 12)
        with self.assertRaisesRegex(ValueError, "uint64"):
            _stable_scalar_ids((-1,) + axis[1:], 1)


    def test_pruned_gate_has_hard_forward_and_explicit_ste_gradient(self):
        score = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
        p, admitted, q = learned_gates(score, temperature=1.0, threshold=0.7,
                                       prune=True, straight_through=True)
        self.assertTrue(torch.equal(admitted, torch.tensor([False, False, True])))
        self.assertTrue(torch.equal(q.detach(), torch.tensor([0.0, 0.0, p.detach()[2]])))
        q.sum().backward()
        self.assertTrue(torch.allclose(score.grad, p.detach() * (1.0 - p.detach())))
