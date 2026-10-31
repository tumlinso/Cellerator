import unittest

import torch

from cellerator.adapters.torch import GatedInteractor
from cellerator.torch.programs import InteractionPrograms
from cellerator.torch.relations import PreparedRelation


class InteractionProgramsTests(unittest.TestCase):
    def test_dense_programs_return_feature_updates_and_train_normally(self):
        torch.manual_seed(101)
        model = InteractionPrograms(4, 3, heads=2, program_ids=("p0", "p1", "p2")).cpu()
        features = torch.randn(2, 5, 4, requires_grad=True)

        output = model(features)

        self.assertEqual(set(output), {"program_features", "feature_updates"})
        self.assertEqual(tuple(output["program_features"].shape), (2, 3, 4))
        self.assertEqual(tuple(output["feature_updates"].shape), (2, 5, 4))
        (output["program_features"].square().mean() + output["feature_updates"].square().mean()).backward()
        self.assertIsNotNone(model.program_queries.grad)
        self.assertIsNotNone(features.grad)
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_prepared_overlapping_membership_preserves_overlap_and_empty_support(self):
        torch.manual_seed(102)
        program_ids = ("module-a", "module-b", "module-empty")
        feature_order = ("g0", "g1", "g2", "g3")
        # g1 participates in two modules. Module-empty and g3 have no support.
        membership = PreparedRelation.from_dense(
            [
                [True, True, False, False],
                [False, True, True, False],
                [False, False, False, False],
            ],
            source_order=feature_order,
            target_order=program_ids,
            structural_generation=7,
        )
        model = InteractionPrograms(4, 3, heads=2, program_ids=program_ids).cpu()
        features = torch.randn(4, 4, requires_grad=True)

        output = model(
            features, relation=membership, feature_order=feature_order,
            structural_generation=7,
        )

        self.assertEqual(tuple(output["program_features"].shape), (3, 4))
        self.assertEqual(tuple(output["feature_updates"].shape), (4, 4))
        torch.testing.assert_close(output["program_features"][2], torch.zeros(4))
        torch.testing.assert_close(output["feature_updates"][3], torch.zeros(4))
        output["feature_updates"].square().sum().backward()
        self.assertIsNotNone(features.grad)
        self.assertIsNotNone(model.program_queries.grad)

    def test_duplicate_membership_is_preserved_through_prepared_composition(self):
        relation = PreparedRelation.from_csr(
            [0, 3], [0, 0, 1], source_order=("g0", "g1"), target_order=("p",)
        )
        self.assertEqual(relation.col_indices, (0, 0, 1))
        self.assertEqual(relation.transpose().row_ptr, (0, 2, 3))
        self.assertEqual(relation.transpose().col_indices, (0, 0, 0))
        model = InteractionPrograms(2, 1, program_ids=("p",)).cpu()
        output = model(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]), relation=relation,
            feature_order=("g0", "g1"),
        )
        self.assertTrue(torch.isfinite(output["feature_updates"]).all())

    def test_dense_all_to_all_and_prepared_full_support_agree(self):
        torch.manual_seed(103)
        model = InteractionPrograms(4, 3, heads=2, program_ids=("a", "b", "c")).cpu().eval()
        feature_order = ("g0", "g1", "g2", "g3")
        features = torch.randn(4, 4)
        full = PreparedRelation.from_dense(
            [[True] * 4 for _ in range(3)],
            source_order=feature_order, target_order=("a", "b", "c"),
        )

        dense = model(features)
        prepared = model(features, relation=full, feature_order=feature_order)

        torch.testing.assert_close(prepared["program_features"], dense["program_features"])
        torch.testing.assert_close(prepared["feature_updates"], dense["feature_updates"])

    def test_context_changes_queries_and_receives_gradients(self):
        torch.manual_seed(104)
        model = InteractionPrograms(4, 3, context_size=2, program_ids=("a", "b", "c")).cpu().eval()
        features = torch.randn(5, 4)
        zero_context = torch.zeros(2)
        context = torch.tensor([0.2, 0.7], requires_grad=True)

        zero = model(features, context=zero_context)["feature_updates"]
        conditioned = model(features, context=context)["feature_updates"]
        self.assertFalse(torch.allclose(conditioned, zero))
        conditioned.square().sum().backward()
        self.assertIsNotNone(context.grad)
        self.assertIsNotNone(model.context_projection.weight.grad)

    def test_unconfigured_context_is_rejected_and_stale_binding_fails(self):
        model = InteractionPrograms(3, 2, program_ids=("p0", "p1")).cpu()
        features = torch.randn(2, 3)
        with self.assertRaisesRegex(ValueError, "context_size"):
            model(features, context=torch.ones(2))

        relation = PreparedRelation.from_csr(
            [0, 1, 2], [0, 1], source_order=("a", "b"),
            target_order=("p0", "p1"), structural_generation=2,
        )
        with self.assertRaisesRegex(ValueError, "source occurrence order"):
            model(features, relation=relation, feature_order=("b", "a"), structural_generation=2)
        with self.assertRaisesRegex(ValueError, "generation"):
            model(features, relation=relation, feature_order=("a", "b"), structural_generation=3)

    def test_program_descriptors_are_explicit_and_require_codec_for_custom_ids(self):
        model = InteractionPrograms(3, 2, program_ids=("p0", "p1")).cpu()
        self.assertEqual(
            model.cellerator_biological_descriptors()["interaction_programs"]["program_ids"],
            ["p0", "p1"],
        )
        custom = InteractionPrograms(3, 1, program_ids=(object(),)).cpu()
        with self.assertRaisesRegex(TypeError, "codec"):
            custom.cellerator_biological_descriptors()


class GatedInteractorContextTests(unittest.TestCase):
    def test_no_context_preserves_legacy_computation_and_unconfigured_context_errors(self):
        torch.manual_seed(105)
        module = GatedInteractor(3, 5, 2).cpu()
        participants = torch.randn(4, 3)
        hidden = torch.tanh(module.encode(participants))
        weights = torch.sigmoid(module.gate(participants))
        expected = module.joint((weights * hidden).sum(dim=-2))

        actual = module(participants, context=None)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "no context_size"):
            module(participants, context=torch.ones(2))

    def test_registered_context_conditions_participants_and_trains(self):
        torch.manual_seed(106)
        module = GatedInteractor(3, 5, 2, context_size=2).cpu().eval()
        participants = torch.randn(2, 4, 3)
        zero = torch.zeros(2, 2)
        context = torch.tensor([[0.1, 0.7], [0.4, 0.3]], requires_grad=True)

        baseline = module(participants, context=zero)
        conditioned = module(participants, context=context)
        self.assertFalse(torch.allclose(conditioned, baseline))
        conditioned.square().sum().backward()
        self.assertIsNotNone(context.grad)
        self.assertIsNotNone(module.context_projection.weight.grad)
        with self.assertRaisesRegex(ValueError, "align with participant groups"):
            module(participants, context=torch.ones(2, 4, 2))


if __name__ == "__main__":
    unittest.main()
