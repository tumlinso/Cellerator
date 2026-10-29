"""Focused CPU checks for the optional learnable-model adapter."""
import unittest

import torch
from torch import nn

from cellerator.interactions import Interaction, InteractionGraph
from cellerator.modeling import Cell, StepContext
from cellerator.adapters.torch import (
    EventPredictionHead,
    GatedInteractor,
    ResidualStateUpdate,
    SparseExpertBank,
    SparseRouting,
    TorchModel,
)


class TorchAdapterTests(unittest.TestCase):
    def test_shared_torch_dynamics_advances_independent_cell_states_and_backpropagates(self):
        module = ResidualStateUpdate(3, hidden_size=5, input_size=2)
        model = TorchModel(module).as_model()
        x1 = torch.tensor([0.2, 0.5, 0.7], requires_grad=True)
        x2 = torch.tensor([0.9, 0.1, 0.3], requires_grad=True)
        first = Cell(model, x1, id="a")
        second = Cell(model, x2, id="b")

        result = first.advance(0.25, inputs=torch.tensor([0.1, 0.8]))
        self.assertEqual(first.state.shape, x1.shape)
        self.assertTrue(torch.equal(second.state, x2))
        self.assertEqual(first.time, 0.25)
        result.state.sum().backward()
        self.assertIsNotNone(x1.grad)
        self.assertTrue(any(p.grad is not None for p in module.parameters()))
        self.assertIs(first.model, second.model)  # states remain private under shared dynamics

    def test_registered_interactor_executes_over_sparse_relation_with_gradients(self):
        interactor = GatedInteractor(2, hidden_size=4, output_size=1)
        relation = Interaction.hyperedge(
            "joint",
            ["tf-a", "tf-b"],
            ["target"],
            interactor,
        )
        graph = InteractionGraph([relation])
        left = torch.tensor([0.5, 0.2], requires_grad=True)
        right = torch.tensor([0.1, 0.9], requires_grad=True)
        output = graph.evaluate({"tf-a": left, "tf-b": right}).state["target"]
        self.assertEqual(output.shape, (1,))
        output.sum().backward()
        self.assertIsNotNone(left.grad)
        self.assertTrue(any(p.grad is not None for p in interactor.parameters()))

    def test_event_head_keeps_identity_mapping_and_samples(self):
        head = EventPredictionHead(3, ["gene-A", "gene-B"])
        state = torch.ones(3)
        self.assertEqual(head(state).shape, (2,))
        self.assertIn(head.sample(state), head.event_ids)
        self.assertTrue(torch.allclose(head.probabilities(state).sum(-1), torch.tensor(1.0)))

    def test_ragged_sparse_experts_route_only_assigned_rows_and_combine_overlaps(self):
        class ScaleExpert(nn.Module):
            def __init__(self, scale):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(float(scale)))
                self.rows_seen = 0

            def forward(self, values):
                self.rows_seen += values.shape[0]
                return values * self.scale

        experts = [ScaleExpert(i + 1) for i in range(4)]
        bank = SparseExpertBank(experts, combine="mean", output_size=1)
        ids = torch.tensor([[0, 1], [1, 0], [0, 0], [0, 2]])
        gates = torch.tensor([[0.25, 0.75], [0.5, 0.0], [0.0, 0.0], [0.2, 0.8]],
                             requires_grad=True)
        mask = torch.tensor([[True, True], [True, False], [False, False], [True, True]])
        route = SparseRouting.from_dense(ids, gates, mask=mask)
        x = torch.tensor([[1.0], [2.0], [3.0], [4.0]], requires_grad=True)

        output = bank(x, route)
        self.assertTrue(torch.allclose(output, torch.tensor([[1.75], [4.0], [0.0], [10.4]])))
        self.assertEqual([expert.rows_seen for expert in experts], [2, 2, 1, 0])
        output.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(gates.grad)
        self.assertTrue(all(expert.scale.grad is not None for expert in experts[:3]))
        self.assertIsNone(experts[3].scale.grad)

    def test_learned_router_returns_sparse_identity_choices_and_receives_gradient(self):
        torch.manual_seed(7)
        experts = [nn.Linear(2, 2, bias=False) for _ in range(3)]
        router = nn.Linear(2, 3, bias=False)
        bank = SparseExpertBank(experts, router=router, top_k=1, gate_mode="sigmoid")
        x = torch.tensor([[0.3, 0.9], [0.5, 0.2]], requires_grad=True)

        values, route = bank(x, return_routing=True)
        self.assertEqual(route.expert_ids.numel(), 2)
        self.assertEqual(route.row_offsets.tolist(), [0, 1, 2])
        values.square().sum().backward()
        self.assertIsNotNone(router.weight.grad)
        self.assertTrue(any(parameter.grad is not None for parameter in experts[0].parameters())
                        or any(parameter.grad is not None for parameter in experts[1].parameters())
                        or any(parameter.grad is not None for parameter in experts[2].parameters()))
        self.assertIsNotNone(x.grad)


if __name__ == "__main__":
    unittest.main()
