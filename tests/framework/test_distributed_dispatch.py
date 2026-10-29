"""Focused CPU oracle for owner-compute routing and reverse redistribution."""
from __future__ import annotations

import copy
from datetime import timedelta
import os
from pathlib import Path
import tempfile
import unittest

import torch
from torch import distributed as dist, nn
import torch.multiprocessing as mp

from cellerator.adapters.distributed import DistributedSparseExpertBank
from cellerator.adapters.torch import SparseRouting


def _experts():
    result = [nn.Linear(2, 3, dtype=torch.float64) for _ in range(3)]
    with torch.no_grad():
        for expert_id, expert in enumerate(result):
            expert.weight.copy_(torch.arange(6, dtype=torch.float64).reshape(3, 2) / 7 + expert_id)
            expert.bias.fill_(expert_id / 3)
    return result


def _oracle(x, experts, routing, combine="sum"):
    # Deliberately simple row/assignment loops: independent of packing and
    # inverse permutation used by the implementation.
    rows = []
    for row in range(x.shape[0]):
        value = x.new_zeros((*x.shape[1:-1], 3)) + x[row].reshape(-1)[:0].sum()
        denominator = x.new_zeros(())
        for assignment in range(int(routing.row_offsets[row]), int(routing.row_offsets[row + 1])):
            weight = x.new_ones(()) if routing.weights is None else routing.weights[assignment]
            value = value + weight * experts[int(routing.expert_ids[assignment])](x[row])
            denominator = denominator + weight
        if combine == "mean":
            value = value / torch.where(denominator != 0, denominator, torch.ones_like(denominator))
        rows.append(value)
    if not rows:
        return x.new_zeros((0, *x.shape[1:-1], 3)) + x.sum() * 0
    return torch.stack(rows)


def _case(rank, case):
    if case in {"uneven", "no_local_experts"}:
        ids = [2, 0, 2, 1] if rank == 0 else [0, 1, 1]
        offsets = [0, 3, 4] if rank == 0 else [0, 0, 1, 3]
    elif case == "empty_origin":
        ids, offsets = ([1, 1, 0], [0, 2, 3]) if rank == 0 else ([], [0])
    elif case == "empty_owner":
        ids, offsets = ([0, 2], [0, 2]) if rank == 0 else ([2], [0, 1])
    else:
        ids, offsets = [], [0, 0, 0]
    rows = len(offsets) - 1
    x = (torch.arange(rows * 4, dtype=torch.float64).reshape(rows, 2, 2) + rank + 1) / 5
    weights = torch.arange(len(ids), dtype=torch.float64) / 5 + 0.2
    return x, SparseRouting(torch.tensor(ids, dtype=torch.long),
                            torch.tensor(offsets, dtype=torch.long), weights)


def _distributed_worker(rank, init_path):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=Path(init_path).as_uri(), rank=rank,
                            world_size=2, timeout=timedelta(seconds=40))
    try:
        for case in ("uneven", "empty_origin", "empty_owner", "all_empty", "no_local_experts"):
            for combine in ("sum", "mean"):
                reference_experts = _experts()
                owners = [0, 0, 0] if case == "no_local_experts" else [0, 1, 0]
                local = {e: copy.deepcopy(reference_experts[e]) for e in range(3) if owners[e] == rank}
                bank = DistributedSparseExpertBank(local, owners, output_size=3, combine=combine)
                original_x, original_routing = _case(rank, case)
                # Rank 1 intentionally has no input gradients. Its experts must
                # nevertheless receive gradients from losses on rank zero.
                x = original_x.clone().requires_grad_(rank == 0)
                weights = original_routing.weights.clone().requires_grad_()
                route = SparseRouting(original_routing.expert_ids, original_routing.row_offsets, weights)
                actual, plan = bank(x, route, return_plan=True)
                expected_inputs, expected_weights, expected_outputs = [], [], []
                for source in range(2):
                    source_x, source_route = _case(source, case)
                    source_x.requires_grad_()
                    source_weights = source_route.weights.clone().requires_grad_()
                    expected_inputs.append(source_x)
                    expected_weights.append(source_weights)
                    expected_outputs.append(_oracle(source_x, reference_experts,
                        SparseRouting(source_route.expert_ids, source_route.row_offsets, source_weights), combine))
                torch.testing.assert_close(actual, expected_outputs[rank])
                assert sum(plan.send_splits) == route.expert_ids.numel()
                assert sorted(plan.assignment_order.tolist()) == list(range(route.expert_ids.numel()))
                # Nonuniform local cotangents ensure returning to the right
                # origin row matters, rather than merely summing all values.
                actual_scale = torch.arange(actual.numel(), dtype=actual.dtype).reshape_as(actual) / 9 + rank + 1
                loss = (actual * actual_scale).sum()
                loss.backward()
                reference_loss = sum((value * (torch.arange(value.numel(), dtype=value.dtype).reshape_as(value) / 9 + source + 1)).sum()
                                     for source, value in enumerate(expected_outputs))
                reference_loss.backward()
                if x.requires_grad:
                    torch.testing.assert_close(x.grad, expected_inputs[rank].grad)
                if weights.numel():
                    torch.testing.assert_close(weights.grad, expected_weights[rank].grad)
                for expert_id, expert in local.items():
                    for actual_parameter, expected_parameter in zip(expert.parameters(), reference_experts[expert_id].parameters()):
                        if expected_parameter.grad is None:
                            assert actual_parameter.grad is None
                        else:
                            torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad)
    finally:
        dist.destroy_process_group()


class DistributedDispatchTests(unittest.TestCase):
    def test_single_rank_overlapping_routes_and_gradients(self):
        experts = _experts()
        reference = copy.deepcopy(experts)
        bank = DistributedSparseExpertBank(dict(enumerate(experts)), [0, 0, 0], output_size=3)
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64, requires_grad=True)
        gate = torch.tensor([0.2, 0.3, 0.5, 0.7], dtype=torch.float64, requires_grad=True)
        route = SparseRouting(torch.tensor([2, 0, 2, 1]), torch.tensor([0, 3, 4]), gate)
        actual, plan = bank(x, route, return_plan=True)
        expected_x = x.detach().clone().requires_grad_()
        expected_gate = gate.detach().clone().requires_grad_()
        expected = _oracle(expected_x, reference, SparseRouting(route.expert_ids, route.row_offsets, expected_gate))
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(x.grad, expected_x.grad)
        torch.testing.assert_close(gate.grad, expected_gate.grad)
        for expert, ref in zip(experts, reference):
            for parameter, ref_parameter in zip(expert.parameters(), ref.parameters()):
                torch.testing.assert_close(parameter.grad, ref_parameter.grad)
        self.assertEqual(plan.send_splits, (4,))
        self.assertEqual(plan.packed_expert_ids.tolist(), [0, 1, 2, 2])
        self.assertEqual(plan.origin_rows.tolist(), [0, 0, 0, 1])

    def test_empty_routes_preserve_input_gradient_and_shape(self):
        bank = DistributedSparseExpertBank({0: nn.Identity()}, [0], output_shape=(2, 2))
        x = torch.ones(3, 2, 2, requires_grad=True)
        gates = torch.empty(0, requires_grad=True)
        route = SparseRouting(torch.empty(0, dtype=torch.long), torch.zeros(4, dtype=torch.long), gates)
        output = bank(x, route)
        self.assertEqual(tuple(output.shape), (3, 2, 2))
        output.sum().backward()
        torch.testing.assert_close(x.grad, torch.zeros_like(x))
        self.assertIsNotNone(gates.grad)

    def test_declared_arbitrary_trailing_output_shape(self):
        class Flatten(nn.Module):
            def forward(self, x):
                return x.reshape(x.shape[0], -1)
        bank = DistributedSparseExpertBank({0: Flatten()}, [0], output_shape=(4,))
        x = torch.arange(8.0).reshape(2, 2, 2).requires_grad_()
        routing = SparseRouting(torch.tensor([0, 0]), torch.tensor([0, 1, 2]))
        torch.testing.assert_close(bank(x, routing), x.reshape(2, 4))

    def test_mean_handles_tiny_gates_without_epsilon_distortion(self):
        bank = DistributedSparseExpertBank({0: nn.Identity()}, [0], output_size=2, combine="mean")
        x = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
        routing = SparseRouting(torch.tensor([0]), torch.tensor([0, 1]), torch.tensor([1e-30], dtype=x.dtype))
        torch.testing.assert_close(bank(x, routing), x)

    def test_rejects_invalid_ownership_and_routes(self):
        with self.assertRaises(ValueError):
            DistributedSparseExpertBank({}, [0], output_size=2)
        bank = DistributedSparseExpertBank({0: nn.Identity()}, [0], output_size=2)
        with self.assertRaises(ValueError):
            bank(torch.zeros(1, 2), SparseRouting(torch.tensor([1]), torch.tensor([0, 1])))
        with self.assertRaises(ValueError):
            bank(torch.zeros(1, 2), SparseRouting(torch.tensor([0]), torch.tensor([1, 1])))

    @unittest.skipUnless(os.environ.get("CELLERATOR_TEST_DISTRIBUTED") == "1",
                         "set CELLERATOR_TEST_DISTRIBUTED=1 for the real two-process CPU smoke")
    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo unavailable")
    def test_two_process_gloo_outputs_and_all_gradients(self):
        with tempfile.TemporaryDirectory(prefix="cellerator-dispatch-") as directory:
            mp.spawn(_distributed_worker, args=(str(Path(directory) / "init"),), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
