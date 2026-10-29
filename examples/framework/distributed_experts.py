"""Sparse owner-compute experts, independent of transformer or cell ontology.

Single process:
    PYTHONPATH=python python examples/framework/distributed_experts.py

Two CPU processes:
    PYTHONPATH=python torchrun --standalone --nproc-per-node=2 \
        examples/framework/distributed_experts.py

NCCL requires a separately authorized GPU allocation and one GPU per process.
The dispatcher itself never initializes groups or moves modules/devices.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import os

import torch
from torch import distributed as dist, nn

from cellerator.adapters.distributed import DistributedSparseExpertBank
from cellerator.adapters.torch import SparseRouting


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    args = parser.parse_args()
    launched = int(os.environ.get("WORLD_SIZE", "1")) > 1
    device = torch.device("cpu")
    if args.backend == "nccl":
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    if launched:
        dist.init_process_group(args.backend, timeout=timedelta(seconds=60))
    try:
        rank = dist.get_rank() if launched else 0
        world_size = dist.get_world_size() if launched else 1
        owners = [expert_id % world_size for expert_id in range(3)]
        local_experts = {}
        for expert_id, owner in enumerate(owners):
            if owner == rank:
                expert = nn.Linear(2, 2, device=device, dtype=torch.float32)
                with torch.no_grad():
                    expert.weight.copy_(torch.eye(2, device=device) * (expert_id + 1))
                    expert.bias.zero_()
                local_experts[expert_id] = expert
        bank = DistributedSparseExpertBank(local_experts, owners, output_size=2)
        optimizer = torch.optim.SGD(bank.parameters(), lr=0.01) if local_experts else None

        # Private state stays at its origin. These two rows are separate cells
        # sharing one logical model, whose parameters are owned by expert ranks.
        private_state = torch.tensor([[0.1, 0.3], [0.7, 0.2]], device=device,
                                     requires_grad=True)
        ids = torch.tensor([[rank % 3, (rank + 1) % 3], [2, 0]], device=device)
        gates = torch.tensor([[0.7, 0.3], [0.4, 0.6]], device=device, requires_grad=True)
        routing = SparseRouting.from_dense(ids, gates)
        output, plan = bank(private_state, routing, return_plan=True)
        loss = output.square().sum()
        loss.backward()  # Every rank enters, even with zero local loss/work.
        if optimizer is not None:
            optimizer.step()
        print(f"rank={rank} owned={list(local_experts)} send={plan.send_splits} "
              f"receive={plan.receive_splits} loss={loss.item():.6f} "
              f"state_gradient={private_state.grad.tolist()} gate_gradient={gates.grad.tolist()}")
    finally:
        if launched:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
