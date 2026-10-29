"""Owner-compute sparse experts using optional PyTorch collectives.

Routing is a model decision, supplied as :class:`SparseRouting`. This module
only permutes assignments, transports activations to their parameter owner,
and restores contributions to the originating rows. Individual cell state is
never migrated. Importing this provider requires PyTorch, not a native build.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from operator import index
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch import distributed as dist
from torch.autograd.function import once_differentiable

from .torch import SparseRouting


@dataclass(frozen=True)
class ExpertDispatchPlan:
    """Physical permutation of one rank's unchanged logical assignments.

    ``assignment_order`` maps packed positions to original assignment positions;
    ``inverse_order`` restores logical assignment order. ``origin_rows`` is in
    that logical order. Split counts use process-group rank order, not global
    ranks. Tensor fields are inspection records and must not be mutated.
    """

    assignment_order: Tensor
    inverse_order: Tensor
    origin_rows: Tensor
    packed_expert_ids: Tensor
    send_splits: tuple[int, ...]
    receive_splits: tuple[int, ...] = ()


def _exchange(value: Tensor, send: tuple[int, ...], receive: tuple[int, ...],
              group: dist.ProcessGroup | None) -> Tensor:
    if len(send) == 1:
        return value.clone()
    output = value.new_empty((sum(receive), *value.shape[1:]))
    dist.all_to_all_single(output, value.contiguous(),
                           output_split_sizes=list(receive),
                           input_split_sizes=list(send), group=group)
    return output


class _AllToAll(torch.autograd.Function):
    """First-order variable-row exchange; VJP reverses the split directions."""

    @staticmethod
    def forward(ctx, value, send, receive, group):
        ctx.send, ctx.receive, ctx.group = send, receive, group
        return _exchange(value, send, receive, group)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        return _exchange(gradient, ctx.receive, ctx.send, ctx.group), None, None, None


def _integers(values: Sequence[int], name: str) -> tuple[int, ...]:
    result = []
    for value in values:
        if isinstance(value, bool):
            raise TypeError(f"{name} must contain integers")
        try:
            result.append(index(value))
        except TypeError as exc:
            raise TypeError(f"{name} must contain integers") from exc
    return tuple(result)


class DistributedSparseExpertBank(nn.Module):
    """One logical expert bank whose parameters are sharded across ranks.

    ``expert_owners[e]`` selects a process-group rank for logical expert ``e``.
    Each rank registers exactly the modules it owns in ``local_experts``. All
    ranks must use the same owner map, output shape, input trailing shape and
    dtype, and call forward/backward in the same order, including ranks with
    no originating rows or assignments. Losses may differ by rank; each rank
    must backpropagate through its returned output (``output.sum() * 0`` is a
    suitable empty local objective). Gradients sum contributions from all
    origin ranks into each owner's parameters, without DDP averaging.

    Inputs have shape ``[rows, *input_shape]``. Experts receive batches of
    assigned rows and return ``[assignments, *output_shape]`` with the input
    dtype/device. Specify ``output_shape`` explicitly, or ``output_size`` to
    preserve intermediate dimensions and replace the last feature dimension.
    No activation dtype conversion, parameter migration, replication, capacity
    dropping, cell-state gathering or model routing is performed. Gates use
    the input dtype for multiplication and the weighted-mean denominator.

    Gloo uses CPU tensors; NCCL uses caller-selected CUDA tensors and an
    initialized process group. Without distributed initialization the bank
    executes as rank zero in a one-rank group. A bank's group membership must
    remain unchanged for its lifetime. Only first-order autograd is supported.
    """

    def __init__(self, local_experts: Mapping[int, nn.Module],
                 expert_owners: Sequence[int], *,
                 output_shape: Sequence[int] | None = None,
                 output_size: int | None = None,
                 combine: str = "sum", group: dist.ProcessGroup | None = None):
        super().__init__()
        if group is not None and (not dist.is_available() or not dist.is_initialized()):
            raise ValueError("an explicit process group must be initialized")
        initialized = dist.is_available() and dist.is_initialized()
        self.world_size = dist.get_world_size(group) if initialized else 1
        self.rank = dist.get_rank(group) if initialized else 0
        if self.rank < 0 or self.world_size < 1:
            raise ValueError("this process must belong to the process group")
        owners = _integers(expert_owners, "expert_owners")
        if not owners or any(owner < 0 or owner >= self.world_size for owner in owners):
            raise ValueError("expert_owners must be nonempty and contain group ranks")
        local_ids = _integers(tuple(local_experts), "local expert IDs")
        expected = {expert for expert, owner in enumerate(owners) if owner == self.rank}
        if set(local_ids) != expected:
            raise ValueError("local_experts must contain exactly the experts owned by this rank")
        if not all(isinstance(expert, nn.Module) for expert in local_experts.values()):
            raise TypeError("local experts must be torch.nn.Module instances")
        if combine not in {"sum", "mean"}:
            raise ValueError("combine must be 'sum' or 'mean'")
        if (output_shape is None) == (output_size is None):
            raise ValueError("specify exactly one of output_shape or output_size")
        shape = None if output_shape is None else _integers(output_shape, "output_shape")
        size = None if output_size is None else _integers((output_size,), "output_size")[0]
        if shape is not None and any(extent <= 0 for extent in shape):
            raise ValueError("output_shape extents must be positive")
        if size is not None and size <= 0:
            raise ValueError("output_size must be positive")
        self.local_experts = nn.ModuleDict({str(e): local_experts[e] for e in sorted(expected)})
        self.expert_owners = owners
        self.output_shape, self.output_size = shape, size
        self.combine, self.group = combine, group

    @property
    def expert_count(self) -> int:
        return len(self.expert_owners)

    def plan(self, x: Tensor, routing: SparseRouting) -> ExpertDispatchPlan:
        """Validate local routing and pack by owner, then by expert identity.

        This method performs no collectives or expert computation. It preserves
        repeated assignments and never changes the selected expert identities.
        """
        if not isinstance(x, Tensor) or x.ndim < 2 or not x.is_floating_point():
            raise ValueError("x must be floating point with shape [rows, *features]")
        if not isinstance(routing, SparseRouting):
            raise TypeError("routing must be SparseRouting")
        ids, offsets, weights = routing.expert_ids, routing.row_offsets, routing.weights
        if ids.ndim != 1 or ids.dtype != torch.long:
            raise ValueError("expert_ids must be a rank-1 torch.long tensor")
        if offsets.ndim != 1 or offsets.dtype != torch.long or offsets.numel() != x.shape[0] + 1:
            raise ValueError("row_offsets must be torch.long with rows + 1 entries")
        if ids.device != x.device or offsets.device != x.device:
            raise ValueError("routing and x must be on the same device")
        if int(offsets[0]) != 0 or int(offsets[-1]) != ids.numel():
            raise ValueError("row_offsets must span the assignment list from zero")
        if bool(torch.any(offsets[1:] < offsets[:-1])):
            raise ValueError("row_offsets must be monotone")
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= self.expert_count):
            raise ValueError("routing contains an expert outside the logical bank")
        if weights is not None:
            if (weights.ndim != 1 or weights.numel() != ids.numel()
                    or not weights.is_floating_point() or weights.device != x.device):
                raise ValueError("weights must be floating point, on x.device, and align with assignments")
            if not bool(torch.all(torch.isfinite(weights))):
                raise ValueError("routing weights must be finite")
            if self.combine == "mean" and bool(torch.any(weights < 0)):
                raise ValueError("weighted mean requires nonnegative routing weights")
        owners = torch.tensor(self.expert_owners, device=x.device, dtype=torch.long)
        destinations = owners.index_select(0, ids)
        order = torch.argsort(destinations * self.expert_count + ids, stable=True)
        counts = torch.bincount(destinations, minlength=self.world_size)
        return ExpertDispatchPlan(order, torch.argsort(order), routing.row_indices(),
                                  ids.index_select(0, order), tuple(counts.tolist()))

    def forward(self, x: Tensor, routing: SparseRouting, *,
                return_plan: bool = False) -> Tensor | tuple[Tensor, ExpertDispatchPlan]:
        plan = self.plan(x, routing)
        shape = self.output_shape
        if shape is None:
            shape = (*x.shape[1:-1], self.output_size)
        if self.world_size > 1:
            counts = torch.tensor(plan.send_splits, dtype=torch.long, device=x.device)
            received_counts = torch.empty_like(counts)
            dist.all_to_all_single(received_counts, counts, group=self.group)
            receive = tuple(received_counts.tolist())
        else:
            receive = plan.send_splits
        plan = replace(plan, receive_splits=receive)
        rows = plan.origin_rows.index_select(0, plan.assignment_order)
        payload = x.index_select(0, rows)
        # Every training rank must participate in both reverse collectives even
        # when its inputs do not require grad or its local assignment set is
        # empty. This leaf anchors that participation without owning parameters.
        if torch.is_grad_enabled():
            anchor = x.new_zeros((), requires_grad=True)
            payload = payload + anchor * 0
        received = _AllToAll.apply(payload, plan.send_splits, receive, self.group)
        expert_ids = _exchange(plan.packed_expert_ids, plan.send_splits, receive, self.group)
        # The zero dependency keeps the activation VJP reachable even when an
        # expert ignores its input or this rank owns no active experts.
        zero_dependency = received.reshape(-1)[:0].sum()
        local_output = x.new_zeros((sum(receive), *shape)) + zero_dependency
        for expert_id, expert in self.local_experts.items():
            positions = torch.nonzero(expert_ids == int(expert_id), as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            value = expert(received.index_select(0, positions))
            if (not isinstance(value, Tensor) or value.shape != (positions.numel(), *shape)
                    or value.device != x.device or value.dtype != x.dtype):
                raise ValueError("expert outputs must have declared shape and input dtype/device")
            local_output = local_output.index_copy(0, positions, value)
        returned = _AllToAll.apply(local_output, receive, plan.send_splits, self.group)
        logical_values = returned.index_select(0, plan.inverse_order)
        weights = routing.weights
        if weights is None:
            weights = x.new_ones((routing.expert_ids.numel(),))
        else:
            weights = weights.to(dtype=x.dtype)
        broadcasts = weights.reshape((-1, *([1] * len(shape))))
        output = x.new_zeros((x.shape[0], *shape)).index_add(
            0, plan.origin_rows, logical_values * broadcasts)
        if self.combine == "mean":
            denominator = x.new_zeros((x.shape[0],)).index_add(0, plan.origin_rows, weights)
            # All-zero gates yield zero; a positive denominator is not clamped
            # to epsilon, preserving exact weighted-mean semantics for tiny gates.
            safe = torch.where(denominator != 0, denominator, torch.ones_like(denominator))
            output = output / safe.reshape((-1, *([1] * len(shape))))
        return (output, plan) if return_plan else output


__all__ = ["DistributedSparseExpertBank", "ExpertDispatchPlan"]
