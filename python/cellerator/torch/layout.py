"""Logical edge-value assembly helpers over canonical entity ordinals."""
from __future__ import annotations

import torch
from torch import Tensor


def logical_edge_values(embedding: Tensor, scorer: Tensor,
                        source_indices: Tensor, destination_indices: Tensor,
                        ordinals: Tensor, edge_count: int, *,
                        normalize: bool = True) -> Tensor:
    """CPU-testable logical E-vector assembly with differentiable sparse scatter."""
    ordinals = ordinals.to(dtype=torch.int64)
    if normalize:
        ordinals = torch.unique(ordinals, sorted=True)
    source = embedding.index_select(0, source_indices.index_select(0, ordinals))
    destination = embedding.index_select(0, destination_indices.index_select(0, ordinals))
    values = (source * destination * scorer).sum(dim=-1) / (embedding.shape[-1] ** 0.5)
    if ordinals.numel() == edge_count:
        return values
    return embedding.new_zeros((edge_count,), dtype=torch.float32).index_copy(
        0, ordinals, values)


_logical_values_for_ordinals = logical_edge_values

__all__ = ["logical_edge_values"]
