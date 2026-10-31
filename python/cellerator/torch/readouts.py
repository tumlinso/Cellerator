"""Identity-aware, on-demand readout modules for biological features."""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from cellerator.biology import FeatureBinding, PredictionQuery
from .relation_attention import RelationAttention


class FeatureReadout(nn.Module):
    """Predict selected biological features from learned model representations.

    The readout owns target-feature embeddings indexed in the model coordinate
    system declared by ``binding``. A query selects only requested target
    embeddings, lets them attend to the supplied context features, and projects
    the result. The output feature IDs remain biological identities; model
    indices are internal embedding rows.

    ``context`` may be ``[batch, context_features, width]`` or
    ``[batch, width]`` (one context vector per sample). The latter is promoted
    to a single context feature. This module does not define a cell state or
    assume a particular measurement scale.
    """

    def __init__(
        self,
        binding: FeatureBinding,
        width: int,
        *,
        output_size: int = 1,
        heads: int = 1,
    ) -> None:
        super().__init__()
        if not isinstance(binding, FeatureBinding):
            raise TypeError("binding must be a FeatureBinding")
        if width <= 0 or output_size <= 0:
            raise ValueError("width and output_size must be positive")
        capacity = binding.model_capacity
        if capacity is None:
            capacity = max(binding.model_indices) + 1
        binding.validate_model_capacity(capacity)
        self.binding = binding
        self.width = int(width)
        self.output_size = int(output_size)
        self.target_embedding = nn.Embedding(capacity, width)
        self.interactions = RelationAttention(width, heads=heads)
        self.projection = nn.Linear(width, output_size)

    def forward(self, context: Tensor) -> Tensor:
        """Read all bound output features in binding order."""
        return self.query(context, self.binding.model_indices)

    def query(
        self,
        context: Tensor,
        target_indices: Sequence[int],
        *,
        query: PredictionQuery | None = None,
    ) -> Tensor:
        """Read selected model-index rows, preserving request order and gradients."""
        del query  # query metadata labels the result; it is not model conditioning.
        if not isinstance(context, Tensor) or context.ndim not in (2, 3):
            raise ValueError("context must have shape [batch, width] or [batch, features, width]")
        if context.shape[-1] != self.width:
            raise ValueError("context final dimension does not match readout width")
        if not context.is_floating_point():
            raise TypeError("context must have a floating-point dtype")
        if context.device != self.target_embedding.weight.device:
            raise ValueError("context and readout parameters must share a device")
        if context.dtype != self.target_embedding.weight.dtype:
            raise TypeError("context and readout parameters must share a dtype")
        if isinstance(target_indices, (str, bytes, Tensor)) or not isinstance(target_indices, Sequence):
            raise TypeError("target_indices must be a prepared host sequence of model indices")
        indices = tuple(target_indices)
        if any(not isinstance(index, int) or isinstance(index, bool) for index in indices):
            raise TypeError("target_indices entries must be integers")
        if any(index < 0 or index >= self.target_embedding.num_embeddings for index in indices):
            raise ValueError("target index is outside the model feature capacity")

        if context.ndim == 2:
            context = context.unsqueeze(1)
        target_index_tensor = torch.tensor(
            indices, dtype=torch.long, device=context.device
        )
        targets = self.target_embedding(target_index_tensor)
        targets = targets.unsqueeze(0).expand(context.shape[0], -1, -1)
        features = self.interactions(context, target=targets)
        values = self.projection(features)
        return values.squeeze(-1) if self.output_size == 1 else values


__all__ = ["FeatureReadout"]
