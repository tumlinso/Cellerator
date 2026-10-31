"""Learned shared interaction programs over biological feature occurrences.

Programs are ordinary Torch parameters with caller-supplied identities. They
attend to feature occurrences and send contextual updates back through either
the complete dense relation or an optional prepared overlapping relation.
Membership is supplied, not inferred or interpreted as a known cell type,
regulatory complex, or causal mechanism.
"""
from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from cellerator._declarations import Axis
from .relation_attention import RelationAttention
from .relations import PreparedRelation, _axis_manifest, _encode_identity


class InteractionPrograms(nn.Module):
    """Compose features through shared, potentially overlapping modules.

    A learned program query first aggregates its supplied feature members.
    A second relation-attention call returns program-derived feature updates.
    When ``relation`` is omitted, attention is dense within supplied groups.
    When present, it is a :class:`PreparedRelation` from feature occurrences to
    program identities; its transposed support sends updates back. Both
    ``[N, width]`` packed and ``[B, N, width]`` dense feature inputs are
    supported. Prepared support is packed and therefore uses the former.

    Context is optional and becomes an actual additive condition on learned
    queries through a registered projection. For packed inputs it may be
    ``[context_size]`` (shared) or ``[num_programs, context_size]`` (per
    program). For dense inputs it may be ``[context_size]``, ``[batch,
    context_size]``, or ``[batch, num_programs, context_size]``.

    The returned ``feature_updates`` are interaction contributions; residual
    composition with the input features is left to the caller.
    """

    def __init__(
        self,
        width: int,
        num_programs: int,
        *,
        heads: int = 1,
        dropout: float = 0.0,
        program_ids: Sequence[Hashable] | None = None,
        context_size: int | None = None,
        program_axis: Axis | None = None,
    ) -> None:
        super().__init__()
        if width <= 0 or num_programs <= 0:
            raise ValueError("width and num_programs must be positive")
        if context_size is not None and context_size <= 0:
            raise ValueError("context_size must be positive or None")
        ids = tuple(range(num_programs)) if program_ids is None else tuple(program_ids)
        if len(ids) != num_programs:
            raise ValueError("program_ids must contain one identity per program")
        try:
            if len(set(ids)) != len(ids):
                raise ValueError("program_ids must be unique")
        except TypeError as exc:
            raise TypeError("program_ids must be hashable") from exc
        if program_axis is not None and not isinstance(program_axis, Axis):
            raise TypeError("program_axis must be a Cellerator Axis or None")
        if program_axis is not None and program_axis.extent != num_programs:
            raise ValueError("program_axis extent must equal num_programs")

        self.width = int(width)
        self.num_programs = int(num_programs)
        self.program_ids = ids
        self.program_axis = program_axis
        self.context_size = context_size
        self.program_queries = nn.Parameter(torch.empty(num_programs, width))
        nn.init.normal_(self.program_queries, mean=0.0, std=width ** -0.5)
        self.context_projection = (
            nn.Linear(context_size, width, bias=False) if context_size is not None else None
        )
        self.feature_to_program = RelationAttention(width, heads=heads, dropout=dropout)
        self.program_to_feature = RelationAttention(width, heads=heads, dropout=dropout)

    def _condition_queries(self, features: Tensor, context: Tensor | None) -> Tensor:
        if features.ndim == 2:
            queries = self.program_queries
        else:
            queries = self.program_queries.unsqueeze(0).expand(features.shape[0], -1, -1)
        if context is None:
            return queries
        if self.context_projection is None:
            raise ValueError("context was supplied but InteractionPrograms has no context_size")
        if not isinstance(context, Tensor) or not context.is_floating_point():
            raise TypeError("context must be a floating-point Torch Tensor")
        if context.device != features.device:
            raise ValueError("context and features must share a device")
        if context.dtype != features.dtype:
            raise TypeError("context and features must share a dtype")
        if context.shape[-1] != self.context_size:
            raise ValueError("context's final dimension must equal context_size")

        projected = self.context_projection(context)
        if features.ndim == 2:
            if context.ndim == 1:
                projected = projected.unsqueeze(0)
            elif context.ndim != 2 or context.shape[0] not in (1, self.num_programs):
                raise ValueError(
                    "packed context must be [context_size] or [num_programs, context_size]"
                )
        else:
            batch = features.shape[0]
            if context.ndim == 1:
                projected = projected.reshape(1, 1, self.width)
            elif context.ndim == 2:
                if context.shape[0] not in (1, batch):
                    raise ValueError("dense group context must have one row per batch group")
                projected = projected.unsqueeze(1)
            elif context.ndim == 3:
                if context.shape[:2] != (batch, self.num_programs):
                    raise ValueError(
                        "dense per-program context must be [batch, num_programs, context_size]"
                    )
            else:
                raise ValueError("context rank must be one, two, or three")
        try:
            return queries + projected
        except RuntimeError as exc:
            raise ValueError("context does not broadcast over program queries") from exc

    def forward(
        self,
        features: Tensor,
        *,
        relation: PreparedRelation | None = None,
        feature_order: Sequence[Hashable] | None = None,
        feature_offsets: Sequence[int] | None = None,
        program_offsets: Sequence[int] | None = None,
        feature_axis: Axis | None = None,
        structural_generation: int = 0,
        context: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if not isinstance(features, Tensor) or features.ndim not in (2, 3):
            raise ValueError("features must be packed [N, width] or dense [B, N, width]")
        if features.shape[-1] != self.width:
            raise ValueError("features final dimension must equal width")
        if not features.is_floating_point():
            raise TypeError("features must use a floating-point dtype")
        if features.device != self.program_queries.device:
            raise ValueError("features and InteractionPrograms parameters must share a device")
        if features.dtype != self.program_queries.dtype:
            raise TypeError("features and InteractionPrograms parameters must share a dtype")
        if relation is not None and features.ndim != 2:
            raise ValueError("prepared overlapping relations require packed [N, width] features")
        if relation is not None and not isinstance(relation, PreparedRelation):
            raise TypeError("relation must be a PreparedRelation or None")

        queries = self._condition_queries(features, context)
        if relation is None:
            program_features = self.feature_to_program(
                features, queries,
                source_offsets=feature_offsets, target_offsets=program_offsets,
            )
            feature_updates = self.program_to_feature(
                program_features, features,
                source_offsets=program_offsets, target_offsets=feature_offsets,
            )
        else:
            if feature_order is None:
                raise ValueError("feature_order is required with a prepared relation")
            program_features = self.feature_to_program(
                features, queries,
                source_offsets=feature_offsets, target_offsets=program_offsets,
                relation=relation,
                source_order=feature_order,
                target_order=self.program_ids,
                source_axis=feature_axis,
                target_axis=self.program_axis,
                structural_generation=structural_generation,
            )
            reverse = relation.transpose()
            feature_updates = self.program_to_feature(
                program_features, features,
                source_offsets=program_offsets, target_offsets=feature_offsets,
                relation=reverse,
                source_order=self.program_ids,
                target_order=feature_order,
                source_axis=self.program_axis,
                target_axis=feature_axis,
                structural_generation=structural_generation,
            )
        return {
            "program_features": program_features,
            "feature_updates": feature_updates,
        }

    def cellerator_biological_descriptors(
        self, *, identity_codec: Any = None
    ) -> dict[str, Any]:
        """Describe declared program identities without assuming their biology."""
        descriptor = {
            "kind": "cellerator.interaction_programs.v1",
            "program_ids": [
                _encode_identity(value, identity_codec) for value in self.program_ids
            ],
            "program_axis": _axis_manifest(self.program_axis, identity_codec),
            "width": self.width,
            "num_programs": self.num_programs,
            "context_size": self.context_size,
        }
        return {"interaction_programs": descriptor}


__all__ = ["InteractionPrograms"]
