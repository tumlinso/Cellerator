"""Expression and event/history reference models built from reusable modules.

These are engineering examples, not prescribed biological architectures or
scientific objectives. They deliberately use ordinary PyTorch modules and
optimizers around Cellerator's feature and interaction primitives.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from cellerator.biology import FeatureBinding
from cellerator.adapters.torch import EventPredictionHead
from cellerator.torch.features import FeatureEncoder
from cellerator.torch.programs import InteractionPrograms
from cellerator.torch.relation_attention import RelationAttention
from cellerator.torch.readouts import FeatureReadout
from cellerator.torch.views import ModelWithViews


class ExpressionReferenceModel(nn.Module):
    """Feature-aware expression reconstruction with learned interactions.

    The model makes no normalization or feature-order assumption. A batch uses
    ``feature_ids``, ``values``, and an explicit Boolean ``observed`` mask;
    hidden measurements are sanitized by :class:`FeatureEncoder` before they
    reach learned projections.
    """

    def __init__(
        self,
        vocabulary_size: int,
        width: int,
        *,
        heads: int = 1,
        binding: FeatureBinding | None = None,
        num_programs: int = 0,
    ) -> None:
        super().__init__()
        self.width = int(width)
        if num_programs < 0:
            raise ValueError("num_programs must be nonnegative")
        self.binding = binding
        self.encoder = FeatureEncoder(
            vocabulary_size, width, value_size=1, binding=binding
        )
        self.interactions = RelationAttention(width, heads=heads)
        self.interaction_programs = (
            InteractionPrograms(width, num_programs, heads=heads,
                                program_ids=tuple(f"learned-program-{i}" for i in range(num_programs)))
            if num_programs else None
        )
        self.expression_readout = nn.Linear(width, 1)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        values = batch["values"]
        observed = batch["observed"]
        if self.binding is not None and "feature_identities" in batch:
            raw_identities = batch["feature_identities"]
            if isinstance(raw_identities, Tensor) or isinstance(raw_identities, (str, bytes)):
                raise TypeError("feature_identities must be a host sequence of feature identities")
            source_ids = tuple(batch["feature_identities"])
            alignment = self.binding.prepare(source_ids, self.binding.identities)
            positions = alignment.index_tensor(device=values.device)
            values = values.index_select(-1, positions)
            observed = observed.index_select(-1, positions)
            named_ids = tuple(self.binding.identities for _ in range(values.shape[0]))
            encoded = self.encoder.encode_bound(
                named_ids, values, observed=observed
            )
            output_feature_ids = self.binding.identities
        else:
            try:
                feature_ids = batch["feature_ids"]
            except KeyError as exc:
                if self.binding is not None:
                    raise ValueError(
                        "bound expression batches require feature_identities; raw batches require feature_ids"
                    ) from exc
                raise
            encoded = self.encoder(feature_ids, values, observed=observed)
            output_feature_ids = feature_ids
        features = self.interactions(encoded)
        program_features = None
        if self.interaction_programs is not None:
            program_result = self.interaction_programs(features)
            # The program contribution is deliberately composed as a residual
            # update; the optional module does not prescribe the model's full
            # feature dynamics.
            features = features + program_result["feature_updates"]
            program_features = program_result["program_features"]
        prediction = self.expression_readout(features).squeeze(-1)
        # This simple pooled representation is exposed only as a reusable
        # feature source for auxiliary views; it is not a canonical cell state.
        cell_features = features.mean(dim=-2)
        result = {
            "expression": prediction,
            "features": features,
            "cell_features": cell_features,
        }
        result["feature_ids"] = output_feature_ids
        if program_features is not None:
            result["program_features"] = program_features
        return result

    def cellerator_biological_descriptors(self, *, identity_codec=None):
        if self.binding is None:
            return {}
        return {"expression_output_binding": self.binding.to_manifest(codec=identity_codec)}


def masked_expression_mse(
    prediction: Tensor, target: Tensor, target_mask: Tensor
) -> Tensor | None:
    """Mean squared error on explicitly selected targets, or ``None`` if empty.

    The caller decides which observed measurements are hidden from the model.
    Unselected target values are sanitized before arithmetic, so missing or
    otherwise invalid sentinels outside the objective cannot affect the loss.
    """
    if prediction.shape != target.shape or target_mask.shape != target.shape:
        raise ValueError("prediction, target, and target_mask shapes must match")
    if target_mask.dtype != torch.bool:
        raise TypeError("target_mask must be Boolean")
    if not bool(target_mask.any().item()):
        return None
    visible_target = torch.where(target_mask, target, torch.zeros_like(target))
    squared_error = torch.where(
        target_mask, (prediction - visible_target).square(), torch.zeros_like(prediction)
    )
    return squared_error.sum() / target_mask.sum()


def attach_expression_view(
    base: ExpressionReferenceModel, *, output_binding: FeatureBinding | None = None,
) -> ModelWithViews:
    """Expose expression outputs as an identity-aware, queryable model view."""
    binding = output_binding if output_binding is not None else base.binding
    model = ModelWithViews(base)
    if binding is None:
        source: Any = "expression"
    else:
        def source(result, _inputs):
            declared = result.get("feature_ids") if isinstance(result, Mapping) else None
            if not isinstance(declared, tuple) or declared != binding.identities:
                raise ValueError(
                    "identity-aware expression queries require the bound feature_identities input"
                )
            return result["expression"]

    model.add_view(
        "expression", nn.Identity(), source=source, output_binding=binding,
        queryable=False, source_id="reference_models.expression",
    )
    return model


def attach_atac_view(
    base: ExpressionReferenceModel,
    peak_count: int,
    *,
    output_binding: FeatureBinding | None = None,
) -> ModelWithViews:
    """Attach a shared binary ATAC readout to the base model's pooled features."""
    if peak_count <= 0:
        raise ValueError("peak_count must be positive")
    model = ModelWithViews(base)
    if output_binding is not None and len(output_binding.identities) != peak_count:
        raise ValueError("peak_count must match the ATAC output binding")
    readout = (
        FeatureReadout(output_binding, base.width)
        if output_binding is not None else nn.Linear(base.width, peak_count)
    )
    model.add_view(
        "atac", readout, source="cell_features",
        output_binding=output_binding, queryable=output_binding is not None,
        source_id="reference_models.atac",
    )
    return model


def attach_history_view(
    base: "EventHistoryReferenceModel | ModelWithViews", view_name: str, output_size: int,
    *, output_binding: FeatureBinding | None = None,
) -> ModelWithViews:
    """Attach a shared readout to an explicitly exposed pooled history feature."""
    if output_size <= 0:
        raise ValueError("output_size must be positive")
    if isinstance(base, ModelWithViews):
        model = base
        base_width = base.base.width
    else:
        model = ModelWithViews(base)
        base_width = base.width
        if base.binding is not None:
            model.add_view(
                "events", nn.Identity(), source="logits", output_binding=base.binding,
                queryable=False, source_id="reference_models.events",
            )
    if output_binding is not None and len(output_binding.identities) != output_size:
        raise ValueError("output_size must match the history-view binding")
    readout = (
        FeatureReadout(output_binding, base_width)
        if output_binding is not None else nn.Linear(base_width, output_size)
    )
    model.add_view(
        view_name, readout, source="history_features",
        output_binding=output_binding, queryable=output_binding is not None,
        source_id=f"reference_models.{view_name}",
    )
    return model


def attach_event_view(base: "EventHistoryReferenceModel") -> ModelWithViews:
    """Expose next-event logits with their declared event identity axis."""
    model = ModelWithViews(base)
    model.add_view(
        "events", nn.Identity(), source="logits", output_binding=base.binding,
        queryable=False, source_id="reference_models.events",
    )
    return model


def atac_binary_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Binary auxiliary-view objective for paired synthetic/experimental peaks."""
    if logits.shape != targets.shape:
        raise ValueError("ATAC logits and targets must have matching shapes")
    if not targets.is_floating_point():
        raise TypeError("ATAC targets must be floating point in [0, 1]")
    return F.binary_cross_entropy_with_logits(logits, targets)


def explicit_dense_anndata_minibatch(
    view: Any, row_indices: Sequence[int], *, max_elements: int,
    device: torch.device | str = "cpu",
    binding: FeatureBinding | None = None,
    requested_ids: Sequence[Any] | None = None,
) -> tuple[Tensor, tuple[Any, ...], tuple[Any, ...]]:
    """Convert selected AnnDataView rows only after an explicit size check.

    The returned tensors contain raw selected values; feature identities and
    cell identities are returned alongside them. No normalization, imputation,
    or vocabulary mapping is inferred. Sparse storage remains untouched until
    this bounded call.
    """
    if max_elements <= 0:
        raise ValueError("max_elements must be positive")
    rows = tuple(row_indices)
    if any(not isinstance(index, int) or isinstance(index, bool) for index in rows):
        raise TypeError("row_indices must contain integers")
    if any(index < 0 or index >= len(view.cell_ids) for index in rows):
        raise IndexError("row_indices contain a row outside the selected AnnDataView")
    if binding is None:
        aligned_feature_ids = tuple(view.feature_ids)
        selected = view.values[list(rows), :]
    else:
        requested = binding.identities if requested_ids is None else tuple(requested_ids)
        alignment = binding.prepare(view.feature_ids, requested)
        aligned_feature_ids = alignment.requested_ids
        # Keep the source sparse while selecting and aligning its columns.
        selected = view.values[list(rows), :][:, list(alignment.source_positions)]
    row_count, feature_count = selected.shape
    if row_count * feature_count > max_elements:
        raise ValueError("selected dense minibatch exceeds max_elements")
    dense = selected.toarray() if hasattr(selected, "toarray") else selected
    values = torch.as_tensor(dense, dtype=torch.float32, device=device)
    cell_ids = tuple(view.cell_ids[index] for index in rows)
    return values, aligned_feature_ids, cell_ids


def _validated_offsets(offsets: Sequence[int], extent: int) -> tuple[int, ...]:
    if isinstance(offsets, (str, bytes)):
        raise TypeError("event_offsets must be a sequence of integers")
    result = tuple(offsets)
    if not result or result[0] != 0 or result[-1] != extent:
        raise ValueError("event_offsets must start at zero and end at len(event_ids)")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in result):
        raise TypeError("event_offsets entries must be integers")
    if any(left > right for left, right in zip(result, result[1:])):
        raise ValueError("event_offsets must be nondecreasing")
    return result


def shifted_event_batch(
    event_ids: Tensor, event_offsets: Sequence[int], *, start_id: int
) -> tuple[Tensor, tuple[int, ...], Tensor, Tensor]:
    """Create packed teacher-forcing inputs without losing order or repeats.

    Every row gets a start token. A length-``n`` history becomes ``[START,
    e0, ..., e(n-2)]`` with the original events as next-event targets. Empty
    rows retain a start token but add no target. The returned prediction indices
    select the context positions aligned with the flattened targets.
    """
    if event_ids.ndim != 1 or event_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("event_ids must be a one-dimensional integer tensor")
    offsets = _validated_offsets(event_offsets, event_ids.numel())
    if start_id < 0:
        raise ValueError("start_id must be nonnegative")
    starts: list[Tensor] = []
    context_offsets = [0]
    prediction_positions: list[Tensor] = []
    for left, right in zip(offsets, offsets[1:]):
        length = right - left
        preceding = event_ids[left:right - 1] if length else event_ids[left:left]
        prefix = torch.cat((event_ids.new_tensor([start_id]), preceding))
        starts.append(prefix)
        context_start = context_offsets[-1]
        context_offsets.append(context_start + prefix.numel())
        if length:
            prediction_positions.append(
                torch.arange(context_start, context_start + length, device=event_ids.device)
            )
    context_ids = torch.cat(starts) if starts else event_ids.new_empty((0,))
    predictions = (
        torch.cat(prediction_positions)
        if prediction_positions else torch.empty(0, dtype=torch.long, device=event_ids.device)
    )
    return context_ids, tuple(context_offsets), event_ids, predictions


class EventHistoryReferenceModel(nn.Module):
    """Packed autoregressive event model with causal, row-local interactions.

    Integer identities and offsets are the event representation; repeated IDs
    remain separate occurrences. A learned positional term captures order in
    this example. Physical time and event payloads are optional model inputs and
    are deliberately not inferred here.
    """

    def __init__(
        self, event_vocabulary_size: int, width: int, *, heads: int = 1,
        max_history: int = 4096,
        binding: FeatureBinding | None = None,
    ) -> None:
        super().__init__()
        if event_vocabulary_size <= 0 or max_history <= 0:
            raise ValueError("event_vocabulary_size and max_history must be positive")
        if binding is not None and len(binding.identities) != event_vocabulary_size:
            raise ValueError("event_vocabulary_size must match the event binding")
        self.binding = binding
        self.event_vocabulary_size = int(event_vocabulary_size)
        self.width = int(width)
        self.start_id = (
            max(binding.model_indices + (() if binding.unknown_model_index is None
                                         else (binding.unknown_model_index,))) + 1
            if binding is not None else self.event_vocabulary_size
        )
        self.max_history = int(max_history)
        self.encoder = FeatureEncoder(
            self.start_id + 1, width, value_size=None, binding=binding
        )
        self.order_embedding = nn.Embedding(max_history, width)
        self.interactions = RelationAttention(width, heads=heads)
        event_output_ids = (
            binding.identities if binding is not None
            else tuple(range(event_vocabulary_size))
        )
        self.event_readout = EventPredictionHead(width, event_output_ids)

    def forward(
        self,
        event_ids: Tensor | Sequence[Any] | Mapping[str, Any],
        event_offsets: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        """Accept either a packed batch mapping or the two explicit arrays."""
        id_space = "identity"
        if isinstance(event_ids, Mapping):
            if event_offsets is not None:
                raise ValueError("event_offsets must be omitted when passing a batch mapping")
            batch = event_ids
            event_ids = batch["event_ids"]
            event_offsets = batch["event_offsets"]
            id_space = batch.get("event_id_space", "identity")
        if event_offsets is None:
            raise TypeError("event_offsets is required with flat event_ids")
        if self.binding is None:
            if not isinstance(event_ids, Tensor):
                event_ids = torch.as_tensor(event_ids, dtype=torch.long)
            if event_ids.ndim != 1 or event_ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("event_ids must be a one-dimensional integer tensor")
            offsets = _validated_offsets(event_offsets, event_ids.numel())
            if event_ids.numel() and bool(
                ((event_ids < 0) | (event_ids >= self.event_vocabulary_size)).any().item()
            ):
                raise ValueError("event_ids contain an identity outside the event vocabulary")
            event_input = event_ids
            target_ids: tuple[Any, ...] = tuple(event_ids.tolist())
            class_targets = None
        else:
            if id_space not in {"identity", "model"}:
                raise ValueError("event_id_space must be 'identity' or 'model'")
            if isinstance(event_ids, Tensor) and event_ids.device.type != "cpu":
                raise ValueError(
                    "prepare bound event occurrences from host identities before device transfer"
                )
            raw_ids = tuple(event_ids.tolist()) if isinstance(event_ids, Tensor) else tuple(event_ids)
            offsets = _validated_offsets(event_offsets, len(raw_ids))
            alignment = self.binding.map_occurrences(raw_ids, id_space=id_space)
            if alignment.dropped_ids:
                raise ValueError("event occurrences cannot be dropped by their binding")
            event_input = torch.tensor(
                alignment.model_indices, dtype=torch.long,
                device=(event_ids.device if isinstance(event_ids, Tensor) else self.encoder.feature_embedding.weight.device),
            )
            target_ids = tuple(alignment.requested_ids)
            class_positions = {identity: i for i, identity in enumerate(self.binding.identities)}
            try:
                class_targets = torch.tensor(
                    [class_positions[identity] for identity in target_ids],
                    dtype=torch.long, device=event_input.device,
                )
            except KeyError as exc:
                raise ValueError(
                    "an input event has no categorical output column in the event binding"
                ) from exc
        context_ids, context_offsets, targets, prediction_positions = shifted_event_batch(
            event_input, offsets, start_id=self.start_id
        )
        order_positions = torch.cat([
            torch.arange(right - left, device=event_input.device)
            for left, right in zip(context_offsets, context_offsets[1:])
        ]) if context_offsets[-1] else torch.empty(0, dtype=torch.long, device=event_input.device)
        if order_positions.numel() and int(order_positions.max().item()) >= self.max_history:
            raise ValueError("event context exceeds max_history")
        encoded = self.encoder(context_ids)
        encoded = encoded + self.order_embedding(order_positions)
        interacted = self.interactions(
            encoded, source_offsets=context_offsets, causal=True
        )
        target_context = interacted.index_select(0, prediction_positions)
        logits = self.event_readout(target_context)
        if class_targets is not None:
            targets = class_targets
        segment_lengths = torch.tensor(
            [right - left for left, right in zip(context_offsets, context_offsets[1:])],
            dtype=torch.long,
            device=event_input.device,
        )
        segment_ids = torch.repeat_interleave(
            torch.arange(segment_lengths.numel(), device=event_input.device),
            segment_lengths,
            output_size=interacted.shape[0],
        )
        history_sums = interacted.new_zeros((segment_lengths.numel(), self.width)).index_add(
            0, segment_ids, interacted
        )
        history_features = history_sums / segment_lengths.clamp_min(1).unsqueeze(-1)
        result = {
            "logits": logits,
            "targets": targets,
            # This caller-visible representation is available to auxiliary
            # observations; it is not a required or canonical biological state.
            "history_features": history_features,
            "context_ids": context_ids,
            "context_offsets": context_offsets,
            "prediction_positions": prediction_positions,
        }
        if self.binding is not None:
            result["target_ids"] = target_ids
            result["event_feature_ids"] = self.binding.identities
        return result

    def cellerator_biological_descriptors(self, *, identity_codec=None):
        if self.binding is None:
            return {}
        return {"event_output_binding": self.binding.to_manifest(codec=identity_codec)}


def event_next_token_loss(logits: Tensor, targets: Tensor) -> Tensor | None:
    """Teacher-forced categorical loss; empty histories have no supervised loss."""
    if targets.ndim != 1 or logits.ndim != 2 or logits.shape[0] != targets.numel():
        raise ValueError("logits must have one row per flattened event target")
    if targets.numel() == 0:
        return None
    return F.cross_entropy(logits, targets.to(dtype=torch.long))


__all__ = [
    "ExpressionReferenceModel", "EventHistoryReferenceModel", "attach_expression_view",
    "attach_event_view", "attach_atac_view", "attach_history_view", "masked_expression_mse",
    "atac_binary_loss", "shifted_event_batch",
    "event_next_token_loss", "explicit_dense_anndata_minibatch",
]
