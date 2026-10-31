"""Biological-identity alignment and explicit loss reduction utilities."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import torch
from torch import Tensor

from cellerator.biology import Prediction, PreparedAlignment


def _host_ids(values: Sequence[Any] | None, name: str) -> tuple[Any, ...] | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes, Tensor)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a host sequence of identities")
    result = tuple(values)
    try:
        set(result)
    except TypeError as exc:
        raise TypeError(f"{name} values must be hashable") from exc
    return result


def _unique_ids(values: tuple[Any, ...], name: str) -> dict[Any, int]:
    try:
        if len(set(values)) != len(values):
            raise ValueError(f"ambiguous repeated {name}; provide an explicit row_map")
        return {identity: index for index, identity in enumerate(values)}
    except TypeError as exc:
        raise TypeError(f"{name} values must be hashable") from exc


def _validated_offsets(
    offsets: Sequence[int] | None,
    row_count: int,
    name: str,
) -> tuple[int, ...] | None:
    if offsets is None:
        return None
    if isinstance(offsets, (str, bytes, Tensor)) or not isinstance(offsets, Sequence):
        raise TypeError(f"{name} must be a host sequence of integers")
    result = tuple(offsets)
    if not result or result[0] != 0 or result[-1] != row_count:
        raise ValueError(f"{name} must start at zero and end at the packed row count")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in result):
        raise TypeError(f"{name} entries must be integers")
    if any(left > right for left, right in zip(result, result[1:])):
        raise ValueError(f"{name} must be nondecreasing")
    return result


def _group_ranges(offsets: tuple[int, ...] | None, rows: int) -> tuple[tuple[int, int], ...]:
    if offsets is None:
        return tuple((index, index + 1) for index in range(rows))
    return tuple(zip(offsets, offsets[1:]))


def _row_mapping(
    prediction: Prediction,
    target_rows: int,
    target_sample_ids: Sequence[Any] | None,
    target_row_offsets: Sequence[int] | None,
    row_map: Sequence[int] | None,
) -> tuple[int, ...]:
    prediction_values = prediction.values
    if not isinstance(prediction_values, Tensor):
        raise TypeError("prediction.values must be a Torch Tensor for tensor supervision")
    if prediction_values.ndim < 2:
        raise ValueError("prediction values need a sample/occurrence axis and feature axis")
    prediction_rows = int(prediction_values.shape[0])
    pred_offsets = _validated_offsets(prediction.row_offsets, prediction_rows, "prediction.row_offsets")
    target_offsets = _validated_offsets(target_row_offsets, target_rows, "target_row_offsets")
    prediction_ranges = _group_ranges(pred_offsets, prediction_rows)
    target_ranges = _group_ranges(target_offsets, target_rows)
    prediction_groups, target_groups = len(prediction_ranges), len(target_ranges)

    if row_map is not None:
        if isinstance(row_map, (str, bytes, Tensor)) or not isinstance(row_map, Sequence):
            raise TypeError("row_map must be a host sequence of target group indices")
        mapping = tuple(row_map)
        if len(mapping) != prediction_groups:
            raise ValueError("row_map must contain one target-group index per prediction group")
        if any(not isinstance(index, int) or isinstance(index, bool) for index in mapping):
            raise TypeError("row_map entries must be integers")
        if any(index < 0 or index >= target_groups for index in mapping):
            raise IndexError("row_map contains a target group outside the target rows")
    else:
        prediction_ids = _host_ids(prediction.sample_ids, "prediction.sample_ids")
        target_ids = _host_ids(target_sample_ids, "target_sample_ids")
        if (prediction_ids is None) != (target_ids is None):
            raise ValueError("sample IDs must be supplied on both sides or resolved by an explicit row_map")
        if prediction_ids is None:
            if prediction_groups != target_groups:
                raise ValueError("unlabelled prediction and target group counts differ")
            mapping = tuple(range(prediction_groups))
        else:
            if len(prediction_ids) != prediction_groups or len(target_ids) != target_groups:
                raise ValueError("sample IDs must align with their row-offset groups")
            target_positions = _unique_ids(target_ids, "target sample IDs")
            _unique_ids(prediction_ids, "prediction sample IDs")
            try:
                mapping = tuple(target_positions[identity] for identity in prediction_ids)
            except KeyError as exc:
                raise KeyError(f"prediction sample ID is absent from targets: {exc.args[0]!r}") from exc

    row_positions: list[int] = []
    for prediction_group, target_group in enumerate(mapping):
        prediction_start, prediction_end = prediction_ranges[prediction_group]
        target_start, target_end = target_ranges[target_group]
        if prediction_end - prediction_start != target_end - target_start:
            raise ValueError("matched prediction and target groups have different row counts")
        row_positions.extend(range(target_start, target_end))
    if len(row_positions) != prediction_rows:
        raise ValueError("row alignment did not cover every prediction row")
    return tuple(row_positions)


def align_prediction_targets(
    prediction: Prediction,
    target_values: Tensor,
    prepared_alignment: PreparedAlignment,
    *,
    target_sample_ids: Sequence[Any] | None = None,
    target_row_offsets: Sequence[int] | None = None,
    row_map: Sequence[int] | None = None,
    feature_axis: int = -1,
) -> tuple[Tensor, Tensor]:
    """Align feature targets to prediction feature IDs and sample/window order.

    ``prepared_alignment`` is prepared from the target source feature order to
    the requested prediction feature IDs. Repeated sample/window identities
    are ambiguous unless the caller supplies ``row_map``. The returned target
    tensor remains differentiable when its source is differentiable.
    """
    if not isinstance(prediction, Prediction):
        raise TypeError("prediction must be a Cellerator Prediction")
    if not isinstance(prepared_alignment, PreparedAlignment):
        raise TypeError("prepared_alignment must be a PreparedAlignment")
    if not isinstance(prediction.values, Tensor) or not isinstance(target_values, Tensor):
        raise TypeError("prediction and target values must be Torch Tensors")
    if prediction.feature_ids is None:
        raise ValueError("prediction must declare feature_ids for target alignment")
    if tuple(prediction.feature_ids) != prepared_alignment.requested_ids:
        raise ValueError("prepared feature destination order must match Prediction.feature_ids")
    if target_values.ndim != prediction.values.ndim:
        raise ValueError("prediction and target values must have the same rank")
    axis = feature_axis + target_values.ndim if feature_axis < 0 else feature_axis
    if axis <= 0 or axis >= target_values.ndim:
        raise ValueError("feature_axis must be a non-leading tensor axis")
    if prediction.values.shape[axis] != len(prepared_alignment.requested_ids):
        raise ValueError("prediction feature axis does not match Prediction.feature_ids")
    aligned_targets = prepared_alignment.gather(target_values, axis=axis)
    row_positions = _row_mapping(
        prediction, aligned_targets.shape[0], target_sample_ids,
        target_row_offsets, row_map,
    )
    row_indices = torch.tensor(row_positions, dtype=torch.long, device=aligned_targets.device)
    aligned_targets = aligned_targets.index_select(0, row_indices)
    aligned_predictions = prediction.values
    if aligned_predictions.shape != aligned_targets.shape:
        raise ValueError("aligned prediction and target tensor shapes must match exactly")
    return aligned_predictions, aligned_targets


def categorical_target_columns(
    prediction: Prediction,
    event_labels: Sequence[Any],
) -> Tensor:
    """Map categorical biological labels to Prediction class-column positions.

    Event/token identities are looked up in ``Prediction.feature_ids``; their
    integer token values are never assumed to equal output-column offsets.
    """
    if not isinstance(prediction, Prediction):
        raise TypeError("prediction must be a Cellerator Prediction")
    if prediction.feature_ids is None:
        raise ValueError("categorical Prediction must declare class-column feature_ids")
    if not isinstance(prediction.values, Tensor):
        raise TypeError("categorical prediction values must be a Torch Tensor")
    if isinstance(event_labels, (str, bytes, Tensor)) or not isinstance(event_labels, Sequence):
        raise TypeError("event_labels must be a host sequence of biological/token identities")
    columns = {identity: index for index, identity in enumerate(prediction.feature_ids)}
    try:
        indices = tuple(columns[identity] for identity in event_labels)
    except TypeError as exc:
        raise TypeError("event labels must be hashable identities") from exc
    except KeyError as exc:
        raise KeyError(f"event label has no Prediction class column: {exc.args[0]!r}") from exc
    return torch.tensor(indices, dtype=torch.long, device=prediction.values.device)


def _group_ids(offsets: tuple[int, ...] | None, rows: int) -> tuple[int, ...]:
    if offsets is None:
        return tuple(range(rows))
    return tuple(
        group
        for group, (start, end) in enumerate(zip(offsets, offsets[1:]))
        for _ in range(end - start)
    )


def reduce_prediction_loss(
    losses: Tensor,
    *,
    unit: Literal["observation", "cell"],
    valid_mask: Tensor | None = None,
    sample_ids: Sequence[Any] | None = None,
    row_offsets: Sequence[int] | None = None,
    cell_ids: Sequence[Any] | None = None,
) -> tuple[Tensor, Tensor]:
    """Reduce unreduced losses over explicit observations or physical cells.

    Returns ``(mean_loss, valid_unit_count)``. Cell reduction first pools every
    valid scalar observation across all windows of one explicitly supplied cell,
    then averages cells equally. ``sample_ids`` identify distinct observation
    or window groups; ``cell_ids`` separately identify physical owners and may
    repeat. Empty validity returns a differentiable zero and a device-local
    zero count. Optimizer behavior remains the caller's decision.
    """
    if not isinstance(losses, Tensor) or not losses.is_floating_point():
        raise TypeError("losses must be a floating-point Torch Tensor")
    if unit not in {"observation", "cell"}:
        raise ValueError("unit must be 'observation' or 'cell'")
    if valid_mask is None:
        mask = torch.ones_like(losses, dtype=torch.bool)
    else:
        if not isinstance(valid_mask, Tensor) or valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must be a Boolean Torch Tensor")
        if valid_mask.shape != losses.shape or valid_mask.device != losses.device:
            raise ValueError("valid_mask must match losses shape and device")
        mask = valid_mask
    safe = torch.where(mask, losses, torch.zeros_like(losses))

    if unit == "observation":
        count = mask.sum(dtype=torch.long)
        mean = safe.sum() / count.clamp_min(1).to(dtype=losses.dtype)
        if sample_ids is not None:
            ids = _host_ids(sample_ids, "sample_ids")
            row_count = losses.shape[0] if losses.ndim else 1
            offsets = _validated_offsets(row_offsets, row_count, "row_offsets")
            group_count = len(offsets) - 1 if offsets is not None else row_count
            if len(ids) != group_count:
                raise ValueError("sample_ids must label observation groups")
        return mean, count

    if losses.ndim == 0:
        row_count = 1
        flattened = safe.reshape(1)
        flat_mask = mask.reshape(1)
        terms_per_row = 1
    else:
        row_count = int(losses.shape[0])
        terms_per_row = 1
        for extent in losses.shape[1:]:
            terms_per_row *= int(extent)
        flattened = safe.reshape(-1)
        flat_mask = mask.reshape(-1)
    offsets = _validated_offsets(row_offsets, row_count, "row_offsets")
    row_group_ids = _group_ids(offsets, row_count)
    group_count = len(offsets) - 1 if offsets is not None else row_count
    window_ids = _host_ids(sample_ids, "sample_ids")
    owners = _host_ids(cell_ids, "cell_ids")
    if window_ids is None or owners is None:
        raise ValueError("cell reduction requires separate sample_ids and physical cell_ids")
    if len(window_ids) != group_count or len(owners) != group_count:
        raise ValueError("sample_ids and cell_ids must label every row-offset group")
    _unique_ids(window_ids, "sample/window IDs")

    owner_positions: dict[Any, int] = {}
    group_owner_indices: list[int] = []
    for owner in owners:
        try:
            index = owner_positions.get(owner)
        except TypeError as exc:
            raise TypeError("cell_ids must be hashable physical identities") from exc
        if index is None:
            index = len(owner_positions)
            owner_positions[owner] = index
        group_owner_indices.append(index)

    row_owner_indices = [group_owner_indices[group] for group in row_group_ids]
    term_owner_indices = [
        owner for owner in row_owner_indices for _ in range(terms_per_row)
    ]
    owner_index_tensor = torch.tensor(term_owner_indices, dtype=torch.long, device=losses.device)
    owner_count = len(owner_positions)
    sums = losses.new_zeros((owner_count,)).index_add(
        0, owner_index_tensor, flattened
    )
    counts = torch.zeros((owner_count,), dtype=torch.long, device=losses.device).index_add(
        0, owner_index_tensor, flat_mask.to(dtype=torch.long)
    )
    valid_owners = counts > 0
    cell_means = sums / counts.clamp_min(1).to(dtype=losses.dtype)
    count = valid_owners.sum(dtype=torch.long)
    mean = torch.where(valid_owners, cell_means, torch.zeros_like(cell_means)).sum() / count.clamp_min(1).to(
        dtype=losses.dtype
    )
    # Keep the empty return connected to the original loss graph even when
    # there are no owner groups at all.
    mean = mean + losses.sum() * 0.0 if owner_count == 0 else mean
    return mean, count


__all__ = [
    "align_prediction_targets", "categorical_target_columns", "reduce_prediction_loss",
]
