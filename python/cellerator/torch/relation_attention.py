"""Attention over biological feature occurrences and supplied relations.

This module is one reusable interaction family, not a prescribed biological
ontology. Feature identities, context, and support are chosen by the caller;
attention learns weights only among the participants presented to it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Hashable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .relations import PreparedRelation
from ._native_nn import native_nn, validate_implementation, functional_children_supported


def _host_offsets(
    offsets: Sequence[int] | None,
    extent: int,
    name: str,
) -> tuple[int, ...] | None:
    if offsets is None:
        return None
    if isinstance(offsets, Tensor) or not isinstance(offsets, Sequence):
        raise TypeError(f"{name} must be an immutable host sequence of integers")
    result = tuple(offsets)
    if len(result) == 0:
        raise ValueError(f"{name} must contain the initial zero offset")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in result):
        raise TypeError(f"{name} entries must be integers")
    if result[0] != 0 or result[-1] != extent:
        raise ValueError(f"{name} must start at 0 and end at the packed extent")
    if any(left > right for left, right in zip(result, result[1:])):
        raise ValueError(f"{name} must be nondecreasing")
    return result


def _group_ids(offsets: tuple[int, ...], device: torch.device) -> Tensor:
    lengths = torch.tensor(
        [right - left for left, right in zip(offsets, offsets[1:])],
        device=device,
        dtype=torch.long,
    )
    return torch.repeat_interleave(
        torch.arange(len(offsets) - 1, device=device), lengths,
        output_size=offsets[-1],
    )


def _csr_tensors(
    row_ptr: Tensor | Sequence[int],
    col_indices: Tensor | Sequence[int],
    n_targets: int,
    n_sources: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    def structural_tensor(value: Tensor | Sequence[int], name: str) -> Tensor:
        if isinstance(value, Tensor):
            if value.dtype not in (torch.int32, torch.int64):
                raise TypeError(f"{name} must have dtype int32 or int64")
            if value.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional")
            return value.to(device=device, dtype=torch.long)
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(f"{name} must be an integer sequence or tensor")
        if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
            raise TypeError(f"{name} entries must be integers")
        return torch.tensor(tuple(value), device=device, dtype=torch.long)

    ptr = structural_tensor(row_ptr, "row_ptr")
    columns = structural_tensor(col_indices, "col_indices")
    if ptr.shape != (n_targets + 1,):
        raise ValueError("row_ptr must have one entry per target plus the terminal offset")
    if ptr.numel() == 0 or int(ptr[0].item()) != 0:
        raise ValueError("row_ptr must start at 0")
    if int(ptr[-1].item()) != columns.numel():
        raise ValueError("row_ptr terminal offset must equal len(col_indices)")
    if bool(torch.any(ptr[1:] < ptr[:-1]).item()):
        raise ValueError("row_ptr must be nondecreasing")
    if columns.numel() and bool(torch.any((columns < 0) | (columns >= n_sources)).item()):
        raise ValueError("col_indices contain a source index outside the input")
    rows = torch.repeat_interleave(
        torch.arange(n_targets, device=device), ptr[1:] - ptr[:-1],
        output_size=columns.numel(),
    )
    return rows, columns


class RelationAttention(nn.Module):
    """Multi-head attention within dense groups or a supplied sparse relation.

    Dense inputs have shape ``[batch, sequence, width]``. Packed inputs have
    shape ``[occurrences, width]`` and may provide source and target offsets to
    define independent groups. In the packed form, no offsets means one group.
    Optional CSR support has one row per flattened target and uses global source
    occurrence indices; duplicate columns remain duplicate participants.
    Packed grouping is computational, not automatically one-group-per-cell: no
    offsets defines one caller-selected global group, while supplied offsets
    define boundaries that CSR edges must respect. Callers can therefore model
    cross-cell relations explicitly by leaving packed inputs in one group and
    supplying the desired CSR support.

    ``target=None`` selects self-attention. Causal masking is only available for
    self-attention and compares source and target positions inside each group.
    The output is an ordinary Torch tensor aligned with the target axes. Empty
    neighborhoods contribute zero; residual updates are left to the caller.
    Context conditioning is also caller-composed: condition source and target
    features before passing them here.
    """

    def __init__(self, width: int, heads: int = 1, dropout: float = 0.0, *,
                 implementation: str = "auto", kernel: str = "aten") -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("width must be positive")
        if heads <= 0 or width % heads:
            raise ValueError("heads must be positive and divide width")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.dropout = float(dropout)
        self.implementation = validate_implementation(implementation)
        if kernel not in {"aten", "cellerator_csr"}:
            raise ValueError("kernel must be 'aten' or 'cellerator_csr'")
        self.kernel = kernel
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width, bias=False)

    def _project(self, module: nn.Linear, values: Tensor) -> Tensor:
        return module(values).reshape(*values.shape[:-1], self.heads, self.head_width)

    def _dropout_probability(self) -> float:
        return self.dropout if self.training else 0.0

    def _sparse_aggregate(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        rows: Tensor,
        columns: Tensor,
        n_targets: int,
    ) -> Tensor:
        native = native_nn(self.implementation)
        if native is not None:
            return native.sparse_attention(
                query, key, value, rows, columns, self.head_width ** -0.5,
                self.dropout, self.training,
            )
        if columns.numel() == 0:
            # Keep a zero-gradient path to source and target projections. This
            # makes empty support an ordinary differentiable case. Sanitizing
            # before multiplying by zero keeps an unused NaN/Inf from leaking
            # out of an empty neighborhood.
            zero = (
                torch.nan_to_num(query) * 0.0
                + torch.nan_to_num(key).sum() * 0.0
                + torch.nan_to_num(value).sum() * 0.0
            )
            return zero

        scale = self.head_width ** -0.5
        edge_query = query.index_select(0, rows)
        edge_key = key.index_select(0, columns)
        scores = (edge_query * edge_key).sum(dim=-1) * scale

        row_heads = rows.unsqueeze(-1).expand(-1, self.heads)
        maxima = scores.new_full((n_targets, self.heads), -torch.inf)
        maxima = maxima.scatter_reduce(
            0, row_heads, scores, reduce="amax", include_self=True
        )
        numerators = torch.exp(scores - maxima.index_select(0, rows))
        denominators = scores.new_zeros((n_targets, self.heads)).index_add(
            0, rows, numerators
        )
        weights = numerators / denominators.index_select(0, rows).clamp_min(
            torch.finfo(scores.dtype).tiny
        )
        if self._dropout_probability():
            weights = F.dropout(weights, p=self.dropout, training=True)
        messages = value.index_select(0, columns) * weights.unsqueeze(-1)
        return value.new_zeros((n_targets, self.heads, self.head_width)).index_add(
            0, rows, messages
        )

    def _packed_dense_edges(
        self,
        source_offsets: tuple[int, ...],
        target_offsets: tuple[int, ...],
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        source_lengths = torch.tensor(
            [b - a for a, b in zip(source_offsets, source_offsets[1:])],
            dtype=torch.long, device=device,
        )
        target_lengths = torch.tensor(
            [b - a for a, b in zip(target_offsets, target_offsets[1:])],
            dtype=torch.long, device=device,
        )
        edge_counts = source_lengths * target_lengths
        n_edges = sum(
            (source_offsets[i + 1] - source_offsets[i])
            * (target_offsets[i + 1] - target_offsets[i])
            for i in range(len(source_lengths))
        )
        groups = torch.repeat_interleave(
            torch.arange(len(source_lengths), device=device), edge_counts,
            output_size=n_edges,
        )
        edge_starts = edge_counts.cumsum(0) - edge_counts
        local_edge = torch.arange(n_edges, device=device) - edge_starts.index_select(0, groups)
        local_source_count = source_lengths.index_select(0, groups).clamp_min(1)
        target_local = torch.div(local_edge, local_source_count, rounding_mode="floor")
        source_local = torch.remainder(local_edge, local_source_count)
        target_base = torch.tensor(target_offsets[:-1], device=device, dtype=torch.long)
        source_base = torch.tensor(source_offsets[:-1], device=device, dtype=torch.long)
        target_rows = target_base.index_select(0, groups) + target_local
        source_columns = source_base.index_select(0, groups) + source_local
        return target_rows, source_columns, target_local, source_local

    def forward(
        self,
        source: Tensor,
        target: Tensor | None = None,
        *,
        source_offsets: Sequence[int] | None = None,
        target_offsets: Sequence[int] | None = None,
        row_ptr: Tensor | Sequence[int] | None = None,
        col_indices: Tensor | Sequence[int] | None = None,
        causal: bool = False,
        relation: PreparedRelation | None = None,
        source_order: Sequence[Hashable] | None = None,
        target_order: Sequence[Hashable] | None = None,
        source_axis: Any = None,
        target_axis: Any = None,
        structural_generation: int = 0,
    ) -> Tensor:
        if not isinstance(source, Tensor):
            raise TypeError("source must be a Torch Tensor")
        if target is None:
            target = source
        elif not isinstance(target, Tensor):
            raise TypeError("target must be a Torch Tensor")
        if source.ndim not in (2, 3) or target.ndim != source.ndim:
            raise ValueError("source and target must both be packed 2D or dense 3D tensors")
        if source.shape[-1] != self.width or target.shape[-1] != self.width:
            raise ValueError("source and target final dimensions must equal width")
        if source.device != target.device:
            raise ValueError("source and target must share a device")
        if source.dtype != target.dtype:
            raise TypeError("source and target must share a dtype")
        if (row_ptr is None) != (col_indices is None):
            raise ValueError("row_ptr and col_indices must be supplied together")
        if relation is not None and not isinstance(relation, PreparedRelation):
            raise TypeError("relation must be a PreparedRelation or None")
        if relation is not None and row_ptr is not None:
            raise ValueError("supply either a prepared relation or raw CSR, not both")
        if causal and target is not source:
            raise ValueError("causal attention is only supported when target is source")

        native = native_nn(self.implementation)
        if self.kernel == "cellerator_csr":
            if native is None or not hasattr(native, "csr_scaled_dot_product_attention_cuda_prepared"):
                raise RuntimeError("Cellerator CSR attention requires the optional native NN CUDA component")
            if relation is None:
                raise ValueError("Cellerator CSR attention requires a PreparedRelation")
            if self._dropout_probability():
                raise ValueError("Cellerator CSR attention does not support active dropout")
        elif (native is not None and relation is None and row_ptr is None
              and functional_children_supported(self.query, self.key, self.value, self.output)):
            src_offsets, dst_offsets = (), ()
            if source.ndim == 2:
                src_offsets = _host_offsets(source_offsets, source.shape[0], "source_offsets")
                dst_offsets = _host_offsets(target_offsets, target.shape[0], "target_offsets")
                if target is source:
                    src_offsets = src_offsets if src_offsets is not None else dst_offsets
                    dst_offsets = dst_offsets if dst_offsets is not None else src_offsets
                elif (src_offsets is None) != (dst_offsets is None):
                    raise ValueError("packed source and target offsets must be supplied together")
            elif source_offsets is not None or target_offsets is not None:
                raise ValueError("offsets are only used with packed 2D inputs")
            return native.relation_attention(
                source, target,
                [self.query.weight, self.query.bias, self.key.weight, self.key.bias,
                 self.value.weight, self.value.bias, self.output.weight],
                self.width, self.heads, self.dropout,
                src_offsets or (), dst_offsets or (), causal, self.training,
            )

        dense = source.ndim == 3
        if dense:
            if source_offsets is not None or target_offsets is not None:
                raise ValueError("offsets are only used with packed 2D inputs")
            if source.shape[0] != target.shape[0]:
                raise ValueError("dense source and target batch sizes must match")
            batch, n_sources, _ = source.shape
            _, n_targets, _ = target.shape
            if (row_ptr is None and relation is None and batch > 0
                    and n_sources > 0 and n_targets > 0):
                q = self._project(self.query, target).transpose(1, 2)
                k = self._project(self.key, source).transpose(1, 2)
                v = self._project(self.value, source).transpose(1, 2)
                heads = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self._dropout_probability(),
                    is_causal=causal,
                )
                combined = heads.transpose(1, 2).reshape(batch, n_targets, self.width)
                return self.output(combined)

            source_flat = source.reshape(-1, self.width)
            target_flat = target.reshape(-1, self.width)
            source_offsets_tuple = tuple(i * n_sources for i in range(batch + 1))
            target_offsets_tuple = tuple(i * n_targets for i in range(batch + 1))
            n_sources = source_flat.shape[0]
            n_targets = target_flat.shape[0]
        else:
            n_sources, n_targets = source.shape[0], target.shape[0]
            source_offsets_tuple = _host_offsets(source_offsets, n_sources, "source_offsets")
            target_offsets_tuple = _host_offsets(target_offsets, n_targets, "target_offsets")
            if (source_offsets_tuple is None) != (target_offsets_tuple is None):
                if target is source and target_offsets_tuple is None:
                    target_offsets_tuple = source_offsets_tuple
                elif source is target and source_offsets_tuple is None:
                    source_offsets_tuple = target_offsets_tuple
                else:
                    raise ValueError("packed source and target offsets must be supplied together")
            if source_offsets_tuple is None:
                source_offsets_tuple = (0, n_sources)
                target_offsets_tuple = (0, n_targets)
            if len(source_offsets_tuple) != len(target_offsets_tuple):
                raise ValueError("source and target offsets must describe the same groups")
            source_flat, target_flat = source, target

        q = self._project(self.query, target_flat)
        k = self._project(self.key, source_flat)
        v = self._project(self.value, source_flat)

        if relation is not None:
            rows, columns = relation.validate_for(
                source_order=source_order,
                target_order=target_order,
                source_offsets=source_offsets_tuple,
                target_offsets=target_offsets_tuple,
                source_axis=source_axis,
                target_axis=target_axis,
                structural_generation=structural_generation,
                n_sources=n_sources,
                n_targets=n_targets,
                device=source.device,
            )
            if self.kernel == "cellerator_csr":
                if causal and relation.has_future_edges:
                    raise ValueError("prepare relation.causal() before selecting native causal CSR attention")
                combined_heads = native.csr_scaled_dot_product_attention_cuda_prepared(
                    q, k, v, relation._row_ptr, relation._col_indices,
                    self.head_width ** -0.5, 0.0,
                )
                combined = combined_heads.reshape(n_targets, self.width)
                output = self.output(combined)
                return output.reshape(*target.shape[:-1], self.width) if dense else output
            if causal:
                if source_offsets_tuple != target_offsets_tuple:
                    raise ValueError("causal self-attention requires identical group offsets")
                keep = relation._device_causal_mask(source.device)
                rows, columns = rows[keep], columns[keep]
            combined_heads = self._sparse_aggregate(q, k, v, rows, columns, n_targets)
        elif row_ptr is not None:
            # Raw CSR remains a convenient numerical interface. It validates
            # structure on each call; callers with stable support can prepare a
            # PreparedRelation to avoid device-to-host validation synchronizes.
            source_group = _group_ids(source_offsets_tuple, source.device)
            target_group = _group_ids(target_offsets_tuple, target.device)
            rows, columns = _csr_tensors(
                row_ptr, col_indices, n_targets, n_sources, source.device
            )
            if columns.numel() and bool(torch.any(
                source_group.index_select(0, columns)
                != target_group.index_select(0, rows)
            ).item()):
                raise ValueError("CSR support crosses a declared group boundary")
            if causal:
                if source_offsets_tuple != target_offsets_tuple:
                    raise ValueError("causal self-attention requires identical group offsets")
                if columns.numel():
                    if dense:
                        source_local = columns.remainder(source.shape[1])
                        target_local = rows.remainder(target.shape[1])
                    else:
                        offsets = torch.tensor(source_offsets_tuple[:-1], device=source.device)
                        source_local = columns - offsets.index_select(
                            0, source_group.index_select(0, columns)
                        )
                        target_local = rows - offsets.index_select(
                            0, target_group.index_select(0, rows)
                        )
                    keep = source_local <= target_local
                    rows, columns = rows[keep], columns[keep]
            combined_heads = self._sparse_aggregate(q, k, v, rows, columns, n_targets)
        elif dense:
            # This case is reached when a dense source or target axis is empty.
            # Preserve zero-gradient paths while returning a bias-free zero.
            combined_heads = (
                torch.nan_to_num(q) * 0.0
                + torch.nan_to_num(k).sum() * 0.0
                + torch.nan_to_num(v).sum() * 0.0
            )
        else:
            rows, columns, target_local, source_local = self._packed_dense_edges(
                source_offsets_tuple, target_offsets_tuple, source.device
            )
            if causal:
                keep = source_local <= target_local
                rows, columns = rows[keep], columns[keep]
            combined_heads = self._sparse_aggregate(q, k, v, rows, columns, n_targets)

        combined = combined_heads.reshape(n_targets, self.width)
        output = self.output(combined)
        return output.reshape(*target.shape[:-1], self.width) if dense else output


__all__ = ["RelationAttention"]
