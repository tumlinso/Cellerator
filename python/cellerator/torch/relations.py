"""Prepared, identity-bound occurrence relations for Torch attention.

The numerical relation is a CSR list of *occurrence positions*. Biological
feature identity and occurrence identity are deliberately separate: source
and target order tuples may contain repeated identities, and duplicate CSR
edges remain duplicate contributions. Preparation validates host structure
once and owns the resulting indices. Move a relation explicitly with
``relation.to(device)`` before using it on that device.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
import operator
from typing import Any, Hashable

import torch
from torch import Tensor

from cellerator._declarations import Axis, Identity


def _host_ints(values: Tensor | Sequence[int], name: str) -> tuple[int, ...]:
    if isinstance(values, Tensor):
        if values.device.type != "cpu":
            raise ValueError(f"{name} must be supplied on the host during preparation")
        if values.dtype not in (torch.int32, torch.int64) or values.ndim != 1:
            raise TypeError(f"{name} must be a one-dimensional integer tensor or sequence")
        values = values.tolist()
    elif isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a one-dimensional host integer sequence")
    result: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise TypeError(f"{name} entries must be integers, not bool")
        try:
            result.append(operator.index(value))
        except TypeError as exc:
            raise TypeError(f"{name} entries must be integers") from exc
    return tuple(result)


def _order(values: Sequence[Hashable], name: str) -> tuple[Hashable, ...]:
    if isinstance(values, (str, bytes)) or isinstance(values, Tensor) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be an immutable host sequence of occurrence identities")
    result = tuple(values)
    for identity in result:
        try:
            hash(identity)
        except TypeError as exc:
            raise TypeError(f"{name} entries must be hashable") from exc
    return result


def _offsets(values: Sequence[int] | None, extent: int, name: str) -> tuple[int, ...]:
    if values is None:
        return (0, extent)
    result = _host_ints(values, name)
    if not result or result[0] != 0 or result[-1] != extent:
        raise ValueError(f"{name} must start at 0 and end at extent {extent}")
    if any(left > right for left, right in zip(result, result[1:])):
        raise ValueError(f"{name} must be nondecreasing")
    return result


def _axis_extent(axis: Any, name: str) -> None:
    if axis is not None and not isinstance(axis, Axis):
        raise TypeError(f"{name} must be a Cellerator Axis or None")


def _encode_identity(value: Hashable, codec: Any = None) -> Any:
    """Encode stable built-ins directly; require an explicit codec otherwise."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float identities cannot be serialized")
        return value
    if isinstance(value, Identity):
        return {"kind": "cellerator.identity.v1", "low": value.low, "high": value.high}
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_encode_identity(item, codec) for item in value]}
    if codec is None:
        raise TypeError(
            f"identity of type {type(value).__name__} needs an explicit manifest codec"
        )
    if callable(codec):
        encoded = codec(value)
    elif isinstance(codec, Mapping):
        try:
            encoded = codec[value]
        except KeyError as exc:
            raise KeyError("manifest codec has no entry for this identity") from exc
    elif callable(getattr(codec, "encode", None)):
        encoded = codec.encode(value)
    else:
        raise TypeError("codec must be callable, a mapping, or expose encode(identity)")
    try:
        json.dumps(encoded, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("manifest codec must return a JSON-serializable value") from exc
    return encoded


def _axis_manifest(axis: Any, codec: Any = None) -> dict[str, Any] | None:
    if axis is None:
        return None
    def encode_axis_identity(identity: Identity) -> dict[str, int]:
        return {"low": identity.low, "high": identity.high}
    return {
        "domain": encode_axis_identity(axis.domain),
        "order": encode_axis_identity(axis.order),
        "geometry": encode_axis_identity(axis.geometry),
        "partition": encode_axis_identity(axis.partition),
        "extent": axis.extent,
    }


class PreparedRelation:
    """Validated CSR support bound to source/target occurrence order.

    ``from_csr`` accepts row pointers and columns on CPU (or host sequences),
    validates CSR/group geometry, preserves duplicate edges and empty rows, and
    materializes owned long indices on ``device``. ``source_order`` and
    ``target_order`` are occurrence signatures, not unique vocabulary lists.
    Values may change between calls; topology, order, offsets, axes, generation,
    extents, and device must match the prepared declaration.
    """

    __slots__ = (
        "_row_ptr_host", "_col_indices_host", "_rows_host", "_causal_host",
        "source_order", "target_order", "source_offsets", "target_offsets",
        "source_axis", "target_axis", "structural_generation", "n_sources",
        "n_targets", "_device", "_row_ptr", "_col_indices", "_rows",
        "_causal_mask", "_transpose_cache", "_structure_versions", "_has_future_edges", "_frozen",
    )

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("PreparedRelation is immutable; prepare a new relation")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        row_ptr: Tensor | Sequence[int],
        col_indices: Tensor | Sequence[int],
        *,
        source_order: Sequence[Hashable],
        target_order: Sequence[Hashable],
        source_offsets: Sequence[int] | None = None,
        target_offsets: Sequence[int] | None = None,
        source_axis: Axis | None = None,
        target_axis: Axis | None = None,
        structural_generation: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        src_order = _order(source_order, "source_order")
        dst_order = _order(target_order, "target_order")
        n_sources, n_targets = len(src_order), len(dst_order)
        ptr = _host_ints(row_ptr, "row_ptr")
        columns = _host_ints(col_indices, "col_indices")
        if len(ptr) != n_targets + 1:
            raise ValueError("row_ptr must have one entry per target plus terminal offset")
        if not ptr or ptr[0] != 0:
            raise ValueError("row_ptr must start at 0")
        if ptr[-1] != len(columns):
            raise ValueError("row_ptr terminal offset must equal len(col_indices)")
        if any(left > right for left, right in zip(ptr, ptr[1:])):
            raise ValueError("row_ptr must be nondecreasing")
        if any(column < 0 or column >= n_sources for column in columns):
            raise ValueError("col_indices contain a source index outside source_order")

        src_offsets = _offsets(source_offsets, n_sources, "source_offsets")
        dst_offsets = _offsets(target_offsets, n_targets, "target_offsets")
        if len(src_offsets) != len(dst_offsets):
            raise ValueError("source and target offsets must define the same number of groups")
        _axis_extent(source_axis, "source_axis")
        _axis_extent(target_axis, "target_axis")
        if source_axis is not None and source_axis.extent != n_sources:
            raise ValueError("source_axis extent must equal source occurrence count")
        if target_axis is not None and target_axis.extent != n_targets:
            raise ValueError("target_axis extent must equal target occurrence count")
        if isinstance(structural_generation, bool) or not isinstance(structural_generation, int):
            raise TypeError("structural_generation must be a nonnegative integer")
        if structural_generation < 0:
            raise ValueError("structural_generation must be nonnegative")

        rows: list[int] = []
        for target, (left, right) in enumerate(zip(ptr, ptr[1:])):
            rows.extend([target] * (right - left))
        src_group = [0] * n_sources
        dst_group = [0] * n_targets
        src_local = [0] * n_sources
        dst_local = [0] * n_targets
        for group, (left, right) in enumerate(zip(src_offsets, src_offsets[1:])):
            for position in range(left, right):
                src_group[position] = group
                src_local[position] = position - left
        for group, (left, right) in enumerate(zip(dst_offsets, dst_offsets[1:])):
            for position in range(left, right):
                dst_group[position] = group
                dst_local[position] = position - left
        causal = []
        for row, column in zip(rows, columns):
            if src_group[column] != dst_group[row]:
                raise ValueError("CSR support crosses a declared group boundary")
            causal.append(src_local[column] <= dst_local[row])

        target_device = torch.device(device)
        self._row_ptr_host = ptr
        self._col_indices_host = columns
        self._rows_host = tuple(rows)
        self._causal_host = tuple(causal)
        self.source_order = src_order
        self.target_order = dst_order
        self.source_offsets = src_offsets
        self.target_offsets = dst_offsets
        self.source_axis = source_axis
        self.target_axis = target_axis
        self.structural_generation = structural_generation
        self.n_sources = n_sources
        self.n_targets = n_targets
        self._device = target_device
        # Index tensors need version counters even when preparation is called
        # inside inference_mode. Checking counters never reads device contents.
        with torch.inference_mode(False):
            self._row_ptr = torch.tensor(ptr, dtype=torch.long, device=target_device)
            self._col_indices = torch.tensor(columns, dtype=torch.long, device=target_device)
            self._rows = torch.tensor(rows, dtype=torch.long, device=target_device)
            self._causal_mask = torch.tensor(causal, dtype=torch.bool, device=target_device)
        self._structure_versions = tuple(tensor._version for tensor in (
            self._row_ptr, self._col_indices, self._rows, self._causal_mask
        ))
        self._has_future_edges = not all(causal)
        self._transpose_cache = None
        self._frozen = True

    @classmethod
    def from_csr(
        cls,
        row_ptr: Tensor | Sequence[int],
        col_indices: Tensor | Sequence[int],
        *,
        source_order: Sequence[Hashable],
        target_order: Sequence[Hashable],
        source_offsets: Sequence[int] | None = None,
        target_offsets: Sequence[int] | None = None,
        source_axis: Axis | None = None,
        target_axis: Axis | None = None,
        structural_generation: int = 0,
        device: torch.device | str = "cpu",
    ) -> "PreparedRelation":
        """Create an immutable relation from host CSR structure."""
        return cls(
            row_ptr, col_indices, source_order=source_order, target_order=target_order,
            source_offsets=source_offsets, target_offsets=target_offsets,
            source_axis=source_axis, target_axis=target_axis,
            structural_generation=structural_generation, device=device,
        )

    @classmethod
    def from_dense(
        cls,
        membership: Tensor | Sequence[Sequence[bool]],
        *,
        source_order: Sequence[Hashable],
        target_order: Sequence[Hashable],
        source_offsets: Sequence[int] | None = None,
        target_offsets: Sequence[int] | None = None,
        source_axis: Axis | None = None,
        target_axis: Axis | None = None,
        structural_generation: int = 0,
        device: torch.device | str = "cpu",
    ) -> "PreparedRelation":
        """Prepare a host Boolean ``[targets, sources]`` support matrix."""
        if isinstance(membership, Tensor):
            if membership.device.type != "cpu":
                raise ValueError("dense membership must be supplied on CPU during preparation")
            if membership.dtype != torch.bool or membership.ndim != 2:
                raise TypeError("membership must be a two-dimensional Boolean tensor")
            matrix = membership.tolist()
        else:
            if isinstance(membership, (str, bytes)) or not isinstance(membership, Sequence):
                raise TypeError("membership must be a two-dimensional Boolean host sequence")
            matrix = [list(row) for row in membership]
            if any(any(not isinstance(value, bool) for value in row) for row in matrix):
                raise TypeError("membership entries must be Boolean")
        src_order, dst_order = _order(source_order, "source_order"), _order(target_order, "target_order")
        if len(matrix) != len(dst_order) or any(len(row) != len(src_order) for row in matrix):
            raise ValueError("membership shape must be [len(target_order), len(source_order)]")
        ptr, columns = [0], []
        for row in matrix:
            columns.extend(index for index, present in enumerate(row) if present)
            ptr.append(len(columns))
        return cls.from_csr(
            ptr, columns, source_order=src_order, target_order=dst_order,
            source_offsets=source_offsets, target_offsets=target_offsets,
            source_axis=source_axis, target_axis=target_axis,
            structural_generation=structural_generation, device=device,
        )

    @property
    def row_ptr(self) -> tuple[int, ...]:
        return self._row_ptr_host

    @property
    def col_indices(self) -> tuple[int, ...]:
        return self._col_indices_host

    @property
    def row_indices(self) -> tuple[int, ...]:
        return self._rows_host

    @property
    def device(self) -> torch.device:
        return self._device

    def _device_edges(self, device: torch.device) -> tuple[Tensor, Tensor]:
        if torch.device(device) != self._device:
            raise ValueError(
                f"prepared relation is on {self._device}; call relation.to({device!s}) explicitly"
            )
        return self._rows, self._col_indices

    def _device_causal_mask(self, device: torch.device) -> Tensor:
        self._validate_storage()
        if torch.device(device) != self._device:
            raise ValueError(
                f"prepared relation is on {self._device}; call relation.to({device!s}) explicitly"
            )
        return self._causal_mask

    def validate_for(
        self,
        *,
        source_order: Sequence[Hashable] | None,
        target_order: Sequence[Hashable] | None,
        source_offsets: Sequence[int] | None,
        target_offsets: Sequence[int] | None,
        source_axis: Axis | None,
        target_axis: Axis | None,
        structural_generation: int,
        n_sources: int,
        n_targets: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        self._validate_storage()
        if isinstance(structural_generation, bool) or not isinstance(structural_generation, int):
            raise TypeError("structural_generation must be a nonnegative integer")
        if structural_generation < 0:
            raise ValueError("structural_generation must be nonnegative")
        if n_sources != self.n_sources or n_targets != self.n_targets:
            raise ValueError("prepared relation extents do not match source/target tensors")
        if source_order is None or target_order is None:
            raise ValueError("prepared relations require source_order and target_order at execution")
        checked_source_order = (
            source_order if source_order is self.source_order
            else _order(source_order, "source_order")
        )
        checked_target_order = (
            target_order if target_order is self.target_order
            else _order(target_order, "target_order")
        )
        if checked_source_order != self.source_order:
            raise ValueError("source occurrence order differs from prepared relation")
        if checked_target_order != self.target_order:
            raise ValueError("target occurrence order differs from prepared relation")
        if _offsets(source_offsets, n_sources, "source_offsets") != self.source_offsets:
            raise ValueError("source group offsets differ from prepared relation")
        if _offsets(target_offsets, n_targets, "target_offsets") != self.target_offsets:
            raise ValueError("target group offsets differ from prepared relation")
        if structural_generation != self.structural_generation:
            raise ValueError("structural generation differs from prepared relation")
        if source_axis != self.source_axis:
            raise ValueError("source Axis differs from prepared relation")
        if target_axis != self.target_axis:
            raise ValueError("target Axis differs from prepared relation")
        return self._device_edges(device)

    def _validate_storage(self) -> None:
        versions = tuple(tensor._version for tensor in (
            self._row_ptr, self._col_indices, self._rows, self._causal_mask
        ))
        if versions != self._structure_versions:
            raise ValueError("prepared relation buffers were mutated; prepare a new relation")

    @property
    def has_future_edges(self) -> bool:
        self._validate_storage()
        return self._has_future_edges

    def causal(self) -> "PreparedRelation":
        """Explicitly prepare support containing only group-local causal edges."""
        self._validate_storage()
        columns = []
        offsets = [0]
        for left, right in zip(self._row_ptr_host, self._row_ptr_host[1:]):
            columns.extend(self._col_indices_host[index]
                           for index in range(left, right) if self._causal_host[index])
            offsets.append(len(columns))
        return PreparedRelation(
            offsets, columns, source_order=self.source_order, target_order=self.target_order,
            source_offsets=self.source_offsets, target_offsets=self.target_offsets,
            source_axis=self.source_axis, target_axis=self.target_axis,
            structural_generation=self.structural_generation, device=self._device,
        )

    def to(self, device: torch.device | str) -> "PreparedRelation":
        """Return this relation prepared on ``device`` (without implicit copies)."""
        self._validate_storage()
        target_device = torch.device(device)
        if target_device == self._device:
            return self
        return PreparedRelation(
            self._row_ptr_host, self._col_indices_host,
            source_order=self.source_order, target_order=self.target_order,
            source_offsets=self.source_offsets, target_offsets=self.target_offsets,
            source_axis=self.source_axis, target_axis=self.target_axis,
            structural_generation=self.structural_generation, device=target_device,
        )

    def transpose(self) -> "PreparedRelation":
        """Return the reversed relation, retaining every duplicate edge.

        The derived prepared layout is memoized, so repeated model calls do not
        rebuild or recopy its indices.
        """
        cached = self._transpose_cache
        if cached is not None:
            return cached
        buckets: list[list[int]] = [[] for _ in range(self.n_sources)]
        for target, source in zip(self._rows_host, self._col_indices_host):
            buckets[source].append(target)
        ptr, columns = [0], []
        for bucket in buckets:
            columns.extend(bucket)
            ptr.append(len(columns))
        reversed_relation = PreparedRelation(
            ptr, columns,
            source_order=self.target_order, target_order=self.source_order,
            source_offsets=self.target_offsets, target_offsets=self.source_offsets,
            source_axis=self.target_axis, target_axis=self.source_axis,
            structural_generation=self.structural_generation, device=self._device,
        )
        # This cache does not change the declared relation or its indices; it
        # only retains a derived layout so high-level compositions can reuse it.
        object.__setattr__(self, "_transpose_cache", reversed_relation)
        object.__setattr__(reversed_relation, "_transpose_cache", self)
        return reversed_relation

    def to_manifest(self, *, codec: Any = None) -> dict[str, Any]:
        """Return a JSON-safe semantic declaration; custom IDs need ``codec``."""
        result = {
            "kind": "cellerator.prepared_relation.v1",
            "source_order": [_encode_identity(value, codec) for value in self.source_order],
            "target_order": [_encode_identity(value, codec) for value in self.target_order],
            "source_offsets": list(self.source_offsets),
            "target_offsets": list(self.target_offsets),
            "source_extent": self.n_sources,
            "target_extent": self.n_targets,
            "row_ptr": list(self._row_ptr_host),
            "col_indices": list(self._col_indices_host),
            "source_axis": _axis_manifest(self.source_axis, codec),
            "target_axis": _axis_manifest(self.target_axis, codec),
            "structural_generation": self.structural_generation,
        }
        json.dumps(result, allow_nan=False)
        return result

    def cellerator_biological_descriptors(
        self, *, identity_codec: Any = None
    ) -> Mapping[str, Any]:
        """Explicit hook for model manifests; no implicit ID ``repr`` is used."""
        return {"prepared_relation": self.to_manifest(codec=identity_codec)}


__all__ = ["PreparedRelation"]
