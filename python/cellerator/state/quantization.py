"""Explicit quantization for compact model-native cellular state.

This is a NumPy reference encoding for biological state values. It is opt-in,
keeps feature identity with the packed payload, and makes no assumptions about
what a value means. It is separate from quantization of learned model weights.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np


def _unique_ids(feature_ids: Sequence[Any] | None, size: int) -> tuple[Any, ...]:
    ids = tuple(range(size)) if feature_ids is None else tuple(feature_ids)
    if len(ids) != size:
        raise ValueError("feature identity count must match the final values axis")
    try:
        if len(set(ids)) != size:
            raise ValueError("feature identities must be unique")
    except TypeError as exc:
        raise TypeError("feature identities must be hashable") from exc
    return ids


def _per_feature(value: Any, ids: tuple[Any, ...], *, name: str,
                 pair: bool = False) -> np.ndarray:
    """Convert identity keyed or position-aligned metadata to a float array."""
    if isinstance(value, Mapping):
        if set(value) != set(ids):
            raise ValueError(f"{name} must cover exactly the feature identities")
        raw = [value[i] for i in ids]
    else:
        raw = value
    try:
        result = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    expected = (len(ids), 2) if pair else (len(ids),)
    if result.shape != expected:
        raise ValueError(f"{name} must have shape {expected}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _pack(codes: np.ndarray, bits: int) -> np.ndarray:
    """Pack flat integer codes little-endian within each byte."""
    flat = np.asarray(codes, dtype=np.uint8).reshape(-1)
    packed = np.zeros((flat.size * bits + 7) // 8, dtype=np.uint8)
    if not flat.size:
        return packed
    bit_positions = np.arange(flat.size, dtype=np.uint64) * bits
    byte_positions = (bit_positions >> 3).astype(np.intp)
    shifts = (bit_positions & 7).astype(np.uint8)
    words = flat.astype(np.uint16) << shifts.astype(np.uint16)
    np.bitwise_or.at(packed, byte_positions, (words & 255).astype(np.uint8))
    crosses = shifts + bits > 8
    if np.any(crosses):
        np.bitwise_or.at(packed, byte_positions[crosses] + 1,
                         (words[crosses] >> 8).astype(np.uint8))
    packed.flags.writeable = False
    return packed


def _unpack(packed: np.ndarray, count: int, bits: int) -> np.ndarray:
    if bits == 8:
        return np.asarray(packed, dtype=np.uint8).copy()
    if count == 0:
        return np.empty(0, dtype=np.uint8)
    positions = np.arange(count, dtype=np.uint64) * bits
    byte_positions = (positions >> 3).astype(np.intp)
    shifts = (positions & 7).astype(np.uint8)
    padded = np.concatenate((packed, np.zeros(1, dtype=np.uint8)))
    words = padded[byte_positions].astype(np.uint16) | (
        padded[byte_positions + 1].astype(np.uint16) << 8)
    return ((words >> shifts.astype(np.uint16)) & ((1 << bits) - 1)).astype(np.uint8)


@dataclass(frozen=True)
class QuantizationDiagnostics:
    """Reconstruction error measured on the values passed to the encoder."""

    elements: int
    packed_bytes: int
    uncompressed_bytes: int
    mean_absolute_error: float
    root_mean_square_error: float
    maximum_absolute_error: float
    clipped_values: int = 0
    updates_since_measurement: int = 0
    metadata_array_bytes: int = 0
    override_count: int = 0
    override_value_bytes: int = 0
    shared_profile_bytes: int = 0

    @property
    def error_is_current(self) -> bool:
        return self.updates_since_measurement == 0

    @property
    def compression_ratio(self) -> float:
        """Ratio including per-state numeric arrays, excluding shared profile arrays."""
        return (self.uncompressed_bytes / self.per_state_array_bytes
                if self.per_state_array_bytes else float("inf"))

    @property
    def per_state_array_bytes(self) -> int:
        """Packed codes, per-state decode arrays, and exact override values."""
        return self.packed_bytes + self.metadata_array_bytes + self.override_value_bytes

    def population_array_bytes(self, state_count: int) -> int:
        """Array bytes for a population plus its once-shared profile arrays."""
        if state_count < 0:
            raise ValueError("state_count must be nonnegative")
        return self.shared_profile_bytes + state_count * self.per_state_array_bytes


@dataclass(frozen=True)
class QuantizedState:
    """Packed state values with per-feature decode metadata and identity order."""

    packed_codes: np.ndarray
    shape: tuple[int, ...]
    bits: int
    feature_ids: tuple[Any, ...]
    mode: str
    source_dtype: str
    value_min: np.ndarray | None = None
    value_max: np.ndarray | None = None
    thresholds: np.ndarray | None = None
    binary_levels: np.ndarray | None = None
    codebooks: tuple[np.ndarray, ...] | None = None
    diagnostics: QuantizationDiagnostics | None = None
    profile_id: str | None = None
    exact_overrides: Mapping[int, Any] | None = None

    @property
    def element_count(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def feature_count(self) -> int:
        return self.shape[-1]

    @property
    def leading_shape(self) -> tuple[int, ...]:
        return self.shape[:-1]

    def _codes(self) -> np.ndarray:
        return _unpack(self.packed_codes, self.element_count, self.bits).reshape(self.shape)

    def dequantize(self, *, feature_ids: Sequence[Any] | None = None,
                   dtype: Any | None = None) -> np.ndarray:
        """Decode values; optional identities request another output ordering."""
        codes = self._codes()
        if self.mode == "binary":
            assert self.thresholds is not None and self.binary_levels is not None
            decoded = np.where(codes != 0, self.binary_levels[:, 1], self.binary_levels[:, 0])
        elif self.mode == "uniform":
            assert self.value_min is not None and self.value_max is not None
            scale = (self.value_max - self.value_min) / ((1 << self.bits) - 1)
            decoded = self.value_min + codes.astype(np.float64) * scale
        else:
            assert self.codebooks is not None
            decoded = np.empty(self.shape, dtype=np.float64)
            for feature, book in enumerate(self.codebooks):
                decoded[..., feature] = book[codes[..., feature]]
        if self.exact_overrides:
            flat = decoded.reshape(-1)
            for index, value in self.exact_overrides.items():
                flat[index] = value
        requested = self.feature_ids if feature_ids is None else tuple(feature_ids)
        if len(requested) != self.feature_count or set(requested) != set(self.feature_ids):
            raise ValueError("requested feature identities must be a permutation of stored identities")
        if requested != self.feature_ids:
            positions = [self.feature_ids.index(identity) for identity in requested]
            decoded = decoded[..., positions]
        default_dtype = np.result_type(np.dtype(self.source_dtype), np.float32)
        return decoded.astype(default_dtype if dtype is None else dtype, copy=False)

    def code(self, feature_id: Any, value: float) -> int:
        """Encode one value using the stored configuration for its feature."""
        try:
            feature = self.feature_ids.index(feature_id)
        except ValueError as exc:
            raise KeyError(feature_id) from exc
        if not np.isfinite(value):
            raise ValueError("state value must be finite")
        if self.mode == "binary":
            assert self.thresholds is not None
            return int(value >= self.thresholds[feature])
        if self.mode == "uniform":
            assert self.value_min is not None and self.value_max is not None
            low, high = self.value_min[feature], self.value_max[feature]
            if high == low:
                return 0
            normalized = np.clip((float(value) - low) / (high - low), 0.0, 1.0)
            return int(np.rint(normalized * ((1 << self.bits) - 1)))
        assert self.codebooks is not None
        book = self.codebooks[feature]
        return int(np.argmin(np.abs(book - float(value))))

    def update_at(self, feature_id: Any, value: float,
                  leading_index: Sequence[int] = (), *, exact: bool | None = None) -> "QuantizedState":
        """Return an updated state, changing only the addressed packed code.

        ``leading_index`` selects an entity/time/group coordinate before the
        final feature axis. Diagnostics from the original encoding are marked
        stale because untouched source values are intentionally not retained.
        """
        if len(leading_index) != len(self.leading_shape):
            raise ValueError("leading_index must address every non-feature axis")
        try:
            flat_index = int(np.ravel_multi_index(tuple(leading_index) +
                                                  (self.feature_ids.index(feature_id),),
                                                  self.shape))
        except ValueError as exc:
            if feature_id not in self.feature_ids:
                raise KeyError(feature_id) from exc
            raise ValueError("leading_index is outside the state shape") from exc
        code = self.code(feature_id, float(value))
        data = self.packed_codes.copy()
        bit = flat_index * self.bits
        byte = bit >> 3
        shift = bit & 7
        word = int(data[byte])
        if byte + 1 < data.size:
            word |= int(data[byte + 1]) << 8
        mask = ((1 << self.bits) - 1) << shift
        word = (word & ~mask) | (code << shift)
        data[byte] = word & 255
        if byte + 1 < data.size and shift + self.bits > 8:
            data[byte + 1] = (word >> 8) & 255
        data.flags.writeable = False
        diagnostics = self.diagnostics
        if diagnostics is not None:
            diagnostics = replace(diagnostics,
                                  updates_since_measurement=diagnostics.updates_since_measurement + 1)
        overrides = dict(self.exact_overrides or {})
        keep_exact = flat_index in overrides if exact is None else exact
        if keep_exact:
            overrides[flat_index] = value
        else:
            overrides.pop(flat_index, None)
        return replace(self, packed_codes=data, diagnostics=diagnostics,
                       exact_overrides=MappingProxyType(overrides) if overrides else None)

    def update_many(self, updates: Iterable[tuple[Sequence[int], Any, float]], *,
                    exact: bool | None = None) -> "QuantizedState":
        """Apply ``(leading_index, feature_id, value)`` updates incrementally."""
        result = self
        for leading_index, feature_id, value in updates:
            result = result.update_at(feature_id, value, leading_index, exact=exact)
        return result


def quantize_state(values: Any, *, bits: int = 4,
                   feature_ids: Sequence[Any] | None = None,
                   thresholds: Mapping[Any, float] | Sequence[float] | None = None,
                   binary_levels: Sequence[float] | Mapping[Any, Sequence[float]] = (0.0, 1.0),
                   value_range: tuple[Any, Any] | Mapping[Any, Sequence[float]] | None = None,
                   codebooks: Mapping[Any, Sequence[float]] | Sequence[Sequence[float]] | None = None,
                   exact_mask: Any | None = None, override_dtype: Any | None = None,
                   ) -> QuantizedState:
    """Quantize numeric state along its final (feature) axis.

    Supply ``thresholds`` for binary state, ``codebooks`` for caller-defined
    levels, or ``value_range`` for uniform quantization. If uniform ranges are
    omitted, observed per-feature minima and maxima are stored. Feature
    metadata may be keyed by identity so a caller can safely reorder inputs.
    """
    if isinstance(bits, bool) or bits not in (1, 2, 4, 8):
        raise ValueError("bits must be one of 1, 2, 4, or 8")
    array = np.asarray(values)
    if array.ndim < 1 or array.shape[-1] < 1:
        raise ValueError("values must have a non-empty final feature axis")
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError("values must contain finite real numeric data")
    array64 = array.astype(np.float64, copy=False)
    ids = _unique_ids(feature_ids, array.shape[-1])
    if exact_mask is None:
        overrides = None
        override_bytes = 0
    else:
        mask = np.asarray(exact_mask, dtype=np.bool_)
        if mask.shape != array.shape:
            raise ValueError("exact_mask must have the same shape as values")
        preserved = array.astype(array.dtype if override_dtype is None else override_dtype, copy=False)
        if preserved.dtype.kind not in "fiu" or not np.isfinite(preserved[mask]).all():
            raise ValueError("override_dtype must be a finite real numeric dtype")
        flat_values = preserved.reshape(-1)
        overrides = MappingProxyType({int(index): flat_values[index].item()
                                      for index in np.flatnonzero(mask.reshape(-1))})
        override_bytes = len(overrides) * preserved.dtype.itemsize

    if thresholds is not None:
        if bits != 1 or codebooks is not None or value_range is not None:
            raise ValueError("threshold mode uses 1 bit and cannot be combined with other modes")
        threshold_values = _per_feature(thresholds, ids, name="thresholds")
        if isinstance(binary_levels, Mapping):
            if set(binary_levels) != set(ids):
                raise ValueError("binary_levels must cover exactly the feature identities")
            level_values = np.asarray([binary_levels[i] for i in ids], dtype=np.float64)
        else:
            levels = np.asarray(binary_levels, dtype=np.float64)
            if levels.shape == (2,):
                level_values = np.broadcast_to(levels, (len(ids), 2)).copy()
            else:
                level_values = levels
        if level_values.shape != (len(ids), 2) or not np.isfinite(level_values).all():
            raise ValueError("binary_levels must provide finite low/high levels per feature")
        codes = (array64 >= threshold_values).astype(np.uint8)
        decoded = np.where(codes != 0, level_values[:, 1], level_values[:, 0])
        mode, mins, maxs, books = "binary", None, None, None
        threshold_values.flags.writeable = level_values.flags.writeable = False
    elif codebooks is not None:
        if thresholds is not None or value_range is not None:
            raise ValueError("codebooks cannot be combined with thresholds or value_range")
        if isinstance(codebooks, Mapping):
            if set(codebooks) != set(ids):
                raise ValueError("codebooks must cover exactly the feature identities")
            raw_books = [codebooks[i] for i in ids]
        elif isinstance(codebooks, np.ndarray):
            if codebooks.ndim != 2 or codebooks.shape[0] != len(ids):
                raise ValueError("codebook array must have shape [features, levels]")
            raw_books = tuple(codebooks[i] for i in range(len(ids)))
        else:
            raw_books = codebooks
        if len(raw_books) != len(ids):
            raise ValueError("one codebook is required per feature")
        book_list = []
        for book in raw_books:
            levels = np.asarray(book, dtype=np.float64)
            if (levels.ndim != 1 or levels.size < 1 or levels.size > (1 << bits)
                    or not np.isfinite(levels).all() or np.any(np.diff(levels) <= 0)):
                raise ValueError("each codebook must be strictly increasing, finite, and fit the bit width")
            levels = levels.copy()
            levels.flags.writeable = False
            book_list.append(levels)
        books = tuple(book_list)
        threshold_values = None
        level_values = None
        codes = np.empty(array.shape, dtype=np.uint8)
        decoded = np.empty(array.shape, dtype=np.float64)
        for feature, book in enumerate(books):
            # Nearest-level selection with the lower level winning exact ties.
            upper = np.searchsorted(book, array64[..., feature], side="left")
            upper = np.clip(upper, 0, len(book) - 1)
            lower = np.maximum(upper - 1, 0)
            choose_upper = np.abs(book[upper] - array64[..., feature]) < np.abs(
                book[lower] - array64[..., feature])
            selected = np.where(choose_upper, upper, lower)
            codes[..., feature] = selected.astype(np.uint8)
            decoded[..., feature] = book[selected]
        mode, mins, maxs = "codebook", None, None
    else:
        if thresholds is not None:
            raise ValueError("thresholds must be used in binary mode")
        maximum_code = (1 << bits) - 1
        if value_range is None:
            mins = np.min(array64, axis=tuple(range(array.ndim - 1))) if array.ndim > 1 else array64.copy()
            maxs = np.max(array64, axis=tuple(range(array.ndim - 1))) if array.ndim > 1 else array64.copy()
        else:
            if isinstance(value_range, Mapping):
                if set(value_range) != set(ids):
                    raise ValueError("value_range must cover exactly the feature identities")
                mins = _per_feature({i: value_range[i][0] for i in ids}, ids, name="value_range minima")
                maxs = _per_feature({i: value_range[i][1] for i in ids}, ids, name="value_range maxima")
            elif isinstance(value_range, np.ndarray):
                if value_range.shape != (2, len(ids)):
                    raise ValueError("value_range array must have shape [2, features]")
                mins = _per_feature(value_range[0], ids, name="value_range minima")
                maxs = _per_feature(value_range[1], ids, name="value_range maxima")
            elif isinstance(value_range, Sequence) and len(value_range) == 2:
                mins = _per_feature(value_range[0], ids, name="value_range minima")
                maxs = _per_feature(value_range[1], ids, name="value_range maxima")
            else:
                raise ValueError("value_range must be (per-feature minima, per-feature maxima)")
        if np.any(maxs < mins):
            raise ValueError("value_range maxima must be greater than or equal to minima")
        span = maxs - mins
        normalized = np.divide(array64 - mins, span,
                               out=np.zeros_like(array64), where=span != 0)
        clipped = np.count_nonzero((normalized < 0) | (normalized > 1))
        codes = np.rint(np.clip(normalized, 0, 1) * maximum_code).astype(np.uint8)
        scale = span / maximum_code
        decoded = mins + codes.astype(np.float64) * scale
        decoded[..., span == 0] = mins[span == 0]
        mins = mins.copy(); maxs = maxs.copy()
        mins.flags.writeable = maxs.flags.writeable = False
        mode, books, threshold_values, level_values = "uniform", None, None, None

    packed = _pack(codes, bits)
    error = np.abs(decoded - array64)
    if overrides:
        for flat_index, value in overrides.items():
            error.reshape(-1)[flat_index] = abs(float(value) - array64.reshape(-1)[flat_index])
    mean_error = float(error.mean()) if error.size else 0.0
    mean_squared_error = float(np.mean(error * error)) if error.size else 0.0
    metadata_bytes = sum(value.nbytes for value in (mins, maxs, threshold_values, level_values)
                         if isinstance(value, np.ndarray))
    if books is not None:
        metadata_bytes += sum(book.nbytes for book in books)
    diagnostics = QuantizationDiagnostics(
        elements=int(array.size), packed_bytes=int(packed.nbytes),
        uncompressed_bytes=int(array.nbytes),
        mean_absolute_error=mean_error,
        root_mean_square_error=float(np.sqrt(mean_squared_error)),
        maximum_absolute_error=float(error.max(initial=0.0)),
        clipped_values=int(clipped) if mode == "uniform" else 0,
        metadata_array_bytes=metadata_bytes,
        override_count=len(overrides or {}),
        override_value_bytes=override_bytes,
    )
    return QuantizedState(packed, tuple(array.shape), bits, ids, mode,
                          array.dtype.str, mins, maxs, threshold_values,
                          level_values, books, diagnostics,
                          exact_overrides=overrides)




__all__ = ["QuantizationDiagnostics", "QuantizedState", "quantize_state"]
