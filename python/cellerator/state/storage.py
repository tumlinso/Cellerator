"""Dynamically sized structure-of-arrays storage for independently evolving entities.

Logical IDs and handles remain stable when capacity grows or rows are compacted.
Physical row indices are ephemeral and are available explicitly when kernels or
batching need them. NumPy is the reference backend; this storage is optional and
is never substituted for caller-owned Tensor, sparse, or custom state.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Hashable

import numpy as np

from .core import LayoutMap, StateReference


@dataclass(frozen=True, slots=True)
class StorageSpec:
    """One SoA field's per-entity trailing shape and NumPy storage dtype."""

    shape: tuple[int, ...] = ()
    dtype: Any = np.float32

    def __post_init__(self) -> None:
        shape = tuple(self.shape)
        if any(isinstance(size, bool) or int(size) != size or size < 0 for size in shape):
            raise ValueError("field shape dimensions must be nonnegative integers")
        object.__setattr__(self, "shape", tuple(int(s) for s in shape))
        try:
            dtype = np.dtype(self.dtype)
        except TypeError as exc:
            raise TypeError("field dtype must be a NumPy dtype") from exc
        if dtype.hasobject:
            raise TypeError("packed fields cannot use object dtype; store opaque state by reference")
        object.__setattr__(self, "dtype", dtype)


@dataclass(frozen=True, slots=True)
class StateHandle:
    """Incarnation-safe logical handle. It survives row relocation."""

    identity: Hashable
    incarnation: int


class SoAStateStore:
    """Dynamic named fields stored in independent capacity-sized arrays.

    Insert/remove/compaction affect structure or layout generations. Field
    writes affect only value_generation. Views returned by ``view`` are borrowed:
    mutate them only inside a controlled batch and call ``mark_values_changed``.
    Use ``get`` and ``set`` for checked, generation-tracked access.
    """

    def __init__(self, fields: Mapping[str, StorageSpec | tuple[tuple[int, ...], Any]],
                 capacity: int = 16, *, growth_factor: float = 2.0) -> None:
        if isinstance(capacity, bool) or int(capacity) != capacity or capacity < 0:
            raise ValueError("capacity must be a nonnegative integer")
        if growth_factor <= 1.0:
            raise ValueError("growth_factor must be greater than one")
        if not fields or any(not isinstance(k, str) or not k for k in fields):
            raise ValueError("at least one named field is required")
        self._specs = {
            name: spec if isinstance(spec, StorageSpec) else StorageSpec(*spec)
            for name, spec in fields.items()
        }
        self._arrays = {
            name: np.zeros((capacity, *spec.shape), dtype=spec.dtype)
            for name, spec in self._specs.items()
        }
        self._growth_factor = float(growth_factor)
        self._id_to_row: dict[Hashable, int] = {}
        self._row_to_id: list[Hashable | None] = [None] * capacity
        self._incarnations: dict[Hashable, int] = {}
        self._free: list[int] = list(range(capacity - 1, -1, -1))
        self._structure_generation = 0
        self._value_generation = 0
        self._layout_generation = 0

    @property
    def specs(self) -> Mapping[str, StorageSpec]:
        return self._specs

    @property
    def capacity(self) -> int:
        return len(self._row_to_id)

    @property
    def live_count(self) -> int:
        return len(self._id_to_row)

    @property
    def structure_generation(self) -> int:
        return self._structure_generation

    @property
    def value_generation(self) -> int:
        return self._value_generation

    @property
    def layout_generation(self) -> int:
        return self._layout_generation

    def __len__(self) -> int:
        return self.live_count

    def _resolve(self, handle: StateHandle | Hashable) -> tuple[Hashable, int]:
        if isinstance(handle, StateHandle):
            identity = handle.identity
            current = self._incarnations.get(identity)
            if current != handle.incarnation:
                raise KeyError("state handle is stale or the identity is not live")
        else:
            identity = handle
        try:
            return identity, self._id_to_row[identity]
        except (KeyError, TypeError) as exc:
            raise KeyError(f"unknown or removed state identity: {identity!r}") from exc

    def _reserve_one(self) -> int:
        if not self._free:
            old = self.capacity
            new = max(1, int(np.ceil(max(old, 1) * self._growth_factor)))
            self.reserve(new)
        return self._free.pop()

    def reserve(self, capacity: int) -> None:
        """Grow physical capacity without changing any logical identity or value."""
        if isinstance(capacity, bool) or int(capacity) != capacity or capacity < 0:
            raise ValueError("capacity must be a nonnegative integer")
        capacity = int(capacity)
        if capacity <= self.capacity:
            return
        old = self.capacity
        for name, array in tuple(self._arrays.items()):
            expanded = np.zeros((capacity, *self._specs[name].shape), dtype=array.dtype)
            expanded[:old] = array
            self._arrays[name] = expanded
        self._row_to_id.extend([None] * (capacity - old))
        self._free.extend(range(capacity - 1, old - 1, -1))
        self._layout_generation += 1

    def insert(self, identity: Hashable, values: Mapping[str, Any]) -> StateHandle:
        """Add a logical entity, validating the complete field row first."""
        try:
            hash(identity)
        except TypeError as exc:
            raise TypeError("state identity must be hashable") from exc
        if identity in self._id_to_row:
            raise ValueError(f"duplicate live identity: {identity!r}")
        if set(values) != set(self._specs):
            raise ValueError("insert must provide exactly the declared fields")
        rows: dict[str, np.ndarray] = {}
        for name, spec in self._specs.items():
            row = np.asarray(values[name])
            if row.shape != spec.shape:
                raise ValueError(f"field {name!r} expected shape {spec.shape}, got {row.shape}")
            if not np.can_cast(row.dtype, spec.dtype, casting="same_kind"):
                raise TypeError(f"field {name!r} cannot safely store {row.dtype} as {spec.dtype}")
            rows[name] = row.astype(spec.dtype, copy=False)
        row_index = self._reserve_one()
        for name, row in rows.items():
            self._arrays[name][row_index] = row
        self._id_to_row[identity] = row_index
        self._row_to_id[row_index] = identity
        incarnation = self._incarnations.get(identity, 0) + 1
        self._incarnations[identity] = incarnation
        self._structure_generation += 1
        self._value_generation += 1
        return StateHandle(identity, incarnation)

    def remove(self, handle: StateHandle | Hashable) -> None:
        """Remove an entity and make its physical row reusable."""
        identity, row = self._resolve(handle)
        del self._id_to_row[identity]
        self._row_to_id[row] = None
        self._free.append(row)
        self._structure_generation += 1

    def get(self, handle: StateHandle | Hashable, field: str, *, copy: bool = True) -> np.ndarray:
        """Read one field row. Copy by default so mutation cannot skip generation."""
        _, row = self._resolve(handle)
        result = self._arrays[field][row:row + 1].reshape(self._specs[field].shape)
        return result.copy() if copy else result

    def view(self, handle: StateHandle | Hashable, field: str) -> np.ndarray:
        """Borrow a mutable row view; call mark_values_changed after mutations."""
        return self.get(handle, field, copy=False)

    def set(self, handle: StateHandle | Hashable, field: str, value: Any) -> None:
        """Set a shape- and dtype-checked field row, advancing its value epoch."""
        _, row = self._resolve(handle)
        spec = self._specs[field]
        value = np.asarray(value)
        if value.shape != spec.shape:
            raise ValueError(f"field {field!r} expected shape {spec.shape}, got {value.shape}")
        if not np.can_cast(value.dtype, spec.dtype, casting="same_kind"):
            raise TypeError(f"field {field!r} cannot safely store {value.dtype} as {spec.dtype}")
        self._arrays[field][row] = value
        self._value_generation += 1

    def update(self, handle: StateHandle | Hashable, values: Mapping[str, Any]) -> None:
        """Validate then atomically replace a subset of one entity's fields."""
        _, row = self._resolve(handle)
        if not values or not set(values).issubset(self._specs):
            raise ValueError("update needs one or more declared fields")
        checked: dict[str, np.ndarray] = {}
        for name, value in values.items():
            spec = self._specs[name]
            array = np.asarray(value)
            if array.shape != spec.shape:
                raise ValueError(f"field {name!r} expected shape {spec.shape}, got {array.shape}")
            if not np.can_cast(array.dtype, spec.dtype, casting="same_kind"):
                raise TypeError(f"field {name!r} cannot safely store {array.dtype} as {spec.dtype}")
            checked[name] = array.astype(spec.dtype, copy=False)
        for name, array in checked.items():
            self._arrays[name][row] = array
        self._value_generation += 1

    def mark_values_changed(self) -> None:
        """Account for mutations performed through borrowed views or bulk arrays."""
        self._value_generation += 1

    def row(self, handle: StateHandle | Hashable, *, copy: bool = True) -> dict[str, np.ndarray]:
        _, index = self._resolve(handle)
        return {name: self.get(handle, name, copy=copy) for name in self._arrays}

    def handle(self, identity: Hashable) -> StateHandle:
        self._resolve(identity)
        return StateHandle(identity, self._incarnations[identity])

    def row_index(self, handle: StateHandle | Hashable) -> int:
        """Return an explicitly ephemeral physical row index."""
        return self._resolve(handle)[1]

    def state_reference(self, handle: StateHandle | Hashable, field: str) -> StateReference[np.ndarray]:
        """Bind a cell field directly to this store, retaining logical identity."""
        identity, _ = self._resolve(handle)
        self._specs[field]  # fail eagerly for unknown field
        # Capture the incarnation, not just its reusable logical ID: a removed
        # identity may be inserted again while older cells still hold a ref.
        stable_handle = StateHandle(identity, self._incarnations[identity])
        return StateReference(lambda: self.get(stable_handle, field),
                              lambda value: self.set(stable_handle, field, value),
                              identity=identity)

    def handles(self) -> tuple[StateHandle, ...]:
        """Return live handles in logical insertion order, independent of rows."""
        return tuple(StateHandle(identity, self._incarnations[identity])
                     for identity in self._id_to_row)

    def physical_order(self) -> tuple[Hashable, ...]:
        """Return live logical IDs ordered by current row."""
        return tuple(identity for identity in self._row_to_id if identity is not None)

    def select(self, identities: Iterable[Hashable] | None = None) -> tuple[StateHandle, ...]:
        """Build a work selection; identities are returned in caller order."""
        if identities is None:
            return self.handles()
        return tuple(self.handle(identity) for identity in identities)

    def gather(self, handles: Sequence[StateHandle | Hashable], field: str) -> np.ndarray:
        """Gather selected live rows in requested order for a specialized kernel."""
        rows = [self._resolve(handle)[1] for handle in handles]
        return self._arrays[field][rows].copy()

    def layout(self) -> LayoutMap[Hashable]:
        logical = tuple(self._id_to_row)
        return LayoutMap(logical, self.physical_order())

    def compact(self, order: Sequence[Hashable] | None = None) -> LayoutMap[Hashable]:
        """Pack live rows and optionally choose a new explicit physical ordering.

        Returns the mapping after compaction. Survivor handles keep their
        incarnations, regardless of physical row changes.
        """
        logical = tuple(self._id_to_row)
        physical = tuple(logical if order is None else order)
        try:
            if len(physical) != len(logical) or set(physical) != set(logical):
                raise ValueError("compaction order must cover each live identity exactly once")
        except TypeError as exc:
            raise TypeError("compaction identities must be hashable") from exc
        old_rows = self._id_to_row
        staged = {name: array[[old_rows[identity] for identity in physical]].copy()
                  for name, array in self._arrays.items()}
        count = len(physical)
        for name, array in staged.items():
            self._arrays[name][:count] = array
            if count < self.capacity:
                self._arrays[name][count:] = 0
        self._row_to_id = list(physical) + [None] * (self.capacity - count)
        self._id_to_row = {identity: row for row, identity in enumerate(physical)}
        self._free = list(range(self.capacity - 1, count - 1, -1))
        self._layout_generation += 1
        return self.layout()
