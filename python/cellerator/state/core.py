"""Identity preserving, representation neutral cellular state containers.

These containers organize state; they do not assume that a biological value is
a dense tensor. Values may be tensors, arrays, sparse buffers, domain objects,
or handles owned by another runtime.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
import operator
from types import MappingProxyType
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")
K = TypeVar("K")


@dataclass(frozen=True)
class StateSnapshot(Mapping[str, Any]):
    """Shallow immutable snapshot with the generations it observed.

    Values are deliberately not copied: copying a device tensor or external
    state handle has backend-specific semantics. Use ``copy_values`` when an
    independent value snapshot is required and supported by the value type.
    """

    fields: Mapping[str, Any]
    structure_generation: int
    value_generation: int

    def __getitem__(self, key: str) -> Any:
        return self.fields[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    def copy_values(self, copier: Callable[[Any], Any]) -> "StateSnapshot":
        return StateSnapshot(MappingProxyType({k: copier(v) for k, v in self.fields.items()}),
                             self.structure_generation, self.value_generation)


class StructuredState(MutableMapping[str, Any]):
    """Named state fields with distinct structure and value generations."""

    def __init__(self, fields: Mapping[str, Any] | None = None, *,
                 metadata: Mapping[str, Any] | None = None) -> None:
        self._fields = dict(fields or {})
        if any(not isinstance(key, str) or not key for key in self._fields):
            raise ValueError("state field names must be non-empty strings")
        self.metadata = dict(metadata or {})
        self._structure_generation = 0
        self._value_generation = 0

    @property
    def structure_generation(self) -> int:
        return self._structure_generation

    @property
    def value_generation(self) -> int:
        return self._value_generation

    def get_field(self, name: str, default: Any = ...) -> Any:
        if default is ...:
            return self._fields[name]
        return self._fields.get(name, default)

    def set_field(self, name: str, value: Any) -> None:
        self[name] = value

    def __getitem__(self, key: str) -> Any:
        return self._fields[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if not isinstance(key, str) or not key:
            raise ValueError("state field names must be non-empty strings")
        if key not in self._fields:
            self._fields[key] = value
            self._structure_generation += 1
        else:
            self._fields[key] = value
            self._value_generation += 1

    def __delitem__(self, key: str) -> None:
        del self._fields[key]
        self._structure_generation += 1

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def snapshot(self) -> StateSnapshot:
        return StateSnapshot(MappingProxyType(dict(self._fields)),
                             self._structure_generation, self._value_generation)


class SparseState(MutableMapping[K, T], Generic[K, T]):
    """Values keyed by stable logical identity, independent of storage order."""

    def __init__(self, values: Mapping[K, T] | None = None) -> None:
        self._values = dict(values or {})
        self._structure_generation = 0
        self._value_generation = 0

    @property
    def structure_generation(self) -> int:
        return self._structure_generation

    @property
    def value_generation(self) -> int:
        return self._value_generation

    def __getitem__(self, identity: K) -> T:
        return self._values[identity]

    def __setitem__(self, identity: K, value: T) -> None:
        if identity in self._values:
            self._values[identity] = value
            self._value_generation += 1
        else:
            self._values[identity] = value
            self._structure_generation += 1

    def __delitem__(self, identity: K) -> None:
        del self._values[identity]
        self._structure_generation += 1

    def __iter__(self) -> Iterator[K]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def update_values(self, updates: Mapping[K, T]) -> None:
        """Apply identity keyed values and increment each affected generation once."""
        added = any(key not in self._values for key in updates)
        changed = any(key in self._values for key in updates)
        self._values.update(updates)
        self._structure_generation += int(added)
        self._value_generation += int(changed)

    def snapshot(self) -> Mapping[K, T]:
        return MappingProxyType(dict(self._values))


class StateReference(Generic[T]):
    """A lightweight view onto state managed by an external owner."""

    def __init__(self, getter: Callable[[], T], setter: Callable[[T], None] | None = None,
                 *, identity: Any = None) -> None:
        if not callable(getter) or (setter is not None and not callable(setter)):
            raise TypeError("getter and setter must be callable")
        self._getter = getter
        self._setter = setter
        self.identity = identity

    def get(self) -> T:
        return self._getter()

    def set(self, value: T) -> None:
        if self._setter is None:
            raise TypeError("this state reference is read-only")
        self._setter(value)


class LayoutMap(Generic[K]):
    """Bijection between logical identities and a chosen physical ordering."""

    def __init__(self, logical_ids: Sequence[K], physical_ids: Sequence[K] | None = None) -> None:
        logical = tuple(logical_ids)
        physical = tuple(logical if physical_ids is None else physical_ids)
        try:
            logical_set, physical_set = set(logical), set(physical)
        except TypeError as exc:
            raise TypeError("layout identities must be hashable") from exc
        if len(logical_set) != len(logical) or len(physical_set) != len(physical):
            raise ValueError("layout identities must be unique")
        if logical_set != physical_set:
            raise ValueError("physical order must be a permutation of logical identities")
        self.logical_ids = logical
        self.physical_ids = physical
        self._logical_position = {identity: i for i, identity in enumerate(logical)}
        self._physical_position = {identity: i for i, identity in enumerate(physical)}

    def logical_position(self, identity: K) -> int:
        return self._logical_position[identity]

    def physical_position(self, identity: K) -> int:
        return self._physical_position[identity]

    def to_physical(self, values: Mapping[K, T]) -> tuple[T, ...]:
        """Order values by physical identity; missing or extra identities fail."""
        if set(values) != set(self.logical_ids):
            raise ValueError("values must cover exactly the layout identities")
        return tuple(values[identity] for identity in self.physical_ids)

    def to_logical(self, values: Sequence[T]) -> dict[K, T]:
        if len(values) != len(self.physical_ids):
            raise ValueError("physical value count must match layout size")
        return dict(zip(self.physical_ids, values))

    def remap(self, physical_ids: Sequence[K]) -> "LayoutMap[K]":
        """Return a new mapping with the same domain and another physical order."""
        return LayoutMap(self.logical_ids, physical_ids)


class RaggedState(Generic[K, T]):
    """Segmented variable length payloads keyed by stable entity identity.

    ``values`` is retained as supplied (for example a tensor or flat array).
    Offsets count its leading entries; this class does not pad or densify.
    """

    def __init__(self, identities: Sequence[K], offsets: Sequence[int], values: Any) -> None:
        self.identities = tuple(identities)
        if len(set(self.identities)) != len(self.identities):
            raise ValueError("ragged entity identities must be unique")
        normalized = []
        for value in offsets:
            if isinstance(value, bool):
                raise TypeError("ragged offsets must be integers")
            try:
                normalized.append(operator.index(value))
            except TypeError as exc:
                raise TypeError("ragged offsets must be integers") from exc
        self.offsets = tuple(normalized)
        if len(self.offsets) != len(self.identities) + 1 or not self.offsets or self.offsets[0] != 0:
            raise ValueError("offsets need one zero-based boundary per entity plus a final boundary")
        if any(a > b for a, b in zip(self.offsets, self.offsets[1:])):
            raise ValueError("offsets must be nondecreasing")
        if len(values) != self.offsets[-1]:
            raise ValueError("final offset must equal the leading extent of values")
        self.values = values
        self._positions = {identity: i for i, identity in enumerate(self.identities)}

    def segment(self, identity: K) -> Any:
        position = self._positions[identity]
        return self.values[self.offsets[position]:self.offsets[position + 1]]

    def lengths(self) -> dict[K, int]:
        return {identity: self.offsets[i + 1] - self.offsets[i]
                for i, identity in enumerate(self.identities)}
