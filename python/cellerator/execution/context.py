"""Bounded model-visible context, separate from pending inputs and history."""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import operator
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class RoutedInput:
    """One occurrence; sequence distinguishes repeated values at the same time."""

    value: Any
    time: float
    sequence: int
    source: Any = None


@dataclass(frozen=True, slots=True)
class Ejection:
    cell_id: Any
    records: tuple[RoutedInput, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class ContextInputs(Sequence):
    """Immutable context membership, with model-native values left untouched.

    Iteration/indexing exposes values. ``records`` preserves delivery time and
    provenance; ``incoming`` identifies newly delivered, retained occurrences.
    Occurrences ejected during admission are reported separately. ``external``
    is a caller's per-step input and is not retained in the active context.
    """

    records: tuple[RoutedInput, ...] = ()
    incoming: tuple[RoutedInput, ...] = ()
    external: Any = None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(record.value for record in self.records[index])
        return self.records[index].value

    def __iter__(self) -> Iterator[Any]:
        return (record.value for record in self.records)


class ContextCapacityError(ValueError):
    """Admission would exceed a context configured to reject overflow."""


class ActiveContext(Sequence):
    """A bounded occurrence window with explicit FIFO or rejecting admission.

    Capacity counts occurrences, not bytes or tensor elements. State and history
    are not owned by this window, and retained values are never copied/coerced.
    """

    def __init__(self, capacity: int = 128, *, overflow: str = "oldest"):
        if isinstance(capacity, bool):
            raise TypeError("capacity must be a nonnegative integer")
        self.capacity = operator.index(capacity)
        if self.capacity < 0:
            raise ValueError("capacity must be nonnegative")
        if overflow not in ("oldest", "reject"):
            raise ValueError("overflow must be 'oldest' or 'reject'")
        self.overflow = overflow
        self._records: list[RoutedInput] = []

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index):
        return tuple(self._records)[index]

    def __iter__(self) -> Iterator[RoutedInput]:
        return iter(tuple(self._records))

    def admit(self, records) -> tuple[RoutedInput, ...]:
        """Admit records atomically with respect to capacity; return ejections."""
        values = tuple(records)
        if not all(isinstance(value, RoutedInput) for value in values):
            raise TypeError("context accepts RoutedInput records")
        excess = max(0, len(self._records) + len(values) - self.capacity)
        if excess and self.overflow == "reject":
            raise ContextCapacityError("active context capacity exceeded; eject records first")
        combined = self._records + list(values)
        removed = tuple(combined[:excess])
        self._records = combined[excess:]
        return removed

    def eject(self, *, count: int | None = None,
              predicate: Callable[[RoutedInput], bool] | None = None) -> tuple[RoutedInput, ...]:
        """Remove a prefix or predicate-selected records; no arguments clears it."""
        if count is not None and predicate is not None:
            raise ValueError("choose count or predicate")
        if predicate is not None:
            # Evaluate every predicate before mutation, including caller errors.
            selected = tuple(bool(predicate(record)) for record in self._records)
            removed = tuple(record for record, take in zip(self._records, selected) if take)
            self._records = [record for record, take in zip(self._records, selected) if not take]
            return removed
        if count is None:
            count = len(self._records)
        if isinstance(count, bool):
            raise TypeError("count must be a nonnegative integer")
        count = operator.index(count)
        if count < 0:
            raise ValueError("count must be nonnegative")
        removed = tuple(self._records[:count])
        del self._records[:count]
        return removed

    def snapshot(self, *, incoming=(), external=None) -> ContextInputs:
        return ContextInputs(tuple(self._records), tuple(incoming), external)
