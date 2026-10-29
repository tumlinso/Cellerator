"""Append-only event histories with explicit bounded-retention operations."""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from threading import RLock
from typing import Any

from .event import Event


class EventHistory(Sequence[Event]):
    """Thread-safe ordered event log; repeated events remain distinct occurrences."""

    def __init__(self, events: Iterable[Event] = ()) -> None:
        self._events = list(events)
        if not all(isinstance(event, Event) for event in self._events):
            raise TypeError("history entries must be Event objects")
        self._lock = RLock()
        self._base_index = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    def __getitem__(self, index):
        with self._lock:
            return self._events[index]

    def __iter__(self) -> Iterator[Event]:
        with self._lock:
            return iter(tuple(self._events))

    @property
    def base_index(self) -> int:
        """Global occurrence index represented by local offset zero."""
        return self._base_index

    @property
    def next_index(self) -> int:
        return self._base_index + len(self)

    def append(self, event: Event) -> int:
        if not isinstance(event, Event):
            raise TypeError("event must be an Event")
        with self._lock:
            index = self._base_index + len(self._events)
            self._events.append(event)
            return index

    def extend(self, events: Iterable[Event]) -> tuple[int, int]:
        values = tuple(events)
        if not all(isinstance(event, Event) for event in values):
            raise TypeError("history entries must be Event objects")
        with self._lock:
            start = self._base_index + len(self._events)
            self._events.extend(values)
            return start, start + len(values)

    def since(self, occurrence_index: int) -> tuple[Event, ...]:
        """Read events at/after a global cursor; reject cursors already ejected."""
        with self._lock:
            if occurrence_index < self._base_index:
                raise ValueError("cursor refers to events already ejected")
            if occurrence_index > self._base_index + len(self._events):
                raise ValueError("cursor is beyond the history")
            return tuple(self._events[occurrence_index - self._base_index :])

    def through(self, time: float) -> tuple[Event, ...]:
        """Return stream-ordered events whose explicit time is at most ``time``."""
        return tuple(event for event in self if event.time is not None and event.time <= time)

    def select(self, *, kind: Any = ..., target: Any = ..., source: Any = ...) -> tuple[Event, ...]:
        """Select by explicit event fields; ``...`` leaves a field unrestricted."""
        return tuple(
            event for event in self
            if (kind is ... or event.kind == kind)
            and (target is ... or event.target == target)
            and (source is ... or event.source == source)
        )

    def eject_before(self, *, time: float | None = None, keep_last: int | None = None) -> tuple[Event, ...]:
        """Explicitly discard a chronological prefix and return the ejected events.

        Supplying a time retains events with missing timestamps. Supplying
        ``keep_last`` retains the last N occurrences. When both are supplied,
        the stricter prefix is ejected. This changes accessible history, not any
        model state previously computed from it.
        """
        if time is None and keep_last is None:
            raise ValueError("provide time and/or keep_last")
        if keep_last is not None and keep_last < 0:
            raise ValueError("keep_last must be nonnegative")
        with self._lock:
            n = len(self._events)
            cutoff = 0
            if keep_last is not None:
                cutoff = max(cutoff, n - keep_last)
            if time is not None:
                cutoff = max(cutoff, sum(1 for event in self._events if event.time is not None and event.time < time))
                # Only discard a prefix: later stream events are retained even
                # when timestamps are unordered or omitted.
                cutoff = 0
                for event in self._events:
                    if event.time is None or event.time >= time:
                        break
                    cutoff += 1
                if keep_last is not None:
                    cutoff = max(cutoff, n - keep_last)
            removed = tuple(self._events[:cutoff])
            del self._events[:cutoff]
            self._base_index += cutoff
            return removed

    def copy(self) -> "EventHistory":
        with self._lock:
            result = EventHistory(self._events)
            result._base_index = self._base_index
            return result

    @classmethod
    def from_token_batch(cls, batch: Any, *, decode_identity=None) -> "EventHistory":
        """Decode a tokenization ``TokenBatch`` lazily, preserving row order/repeats."""
        ids = batch.token_ids.detach().cpu().tolist()
        offsets = batch.offsets.detach().cpu().tolist()
        attrs = {name: value.detach().cpu().tolist() for name, value in batch.attributes.items()}
        events = []
        for row in range(len(offsets) - 1):
            for index in range(offsets[row], offsets[row + 1]):
                identity = decode_identity(ids[index]) if decode_identity else ids[index]
                metadata = {name: values[index] for name, values in attrs.items()}
                events.append(Event(identity, metadata=metadata))
        return cls(events)

    def to_token_batch(self, vocabulary: Any, *, device: Any = None) -> Any:
        """Encode one history with the optional ATen tokenizer component."""
        try:
            from cellerator.tokenization import event_stream
        except ImportError as exc:
            raise RuntimeError("enable Cellerator's optional tokenization component") from exc
        identities = [event.kind for event in self]
        return event_stream(identities, vocabulary, device=device)
