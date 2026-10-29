"""Caller-defined biological or computational events.

An Event records an occurrence. Its meaning and its effect on model state are
defined by the model that consumes it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class Event:
    """One ordered occurrence with optional time, target, payload and provenance."""

    kind: Any
    time: float | None = None
    target: Any = None
    payload: Any = None
    source: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.time is not None:
            value = float(self.time)
            if value != value or value in (float("inf"), float("-inf")):
                raise ValueError("event time must be finite")
            object.__setattr__(self, "time", value)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def provenance(self) -> Any:
        """The event source, retained without assigning causal semantics."""
        return self.source
