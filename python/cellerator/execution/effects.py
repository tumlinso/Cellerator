"""Explicit local execution requests returned in ``StepResult.effects``.

These are mechanisms, not biological decisions. Models/callers choose whether
to send, suspend, create, remove, or inherit state and context.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class Send:
    target: Any
    value: Any
    time: float | None = None


@dataclass(frozen=True, slots=True)
class Broadcast:
    value: Any
    targets: tuple[Any, ...] | None = None
    time: float | None = None


@dataclass(frozen=True, slots=True)
class Suspend:
    target: Any = None
    reason: Any = None


@dataclass(frozen=True, slots=True)
class Resume:
    target: Any = None


@dataclass(frozen=True, slots=True)
class Remove:
    target: Any = None


@dataclass(frozen=True, slots=True)
class Create:
    model: Any
    state: Any
    id: Any = None
    time: float | None = None


@dataclass(frozen=True, slots=True)
class Fork:
    inherit: Callable[[Any], Any]
    parent: Any = None
    id: Any = None
    model: Any = None
    inherit_context: bool = False


@dataclass(frozen=True, slots=True)
class Eject:
    target: Any = None
    count: int | None = None
    predicate: Callable | None = None
    reason: str = "model"
