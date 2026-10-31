"""Experimental rules for copying and dividing caller-defined cellular state.

Inheritance is an explicit model choice. This module supplies mechanics for
copying, partitioning, resetting or replacing fields, but knows nothing about
genes, proteins, organelles or any preferred inheritance law.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from ..modeling import DynamicalCell

_UNSET = object()


def _clone(value: Any) -> Any:
    """Make independent daughter storage while retaining the value's layout."""
    if isinstance(value, (Mapping, list, tuple, set, bytearray)):
        return deepcopy(value)
    clone = getattr(value, "clone", None)
    if callable(clone):
        return clone()
    copy = getattr(value, "copy", None)
    if callable(copy):
        return copy()
    return deepcopy(value)


def _probability(value: Any, name: str = "fraction") -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


@dataclass(frozen=True, slots=True)
class DivisionContext:
    """Caller-supplied identity, time, randomness and metadata for division."""

    parent_id: Any = None
    daughter_ids: tuple[Any, Any] = (None, None)
    time: float | None = None
    rng: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.daughter_ids) != 2:
            raise ValueError("division requires exactly two daughter identities")
        if self.time is not None and not math.isfinite(float(self.time)):
            raise ValueError("division time must be finite")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class InheritanceRule(Protocol):
    """A rule returning independent state values for two daughters."""

    def divide(self, state: Any, context: DivisionContext) -> tuple[Any, Any]: ...


class CopyInheritance:
    """Copy the complete value state into independent daughter storage."""

    __slots__ = ()

    def divide(self, state: Any, context: DivisionContext) -> tuple[Any, Any]:
        return _clone(state), _clone(state)

    def __call__(self, state: Any, context: DivisionContext | None = None) -> tuple[Any, Any]:
        return self.divide(state, context or DivisionContext())


@dataclass(frozen=True, slots=True)
class FractionPartition:
    """Deterministically split a continuous scalar by an asymmetric fraction."""

    fraction: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "fraction", _probability(self.fraction))

    def divide(self, value: Any, context: DivisionContext) -> tuple[Any, Any]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.generic)):
            dtype = getattr(value, "dtype", None)
            if dtype is None or getattr(dtype, "kind", None) != "f":
                is_float = getattr(value, "is_floating_point", None)
                if not callable(is_float) or not is_float():
                    raise TypeError("FractionPartition expects a continuous numeric field")
        if isinstance(value, (int, float, np.generic)) and not math.isfinite(float(value)):
            raise ValueError("partitioned value must be finite")
        if isinstance(value, int) or (isinstance(value, np.generic) and value.dtype.kind in "iu"):
            raise TypeError("integer counts require BinomialPartition")
        if isinstance(value, np.ndarray) and value.dtype.kind in "iu":
            raise TypeError("integer counts require BinomialPartition")
        left = value * self.fraction
        return left, value - left


@dataclass(frozen=True, slots=True)
class BinomialPartition:
    """Conserve a nonnegative integer count using caller-supplied randomness.

    RNG must implement binomial(n, p), such as NumPy's Generator. The second
    daughter receives the exact remainder, so the parent count is conserved
    for every draw. No global RNG or implicit seed is selected.
    """

    fraction: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "fraction", _probability(self.fraction))

    def divide(self, value: Any, context: DivisionContext) -> tuple[Any, Any]:
        if isinstance(value, bool):
            raise TypeError("BinomialPartition expects integer count state")
        if isinstance(value, np.generic):
            if value.dtype.kind not in "iu":
                raise TypeError("BinomialPartition expects integer count state")
            value = int(value)
        if isinstance(value, np.ndarray):
            if value.dtype.kind not in "iu":
                raise TypeError("BinomialPartition expects integer count arrays")
            if np.any(value < 0):
                raise ValueError("integer counts must be nonnegative")
        elif not isinstance(value, int):
            raise TypeError("BinomialPartition expects integer count state")
        elif value < 0:
            raise ValueError("integer counts must be nonnegative")
        draw = getattr(context.rng, "binomial", None)
        if not callable(draw):
            raise TypeError("BinomialPartition requires rng.binomial(n, p)")
        left = draw(value, self.fraction)
        if isinstance(value, np.ndarray):
            if not isinstance(left, np.ndarray) or left.dtype.kind not in "iu":
                raise ValueError("rng.binomial must return an integer array")
            if left.shape != value.shape or np.any(left < 0) or np.any(left > value):
                raise ValueError("rng.binomial returned invalid daughter counts")
            return left, value - left
        if isinstance(left, bool):
            raise ValueError("rng.binomial returned an invalid count")
        try:
            import operator
            left = operator.index(left)
        except TypeError as exc:
            raise ValueError("rng.binomial returned an invalid count") from exc
        if not 0 <= left <= value:
            raise ValueError("rng.binomial returned a count outside the parent range")
        return left, value - left


@dataclass(frozen=True, slots=True)
class ResetField:
    """Reset one field at division; each daughter value is cloned separately."""

    value: Any | Callable[[Any, DivisionContext, int], Any]

    def divide(self, value: Any, context: DivisionContext) -> tuple[Any, Any]:
        if callable(self.value):
            return tuple(_clone(self.value(value, context, index)) for index in range(2))
        return _clone(self.value), _clone(self.value)


@dataclass(frozen=True, slots=True)
class CustomField:
    """Delegate a field's two-daughter rule to the caller."""

    rule: Callable[[Any, DivisionContext], tuple[Any, Any]]

    def __post_init__(self) -> None:
        if not callable(self.rule):
            raise TypeError("rule must be callable")

    def divide(self, value: Any, context: DivisionContext) -> tuple[Any, Any]:
        pair = self.rule(value, context)
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("custom inheritance rule must return two daughter values")
        return _clone(pair[0]), _clone(pair[1])


@dataclass(frozen=True, slots=True)
class FieldwiseInheritance:
    """Apply explicit per-key rules to mappings; copy all unspecified fields.

    Output mappings are plain dictionaries. State classes needing another
    container or extra invariants should provide a custom rule.
    """

    rules: Mapping[Any, InheritanceRule] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.rules, Mapping):
            raise TypeError("rules must be a mapping")
        if any(not callable(getattr(rule, "divide", None)) for rule in self.rules.values()):
            raise TypeError("each field rule must provide divide(value, context)")
        object.__setattr__(self, "rules", MappingProxyType(dict(self.rules)))

    def divide(self, state: Any, context: DivisionContext) -> tuple[dict[Any, Any], dict[Any, Any]]:
        if not isinstance(state, Mapping):
            raise TypeError("FieldwiseInheritance expects mapping state")
        unknown = self.rules.keys() - state.keys()
        if unknown:
            raise KeyError(f"inheritance rules refer to missing fields: {unknown!r}")
        left: dict[Any, Any] = {}
        right: dict[Any, Any] = {}
        for key, value in state.items():
            rule = self.rules.get(key)
            pair = CopyInheritance().divide(value, context) if rule is None else rule.divide(value, context)
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError(f"inheritance rule for {key!r} must return two values")
            # Protect against a custom rule returning the same mutable value twice.
            left[key], right[key] = _clone(pair[0]), _clone(pair[1])
        return left, right

    def __call__(self, state: Any, context: DivisionContext | None = None) -> tuple[dict[Any, Any], dict[Any, Any]]:
        return self.divide(state, context or DivisionContext())


class _DaughterFork:
    """Single-daughter callback compatible with execution.effects.Fork."""

    __slots__ = ("_owner", "_index")

    def __init__(self, owner: "_PairedFork", index: int):
        self._owner, self._index = owner, index

    def __call__(self, state: Any) -> Any:
        return self._owner.inherit(state, self._index)


class _PairedFork:
    """Share one stochastic split across two ordered Fork callbacks."""

    __slots__ = ("rule", "context", "parent", "pair", "served")

    def __init__(self, rule: InheritanceRule, context: DivisionContext):
        self.rule, self.context = rule, context
        self.parent = _UNSET
        self.pair = None
        self.served = 0

    def inherit(self, state: Any, index: int) -> Any:
        if state is not self.parent:
            if self.served not in (0, 3):
                raise RuntimeError("both daughter Fork effects must run before the next parent")
            self.parent = state
            self.pair = self.rule.divide(state, self.context)
            if not isinstance(self.pair, (tuple, list)) or len(self.pair) != 2:
                raise ValueError("inheritance must return two daughter states")
            self.served = 0
        bit = 1 << index
        if self.served & bit:
            raise RuntimeError("a daughter Fork callback was applied twice")
        self.served |= bit
        result = _clone(self.pair[index])
        if self.served == 3:
            self.parent, self.pair = _UNSET, None
        return result


def paired_fork_callbacks(
    inheritance: InheritanceRule,
    *,
    rng: Any = None,
    daughter_ids: tuple[Any, Any] = (None, None),
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Callable[[Any], Any], Callable[[Any], Any]]:
    """Create unary callbacks for the first and second Population Fork effects.

    Both callbacks share one paired stochastic draw. Apply them consecutively
    to the same parent state in the same model step.
    """
    context = DivisionContext(
        daughter_ids=daughter_ids, rng=rng,
        metadata={} if metadata is None else metadata,
    )
    owner = _PairedFork(inheritance, context)
    return _DaughterFork(owner, 0), _DaughterFork(owner, 1)


def divide_cell(
    parent: DynamicalCell,
    inheritance: InheritanceRule,
    *,
    daughter_ids: tuple[Any, Any] = (None, None),
    rng: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[DynamicalCell, DynamicalCell]:
    """Create two daughter Cells sharing model parameters and owning new states."""
    if not isinstance(parent, DynamicalCell):
        raise TypeError("parent must be a DynamicalCell")
    context = DivisionContext(
        parent_id=parent.id, daughter_ids=daughter_ids, time=parent.time, rng=rng,
        metadata={} if metadata is None else metadata,
    )
    pair = inheritance.divide(parent.state, context)
    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
        raise ValueError("inheritance must return two daughter states")
    left, right = (_clone(pair[0]), _clone(pair[1]))
    return (
        DynamicalCell(parent.model, left, id=context.daughter_ids[0], time=parent.time),
        DynamicalCell(parent.model, right, id=context.daughter_ids[1], time=parent.time),
    )


def divide_population(
    population: Any,
    parent_id: Any,
    inheritance: InheritanceRule,
    *,
    daughter_ids: tuple[Any, Any],
    rng: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[DynamicalCell, DynamicalCell]:
    """Evaluate one paired inheritance draw, then add both daughters.

    Computing the pair once preserves joint conservation for stochastic
    partitions. Population membership and context inheritance remain separate
    policies owned by the population runtime.
    """
    if len(daughter_ids) != 2:
        raise ValueError("division requires exactly two daughter identities")
    if daughter_ids[0] is not None and daughter_ids[1] is not None and daughter_ids[0] == daughter_ids[1]:
        raise ValueError("daughter identities must be distinct")
    members = set(population.ids)
    if any(identity is not None and identity in members for identity in daughter_ids):
        raise ValueError("daughter identity already exists in this population")
    parent = population[parent_id]
    daughters = divide_cell(parent, inheritance, daughter_ids=daughter_ids, rng=rng, metadata=metadata)
    for daughter in daughters:
        population.add(daughter)
    return daughters
