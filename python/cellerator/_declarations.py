"""Torch-free biological identities and prepared mechanism declarations."""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import Iterable

import math

def _identity(value: "Identity | tuple[int, int]") -> "Identity":
    if isinstance(value, Identity):
        return value
    if len(value) != 2:
        raise ValueError("an identity is a (low, high) uint64 pair")
    return Identity(value[0], value[1])


def _signed_word(word: int) -> int:
    """Encode uint64 as signed int64 without losing bits."""
    return word if word < (1 << 63) else word - (1 << 64)


@dataclass(frozen=True, order=True)
class Identity:
    """Stable 128-bit biological or execution identity, encoded low then high."""

    low: int
    high: int

    def __post_init__(self) -> None:
        words = []
        for value in (self.low, self.high):
            if isinstance(value, bool):
                raise TypeError("identity words must be integers")
            try:
                words.append(operator.index(value))
            except TypeError as exc:
                raise TypeError("identity words must be integers") from exc
        object.__setattr__(self, "low", words[0])
        object.__setattr__(self, "high", words[1])
        limit = (1 << 64) - 1
        if not (0 <= self.low <= limit and 0 <= self.high <= limit):
            raise ValueError("identity words must fit uint64")
        if self.low == 0 and self.high == 0:
            raise ValueError("zero is not a valid persistent identity")

    @property
    def words(self) -> tuple[int, int]:
        return self.low, self.high


@dataclass(frozen=True)
class Axis:
    """A biological axis identity plus its logical extent.

    Shape is deliberately separate from identity: two equally sized axes with
    different domain, order, geometry, or partition are not interchangeable.
    """

    domain: Identity
    order: Identity
    geometry: Identity
    partition: Identity
    extent: int

    def __post_init__(self) -> None:
        for name in ("domain", "order", "geometry", "partition"):
            object.__setattr__(self, name, _identity(getattr(self, name)))
        if self.extent <= 0:
            raise ValueError("axis extent must be positive")

    @property
    def words(self) -> list[int]:
        return [_signed_word(word) for identity in
                (self.domain, self.order, self.geometry, self.partition)
                for word in identity.words]


@dataclass(frozen=True)
class Incidence:
    """One ordered argument slot referring to an index on a biological axis."""

    slot: int
    role: Identity
    index: int
    axis: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _identity(self.role))
        if self.slot < 0 or self.axis < 0 or self.index < 0:
            raise ValueError("incidence slot, axis, and index must be nonnegative")


@dataclass(frozen=True)
class OutputContribution:
    """A declared additive contribution to an output coordinate."""

    slot: int
    role: Identity
    index: int
    assembly: Identity
    scale: float = 1.0
    axis: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _identity(self.role))
        object.__setattr__(self, "assembly", _identity(self.assembly))
        if self.slot < 0 or self.axis < 0 or self.index < 0:
            raise ValueError("output slot, axis, and index must be nonnegative")
        if not math.isfinite(float(self.scale)):
            raise ValueError("output contribution scale must be finite")


@dataclass(frozen=True)
class Mechanism:
    """One product law with ordered, possibly repeated arguments."""

    identity: Identity
    coefficient: Identity
    arguments: tuple[Incidence, ...]
    outputs: tuple[OutputContribution, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "identity", _identity(self.identity))
        object.__setattr__(self, "coefficient", _identity(self.coefficient))
        object.__setattr__(self, "arguments", tuple(self.arguments))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        slots = sorted(x.slot for x in self.arguments)
        if slots != list(range(len(slots))):
            raise ValueError("argument slots must be unique and contiguous from zero")
        if not self.outputs:
            raise ValueError("each mechanism must declare at least one output")


@dataclass(frozen=True)
class MechanismSpec:
    """Typed biological axes and a set of prepared product mechanisms."""

    input_axis: Axis
    output_axis: Axis
    coefficient_axis: Axis
    coefficient_ids: tuple[Identity, ...]
    mechanisms: tuple[Mechanism, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "coefficient_ids",
                           tuple(_identity(i) for i in self.coefficient_ids))
        object.__setattr__(self, "mechanisms", tuple(self.mechanisms))
        if len(self.coefficient_ids) != self.coefficient_axis.extent:
            raise ValueError("coefficient identity count must equal coefficient axis extent")
        if len(set(self.coefficient_ids)) != len(self.coefficient_ids):
            raise ValueError("coefficient logical identities must be unique")
        if not self.mechanisms:
            raise ValueError("at least one mechanism is required")
        if len({m.identity for m in self.mechanisms}) != len(self.mechanisms):
            raise ValueError("mechanism identities must be unique")
        coefficients = set(self.coefficient_ids)
        for m in self.mechanisms:
            if m.coefficient not in coefficients:
                raise ValueError(f"mechanism {m.identity} binds an unknown coefficient")
            for arg in m.arguments:
                if arg.axis != 0 or arg.index >= self.input_axis.extent:
                    raise ValueError("argument incidence is outside the declared input axis")
            for out in m.outputs:
                if out.axis != 0 or out.index >= self.output_axis.extent:
                    raise ValueError("output contribution is outside the declared output axis")
        # Multiple writes to a coordinate require one explicit assembly identity.
        writers: dict[int, set[Identity]] = {}
        for m in self.mechanisms:
            for out in m.outputs:
                writers.setdefault(out.index, set()).add(out.assembly)
        if any(len(owners) != 1 for owners in writers.values()):
            raise ValueError("additive writers to an output coordinate need one assembly identity")


@dataclass(frozen=True)
class SharedSupportSpec:
    """Persistent directed support over typed source, target and edge axes.

    Edge order is coefficient order. Equal source/target pairs remain distinct
    relation members with separate logical identities and trainable values.
    """

    source_axis: Axis
    target_axis: Axis
    edge_axis: Axis
    edge_ids: tuple[Identity, ...]
    src: tuple[int, ...]
    dst: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in ("source_axis", "target_axis", "edge_axis"):
            if not isinstance(getattr(self, name), Axis):
                raise TypeError(f"{name} must be an Axis")
        ids = tuple(i if isinstance(i, Identity) else Identity(*i)
                    for i in self.edge_ids)
        object.__setattr__(self, "edge_ids", ids)
        for name in ("src", "dst"):
            try:
                indices = tuple(operator.index(i) for i in getattr(self, name))
            except TypeError as exc:
                raise TypeError(f"{name} indices must be integers") from exc
            object.__setattr__(self, name, indices)
        if not (len(ids) == len(self.src) == len(self.dst) == self.edge_axis.extent):
            raise ValueError("edge identities and src/dst counts must match edge-axis extent")
        if len(set(ids)) != len(ids):
            raise ValueError("edge logical identities must be unique, including duplicate pairs")
        if any(i < 0 or i >= self.source_axis.extent for i in self.src):
            raise ValueError("src index is outside the source axis")
        if any(i < 0 or i >= self.target_axis.extent for i in self.dst):
            raise ValueError("dst index is outside the target axis")

    def mechanism_spec(self) -> MechanismSpec:
        """Lower each persistent edge to one additive product contribution."""
        mechanisms = tuple(
            Mechanism(edge, edge,
                      (Incidence(0, self.source_axis.domain, source),),
                      (OutputContribution(0, self.target_axis.domain, target,
                                          self.target_axis.order),))
            for edge, source, target in zip(self.edge_ids, self.src, self.dst)
        )
        return MechanismSpec(self.source_axis, self.target_axis, self.edge_axis,
                             self.edge_ids, mechanisms)


def _flatten_identity(items: Iterable[Identity]) -> list[int]:
    return [_signed_word(word) for identity in items for word in identity.words]


def constructor_data(spec: MechanismSpec) -> tuple:
    """Serialize a typed declaration for the native handle constructor."""
    arg_offsets = [0]
    arg_slots: list[int] = []
    arg_roles: list[Identity] = []
    arg_axes: list[int] = []
    arg_indices: list[int] = []
    output_offsets = [0]
    output_slots: list[int] = []
    output_roles: list[Identity] = []
    output_axes: list[int] = []
    output_indices: list[int] = []
    output_assemblies: list[Identity] = []
    output_scales: list[float] = []
    bindings = {logical_id: index for index, logical_id in enumerate(spec.coefficient_ids)}
    coefficient_bindings: list[int] = []
    for mechanism in spec.mechanisms:
        for arg in mechanism.arguments:
            arg_slots.append(arg.slot)
            arg_roles.append(arg.role)
            arg_axes.append(arg.axis)
            arg_indices.append(arg.index)
        arg_offsets.append(len(arg_slots))
        coefficient_bindings.append(bindings[mechanism.coefficient])
        for out in mechanism.outputs:
            output_slots.append(out.slot)
            output_roles.append(out.role)
            output_axes.append(out.axis)
            output_indices.append(out.index)
            output_assemblies.append(out.assembly)
            output_scales.append(float(out.scale))
        output_offsets.append(len(output_slots))
    axis_words = spec.input_axis.words + spec.output_axis.words + spec.coefficient_axis.words
    axis_extents = [spec.input_axis.extent, spec.output_axis.extent,
                    spec.coefficient_axis.extent]
    return (
        axis_words, axis_extents, _flatten_identity(spec.coefficient_ids),
        _flatten_identity(m.identity for m in spec.mechanisms),
        arg_offsets, arg_slots, _flatten_identity(arg_roles), arg_axes, arg_indices,
        coefficient_bindings, output_offsets, output_slots,
        _flatten_identity(output_roles), output_axes, output_indices,
        _flatten_identity(output_assemblies), output_scales,
    )
