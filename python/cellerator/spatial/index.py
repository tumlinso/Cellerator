"""Dynamic sparse geometric neighborhoods for model entities.

``SpatialIndex`` is a reference uniform-grid cell list. It indexes arbitrary
stable identities in one to three Euclidean dimensions, updating only the
affected bins on movement. Queries visit neighboring bins and exact-filter
their occupants; they never allocate an all-pairs distance matrix.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
import math
from typing import Any

import numpy as np

from cellerator.interactions import Interaction, InteractionGraph


@dataclass(frozen=True, slots=True)
class SpatialEdge:
    """Stable identity for one directed neighbor relation."""

    source: Hashable
    target: Hashable


@dataclass(frozen=True, slots=True)
class Neighbor:
    identity: Hashable
    distance: float
    displacement: tuple[float, ...]


class SpatialIndex:
    """Incremental sparse neighbor lookup for entities with positions.

    ``search_radius`` is the configured maximum query radius and also the grid
    cell width. Coordinates may move arbitrarily; queries remain exact, and
    only occupied bins are stored. A position update changes value generation;
    inserting/removing an identity changes structure generation.
    """

    def __init__(self, positions: Mapping[Hashable, Sequence[float]] | None = None,
                 *, search_radius: float, dimensions: int | None = None) -> None:
        radius = float(search_radius)
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("search_radius must be positive and finite")
        self.search_radius = radius
        self._dimensions = dimensions
        if dimensions is not None and dimensions not in (1, 2, 3):
            raise ValueError("dimensions must be 1, 2, or 3")
        self._positions: dict[Hashable, np.ndarray] = {}
        self._bins: dict[tuple[int, ...], dict[Hashable, None]] = {}
        self._entity_bin: dict[Hashable, tuple[int, ...]] = {}
        self.structure_generation = 0
        self.value_generation = 0
        self.layout_generation = 0
        if positions:
            for identity, position in positions.items():
                self.insert(identity, position)

    @property
    def dimensions(self) -> int | None:
        return self._dimensions

    @property
    def identities(self) -> tuple[Hashable, ...]:
        return tuple(self._positions)

    def __len__(self) -> int:
        return len(self._positions)

    def _point(self, position: Sequence[float]) -> np.ndarray:
        point = np.asarray(position, dtype=np.float64)
        if point.ndim != 1 or point.size not in (1, 2, 3):
            raise ValueError("position must be a one-dimensional 1D, 2D, or 3D coordinate")
        if not np.isfinite(point).all():
            raise ValueError("position coordinates must be finite")
        if self._dimensions is not None and point.size != self._dimensions:
            raise ValueError(f"position must have {self._dimensions} coordinates")
        if self._dimensions is None:
            self._dimensions = int(point.size)
        if point.size != self._dimensions:
            raise ValueError("position dimensions must agree")
        return np.array(point, copy=True)

    def _bin(self, point: np.ndarray) -> tuple[int, ...]:
        return tuple(math.floor(float(coord) / self.search_radius) for coord in point)

    def insert(self, identity: Hashable, position: Sequence[float]) -> None:
        """Add a new identity, rejecting duplicate live identities."""
        hash(identity)
        if identity in self._positions:
            raise ValueError(f"identity already indexed: {identity!r}")
        point = self._point(position)
        cell = self._bin(point)
        self._positions[identity] = point
        self._entity_bin[identity] = cell
        bucket = self._bins.setdefault(cell, {})
        bucket[identity] = None
        self.structure_generation += 1
        self.layout_generation += 1

    def move(self, identity: Hashable, position: Sequence[float]) -> tuple[int, ...]:
        """Update one position and return the changed grid axes (0-based)."""
        if identity not in self._positions:
            raise KeyError(identity)
        point = self._point(position)
        old_cell = self._entity_bin[identity]
        new_cell = self._bin(point)
        changed = tuple(i for i, (old, new) in enumerate(zip(old_cell, new_cell))
                        if old != new)
        self._positions[identity] = point
        self.value_generation += 1
        if new_cell != old_cell:
            old_bucket = self._bins[old_cell]
            del old_bucket[identity]
            if not old_bucket:
                del self._bins[old_cell]
            self._bins.setdefault(new_cell, {})[identity] = None
            self._entity_bin[identity] = new_cell
            self.layout_generation += 1
        return changed

    def remove(self, identity: Hashable) -> np.ndarray:
        """Remove an indexed identity and return its last coordinate."""
        try:
            point = self._positions.pop(identity)
            cell = self._entity_bin.pop(identity)
        except KeyError as exc:
            raise KeyError(identity) from exc
        bucket = self._bins[cell]
        del bucket[identity]
        if not bucket:
            del self._bins[cell]
        self.structure_generation += 1
        self.layout_generation += 1
        return point

    def position(self, identity: Hashable) -> np.ndarray:
        """Return a defensive coordinate copy."""
        return self._positions[identity].copy()

    def _candidate_ids(self, point: np.ndarray) -> Iterable[Hashable]:
        center = self._bin(point)
        offsets = product((-1, 0, 1), repeat=len(center))
        for offset in offsets:
            bucket = self._bins.get(tuple(base + delta for base, delta in zip(center, offset)))
            if bucket:
                yield from bucket

    def query(self, identity: Hashable, *, radius: float | None = None,
              include_self: bool = False) -> tuple[Neighbor, ...]:
        """Return exact Euclidean neighbors ordered by distance then identity order.

        Stable identity insertion order breaks equal-distance ties; identities
        need not be mutually sortable. Cost is proportional to nearby occupied
        bins and their occupants, rather than the square of population size.
        """
        try:
            point = self._positions[identity]
        except KeyError as exc:
            raise KeyError(identity) from exc
        cutoff = self.search_radius if radius is None else float(radius)
        if not math.isfinite(cutoff) or cutoff < 0 or cutoff > self.search_radius:
            raise ValueError("radius must be finite and between zero and search_radius")
        ordinal = {key: index for index, key in enumerate(self._positions)}
        found: list[tuple[int, Neighbor]] = []
        for other in self._candidate_ids(point):
            if other == identity and not include_self:
                continue
            displacement = self._positions[other] - point
            distance = float(np.linalg.norm(displacement))
            if distance <= cutoff:
                found.append((ordinal[other], Neighbor(
                    other, distance, tuple(float(x) for x in displacement))))
        found.sort(key=lambda pair: (pair[1].distance, pair[0]))
        return tuple(item for _, item in found)

    def pairs(self, *, radius: float | None = None,
              directed: bool = False) -> tuple[tuple[Hashable, Hashable, float], ...]:
        """Materialize the current sparse neighbor pairs with measured distance."""
        ordered = tuple(self._positions)
        rank = {identity: i for i, identity in enumerate(ordered)}
        result = []
        seen: set[tuple[Hashable, Hashable]] = set()
        for source in ordered:
            for neighbor in self.query(source, radius=radius):
                target = neighbor.identity
                pair = (source, target)
                if not directed and rank[source] > rank[target]:
                    continue
                if pair not in seen:
                    seen.add(pair)
                    result.append((source, target, neighbor.distance))
        return tuple(result)

    def interaction_graph(
        self,
        operation: Callable[[Any, Any, float, Any], Any],
        *,
        radius: float | None = None,
        directed: bool = True,
        active_if: Callable[[Mapping[Hashable, Any], Any], bool] | None = None,
    ) -> InteractionGraph:
        """Build a snapshot graph of current spatial neighbors.

        ``operation(source, target, distance, context)`` computes one target
        contribution from source and target values. The returned relation graph
        is tied to this geometry generation; rebuild it after entity movement.
        """
        if not callable(operation):
            raise TypeError("operation must be callable")
        pairs = self.pairs(radius=radius, directed=directed)
        relations = []
        for source, target, distance in pairs:
            edge_id = SpatialEdge(source, target)
            def apply(values, context, *, _distance=distance):
                return operation(values[0], values[1], _distance, context)
            relations.append(Interaction(edge_id, (source, target), (target,),
                                         apply, "directed", active_if))
        return InteractionGraph(relations)


def diffuse(
    index: SpatialIndex,
    values: Mapping[Hashable, float],
    *,
    dt: float,
    diffusivity: float,
    radius: float | None = None,
    weight: Callable[[float], float] | None = None,
) -> dict[Hashable, float]:
    """One conservative graph-diffusion step over current undirected neighbors.

    Each unordered pair exchanges equal and opposite amount. The default pair
    weight is one; callers can supply a nonnegative distance kernel. This is an
    explicit graph update, not an unbounded-space diffusion solver, and large
    steps can overshoot. Physical units and any stability limit belong to the
    scientific model.
    """
    step = float(dt)
    coefficient = float(diffusivity)
    if not math.isfinite(step) or step < 0:
        raise ValueError("dt must be finite and nonnegative")
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("diffusivity must be finite and nonnegative")
    if set(values) != set(index.identities):
        raise ValueError("values must contain exactly the indexed identities")
    output = {identity: float(value) for identity, value in values.items()}
    delta = {identity: 0.0 for identity in values}
    for left, right, distance in index.pairs(radius=radius, directed=False):
        conductance = 1.0 if weight is None else float(weight(distance))
        if not math.isfinite(conductance) or conductance < 0:
            raise ValueError("pair weight must be finite and nonnegative")
        flux = step * coefficient * conductance * (output[right] - output[left])
        delta[left] += flux
        delta[right] -= flux
    return {identity: output[identity] + delta[identity] for identity in output}


__all__ = ["Neighbor", "SpatialEdge", "SpatialIndex", "diffuse"]
