"""State trajectories and explicit lineage branching."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from copy import copy, deepcopy
from dataclasses import dataclass
import math
from typing import Any

from .event import Event
from .history import EventHistory


def _clone_policy(policy: str | Callable[[Any], Any], state: Any) -> Any:
    if policy == "reference":
        return state
    if policy == "copy":
        return copy(state)
    if policy == "deepcopy":
        return deepcopy(state)
    if callable(policy):
        return policy(state)
    raise ValueError("clone_policy must be 'reference', 'copy', 'deepcopy', or callable")


def _validate_clone_policy(policy: str | Callable[[Any], Any]) -> None:
    if not (callable(policy) or policy in ("reference", "copy", "deepcopy")):
        raise ValueError("clone_policy must be 'reference', 'copy', 'deepcopy', or callable")


def _validate_point(point: "TrajectoryPoint") -> None:
    if not isinstance(point, TrajectoryPoint):
        raise TypeError("trajectory entries must be TrajectoryPoint objects")
    if not math.isfinite(float(point.time)):
        raise ValueError("trajectory times must be finite")
    if not all(isinstance(event, Event) for event in point.events):
        raise TypeError("trajectory point events must be Event objects")


def _validate_cell_id(cell_id: Any) -> None:
    try:
        hash(cell_id)
    except TypeError as exc:
        raise TypeError("cell identities must be hashable") from exc


@dataclass(frozen=True, slots=True)
class TrajectoryPoint:
    time: float
    state: Any
    events: tuple[Event, ...] = ()

    def __post_init__(self) -> None:
        time = float(self.time)
        if not math.isfinite(time):
            raise ValueError("trajectory times must be finite")
        events = tuple(self.events)
        if not all(isinstance(event, Event) for event in events):
            raise TypeError("trajectory point events must be Event objects")
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "events", events)


class Trajectory:
    """Ordered state/event snapshots. Snapshot cloning is chosen explicitly."""

    def __init__(self, points: Iterable[TrajectoryPoint] = (), *, clone_policy="reference") -> None:
        _validate_clone_policy(clone_policy)
        self.clone_policy = clone_policy
        self._points = list(points)
        previous = -math.inf
        for point in self._points:
            _validate_point(point)
            if point.time < previous:
                raise ValueError("trajectory times must be nondecreasing")
            previous = point.time

    def __len__(self) -> int:
        return len(self._points)

    def __iter__(self) -> Iterator[TrajectoryPoint]:
        return iter(tuple(self._points))

    def __getitem__(self, index):
        return self._points[index]

    def record(self, time: float, state: Any, events: Iterable[Event] = ()) -> TrajectoryPoint:
        time = float(time)
        if not math.isfinite(time):
            raise ValueError("trajectory times must be finite")
        if self._points and time < self._points[-1].time:
            raise ValueError("trajectory times must be nondecreasing")
        point = TrajectoryPoint(time, _clone_policy(self.clone_policy, state), tuple(events))
        self._points.append(point)
        return point

    def branch(self, *, clone_policy: str | Callable[[Any], Any] | None = None) -> "Trajectory":
        """Fork this path; later appends are independent, prefix points are shared."""
        child = Trajectory(clone_policy=self.clone_policy if clone_policy is None else clone_policy)
        child._points = list(self._points)
        return child

    @property
    def final(self) -> TrajectoryPoint:
        if not self._points:
            raise IndexError("empty trajectory has no final point")
        return self._points[-1]

    def observations(self, observer: Callable[[Any], Any]) -> tuple[Any, ...]:
        """Apply a caller-supplied observation map to every recorded state."""
        return tuple(observer(point.state) for point in self._points)


class CellLineage:
    """Explicit parent/child links plus independent event and state histories."""

    def __init__(self, cell_id: Any, *, parent: "CellLineage | None" = None,
                 state: Any = None, time: float = 0.0,
                 events: EventHistory | None = None,
                 trajectory: Trajectory | None = None) -> None:
        _validate_cell_id(cell_id)
        self.cell_id = cell_id
        self.parent = parent
        self.children: list[CellLineage] = []
        if parent is None:
            self._id_registry = {cell_id}
        else:
            self._id_registry = parent._id_registry
            if cell_id in self._id_registry:
                raise ValueError(f"cell identity {cell_id!r} already exists in this lineage")
            self._id_registry.add(cell_id)
        self.events = events if events is not None else EventHistory()
        self.trajectory = trajectory if trajectory is not None else Trajectory()
        self.trajectory.record(time, state)
        if parent is not None:
            parent.children.append(self)

    def divide(self, child_ids: Iterable[Any], *, states: Iterable[Any] | None = None,
               time: float | None = None, clone_policy="deepcopy") -> tuple["CellLineage", ...]:
        """Create named descendants with an explicit state inheritance policy."""
        ids = tuple(child_ids)
        if not ids:
            raise ValueError("division requires at least one child identity")
        for identity in ids:
            _validate_cell_id(identity)
        if len(set(ids)) != len(ids):
            raise ValueError("child identities must be unique")
        duplicates = self._id_registry.intersection(ids)
        if duplicates:
            raise ValueError(f"cell identities already exist in this lineage: {duplicates!r}")
        values = tuple(states) if states is not None else tuple(
            _clone_policy(clone_policy, self.trajectory.final.state) for _ in ids
        )
        if len(values) != len(ids):
            raise ValueError("one inherited or supplied state is required per child")
        child_time = self.trajectory.final.time if time is None else float(time)
        return tuple(CellLineage(identity, parent=self, state=value, time=child_time,
                                 events=self.events.copy(), trajectory=self.trajectory.branch())
                     for identity, value in zip(ids, values))

    def ancestors(self) -> tuple["CellLineage", ...]:
        result = []
        node = self.parent
        while node is not None:
            result.append(node)
            node = node.parent
        return tuple(result)


__all__ = ["CellLineage", "Event", "EventHistory", "Trajectory", "TrajectoryPoint"]
