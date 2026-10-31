"""Deterministic, in-process reference execution for heterogeneous populations.

State stays with each DynamicalCell; DynamicalModel objects/parameters may be shared. There is no
distributed runtime, implicit biological policy, vectorization or device work.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
import heapq
import math
import operator
from types import MappingProxyType
from typing import Any

from ..events import Event
from ..modeling import DynamicalCell, DynamicalModel, StepResult
from .context import ActiveContext, ContextInputs, Ejection, RoutedInput
from .effects import Broadcast, Create, Eject, Fork, Remove, Resume, Send, Suspend


def _time(value: Any, name: str = "time") -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


class CausalityError(ValueError):
    """A message or transition would cross an already committed time boundary."""


@dataclass(frozen=True, slots=True)
class WorkBatch:
    """A resource grouping; grouping does not select biological eligibility."""

    key: Any
    ids: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    kind: str
    cell_id: Any
    time: float
    value: Any = None


@dataclass(frozen=True, slots=True)
class Emission:
    source: Any
    time: float
    value: Any


@dataclass(frozen=True, slots=True)
class PopulationStep:
    """One explicit selection's committed results and observable execution work."""

    order: tuple[Any, ...]
    results: Mapping[Any, StepResult]
    records: tuple[ExecutionRecord, ...] = ()
    emissions: tuple[Emission, ...] = ()
    unhandled_effects: tuple[Emission, ...] = ()

    @property
    def ejections(self) -> tuple[Ejection, ...]:
        return tuple(record.value for record in self.records if record.kind == "eject")


class ExecutionFailure(RuntimeError):
    """Execution stopped after possible partial progress; inspect ``report``.

    Arbitrary caller-owned states, model callbacks and recorder callbacks cannot
    be rolled back. Committed results and effects remain visible. A failed batch
    is never replayed automatically.
    """

    def __init__(self, phase: str, cell_id: Any, report: PopulationStep):
        self.phase = phase
        self.cell_id = cell_id
        self.report = report
        super().__init__(f"population {phase} failed for {cell_id!r}; inspect report for partial progress")


class Population:
    """Identified cells, bounded active contexts, and timed pending inboxes.

    Cells are held by reference and identities may never be reused after removal.
    Callers provide independent state values/handles; the executor cannot infer
    whether arbitrary state objects alias. Suspension pauses computation without
    changing state, time, active context, or pending inputs.

    ``advance`` evaluates cells in insertion order (or explicit batch order),
    then routes each result's events and effects in that order. Cells starting
    at the same time share the earliest pending-input step boundary. Thus messages
    emitted during one call are first visible in a later call. Same-time routes
    use a monotonically increasing sequence. Conflicting lifecycle requests use
    this same declared order; requests targeting a removed cell fail explicitly.

    Active contexts are bounded by occurrence count. Pending inboxes and optional
    history/recorders have independent retention and can grow. This object is a
    single-owner local executor; externally advancing its Cells concurrently is
    unsupported.
    """

    def __init__(self, cells: Iterable[DynamicalCell] = (), *, context_capacity: int = 128,
                 overflow: str = "oldest", history: Any = None,
                 recorder: Callable[[ExecutionRecord], None] | None = None):
        prototype = ActiveContext(context_capacity, overflow=overflow)
        self.context_capacity = prototype.capacity
        self.overflow = prototype.overflow
        if recorder is not None and not callable(recorder):
            raise TypeError("recorder must be callable")
        if history is not None and not callable(getattr(history, "append", None)):
            raise TypeError("history must provide append(event)")
        self.history = history
        self.recorder = recorder
        self._cells: dict[Any, DynamicalCell] = {}
        self._seen: set[Any] = set()
        self._contexts: dict[Any, ActiveContext] = {}
        self._pending: dict[Any, list[tuple[float, int, RoutedInput]]] = {}
        self._suspended: dict[Any, Any] = {}
        self._sequence = 0
        self._running = False
        self._evaluating = False
        self._records: list[ExecutionRecord] | None = None
        for cell in cells:
            self.add(cell)

    def __len__(self) -> int:
        return len(self._cells)

    def __iter__(self) -> Iterator[DynamicalCell]:
        return iter(tuple(self._cells.values()))

    def __getitem__(self, cell_id: Any) -> DynamicalCell:
        return self._cells[cell_id]

    @property
    def ids(self) -> tuple[Any, ...]:
        return tuple(self._cells)

    @property
    def active_ids(self) -> tuple[Any, ...]:
        return tuple(cell_id for cell_id in self._cells if cell_id not in self._suspended)

    def _mutable(self) -> None:
        if self._evaluating:
            raise RuntimeError("return execution effects instead of mutating population during a model step")

    def _record(self, kind: str, cell_id: Any, time: float, value: Any = None) -> None:
        record = ExecutionRecord(kind, cell_id, time, value)
        if self._records is not None:
            self._records.append(record)
        if self.recorder is not None:
            self.recorder(record)

    def add(self, cell: DynamicalCell) -> DynamicalCell:
        self._mutable()
        if not isinstance(cell, DynamicalCell):
            raise TypeError("population members must be DynamicalCells")
        hash(cell.id)
        if cell.id is None:
            raise ValueError("population cell identity cannot be None")
        if cell.id in self._seen:
            raise ValueError("cell identity already used in this population")
        _time(cell.time)
        self._cells[cell.id] = cell
        self._seen.add(cell.id)
        self._contexts[cell.id] = ActiveContext(self.context_capacity, overflow=self.overflow)
        self._pending[cell.id] = []
        self._record("create", cell.id, cell.time, cell)
        return cell

    def create(self, model: DynamicalModel, state: Any, *, id: Any = None, time: float = 0.0) -> DynamicalCell:
        self._mutable()
        return self.add(DynamicalCell(model, state, id=id, time=time))

    def remove(self, cell_id: Any) -> DynamicalCell:
        """Remove membership; return the still-valid DynamicalCell and report discarded inputs."""
        self._mutable()
        cell = self._cells[cell_id]
        context = self._contexts[cell_id].eject()
        pending = self.pending(cell_id)
        del self._cells[cell_id], self._contexts[cell_id], self._pending[cell_id]
        self._suspended.pop(cell_id, None)
        self._record("remove", cell_id, cell.time, {"cell": cell, "context": context, "pending": pending})
        return cell

    def fork(self, parent: Any, *, inherit: Callable[[Any], Any], id: Any = None,
             model: DynamicalModel | None = None, inherit_context: bool = False) -> DynamicalCell:
        """Create a child using the caller's explicit state inheritance function.

        The model is shared unless overridden. Context membership may be copied
        explicitly; pending inputs are never inherited. Value aliasing is the
        inheritance function's responsibility, and no dense conversion occurs.
        """
        self._mutable()
        if not callable(inherit):
            raise TypeError("inherit must be callable")
        source = self._cells[parent]
        if id is not None and id in self._seen:
            raise ValueError("cell identity already used in this population")
        state = inherit(source.state)
        child = self.create(source.model if model is None else model, state, id=id, time=source.time)
        if inherit_context:
            self._contexts[child.id].admit(self._contexts[parent])
        self._record("fork", child.id, child.time, parent)
        return child

    def suspend(self, cell_id: Any, *, reason: Any = None) -> None:
        self._mutable()
        cell = self._cells[cell_id]
        self._suspended[cell_id] = reason
        self._record("suspend", cell_id, cell.time, reason)

    def resume(self, cell_id: Any) -> None:
        self._mutable()
        cell = self._cells[cell_id]
        self._suspended.pop(cell_id, None)
        self._record("resume", cell_id, cell.time)

    def is_suspended(self, cell_id: Any) -> bool:
        self._cells[cell_id]
        return cell_id in self._suspended

    def context(self, cell_id: Any) -> ContextInputs:
        """Read the current active window without draining pending inputs."""
        return self._contexts[cell_id].snapshot()

    def pending(self, cell_id: Any) -> tuple[RoutedInput, ...]:
        return tuple(record for _, _, record in sorted(self._pending[cell_id]))

    def discard_pending(self, cell_id: Any, *, predicate=None,
                        reason: str = "explicit") -> tuple[RoutedInput, ...]:
        """Explicitly discard queued inputs, preserving occurrence provenance.

        This is independent of active-context ejection. It also lets a caller
        resolve an overfull rejecting inbox without implicit input loss.
        """
        self._mutable()
        cell = self._cells[cell_id]
        ordered = self.pending(cell_id)
        selected = tuple(True if predicate is None else bool(predicate(record)) for record in ordered)
        removed = tuple(record for record, take in zip(ordered, selected) if take)
        self._pending[cell_id] = [(record.time, record.sequence, record)
                                  for record, take in zip(ordered, selected) if not take]
        heapq.heapify(self._pending[cell_id])
        self._record("discard_pending", cell_id, cell.time, {"records": removed, "reason": reason})
        return removed

    def send(self, target: Any, value: Any, *, time: float | None = None,
             source: Any = None) -> RoutedInput:
        """Queue one occurrence without waking a suspended recipient.

        An omitted time uses the recipient's current time for external inputs;
        model-produced Send effects default to their emitter's committed time.
        Inputs in a recipient's committed past are rejected.
        """
        self._mutable()
        cell = self._cells[target]
        at = cell.time if time is None else _time(time)
        if at < cell.time:
            raise CausalityError("cannot route input into a cell's committed past")
        record = RoutedInput(value, at, self._sequence, source)
        self._sequence += 1
        heapq.heappush(self._pending[target], (at, record.sequence, record))
        self._record("enqueue", target, at, record)
        return record

    def broadcast(self, value: Any, *, targets: Iterable[Any] | None = None,
                  time: float | None = None, source: Any = None) -> tuple[RoutedInput, ...]:
        self._mutable()
        selected = self.ids if targets is None else tuple(targets)
        if len(set(selected)) != len(selected):
            raise ValueError("broadcast targets must be unique")
        # Validate destinations before enqueuing the first occurrence.
        at = None if time is None else _time(time)
        for cell_id in selected:
            cell = self._cells[cell_id]
            if at is not None and at < cell.time:
                raise CausalityError("cannot route input into a cell's committed past")
        return tuple(self.send(cell_id, value, time=at, source=source) for cell_id in selected)

    def eject(self, cell_id: Any, *, count: int | None = None, predicate=None,
              reason: str = "explicit") -> Ejection:
        self._mutable()
        cell = self._cells[cell_id]
        removed = self._contexts[cell_id].eject(count=count, predicate=predicate)
        result = Ejection(cell_id, removed, reason)
        self._record("eject", cell_id, cell.time, result)
        return result

    def _selected(self, ids: Iterable[Any] | None) -> tuple[Any, ...]:
        if ids is None:
            return self.active_ids
        selected = tuple(ids)
        if len(set(selected)) != len(selected):
            raise ValueError("selected cell identities must be unique")
        for cell_id in selected:
            self._cells[cell_id]
            if cell_id in self._suspended:
                raise ValueError(f"cell {cell_id!r} is suspended; resume it before selection")
        return selected

    def plan(self, ids: Iterable[Any] | None = None, *,
             group_by: str | Callable[[DynamicalCell], Any] = "model",
             batch_size: int | None = None) -> tuple[WorkBatch, ...]:
        """Group selected work by shared DynamicalModel identity or a caller's work key.

        This is a resource plan only. It never changes membership, suspension or
        model rules. Group and within-group order follow first occurrence. The
        reference executor evaluates each member sequentially without claiming
        vectorized execution or converting heterogeneous state.
        """
        selected = self._selected(ids)
        if group_by != "model" and not callable(group_by):
            raise ValueError("group_by must be 'model' or a callable")
        if batch_size is not None:
            if isinstance(batch_size, bool):
                raise TypeError("batch_size must be a positive integer")
            batch_size = operator.index(batch_size)
            if batch_size < 1:
                raise ValueError("batch_size must be positive")
        groups: dict[Any, list[Any]] = {}
        for cell_id in selected:
            cell = self._cells[cell_id]
            key = cell.model if group_by == "model" else group_by(cell)
            groups.setdefault(key, []).append(cell_id)
        batches = []
        for key, members in groups.items():
            size = len(members) if batch_size is None else batch_size
            batches.extend(WorkBatch(key, tuple(members[index:index + size]))
                           for index in range(0, len(members), size))
        return tuple(batches)

    def _deliver(self, cell_id: Any) -> ContextInputs:
        cell = self._cells[cell_id]
        queue = self._pending[cell_id]
        due = tuple(record for at, _, record in sorted(queue) if at <= cell.time)
        if due and due[0].time < cell.time:
            raise CausalityError("pending input lies before the cell's current time")
        # Rejecting context admission leaves the inbox intact.
        removed = self._contexts[cell_id].admit(due)
        for _ in due:
            heapq.heappop(queue)
        for record in due:
            self._record("deliver", cell_id, cell.time, record)
        if removed:
            self._record("eject", cell_id, cell.time, Ejection(cell_id, removed, "capacity"))
        retained = {record.sequence for record in self._contexts[cell_id]}
        incoming = tuple(record for record in due if record.sequence in retained)
        return self._contexts[cell_id].snapshot(incoming=incoming)

    def _effect(self, emission: Emission) -> bool:
        effect, source, at = emission.value, emission.source, emission.time
        target = getattr(effect, "target", None)
        target = source if target is None else target
        if isinstance(effect, Send):
            when = at if effect.time is None else _time(effect.time)
            if when < at:
                raise CausalityError("an effect cannot send before its emitter's committed time")
            self.send(effect.target, effect.value, time=when, source=source)
        elif isinstance(effect, Broadcast):
            when = at if effect.time is None else _time(effect.time)
            if when < at:
                raise CausalityError("an effect cannot send before its emitter's committed time")
            self.broadcast(effect.value, targets=effect.targets, time=when, source=source)
        elif isinstance(effect, Suspend):
            self.suspend(target, reason=effect.reason)
        elif isinstance(effect, Resume):
            self.resume(target)
        elif isinstance(effect, Remove):
            self.remove(target)
        elif isinstance(effect, Create):
            when = at if effect.time is None else _time(effect.time)
            if when < at:
                raise CausalityError("a created cell cannot precede its causal creation time")
            self.create(effect.model, effect.state, id=effect.id, time=when)
        elif isinstance(effect, Fork):
            parent = source if effect.parent is None else effect.parent
            if self._cells[parent].time != at:
                raise CausalityError("effect fork requires parent and emitter at the same committed time")
            self.fork(parent,
                      inherit=effect.inherit, id=effect.id, model=effect.model,
                      inherit_context=effect.inherit_context)
        elif isinstance(effect, Eject):
            self.eject(target, count=effect.count, predicate=effect.predicate, reason=effect.reason)
        else:
            return False
        return True

    def advance(self, dt: float = 1.0, *, ids: Iterable[Any] | None = None,
                batches: Iterable[WorkBatch] | None = None,
                inputs: Mapping[Any, Any] | None = None, rng: Any = None) -> PopulationStep:
        """Advance selected work once, clipping synchronized cohorts together.

        ``inputs`` maps identities to ephemeral per-step values, available as
        ``context.inputs.external``. Retained routed values are iterable through
        ``context.inputs`` and newly delivered records through ``.incoming``.
        Select ``.incoming`` when a model consumes an occurrence only once.

        A supplied batch plan defines evaluation/effect order. With ``ids`` it
        must cover that selection exactly; otherwise the plan itself selects.
        Invalid selections/dt are rejected before execution. Failures during
        callbacks/routing raise ExecutionFailure with partial-progress evidence.
        """
        if self._running:
            raise RuntimeError("population advance is not reentrant")
        delta = _time(dt, "dt")
        if delta <= 0:
            raise ValueError("dt must be positive")
        if inputs is not None and not isinstance(inputs, Mapping):
            raise TypeError("inputs must map cell identities to per-step values")
        if batches is None:
            order = self._selected(ids)
        else:
            plan = tuple(batches)
            if not all(isinstance(batch, WorkBatch) for batch in plan):
                raise TypeError("batches must contain WorkBatch objects")
            order = self._selected(cell_id for batch in plan for cell_id in batch.ids)
            if ids is not None and set(order) != set(self._selected(ids)):
                raise ValueError("batch plan must cover selected identities exactly")
        for cell_id in order:
            if not math.isfinite(self._cells[cell_id].time + delta):
                raise ValueError("step endpoint must be finite")
        # A pending message for one synchronized member must stop its peers too:
        # otherwise a clipped sender could emit into a peer's committed past.
        # Distinct start times remain independent cohorts; callers must select
        # coherent communication work or accept an explicit causal route error.
        boundaries: dict[float, float] = {}
        for cell_id in order:
            start = self._cells[cell_id].time
            end = start + delta
            for at, _, _ in self._pending[cell_id]:
                if at > start:
                    end = min(end, at)
            boundaries[start] = min(boundaries.get(start, end), end)

        results: dict[Any, StepResult] = {}
        emissions: list[Emission] = []
        unhandled: list[Emission] = []
        self._records = []
        self._running = True
        phase, current = "evaluation", None

        def report() -> PopulationStep:
            return PopulationStep(order, MappingProxyType(dict(results)), tuple(self._records or ()),
                                  tuple(emissions), tuple(unhandled))

        try:
            self._evaluating = True
            for current in order:
                cell = self._cells[current]
                visible = self._deliver(current)
                if inputs is not None:
                    visible = ContextInputs(visible.records, visible.incoming, inputs.get(current))
                start = cell.time
                end = boundaries[start]
                result = cell.advance(end - start, inputs=visible, rng=rng,
                                      metadata={"population_id": current}, until=end)
                results[current] = result
                self._record("advance", current, cell.time, result)
            self._evaluating = False
            phase = "routing"
            for current, result in results.items():
                at = result.time
                for event in result.events:
                    emission = Emission(current, at, event)
                    emissions.append(emission)
                    if isinstance(event, Event):
                        if self.history is not None:
                            self.history.append(event)
                        if event.target is not None:
                            when = at if event.time is None else _time(event.time)
                            if when < at:
                                raise CausalityError("event delivery precedes its emitter's committed time")
                            self.send(event.target, event, time=when, source=current)
                    self._record("event", current, at, event)
                for effect in result.effects:
                    emission = Emission(current, at, effect)
                    if not self._effect(emission):
                        unhandled.append(emission)
                        self._record("unhandled_effect", current, at, effect)
                    else:
                        self._record("effect", current, at, effect)
            return report()
        except Exception as exc:
            raise ExecutionFailure(phase, current, report()) from exc
        finally:
            self._evaluating = False
            self._running = False
            self._records = None

    step = advance


class LocalExecutor:
    """Optional resource grouping facade over the sequential reference engine."""

    def __init__(self, population: Population, *, group_by="model", batch_size=None):
        if not isinstance(population, Population):
            raise TypeError("population must be a Population")
        self.population = population
        self.group_by = group_by
        self.batch_size = batch_size

    def step(self, dt: float = 1.0, *, ids=None, inputs=None, rng=None) -> PopulationStep:
        batches = self.population.plan(ids, group_by=self.group_by, batch_size=self.batch_size)
        return self.population.advance(dt, batches=batches, inputs=inputs, rng=rng)
