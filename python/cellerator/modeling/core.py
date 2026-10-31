"""Backend-neutral building blocks for models of evolving biological systems.

The core deliberately treats state and parameters as opaque values. It accepts
NumPy/Torch tensors, sparse or ragged structures, domain objects, and externally
owned state handles without converting or copying them.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
import math
import threading
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable
import warnings
from uuid import uuid4


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _metadata(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise TypeError("metadata must be a mapping")
    return MappingProxyType(dict(value))


class _UseModelParameters:
    __slots__ = ()

    def __repr__(self) -> str:
        return "USE_MODEL_PARAMETERS"


# Distinguish an omitted parameter override from an intentional ``None``.
USE_MODEL_PARAMETERS = _UseModelParameters()


@dataclass(frozen=True, slots=True)
class StepContext:
    """Inputs visible during one model update.

    ``time`` and ``dt`` use the model's time unit. Inputs, parameters and RNG
    retain their caller-owned representations. Metadata is a shallow read-only
    mapping; its values remain caller-owned.
    """

    time: float = 0.0
    dt: float = 1.0
    inputs: Any = None
    parameters: Any = USE_MODEL_PARAMETERS
    rng: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "time", _finite_real(self.time, "time"))
        object.__setattr__(self, "dt", _finite_real(self.dt, "dt"))
        object.__setattr__(self, "metadata", _metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class StepResult:
    """The result of one update, including optional events and system effects."""

    state: Any
    events: tuple[Any, ...] = ()
    effects: tuple[Any, ...] = ()
    time: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "effects", tuple(self.effects))
        if self.time is not None:
            object.__setattr__(self, "time", _finite_real(self.time, "time"))
        object.__setattr__(self, "metadata", _metadata(self.metadata))


@runtime_checkable
class Dynamics(Protocol):
    """Callable state update accepted by :class:`DynamicalModel`."""

    def __call__(self, state: Any, context: StepContext) -> StepResult | Any: ...


@runtime_checkable
class Observation(Protocol):
    """Map a modeled state to a measurement or derived observation."""

    def __call__(self, state: Any, context: StepContext) -> Any: ...


class DynamicalModel:
    """Reusable transition and observation rules with shared parameters.

    This is an optional dynamics abstraction. General learned programs can be
    ordinary callables or modules and do not need to subclass a Cellerator base.
    ``Model`` remains a legacy alias for this class.
    """

    __slots__ = ("dynamics", "observation", "parameters")

    def __init__(
        self,
        dynamics: Dynamics,
        observation: Observation | None = None,
        parameters: Any = None,
    ) -> None:
        if not callable(dynamics):
            raise TypeError("dynamics must be callable")
        if observation is not None and not callable(observation):
            raise TypeError("observation must be callable or None")
        self.dynamics = dynamics
        self.observation = observation
        self.parameters = parameters

    def context(
        self,
        *,
        time: float = 0.0,
        dt: float = 1.0,
        inputs: Any = None,
        parameters: Any = USE_MODEL_PARAMETERS,
        rng: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> StepContext:
        """Build a context, using this model's shared parameters by default."""
        return StepContext(
            time=time,
            dt=dt,
            inputs=inputs,
            parameters=self.parameters if parameters is USE_MODEL_PARAMETERS else parameters,
            rng=rng,
            metadata={} if metadata is None else metadata,
        )

    def step(self, state: Any, context: StepContext) -> StepResult:
        """Evaluate one transition without copying or coercing ``state``."""
        if not isinstance(context, StepContext):
            raise TypeError("context must be a StepContext")
        context = self._bind_parameters(context)
        result = self.dynamics(state, context)
        if isinstance(result, StepResult):
            return result
        return StepResult(state=result)

    def observe(self, state: Any, context: StepContext | None = None) -> Any:
        """Produce an observation; with no observer, return the modeled state."""
        if context is None:
            context = self.context()
        elif not isinstance(context, StepContext):
            raise TypeError("context must be a StepContext")
        context = self._bind_parameters(context)
        if self.observation is None:
            return state
        return self.observation(state, context)

    def _bind_parameters(self, context: StepContext) -> StepContext:
        if context.parameters is USE_MODEL_PARAMETERS:
            return replace(context, parameters=self.parameters)
        return context


class Cell:
    """A lightweight identity and reference to information about one cell.

    ``data`` is caller-owned and may be a measurement, history, structured
    object, latent handle, or another model-specific representation. A cell
    carries no required clock, transition rule, or materialized state vector.
    Bind an ordinary callable or module with :meth:`bind` when a per-cell
    prediction handle is useful; batched calls to the program remain valid.

    The former transition-shaped ``Cell(model, state, ...)`` constructor is
    retained temporarily as a deprecated compatibility route to
    :class:`DynamicalCell`.
    """

    __slots__ = ("_data", "_source", "_program", "id")

    def __new__(cls, *args: Any, **kwargs: Any):
        # Preserve the old constructor only when both pieces of its signature
        # are explicit. A DynamicalModel alone is still valid caller-owned data.
        if cls is Cell:
            positional_model = bool(args) and isinstance(args[0], DynamicalModel)
            keyword_model = isinstance(kwargs.get("model"), DynamicalModel)
            has_state = (len(args) >= 2 or "state" in kwargs)
            if has_state and (positional_model or keyword_model):
                warnings.warn(
                    "Cell(model, state, ...) is deprecated; use "
                    "DynamicalCell(model, state, ...) instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                return object.__new__(DynamicalCell)
        return object.__new__(cls)

    def __init__(self, data: Any = None, *, id: Any = None) -> None:
        self._data = data
        self._source: Cell | None = None
        self._program: Any = None
        self.id = uuid4().hex if id is None else id

    @property
    def data(self) -> Any:
        """Caller-owned information, read live through a bound source handle."""
        return self._source.data if self._source is not None else self._data

    @data.setter
    def data(self, value: Any) -> None:
        if self._source is not None:
            self._source.data = value
        else:
            self._data = value

    @property
    def program(self) -> Any:
        """The shared callable bound to this handle, if any."""
        return self._program

    def bind(self, program: Any) -> "Cell":
        """Return a handle sharing this identity and live data source.

        ``program`` is stored by reference, so modules and their parameters are
        shared rather than copied. The returned object is a generic ``Cell``
        handle even when the source is a :class:`DynamicalCell`.
        """
        if not callable(program):
            raise TypeError("program must be callable")
        bound = Cell(self.data, id=self.id)
        bound._source = self._source if self._source is not None else self
        bound._program = program
        return bound

    def predict(self, view: str | None = None) -> Any:
        """Evaluate the bound program on current data, optionally requesting a view."""
        if self._program is None:
            raise RuntimeError("no program is bound; call cell.bind(program) first")
        if view is None:
            return self._program(self.data)
        if not isinstance(view, str) or not view:
            raise ValueError("view must be a non-empty string")
        predict = getattr(self._program, "predict", None)
        if not callable(predict):
            raise TypeError("the bound program does not support named prediction views")
        return predict(self.data, view=view)

    def query(self, query: Any) -> Any:
        """Delegate an identity-labelled request to the shared bound program.

        A standard ``PredictionQuery`` without sample identifiers receives this
        cell's identity. The program remains responsible for interpreting its
        data and for any batched computation; no representation is cached here.
        """
        if self._program is None:
            raise RuntimeError("no program is bound; call cell.bind(program) first")
        evaluate = getattr(self._program, "query", None)
        if not callable(evaluate):
            raise TypeError("the bound program does not support biological queries")
        from ..biology import PredictionQuery
        if isinstance(query, PredictionQuery) and query.sample_ids is None:
            query = replace(query, sample_ids=(self.id,))
        return evaluate(self.data, query)

    @classmethod
    def from_reference(cls, model: "DynamicalModel", state_reference: Any, *,
                       id: Any = None, time: float = 0.0) -> "DynamicalCell":
        """Deprecated compatibility factory for externally owned dynamic state."""
        if cls is not Cell:
            raise TypeError("Cell.from_reference is a legacy dynamics factory")
        warnings.warn(
            "Cell.from_reference is deprecated; use "
            "DynamicalCell.from_reference instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return DynamicalCell.from_reference(
            model, state_reference, id=id, time=time
        )


class DynamicalCell(Cell):
    """An identified, persistent instance of a :class:`DynamicalModel`.

    This optional specialization owns a clock and transition behavior. State is
    held by reference; the generic :attr:`Cell.data` view always follows its
    current state, including when the state is externally managed.
    """

    __slots__ = ("model", "_state", "_state_ref", "_time", "_lock")

    def __init__(self, model: DynamicalModel, state: Any, *, id: Any = None,
                 time: float = 0.0) -> None:
        if not isinstance(model, DynamicalModel):
            raise TypeError("model must be a DynamicalModel")
        super().__init__(state, id=id)
        self.model = model
        self._state = state
        self._state_ref = None
        self._time = _finite_real(time, "time")
        self._lock = threading.RLock()

    @property
    def data(self) -> Any:
        """Current modeled information, following the live state reference."""
        return self.state

    @data.setter
    def data(self, value: Any) -> None:
        with self._lock:
            if self._state_ref is not None:
                self._state_ref.set(value)
            else:
                self._state = value

    @property
    def state(self) -> Any:
        """Current model-native state; returned without conversion or copying."""
        with self._lock:
            return self._state_ref.get() if self._state_ref is not None else self._state

    @property
    def time(self) -> float:
        with self._lock:
            return self._time

    def advance(
        self,
        dt: float = 1.0,
        *,
        inputs: Any = None,
        rng: Any = None,
        parameters: Any = USE_MODEL_PARAMETERS,
        metadata: Mapping[str, Any] | None = None,
        until: float | None = None,
    ) -> StepResult:
        """Advance this cell and atomically adopt the returned state reference."""
        delta = _finite_real(dt, "dt")
        upper_bound = None if until is None else _finite_real(until, "until")
        with self._lock:
            if upper_bound is not None and upper_bound < self._time:
                raise ValueError("until must not precede the current cell time")
            context = self.model.context(
                time=self._time,
                dt=delta,
                inputs=inputs,
                parameters=parameters,
                rng=rng,
                metadata=metadata,
            )
            current_state = self._state_ref.get() if self._state_ref is not None else self._state
            result = self.model.step(current_state, context)
            next_time = self._time + delta if result.time is None else result.time
            if not math.isfinite(next_time):
                raise ValueError("result time must be finite")
            if upper_bound is not None:
                if next_time < self._time:
                    raise ValueError("bounded advance moved cell time backwards")
                if next_time > upper_bound:
                    raise ValueError("dynamics advanced past the requested time bound")
                if upper_bound > self._time and next_time <= self._time:
                    raise ValueError("bounded advance did not make forward time progress")
            if self._state_ref is not None:
                self._state_ref.set(result.state)
            else:
                self._state = result.state
            self._time = next_time
            return result if result.time is not None else StepResult(
                state=result.state,
                events=result.events,
                effects=result.effects,
                time=next_time,
                metadata=result.metadata,
            )

    def observe(
        self,
        *,
        inputs: Any = None,
        parameters: Any = USE_MODEL_PARAMETERS,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        """Observe this cell's current state without advancing it."""
        with self._lock:
            context = self.model.context(
                time=self._time,
                dt=0.0,
                inputs=inputs,
                parameters=parameters,
                metadata=metadata,
            )
            current_state = self._state_ref.get() if self._state_ref is not None else self._state
            return self.model.observe(current_state, context)

    @classmethod
    def from_reference(
        cls,
        model: DynamicalModel,
        state_reference: Any,
        *,
        id: Any = None,
        time: float = 0.0,
    ) -> "DynamicalCell":
        """Bind to an external state owner implementing ``get`` and ``set``.

        The reference remains the authoritative state store. This is useful for
        state distributed across layouts/buffers or managed by a simulation
        runtime. Use a read-only reference only if the model does not advance it.
        """
        if not callable(getattr(state_reference, "get", None)):
            raise TypeError("state_reference must provide get()")
        if not callable(getattr(state_reference, "set", None)):
            raise TypeError("state_reference must provide set()")
        reference_identity = getattr(state_reference, "identity", None)
        cell = cls(
            model,
            state_reference.get(),
            id=reference_identity if id is None else id,
            time=time,
        )
        cell._state_ref = state_reference
        return cell


# Retain the original name for transition-model users while making its scope
# explicit in new APIs. Ordinary learned programs need no Cellerator base class.
Model = DynamicalModel


def functional_model(
    dynamics: Callable[[Any, StepContext], StepResult | Any],
    *,
    observation: Callable[[Any, StepContext], Any] | None = None,
    parameters: Any = None,
) -> DynamicalModel:
    """Create a model from ordinary functions without subclassing."""
    return DynamicalModel(dynamics, observation, parameters)


class FunctionalDynamics:
    """Named adapter for a state/context transition function."""

    __slots__ = ("function",)

    def __init__(self, function: Callable[[Any, StepContext], StepResult | Any]):
        if not callable(function):
            raise TypeError("function must be callable")
        self.function = function

    def __call__(self, state: Any, context: StepContext) -> StepResult | Any:
        return self.function(state, context)


class FunctionalObservation:
    """Named adapter for a state/context observation function."""

    __slots__ = ("function",)

    def __init__(self, function: Callable[[Any, StepContext], Any]):
        if not callable(function):
            raise TypeError("function must be callable")
        self.function = function

    def __call__(self, state: Any, context: StepContext) -> Any:
        return self.function(state, context)


class SequentialUpdates:
    """Apply explicitly ordered update functions within one model step.

    This is an operator-splitting convenience: every component receives the
    same context and sees the state returned by its predecessor. Callers choose
    ordering, substep sizes, and validity for their model.
    """

    __slots__ = ("updates",)

    def __init__(self, *updates: Dynamics):
        if not updates or not all(callable(update) for update in updates):
            raise ValueError("provide one or more callable updates")
        self.updates = tuple(updates)

    def __call__(self, state: Any, context: StepContext) -> StepResult:
        events: list[Any] = []
        effects: list[Any] = []
        metadata: list[Mapping[str, Any]] = []
        for update in self.updates:
            value = update(state, context)
            result = value if isinstance(value, StepResult) else StepResult(value)
            state = result.state
            events.extend(result.events)
            effects.extend(result.effects)
            metadata.append(result.metadata)
        return StepResult(
            state,
            events=tuple(events),
            effects=tuple(effects),
            time=context.time + context.dt,
            metadata={"composition": "sequential", "components": tuple(metadata)},
        )


def sequential_updates(*updates: Dynamics) -> SequentialUpdates:
    """Compose updates in declared order within a single step."""
    return SequentialUpdates(*updates)
