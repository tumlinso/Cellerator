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
    """Callable state update accepted by :class:`Model`."""

    def __call__(self, state: Any, context: StepContext) -> StepResult | Any: ...


@runtime_checkable
class Observation(Protocol):
    """Map a modeled state to a measurement or derived observation."""

    def __call__(self, state: Any, context: StepContext) -> Any: ...


class Model:
    """Reusable dynamics and observation rules with shared model parameters.

    A model defines behavior; :class:`Cell` instances supply independent state.
    No tensor framework or biological ontology is required.
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
    """An identified, persistent model instance with its own evolving state.

    State is held by reference. Construct separate state values (or handles)
    for independent cells. This avoids assuming state is copyable or dense.
    A cell lock serializes advances of this identity, but does not make arbitrary
    caller-owned mutable state thread safe.
    """

    __slots__ = ("model", "_state", "_state_ref", "id", "_time", "_lock")

    def __init__(
        self,
        model: Model,
        state: Any,
        *,
        id: Any = None,
        time: float = 0.0,
    ) -> None:
        if not isinstance(model, Model):
            raise TypeError("model must be a Model")
        self.model = model
        self._state = state
        self._state_ref = None
        self.id = uuid4().hex if id is None else id
        self._time = _finite_real(time, "time")
        self._lock = threading.RLock()

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
        model: Model,
        state_reference: Any,
        *,
        id: Any = None,
        time: float = 0.0,
    ) -> "Cell":
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


def functional_model(
    dynamics: Callable[[Any, StepContext], StepResult | Any],
    *,
    observation: Callable[[Any, StepContext], Any] | None = None,
    parameters: Any = None,
) -> Model:
    """Create a model from ordinary functions without subclassing."""
    return Model(dynamics, observation, parameters)


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
