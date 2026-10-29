"""Composable state updates and reference integrators.

The dynamics layer owns update semantics and integration algorithms, not the
physical representation of biological state.  The defaults use the state's
ordinary arithmetic operators (so NumPy and Torch values retain their behavior);
structured states supply a :class:`StateAlgebra` explicitly.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
import math
import random
from typing import Any, Protocol

from ..modeling import StepContext, StepResult


class StateAlgebra(Protocol):
    """The minimal arithmetic required to integrate one state representation."""

    def add(self, left: Any, right: Any) -> Any: ...
    def scale(self, state: Any, scalar: float) -> Any: ...


class OperatorAlgebra:
    """Use ``+`` and scalar ``*`` without converting or copying either value."""

    __slots__ = ()

    def add(self, left: Any, right: Any) -> Any:
        return left + right

    def scale(self, state: Any, scalar: float) -> Any:
        return state * scalar


class MappingAlgebra:
    """Integrate matching key-addressed state fields independently.

    The output is a plain dict so immutable/custom mapping inputs are never
    mutated. Nested mappings can be handled with their own explicitly composed
    algebra when field types require different arithmetic.
    """

    __slots__ = ()

    def add(self, left: Any, right: Any) -> dict[Any, Any]:
        if not hasattr(left, "keys") or not hasattr(right, "keys"):
            raise TypeError("MappingAlgebra requires mappings at every stage")
        if left.keys() != right.keys():
            raise ValueError("mapping state and derivative must have identical keys")
        return {key: left[key] + right[key] for key in left}

    def scale(self, state: Any, scalar: float) -> dict[Any, Any]:
        if not hasattr(state, "keys"):
            raise TypeError("MappingAlgebra requires a mapping state")
        return {key: value * scalar for key, value in state.items()}


_DEFAULT_ALGEBRA = OperatorAlgebra()
VectorField = Callable[[Any, StepContext], Any]


def _field_value(field: VectorField, state: Any, context: StepContext) -> Any:
    """Evaluate a pure field and reject the common accidental full-step result."""
    value = field(state, context)
    if isinstance(value, StepResult):
        raise TypeError("a vector field must return a derivative, not StepResult")
    return value


def _stage_context(context: StepContext, *, time: float, dt: float) -> StepContext:
    return StepContext(
        time=time, dt=dt, inputs=context.inputs, parameters=context.parameters,
        rng=context.rng, metadata=context.metadata,
    )


def discrete(update: Callable[[Any, StepContext], Any]) -> Callable[[Any, StepContext], StepResult]:
    """Wrap a discrete update returning raw state or a complete ``StepResult``."""
    if not callable(update):
        raise TypeError("update must be callable")

    def transition(state: Any, context: StepContext) -> StepResult:
        result = update(state, context)
        return result if isinstance(result, StepResult) else StepResult(state=result)

    return transition


def euler(field: VectorField, *, algebra: StateAlgebra | None = None) -> Callable[[Any, StepContext], StepResult]:
    """Build one explicit Euler transition over ``context.dt``."""
    if not callable(field):
        raise TypeError("field must be callable")
    ops = _DEFAULT_ALGEBRA if algebra is None else algebra

    def transition(state: Any, context: StepContext) -> StepResult:
        derivative = _field_value(field, state, context)
        next_state = ops.add(state, ops.scale(derivative, context.dt))
        return StepResult(state=next_state, time=context.time + context.dt)

    return transition


def rk4(field: VectorField, *, algebra: StateAlgebra | None = None) -> Callable[[Any, StepContext], StepResult]:
    """Build a classical fourth-order Runge–Kutta transition.

    Stages use physical stage times and the same inputs, shared parameters, RNG,
    and metadata.  The method assumes the supplied vector field is smooth enough
    for RK4; for jumps or discontinuities, choose an event/process update.
    """
    if not callable(field):
        raise TypeError("field must be callable")
    ops = _DEFAULT_ALGEBRA if algebra is None else algebra

    def transition(state: Any, context: StepContext) -> StepResult:
        h, t = context.dt, context.time
        half, sixth = h / 2.0, h / 6.0
        k1 = _field_value(field, state, _stage_context(context, time=t, dt=h))
        s2 = ops.add(state, ops.scale(k1, half))
        k2 = _field_value(field, s2, _stage_context(context, time=t + half, dt=h))
        s3 = ops.add(state, ops.scale(k2, half))
        k3 = _field_value(field, s3, _stage_context(context, time=t + half, dt=h))
        s4 = ops.add(state, ops.scale(k3, h))
        k4 = _field_value(field, s4, _stage_context(context, time=t + h, dt=h))
        weighted = ops.add(ops.scale(k1, sixth), ops.scale(k2, h / 3.0))
        weighted = ops.add(weighted, ops.scale(k3, h / 3.0))
        weighted = ops.add(weighted, ops.scale(k4, sixth))
        return StepResult(state=ops.add(state, weighted), time=t + h)

    return transition


@dataclass(frozen=True, slots=True)
class JumpOccurrence:
    """One realized channel firing; meaning remains with the supplied model."""

    channel: str
    time: float


@dataclass(frozen=True, slots=True)
class JumpChannel:
    """A stochastic transition with a state-dependent rate and state update.

    ``rate(state, context)`` returns a propensity in inverse context-time units.
    ``update(state, context)`` returns the state after one occurrence.  The
    channel name and any event meaning are caller-defined.
    """

    name: str
    rate: Callable[[Any, StepContext], float]
    update: Callable[[Any, StepContext], Any]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("channel name must be a nonempty string")
        if not callable(self.rate) or not callable(self.update):
            raise TypeError("channel rate and update must be callable")


def gillespie(
    channels: Iterable[JumpChannel], *, max_events: int = 100_000,
) -> Callable[[Any, StepContext], StepResult]:
    """Create an exact direct-method jump-process transition for one time span.

    Channel propensities are reevaluated after every firing, allowing arbitrary
    user-defined state changes and structured states.  RNG must expose
    ``random()`` on ``[0, 1)``; Python's module RNG is used when omitted.
    """
    entries = tuple(channels)
    if not entries:
        raise ValueError("at least one jump channel is required")
    if any(not isinstance(channel, JumpChannel) for channel in entries):
        raise TypeError("channels must contain JumpChannel values")
    if max_events <= 0:
        raise ValueError("max_events must be positive")

    def transition(state: Any, context: StepContext) -> StepResult:
        if context.dt < 0:
            raise ValueError("jump-process dt must be nonnegative")
        rng = context.rng if context.rng is not None else random
        if not callable(getattr(rng, "random", None)):
            raise TypeError("rng must provide random()")
        end = context.time + context.dt
        now, current = context.time, state
        occurrences: list[JumpOccurrence] = []

        while now < end:
            stage = _stage_context(context, time=now, dt=end - now)
            rates = []
            for channel in entries:
                raw_rate = channel.rate(current, stage)
                if isinstance(raw_rate, bool):
                    raise TypeError(f"rate for {channel.name!r} must be a finite nonnegative number")
                try:
                    rate = float(raw_rate)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise TypeError(f"rate for {channel.name!r} must be numeric") from exc
                if not math.isfinite(rate) or rate < 0:
                    raise ValueError(f"rate for {channel.name!r} must be finite and nonnegative")
                rates.append(rate)
            total = math.fsum(rates)
            if total == 0:
                break
            u_wait = float(rng.random())
            if not 0.0 <= u_wait < 1.0:
                raise ValueError("rng.random() must return a value in [0, 1)")
            wait = -math.log1p(-u_wait) / total
            event_time = now + wait
            if event_time >= end:
                break
            u_pick = float(rng.random()) * total
            cumulative, selected = 0.0, len(entries) - 1
            for index, rate in enumerate(rates):
                cumulative += rate
                if u_pick < cumulative:
                    selected = index
                    break
            channel = entries[selected]
            event_context = _stage_context(context, time=event_time, dt=event_time - now)
            current = channel.update(current, event_context)
            now = event_time
            occurrences.append(JumpOccurrence(channel.name, event_time))
            if len(occurrences) >= max_events and now < end:
                raise RuntimeError("max_events exceeded; interval or propensities may be impractical")

        return StepResult(
            state=current, events=tuple(occurrences), time=end,
            metadata={"jump_count": len(occurrences)},
        )

    return transition


def sequence(*updates: Callable[[Any, StepContext], Any]) -> Callable[[Any, StepContext], StepResult]:
    """Compose update operators in declared order for one context.

    Each operator receives the previous operator's state and the same context.
    Events/effects accumulate; an explicit time returned by a later operator
    takes precedence. Use an integrator when operators are derivatives of a
    shared continuous field.
    """
    if not updates or any(not callable(update) for update in updates):
        raise ValueError("sequence requires one or more callable updates")

    def transition(state: Any, context: StepContext) -> StepResult:
        events: list[Any] = []
        effects: list[Any] = []
        time = None
        metadata: dict[str, Any] = {}
        current = state
        for update in updates:
            raw = update(current, context)
            result = raw if isinstance(raw, StepResult) else StepResult(state=raw)
            current = result.state
            events.extend(result.events)
            effects.extend(result.effects)
            if result.time is not None:
                time = result.time
            metadata.update(result.metadata)
        return StepResult(current, tuple(events), tuple(effects), time, metadata)

    return transition


def additive(*fields: VectorField, algebra: StateAlgebra | None = None) -> VectorField:
    """Sum simultaneous vector-field contributions using the chosen algebra."""
    if not fields or any(not callable(field) for field in fields):
        raise ValueError("additive requires one or more callable fields")
    ops = _DEFAULT_ALGEBRA if algebra is None else algebra

    def combined(state: Any, context: StepContext) -> Any:
        values = (_field_value(field, state, context) for field in fields)
        iterator = iter(values)
        result = next(iterator)
        for value in iterator:
            result = ops.add(result, value)
        return result

    return combined


def apply_events(
    update: Callable[[Any, Any, StepContext], Any],
) -> Callable[[Any, StepContext], StepResult]:
    """Apply caller-supplied ``context.inputs`` events to a state in order.

    ``inputs`` must be an iterable of opaque events. This operator assigns no
    ontology or causal status; it only delegates each state change to ``update``.
    Events lacking a ``time`` attribute inherit the substep context time.
    """
    if not callable(update):
        raise TypeError("update must be callable")

    def transition(state: Any, context: StepContext) -> StepResult:
        if context.inputs is None:
            return StepResult(state=state, time=context.time + context.dt)
        current = state
        applied = []
        for event in context.inputs:
            event_time = getattr(event, "time", None)
            event_context = context if event_time is None else _stage_context(
                context, time=float(event_time), dt=context.dt,
            )
            current = update(current, event, event_context)
            applied.append(event)
        return StepResult(
            state=current, events=tuple(applied), time=context.time + context.dt,
        )

    return transition
