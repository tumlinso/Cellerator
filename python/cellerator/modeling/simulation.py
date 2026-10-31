"""Small experiment runner for sampling one persistent model instance."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import math
from typing import Any

from cellerator.events import Trajectory

from .core import DynamicalCell, StepResult, USE_MODEL_PARAMETERS, _finite_real


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Observed trajectory, emitted event stream, and final live cell state."""

    times: tuple[float, ...]
    observations: tuple[Any, ...]
    trajectory: Trajectory
    events: tuple[Any, ...]
    final_state: Any
    final_time: float
    cell: DynamicalCell


class Experiment:
    """Run a cell to requested times under an optional input schedule.

    ``input_schedule`` may be a mapping from absolute times to values (held
    constant until replaced) or a callback ``(time, state) -> value`` sampled
    at each model update. Mapping boundaries split integration steps so inputs
    become visible at their declared time. The callback chooses values from the
    current cell state and is evaluated before each step.

    The supplied cell is advanced in place. Trajectory state snapshots default
    to ``deepcopy``; pass ``'reference'``, ``'copy'``, or a caller-owned clone
    function when the state type requires different snapshot semantics.
    """

    def __init__(
        self,
        cell: DynamicalCell,
        times: Iterable[float],
        *,
        input_schedule: Mapping[float, Any] | Callable[[float, Any], Any] | None = None,
        clone_policy: str | Callable[[Any], Any] = "deepcopy",
        max_step: float | None = None,
        rng: Any = None,
        parameters: Any = USE_MODEL_PARAMETERS,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(cell, DynamicalCell):
            raise TypeError("cell must be a DynamicalCell")
        requested = tuple(_finite_real(value, "observation time") for value in times)
        if not requested:
            raise ValueError("at least one observation time is required")
        if any(right < left for left, right in zip(requested, requested[1:])):
            raise ValueError("observation times must be nondecreasing")
        if requested[0] < cell.time:
            raise ValueError("observation times must not precede the cell's current time")
        if max_step is not None:
            max_step = _finite_real(max_step, "max_step")
            if max_step <= 0:
                raise ValueError("max_step must be positive")
        if input_schedule is not None and not callable(input_schedule) and not isinstance(input_schedule, Mapping):
            raise TypeError("input_schedule must be a time mapping, callback, or None")
        if isinstance(input_schedule, Mapping):
            schedule = tuple(sorted(
                ((_finite_real(at, "input time"), value) for at, value in input_schedule.items()),
                key=lambda pair: pair[0],
            ))
            if any(right[0] == left[0] for left, right in zip(schedule, schedule[1:])):
                raise ValueError("input schedule times must be unique")
            self._schedule = schedule
            self._input_callback = None
        else:
            self._schedule = ()
            self._input_callback = input_schedule
        self.cell = cell
        self.times = requested
        self.clone_policy = clone_policy
        self.max_step = max_step
        self.rng = rng
        self.parameters = parameters
        self.metadata = {} if metadata is None else dict(metadata)
        # Validate the copy policy before any state advance.
        Trajectory(clone_policy=clone_policy)

    def _input_at(self, time: float, state: Any) -> Any:
        if self._input_callback is not None:
            return self._input_callback(time, state)
        result = None
        for at, value in self._schedule:
            if at > time:
                break
            result = value
        return result

    def _next_input_boundary(self, time: float, end: float) -> float:
        for at, _ in self._schedule:
            if time < at < end:
                return at
        return end

    def run(self) -> ExperimentResult:
        """Advance the cell, recording every requested observation time."""
        trajectory = Trajectory(clone_policy=self.clone_policy)
        all_events: list[Any] = []
        pending_events: list[Any] = []
        observations: list[Any] = []
        start = self.cell.time
        trajectory.record(start, self.cell.state)

        for target in self.times:
            while self.cell.time < target:
                current = self.cell.time
                end = self._next_input_boundary(current, target)
                if self.max_step is not None:
                    end = min(end, current + self.max_step)
                dt = end - current
                if not math.isfinite(dt) or dt <= 0:
                    raise RuntimeError("experiment could not make forward time progress")
                inputs = self._input_at(current, self.cell.state)
                result = self.cell.advance(
                    dt,
                    inputs=inputs,
                    rng=self.rng,
                    parameters=self.parameters,
                    metadata=self.metadata,
                    until=end,
                )
                pending_events.extend(result.events)
                all_events.extend(result.events)

            sample_inputs = self._input_at(target, self.cell.state)
            observations.append(self.cell.observe(
                inputs=sample_inputs,
                parameters=self.parameters,
                metadata=self.metadata,
            ))
            trajectory.record(target, self.cell.state, pending_events)
            pending_events = []

        return ExperimentResult(
            times=self.times,
            observations=tuple(observations),
            trajectory=trajectory,
            events=tuple(all_events),
            final_state=self.cell.state,
            final_time=self.cell.time,
            cell=self.cell,
        )


def simulate(
    cell: DynamicalCell,
    times: Iterable[float],
    **kwargs: Any,
) -> ExperimentResult:
    """Run one :class:`DynamicalCell` through requested observation times."""
    return Experiment(cell, times, **kwargs).run()


__all__ = ["Experiment", "ExperimentResult", "simulate"]
