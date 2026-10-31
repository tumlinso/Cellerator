"""Explicit, ragged truncated-BPTT windows for a shared optional Torch model.

This adapter does not schedule collectives, infer biological losses, or own an
optimizer. A window evaluates functional transitions, reduces observation
losses, completes backward, detaches retained state/history, and only then may
invoke the caller's optimizer. Import this module explicitly to opt into Torch.
"""
from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from itertools import islice
import math
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor, nn

from cellerator.modeling import DynamicalCell, DynamicalModel, StepContext, StepResult


class _NoTarget:
    def __repr__(self):
        return "NO_TARGET"


NO_TARGET = _NoTarget()


def detach_state(value: Any) -> Any:
    """Detach tensor leaves without densifying or copying tensor storage.

    Plain dict/list/tuple, namedtuple and dataclass containers are rebuilt;
    immutable scalar leaves remain unchanged. Unknown objects fail explicitly:
    supply a policy for custom state/native handles rather than silently retaining
    a graph or converting them to tensors. Cyclic containers are unsupported.
    Detached tensors share storage: subsequent dynamics must be functional.
    """
    if isinstance(value, Tensor):
        return value.detach()
    if value is None or isinstance(value, (str, bytes, bool, int, float, complex)):
        return value
    if type(value) is dict:
        return {key: detach_state(item) for key, item in value.items()}
    if isinstance(value, MappingProxyType):
        return MappingProxyType({key: detach_state(item) for key, item in value.items()})
    if type(value) is list:
        return [detach_state(item) for item in value]
    if isinstance(value, tuple):
        items = [detach_state(item) for item in value]
        if hasattr(value, "_fields"):
            return type(value)(*items)
        if type(value) is tuple:
            return tuple(items)
    if is_dataclass(value) and not isinstance(value, type):
        if any(not item.init for item in fields(value)):
            raise TypeError("dataclasses with non-init fields need a custom detachment policy")
        return replace(value, **{item.name: detach_state(getattr(value, item.name))
                                 for item in fields(value)})
    raise TypeError(f"provide a detachment policy for {type(value).__name__}")


def _finite(value, name, *, nonnegative=False):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ValueError(f"{name} must be finite" + (" and nonnegative" if nonnegative else ""))
    return result


@dataclass(frozen=True, slots=True)
class TrainingStep:
    """One caller-selected time increment and optional post-step target.

    ``advance=False`` observes the current state and requires ``dt=0``. A missing
    target skips observation/loss; an explicit ``None`` is a valid target.
    ``weight=0`` skips the loss callback. Inputs and RNG retain caller ownership.
    """
    dt: float = 1.0
    target: Any = NO_TARGET
    inputs: Any = None
    rng: Any = None
    weight: float = 1.0
    advance: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "dt", _finite(self.dt, "dt", nonnegative=True))
        object.__setattr__(self, "weight", _finite(self.weight, "weight", nonnegative=True))
        if not isinstance(self.advance, bool):
            raise TypeError("advance must be bool")
        if not self.advance and self.dt != 0:
            raise ValueError("observation-only steps require dt=0")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class TrainingRecord:
    """Graph-free history; state/observations are not retained automatically."""
    step_index: int
    start_time: float
    end_time: float
    requested_dt: float
    advanced: bool
    loss: float | None
    weight: float
    events: tuple[Any, ...] = ()
    effects: tuple[Any, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrainingTrajectory:
    """Persistent identity and private, model-native state across windows.

    Values are held by reference, not copied. Supply independent values for
    independent cells. Use the returned trajectory after each successful window;
    the input is a previous version, not an object updated by the runner.
    """
    id: Hashable
    state: Any
    time: float = 0.0
    step_index: int = 0
    history: tuple[TrainingRecord, ...] = ()

    def __post_init__(self):
        hash(self.id)
        object.__setattr__(self, "time", _finite(self.time, "time"))
        if isinstance(self.step_index, bool) or not isinstance(self.step_index, int) or self.step_index < 0:
            raise ValueError("step_index must be a nonnegative integer")
        object.__setattr__(self, "history", tuple(self.history))

    @classmethod
    def from_cell(cls, cell: DynamicalCell) -> "TrainingTrajectory":
        """Capture identity/state/time without copying; does not mutate the DynamicalCell.

        The caller must use this trajectory as the state owner while training,
        and must not concurrently advance or mutate the source cell.
        """
        if not isinstance(cell, DynamicalCell):
            raise TypeError("cell must be a DynamicalCell")
        return cls(cell.id, cell.state, cell.time)


@dataclass(frozen=True, slots=True)
class LossContext:
    trajectory_id: Hashable
    step_index: int
    context: StepContext
    transition: StepResult | None


@dataclass(frozen=True, slots=True)
class TrainingWindowResult:
    trajectories: tuple[TrainingTrajectory, ...]
    loss: float | None
    observations: int
    transitions: int
    optimizer_stepped: bool


class TrajectoryWindowRunner:
    """Run one bounded graph window across independently timed trajectories.

    ``model`` is a DynamicalModel or a module exposing ``as_model()`` (TorchModel or
    TorchDynamics). For parameter-mutation detection a DynamicalModel must expose its
    owning nn.Module as ``parameters``. ``max_steps`` bounds each cell's schedule,
    including observation-only steps. Reduction is ``sum``, weighted ``mean``,
    or ``mean_trajectories`` (mean of each observed cell's weighted mean).

    The loss callback takes ``(prediction, target, LossContext)`` and returns one
    real scalar Tensor. It may implement observation error or event likelihood.
    No discrete sampling estimator is supplied. Native/conditional operations
    belong inside the model callable and must supply their own valid autograd.

    Every call ends a graph window, even when ``step_optimizer=False``. Set
    ``zero_grad=False`` to accumulate already-completed windows; use ``loss_scale``
    to establish an explicit global normalization. The runner never clears grads
    implicitly after stepping. Optimizer closure/replay APIs are unsupported.

    Callbacks must not mutate input state, parameters, or optimizer. This runner
    is single-owner and not thread-safe; concurrent parameter updates are invalid.
    Callback/RNG side effects, partial gradients from failed backward, and partial
    optimizer updates cannot be rolled back. No trajectory result is returned on
    failure. See the guide for failure recovery and collective restrictions.
    """
    def __init__(self, model, *, max_steps: int, reduction: str = "mean",
                 detach: Callable[[Any], Any] = detach_state):
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if reduction not in ("sum", "mean", "mean_trajectories"):
            raise ValueError("unknown loss reduction")
        if not callable(detach):
            raise TypeError("detach must be callable")
        if isinstance(model, DynamicalModel):
            bound = model
        else:
            as_model = getattr(model, "as_model", None)
            if not callable(as_model):
                raise TypeError("model must be DynamicalModel or provide as_model()")
            bound = as_model()
        if not isinstance(bound, DynamicalModel) or not isinstance(bound.parameters, nn.Module):
            raise TypeError("training DynamicalModel.parameters must be its owning torch.nn.Module")
        self.model, self.module = bound, bound.parameters
        self.max_steps, self.reduction, self.detach = max_steps, reduction, detach
        self._running = False

    def _versions(self):
        # Tensor version counters detect ordinary in-place optimizer updates.
        # .data writes and external native writes violate the caller contract.
        return tuple((id(p), p._version) for p in self.module.parameters())

    def run_window(self, trajectories: Sequence[TrainingTrajectory],
                   schedules: Mapping[Hashable, Sequence[TrainingStep]],
                   loss_fn: Callable[[Any, Any, LossContext], Tensor], *,
                   optimizer: torch.optim.Optimizer | None = None,
                   zero_grad: bool = True, step_optimizer: bool = True,
                   loss_scale: float = 1.0) -> TrainingWindowResult:
        """Complete forward/backward/detachment before an optional optimizer step.

        Missing/empty schedules leave time, history and step count unchanged;
        their state still crosses the explicit detachment boundary. Empty windows
        do not clear gradients or step an optimizer. ``optimizer=None`` permits
        backward-only accumulation and requires ``step_optimizer=False``.
        """
        if self._running:
            raise RuntimeError("a trajectory window is already running")
        if not callable(loss_fn):
            raise TypeError("loss_fn must be callable")
        if optimizer is None and step_optimizer:
            raise ValueError("step_optimizer=True requires a caller-owned optimizer")
        if optimizer is not None and not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch.optim.Optimizer")
        scale = _finite(loss_scale, "loss_scale", nonnegative=True)
        items = tuple(trajectories)
        if not all(isinstance(item, TrainingTrajectory) for item in items):
            raise TypeError("trajectories must contain TrainingTrajectory values")
        identities = {item.id for item in items}
        if len(identities) != len(items):
            raise ValueError("trajectory identities must be unique")
        if not isinstance(schedules, Mapping) or set(schedules) - identities:
            raise ValueError("schedules must map known trajectory identities")
        batches = []
        for item in items:
            steps = tuple(islice(iter(schedules.get(item.id, ())), self.max_steps + 1))
            if len(steps) > self.max_steps:
                raise ValueError("schedule exceeds max_steps; split it into explicit windows")
            if not all(isinstance(step, TrainingStep) for step in steps):
                raise TypeError("schedules must contain TrainingStep values")
            batches.append(steps)
        self._running = True
        try:
            return self._run(items, batches, loss_fn, optimizer, zero_grad, step_optimizer, scale)
        finally:
            self._running = False

    def _run(self, items, batches, loss_fn, optimizer, zero_grad, step_optimizer, scale):
        versions = self._versions()
        pending, cell_losses, weights = [], [], []
        observations = transitions = 0
        # No no_grad(), as_tensor(), detach(), clone(), or packing occurs in the
        # numerical path. The supplied operations retain their original graphs.
        for item, steps in zip(items, batches):
            state, time, records = item.state, item.time, []
            terms, total_weight = [], 0.0
            for offset, step in enumerate(steps):
                context = self.model.context(time=time, dt=step.dt, inputs=step.inputs,
                                             rng=step.rng, metadata=step.metadata)
                transition = self.model.step(state, context) if step.advance else None
                end_time = time
                if transition is not None:
                    end_time = time + step.dt if transition.time is None else transition.time
                    if not math.isfinite(end_time) or end_time < time:
                        raise ValueError("training transitions must have finite nondecreasing time")
                    state = transition.state
                    transitions += 1
                scalar_loss = None
                if step.target is not NO_TARGET and step.weight > 0:
                    observed_context = replace(context, time=end_time)
                    prediction = self.model.observe(state, observed_context)
                    loss = loss_fn(prediction, step.target, LossContext(
                        item.id, item.step_index + offset, observed_context, transition))
                    if not isinstance(loss, Tensor) or loss.ndim != 0 or not loss.is_floating_point():
                        raise TypeError("loss callback must return a real scalar floating Tensor")
                    if not torch.isfinite(loss).item():
                        raise ValueError("loss callback returned a nonfinite value")
                    scalar_loss = float(loss.detach())
                    terms.append(loss * step.weight)
                    total_weight += step.weight
                    observations += 1
                records.append(TrainingRecord(
                    item.step_index + offset, time, end_time, step.dt, step.advance,
                    scalar_loss, step.weight,
                    () if transition is None else transition.events,
                    () if transition is None else transition.effects,
                    {} if transition is None else transition.metadata))
                time = end_time
            if terms:
                cell_losses.append(sum(terms))
                weights.append(total_weight)
            pending.append(replace(item, state=state, time=time,
                                   step_index=item.step_index + len(steps),
                                   history=item.history + tuple(records)))
        if self._versions() != versions:
            raise RuntimeError("model parameters changed during the graph window")
        objective = None
        if cell_losses:
            if self.reduction == "mean_trajectories":
                objective = sum(loss / weight for loss, weight in zip(cell_losses, weights)) / len(cell_losses)
            else:
                objective = sum(cell_losses)
                if self.reduction == "mean":
                    objective = objective / sum(weights)
            objective = objective * scale
            if not torch.isfinite(objective).item():
                raise ValueError("reduced loss is nonfinite")
            if not objective.requires_grad:
                raise ValueError("objective has no autograd graph; supply differentiable dynamics/loss")
        # Prepare boundary values before mutating gradients or parameters. Custom
        # policies must be pure and detach every graph-bearing retained leaf.
        detached = tuple(replace(item, state=self.detach(item.state),
                                 history=tuple(self.detach(record) for record in item.history))
                         for item in pending)
        if objective is not None:
            if zero_grad:
                if optimizer is None:
                    self.module.zero_grad(set_to_none=True)
                else:
                    optimizer.zero_grad(set_to_none=True)
            objective.backward()
            if self._versions() != versions:
                raise RuntimeError("model parameters changed before backward completed")
            if step_optimizer:
                optimizer.step()
        return TrainingWindowResult(detached,
                                    None if objective is None else float(objective.detach()),
                                    observations, transitions,
                                    objective is not None and step_optimizer)


__all__ = ["NO_TARGET", "detach_state", "TrainingStep", "TrainingRecord",
           "TrainingTrajectory", "LossContext", "TrainingWindowResult", "TrajectoryWindowRunner"]
