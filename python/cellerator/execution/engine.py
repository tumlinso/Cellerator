"""Execute exact selected work against revisioned, pinned parameters.

The CPU callback path is backend-neutral. Distributed Torch banks are collective
phases: every rank must launch the same collective sequence, and a forward
completion is not permission to release parameters needed by backward.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Callable, Mapping
import threading
from typing import Any

from .planning import ExecutionPlan, ExpertWorklist
from .residency import ParameterKey, ResidencyManager


class StaleRevisionError(RuntimeError):
    """An attempted update or execution names a different logical revision."""


@dataclass(frozen=True, slots=True)
class ParameterRevision:
    key: ParameterKey
    parameters: Any
    nbytes: int
    source: Any = "host"
    master_parameters: Any = None
    gradients: Any = None
    optimizer_state: Any = None


class ParameterStore:
    """Logical immutable versions above a separate physical residency manager.

    All payloads are opaque owned references. Callers must never mutate published
    buffers; an optimizer callback allocates a new revision. Master parameters,
    gradients and optimizer state stay distinct from execution parameters. Their
    own allocations are backend-owned; nbytes counts execution parameters only.
    This is a Python reference counterpart of the native ParameterStore, not a
    binding to it or a replacement for a Torch optimizer.
    """
    def __init__(self, residency: ResidencyManager):
        self.residency = residency
        self._records = {}
        self._current = {}
        self._used = set()
        self._lock = threading.RLock()

    def register(self, revision: ParameterRevision) -> None:
        if not isinstance(revision, ParameterRevision):
            raise TypeError("expected ParameterRevision")
        with self._lock:
            if revision.key.expert_id in self._current:
                raise ValueError("expert already registered; publish an update")
            self._insert(revision)

    def _insert(self, revision):
        if not isinstance(revision, ParameterRevision):
            raise TypeError("expected ParameterRevision")
        if revision.key in self._used:
            raise ValueError("parameter revision identity cannot be reused")
        self.residency.register(revision.key, revision.parameters,
                                nbytes=revision.nbytes, source=revision.source)
        self._records[revision.key] = revision
        self._used.add(revision.key)
        self._current[revision.key.expert_id] = revision.key

    def current(self, expert_id) -> ParameterKey:
        with self._lock:
            return self._current[expert_id]

    def snapshot(self, keys) -> tuple[ParameterRevision, ...]:
        with self._lock:
            return tuple(self._records[key] for key in keys)

    def publish(self, expected: ParameterKey, revision: ParameterRevision) -> None:
        if not isinstance(revision, ParameterRevision):
            raise TypeError("expected ParameterRevision")
        with self._lock:
            if self._current.get(expected.expert_id) != expected:
                raise StaleRevisionError("parameter update base is no longer current")
            if revision.key.expert_id != expected.expert_id or revision.key == expected:
                raise ValueError("publish needs a new revision of the same expert")
            self._insert(revision)

    def update(self, expected: ParameterKey, backend: Callable) -> ParameterRevision:
        """backend(old_revision) returns newly allocated ParameterRevision."""
        with self._lock:
            if self._current.get(expected.expert_id) != expected:
                raise StaleRevisionError("parameter update base is no longer current")
            old = self._records[expected]
        result = backend(old)
        self.publish(expected, result)
        return result

    def prune(self, expert_id) -> tuple[ParameterKey, ...]:
        """Remove old registry entries only if no execution/transfer pins remain."""
        with self._lock:
            dropped = self.residency.invalidate(
                expert_id, keep_revision=self._current[expert_id].revision)
            for key in dropped:
                self._records.pop(key, None)
            return dropped

    def _pin(self, keys):
        leases = []
        with self._lock:
            records = self.snapshot(keys)
            try:
                for record in records:
                    leases.append(self.residency.acquire(record.key, record.source))
            except BaseException:
                for lease in leases:
                    lease.release()
                raise
        return records, leases


@dataclass(frozen=True, slots=True)
class BoundExecution:
    """Plan bound to an immutable worklist and caller-defined structure revision."""
    plan: ExecutionPlan
    work: ExpertWorklist
    work_revision: Any


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    decision_index: int
    decision: Any
    selections: tuple
    selection_indices: tuple[int, ...]
    parameters: Any
    inputs: Any


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    # One unweighted output per original selected occurrence. A caller combines
    # using the original weights and explicit domain-specific reduction.
    outputs: tuple
    selections: tuple
    completed_decisions: tuple[int, ...]


class PlanExecutionError(RuntimeError):
    def __init__(self, decision_index, partial):
        self.decision_index = decision_index
        self.partial = partial
        super().__init__(f"execution failed at decision {decision_index}; no work was replayed")


def _validate(bound):
    if not isinstance(bound, BoundExecution):
        raise TypeError("execution requires a BoundExecution")
    # Reject malformed manually constructed metadata, including permuted groups.
    canonical = ExpertWorklist.from_selections(bound.work.selections)
    if canonical != bound.work:
        raise ValueError("worklist is not its canonical immutable expert grouping")
    grouped = {}
    for index, selected in enumerate(bound.work.selections):
        grouped.setdefault((selected.key, selected.origin), []).append(index)
    groups = []
    seen = set()
    for index, decision in enumerate(bound.plan.decisions):
        identity = decision.key, decision.origin
        if identity in seen or identity not in grouped:
            raise ValueError("plan duplicates or substitutes selected work")
        seen.add(identity)
        indices = tuple(grouped[identity])
        demand = decision.demand
        if demand.uses != len(indices) or demand.work_ids != tuple(
                bound.work.selections[i].row_id for i in indices):
            raise ValueError("plan work identities/order do not match selected occurrences")
        if not decision.chosen.feasible:
            raise ValueError("cannot execute an infeasible placement candidate")
        if any(dep < 0 or dep >= index for dep in decision.chosen.dependencies):
            raise ValueError("plan dependencies must refer to preceding decisions")
        groups.append(indices)
    if seen != set(grouped):
        raise ValueError("plan omits selected work")
    return tuple(groups)


class CollectiveExecution:
    """Explicit lease scope through collective forward AND optional backward.

    close() requires the forward future to finish. The caller additionally
    guarantees backward and device work have completed. No destructor silently
    drops these pins; a forgotten close is a visible retained resource.
    """
    def __init__(self, future, leases):
        self.future = future
        self._leases = leases

    def result(self, timeout=None):
        return self.future.result(timeout)

    def close(self):
        if not self.future.done():
            raise RuntimeError("collective execution is still in flight")
        for lease in self._leases:
            lease.release()
        self._leases.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            self.future.result()
        finally:
            self.close()


class ExecutionEngine:
    """Sequential placement-plan executor with asynchronous backend completion.

    backend(ExecutionRequest) returns an iterable of occurrence outputs or a
    Future resolving to that iterable only after all parameter reads finish.
    It owns activation transfer/allocation and numerical execution; the engine
    owns exact revision checks, parameter movement and pins. A raw return means
    execution is complete, not merely enqueued on a device stream.
    """
    def __init__(self, store: ParameterStore, backend: Callable, *, executor=None):
        if not callable(backend):
            raise TypeError("backend must be callable")
        self.store = store
        self.backend = backend
        self._owned_executor = executor is None
        self._executor = ThreadPoolExecutor(max_workers=1) if executor is None else executor

    def submit(self, bound: BoundExecution, inputs: Any, *, work_revision: Any) -> Future:
        groups = _validate(bound)
        if bound.work_revision != work_revision:
            raise StaleRevisionError("selected-work structure revision has changed")
        for decision in bound.plan.decisions:
            if decision.execution_device not in self.store.residency.capacities:
                raise ValueError("planned execution device has no managed residency budget")
        _, pins = self.store._pin(bound.work.expert_keys)
        try:
            future = self._executor.submit(self._run, bound, groups, inputs, pins)
        except BaseException:
            for pin in pins:
                pin.release()
            raise
        # Also runs for a queued future cancelled before its worker starts.
        def cancelled(done):
            if done.cancelled():
                for pin in pins:
                    pin.release()
                pins.clear()
        future.add_done_callback(cancelled)
        return future

    def _run(self, bound, groups, inputs, pins):
        try:
            return self._execute(bound, groups, inputs, pins)
        finally:
            for pin in pins:
                pin.release()
            pins.clear()

    def _execute(self, bound, groups, inputs, pins):
        outputs = [None] * len(bound.work.selections)
        completed = []
        for index, (decision, indices) in enumerate(zip(bound.plan.decisions, groups)):
            lease = None
            try:
                lease = self.store.residency.acquire(
                    decision.key, decision.execution_device,
                    source=decision.chosen.parameter_source)
                parameters = lease.result()
                request = ExecutionRequest(index, decision,
                    tuple(bound.work.selections[i] for i in indices), indices,
                    parameters, inputs)
                values = self.backend(request)
                if isinstance(values, Future):
                    values = values.result()
                values = tuple(values)
                if len(values) != len(indices):
                    raise ValueError("backend output count differs from selected occurrences")
                for occurrence, value in zip(indices, values):
                    outputs[occurrence] = value
                completed.append(index)
                # A replica producer is a resource dependency of later work.
                if any(index in later.chosen.dependencies for later in bound.plan.decisions[index + 1:]):
                    pins.append(lease)
                    lease = None
            except BaseException as exc:
                partial = ExecutionResult(tuple(outputs), bound.work.selections, tuple(completed))
                raise PlanExecutionError(index, partial) from exc
            finally:
                if lease is not None:
                    lease.release()
        return ExecutionResult(tuple(outputs), bound.work.selections, tuple(completed))

    def submit_collective(self, keys, device, launch: Callable, inputs=None) -> CollectiveExecution:
        """Explicit SPMD phase, called in matching order on every process rank.

        launch(parameter_mapping, inputs) must bind these exact revision payloads
        to its collective bank and return its output or completion Future. Never
        invoke a collective bank as an independent per-decision backend. Pins
        survive forward until ticket.close(), including autograd backward.
        """
        keys = tuple(keys)
        if len(set(keys)) != len(keys):
            raise ValueError("collective parameter keys must be unique")
        _, pins = self.store._pin(keys)
        target_pins = []
        try:
            for key in keys:
                target_pins.append(self.store.residency.acquire(key, device))
            payloads = MappingProxyType({key: pin.result() for key, pin in zip(keys, target_pins)})
            result = launch(payloads, inputs)
            if not isinstance(result, Future):
                future = Future()
                future.set_result(result)
            else:
                future = result
            return CollectiveExecution(future, pins + target_pins)
        except BaseException:
            for pin in pins + target_pins:
                pin.release()
            raise

    def close(self):
        if self._owned_executor:
            self._executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class TorchCollectiveBinding:
    """Bind exact resident state dictionaries to a distributed Torch bank.

    key_to_expert maps each local ParameterKey to its bank expert integer.
    Revision payloads are complete module state mappings (parameters + buffers).
    torch.func.functional_call preserves autograd to these supplied tensors.
    Invoke only via a coordinated collective phase; do not overlap calls using
    the same bank, and retain the CollectiveExecution through backward.
    """
    def __init__(self, bank, key_to_expert: Mapping, routing):
        self.bank = bank
        self.key_to_expert = MappingProxyType(dict(key_to_expert))
        self.routing = routing
        experts = tuple(str(value) for value in self.key_to_expert.values())
        if len(set(experts)) != len(experts) or set(experts) != set(bank.local_experts):
            raise ValueError("binding must name each local bank expert exactly once")

    def __call__(self, parameters: Mapping, inputs):
        from torch.func import functional_call
        if set(parameters) != set(self.key_to_expert):
            raise StaleRevisionError("collective payload keys differ from bound expert revisions")
        state = {}
        for key, expert in self.key_to_expert.items():
            payload = parameters[key]
            if not isinstance(payload, Mapping):
                raise TypeError("Torch parameter payload must be a module state mapping")
            prefix = f"local_experts.{expert}."
            state.update((prefix + name, value) for name, value in payload.items())
        return functional_call(self.bank, state, (inputs, self.routing), strict=True)
