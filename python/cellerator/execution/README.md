# Local population execution

`cellerator.execution` is a backend-neutral, sequential reference executor. A
`Population` owns membership, communication and active context. Each `Cell` owns
its time and state reference; multiple cells can share one `Model` and its
parameters. State is never converted into a dense array or tensor.

```python
from cellerator.modeling import Cell, Model
from cellerator.execution import Population

def update(state, context):
    # Newly admitted occurrences are consumed once. Their values remain opaque.
    incoming = [record.value for record in context.inputs.incoming]
    return state + incoming

model = Model(update)
population = Population([Cell(model, [], id="a"), Cell(model, [], id="b")],
                        context_capacity=16)
population.send("a", "pulse", time=0.5)
population.advance(dt=1)   # Both synchronized cells stop at the 0.5 boundary.
population.advance(dt=1)   # The pulse is now admitted and consumed by a.
assert population["a"].state == ["pulse"]
assert population["b"].state == []
```

## Context, pending inputs and history

`context.inputs` is a `ContextInputs` sequence of retained values. Its `records`
carry occurrence sequence, delivery time and source; `incoming` contains newly
delivered occurrences still retained after admission. Per-step caller inputs
are separate: `population.advance(inputs={"a": value})` exposes
`context.inputs.external` without retaining the value.

Capacity counts occurrences, not bytes. The default `overflow="oldest"` ejects
the oldest records, including newly delivered records if necessary. Ejections
appear in `PopulationStep.ejections`. `overflow="reject"` leaves the pending
inbox intact on admission failure. `population.eject(id, count=...,
predicate=...)` explicitly removes active context; choose count or predicate.
`population.discard_pending(id, predicate=...)` separately removes queued inputs.
Both return the removed records, with ejection also carrying its reason.

An optional `EventHistory` records emitted `Event` objects independently of
context membership. Original events are retained without rewriting their
timestamps or provenance. `PopulationStep.emissions` supplies the actual emitter
identity and committed execution time separately. An optional `recorder(record)`
callback receives delivery, ejection, lifecycle and advancement records.
History, pending inboxes and caller recorders have separate retention policies;
they are not bounded by active-context capacity.

## Routing, lifecycle and time

Models return ordinary `StepResult`s. Targeted `Event`s and `Send`/`Broadcast`
effects route model-native values. Effects `Suspend`, `Resume`, `Create`,
`Remove`, `Fork` and `Eject` request explicit execution operations. Unknown
effects remain available in `PopulationStep.unhandled_effects` for the caller.
Untargeted events are recorded without routing. A target denotes one identity;
use `Broadcast` for multiple recipients.

All selected transitions finish before their events/effects are applied. The
default order is population insertion order; events precede effects within a
result. A supplied batch plan changes that order explicitly. Newly created cells
first execute in a later call. Lifecycle conflicts use this same order, and a
request targeting an already removed cell fails. Removed identities cannot be
reused, preventing stale messages from reaching a different individual.

Messages default to their emitter's committed time, and explicit timestamps
cannot precede it or the recipient's committed time. A future pending input
clips the next step to its delivery boundary. Cells starting at the same time
share the earliest such boundary. Inputs due at that boundary are admitted on
the following advance. Cells with different starting times remain independent
cohorts: callers must select coherent communication work, and routing into a
committed past fails explicitly. A model-selected result time is bounded before
Cell adoption; choosing an earlier result time can leave a synchronized cohort.

Suspension preserves state, model time, active context and pending inputs.
Sending does not wake a suspended cell. Resume does not fast-forward its model
time. Fork requires an explicit `inherit(parent.state)` function; callers choose
copying, partitioning or externally owned state. The model is shared by default;
context membership is inherited only with `inherit_context=True`, and pending
inputs are never inherited. Effect-driven forks require the parent and emitter
to have the same committed time.

## Work selection and failure

`population.plan(ids, group_by="model", batch_size=...)` groups selected work by
shared model identity. A callable `group_by(cell)` can select another resource
key. `LocalExecutor(population, ...)` applies these plans. Grouping does not
decide biological eligibility, suspension, or membership. Execution stays
sequential; there is no automatic vectorization, device scheduling or
distributed runtime.

Invalid selections and step sizes fail before execution. Callback, admission
and routing errors raise `ExecutionFailure`, whose `phase`, `cell_id`, `report`
and chained exception expose partial progress. Committed results/effects remain
visible; remaining outputs can be inspected in `report.results`. Arbitrary
mutable states and callbacks cannot be rolled back. No failed work is replayed
automatically. A population has one execution owner; advancing its cells
externally or concurrently is unsupported.

## Topology, sparse demand and parameter residency

Model-selected experts are represented by `ParameterKey(expert_id, revision)`.
`SelectedWork` and `ExpertWorklist.from_selections(...)` retain every selected
occurrence, its origin, row identity and weight. Expert-major offsets and a
permutation provide compact work without a padded cell-by-expert array. The
worklist validates exact revision grouping and preserves repeated selections.

`HardwareTopology` accepts explicit `DeviceSpec` and directed `LinkSpec` values.
It estimates message costs from bandwidth and latency, choosing a deterministic
message-size-dependent path. Hardware discovery and measured link performance
remain outside this API. Device identities can be CPU handles, rank identifiers
or other hashable values; no device-to-rank map is inferred.

`PlacementPlanner` consumes exact `ExpertDemand` records and ready
`ExpertPlacement` snapshots. It compares local computation, remote activation
and return-output movement, temporary weight movement, and persistent replica
proposals. Supplied compute-time estimates and explicit reuse hints affect
resource cost, not model expert selection. A missing exact revision fails with
`PlanningError`; an old or merely planned revision is never substituted.

```python
from cellerator.execution import (
    DeviceSpec, ExpertDemand, ExpertPlacement, ExpertSpec, HardwareTopology,
    LinkSpec, ParameterKey, PlacementPlanner, ResidencyManager,
)

key = ParameterKey("shared_rule", 1)
topology = HardwareTopology(
    [DeviceSpec("host", kind="host", memory_bytes=4096),
     DeviceSpec("cpu", memory_bytes=2048)],
    [LinkSpec("host", "cpu", bandwidth_bytes_per_second=1e9)],
)
planner = PlacementPlanner(topology, [ExpertSpec(key, nbytes=128)],
                           [ExpertPlacement(key, "host", executable=False)])
plan = planner.plan([ExpertDemand(key, "cpu", activation_bytes=4, output_bytes=4)],
                    repeats={(key, "cpu"): 8})

# A CPU reference callback performs an actual copy. A backend can instead return
# a concurrent.futures.Future for an asynchronous transfer.
residency = ResidencyManager({"cpu": 2048},
                            transfer=lambda payload, source, target: bytearray(payload))
residency.register(key, bytes(128), nbytes=128)
decision = plan.decisions[0]
lease = residency.acquire(key, decision.execution_device,
                          source=decision.chosen.parameter_source)
try:
    parameters = lease.result()
    assert len(parameters) == 128
    # Run caller-defined work and wait for dependent work to complete here.
finally:
    lease.release()
```

Plans retain every actual demand. `RollingDemandPredictor` supplies bounded EWMA
forecasts for advisory prefetch/reuse decisions; a forecast is never an expert
gate and cannot remove a cache miss from selected work. Ready replicas and model
demand are separate inputs. Replica proposals have `ready=False` and dependent
decisions identify their producer index; an executor must materialize each
dependency before using it. Planning never updates real residency.

`MemoryCategory` distinguishes parameters, transient activations, cell state and
optimizer state. A supplied baseline memory snapshot includes existing buffers;
planning additionally reserves full temporary I/O buffers, including returned
outputs at a remote demand's origin. Persistent replica bytes affect later
decisions; weight-movement buffers are temporary. Peak category figures are
conservative maxima over sequential decisions. Estimates exclude runtime
contention, overlap, allocator overhead and backend-specific relay/staging
requirements. Reserve those explicitly in baseline usage when required.

`ResidencyManager` retains caller-owned backing payloads through device eviction.
An asynchronous callback must return its `Future` promptly; the manager creates
no transfer threads and imports no GPU framework. In-flight transfers reserve
bytes. Demand leases pin their exact parameter revision through caller work;
explicit source replicas are pinned through copy completion as well. Pins and
pending transfers prevent eviction and revision invalidation. Prefetch is
unpinned advisory work, and may be evicted after completion before consumption.
LRU eviction considers only completed, unpinned, non-source copies. Backend
disposal errors are retained explicitly in `disposal_errors`.

Registering a new revision does not rewrite an old payload. `invalidate(expert)`
removes older revisions while retaining the last registered revision;
`keep_revision=...` chooses an exact retained revision. `purge(expert)` explicitly
removes every version. Both reject pinned or in-flight versions before mutation.
Callers remain responsible for preventing mutation of published payloads and
for releasing leases only after all dependent computation completes. Parameter
versions describe coherence, not automatic optimizer synchronization.

## Binding a plan to shared parameters and private state

`ParameterStore` registers canonical `ParameterRevision` objects above residency.
`BoundExecution(plan, work, work_revision)` binds a plan to exact selected
occurrences and their structure revision. `ExecutionEngine` validates that
binding, pins the exact parameter versions, and invokes a caller backend with
each `ExecutionRequest`. A backend can return outputs directly or a completion
`Future`; parameter pins survive until the dependent reads complete.

`reduce_execution_outputs(result)` restores a weighted sum per
`(selection.origin, selection.row_id)` in original selection order. Pass
`owner=lambda selected: selected.row_id` for globally unique cell identities,
or supply custom `scale` and `combine` callbacks for other value representations.
No dense conversion or weight normalization is implied, and tensor operators
retain their autograd graphs. Incomplete execution reports are rejected before
reduction. The caller applies each reduced value through its state updater;
sharing an expert's parameters never merges individual cell states.

The [scheduled experts example](../../../examples/framework/scheduled_experts.py)
connects private cell states, overlapping expert selection, placement, execution,
and caller-selected state updates. The [execution guide](../../../docs/distributed_execution.qmd)
describes version publication and optional collective execution. Collective
phases retain their pins through forward and backward until explicitly closed.
