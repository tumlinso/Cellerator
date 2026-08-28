# Architecture follow-ups

## Resolved: CellShard-to-Cellerator runtime dependency

Cellerator is the primary compute and execution layer. The intended dependency
direction is `Cellerator -> CellShard`: Cellerator may consume CellShard storage,
layout, export, and device-view interfaces, but CellShard must not depend on
Cellerator compute, distributed-runtime, stream, scheduler, or collective types.

Resolved by the bounded post-remap completion pass. CellShard distributed APIs
now accept neutral borrowed device IDs and CUDA stream handles through a
CellShard-owned binding view. CellShard no longer includes Cellerator runtime or
distributed headers, names Cellerator runtime types, locates `Cellerator::dist`,
or links a Cellerator runtime target.

Cellerator retains execution topology, stream/resource ownership, NCCL policy,
and planner decisions. The Cellerator caller translates its local execution
context into the neutral CellShard binding at the interop call site. CellShard
retains storage, residency, transport, and generic shard assignment.

The remaining Cellerator compatibility-format sources embedded by CellShard do
not link or expose Cellerator runtime machinery; they are retained only for
historical matrix-format compatibility.
