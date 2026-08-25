# Architecture follow-ups

## Remove the CellShard-to-Cellerator runtime dependency

Cellerator is the primary compute and execution layer. The intended dependency
direction is `Cellerator -> CellShard`: Cellerator may consume CellShard storage,
layout, export, and device-view interfaces, but CellShard must not depend on
Cellerator compute, distributed-runtime, stream, scheduler, or collective types.

The current CellShard distributed header includes
`Cellerator/dist/distributed.cuh` and accepts
`cellerator::dist::local_context`. This creates the cycle
`Cellerator -> CellShard -> Cellerator` and causes Cellerator NCCL/runtime defects
to surface while CellShard is being compiled.

Future work should move the Cellerator-specific shard scheduling and collective
adapter into Cellerator's CellShard interop layer. CellShard should expose only
storage-owned shard metadata and narrow neutral bindings such as explicit device
IDs, streams, and shard assignments. Preserve CellShard's storage ownership and
do not duplicate Cellerator compute or collective policy inside CellShard.

This is a deferred architectural follow-up, not part of CE-ARCH-71 through
CE-ARCH-74.
