

<!-- todo-orchestrator:v2-managed:start -->
# CE-PTR-12: Forward-neighbor architectural decomposition

Task revision: `2376`; current project revision is in `todo-status.md`.

## Objective
Split forward-neighbor low-level mathematical kernels, views, routes, workspaces, and fixed-K primitives from downstream index construction, durable ownership, cell lookup, sharding, residency, storage, biological workflow policy, and application results.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `parallel_safe`
- Result: `-`

## Next Action
Determine the live architectural split first, then migrate or remove Cellerator ownership only where the standing CellShard and BioPrep boundaries require it.

## Ownership
- `exclusive`: `include/Cellerator/compute/neighbors/forward_neighbors`
- `exclusive`: `src/compute/neighbors/forward_neighbors`
- `read`: `bench`
- `read`: `include/Cellerator/compute/preprocess`
- `read`: `include/Cellerator/interop/cellshard`
- `read`: `src/interop/cellshard`
- `read`: `src/preprocess`
- `read`: `tests`

## Dependencies
- `checkpoint`: `CE-PTR-SUBSTRATE-CONTRACT-READY`
<!-- todo-orchestrator:v2-managed:end -->
