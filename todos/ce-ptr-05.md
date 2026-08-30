

<!-- todo-orchestrator:v2-managed:start -->
# CE-PTR-05: Packing optimizer state representation

Task revision: `2494`; current project revision is in `todo-status.md`.

## Objective
Replace per-block growable member and cache ownership with a prepared domain representation using bounded fixed-stride slabs where justified, aligned descriptors, direct feature-slot maps, explicit generations, prepared union-cache storage, exact workspaces, and local updates.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
Freeze and validate the optimizer state contract needed by CE-PTR-06 without coupling unrelated workstreams.

## Ownership
- `exclusive`: `src/geometry/optimizer.cc`
- `exclusive`: `src/geometry/optimizer_state.hh`
- `read`: `bench`
- `read`: `include/Cellerator/geometry`
- `read`: `src/geometry/merge_cost.cc`
- `read`: `tests`

## Dependencies
- `checkpoint`: `CE-PTR-SUBSTRATE-CONTRACT-READY`
<!-- todo-orchestrator:v2-managed:end -->
