

<!-- todo-orchestrator:v2-managed:start -->
# CE-JBC-P01: Map CellShard atom planes to projection_value_plane_v1

Task revision: `3602`; current project revision is in `todo-status.md`.

## Objective
Map CellShard atom planes to projection_value_plane_v1. Deliver this as one isolated, reviewable step in the Cellerator atom planes, mutable values, gradients, and atom outputs workstream.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `serial`
- Result: `-`

## Next Action
_None._

## Ownership
- `exclusive`: `include/Cellerator/execution/atom_plane`
- `exclusive`: `include/Cellerator/execution/projection_value_plane`
- `exclusive`: `src/execution/atom_plane`
- `exclusive`: `src/execution/projection_value_plane`
- `exclusive`: `tests/jbc/atom_plane`
- `read`: `include/Cellerator/compute/operation/relation_algebra_v2/relation_algebra.hh`
- `read`: `include/Cellerator/execution/projection_value_plane/composite_plane_v1.hh`
- `read`: `include/Cellerator/execution/projection_value_plane/generation_publication_v1.hh`
- `read`: `include/Cellerator/execution/projection_value_plane/value_plane_v1.hh`

## Dependencies
- `task`: `CE-JBC-I12`
<!-- todo-orchestrator:v2-managed:end -->
