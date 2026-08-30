

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-90: Target-specific transpose cover

Task revision: `2999`; current project revision is in `todo-status.md`.

## Objective
Build a separate target-specific forward/transpose physical geometry while preserving shared logical-edge identity.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `src/compute/architecture/providers/nvidia/sm70/transpose_cover.cc`
- `exclusive`: `tests/tensor_core/sm70/transpose_cover_test.cc`
- `read`: `include/Cellerator/compute/projection/physical_mma_hybrid.hh`
- `read`: `src/compute/projection`

## Dependencies
- `checkpoint`: `CE-GEO-SM70-N64-VERTICAL`
- `interface`: `cellerator-relation-algebra-v1`
<!-- todo-orchestrator:v2-managed:end -->
