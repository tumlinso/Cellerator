

<!-- todo-orchestrator:v2-managed:start -->
# CE-AMP-02: sm_86 physical realization

Task revision: `3281`; current project revision is in `todo-status.md`.

## Objective
Realize the same portable CSG1 semantic geometry as an independent Ampere physical cover, schedules, value maps, capabilities, and CPE2 projection.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `serial`
- Result: `-`

## Next Action
_None._

## Ownership
- `exclusive`: `src/compute/architecture/providers/nvidia/sm86/physical_realization.cc`
- `exclusive`: `tests/tensor_core/sm86/physical_realization_test.cc`
- `read`: `include/Cellerator/compute/projection/physical_mma_hybrid.hh`
- `read`: `include/Cellerator/geometry/persistence/semantic_geometry_image_v1.hh`

## Dependencies
- `task`: `CE-AMP-01`
- `checkpoint`: `CE-EXOP-COMPLETE`
- `decision`: `CE-AMP-PERMISSION`
<!-- todo-orchestrator:v2-managed:end -->
