

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-63: Physical work layout and padding

Task revision: `3281`; current project revision is in `todo-status.md`.

## Objective
Implement provider-specific work layouts with invalid-sentinel padding that never enters semantic work identity.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `src/compute/projection/mma_physical_work_layout.cc`
- `exclusive`: `tests/ce_geo/projection/physical_padding_test.cc`
- `read`: `include/Cellerator/compute/projection/physical_mma_hybrid.hh`
- `read`: `include/Cellerator/geometry/work_layout.hh`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
