

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-41: Prebound projection view v2

Task revision: `2379`; current project revision is in `todo-status.md`.

## Objective
Expose validated capability pointer and bytes through a compatible v2 prebound view while retaining v1 layout, functions, and readers.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `serial`
- Result: `-`

## Next Action
_None._

## Ownership
- `exclusive`: `include/Cellerator/geometry/persistence/execution_image_v2.hh`
- `exclusive`: `src/geometry/persistence/execution_image_v2.cc`
- `exclusive`: `tests/geometry/ce_geo/prebound_projection_v2_test.cc`
- `read`: `tests/execution/projection_activation_test.cu`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
