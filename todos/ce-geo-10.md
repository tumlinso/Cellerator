

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-10: Canonical cold device descriptor

Task revision: `3281`; current project revision is in `todo-status.md`.

## Objective
Implement runtime::device_descriptor_v1 as the single cold hardware truth, derive existing compatibility views, and prove no query after session sealing.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `include/Cellerator/runtime/device_descriptor.hh`
- `exclusive`: `src/runtime/device_descriptor.cu`
- `exclusive`: `tests/ce_geo/hardware/device_descriptor_test.cu`
- `read`: `include/Cellerator/execution`
- `read`: `include/Cellerator/planner`
- `read`: `include/Cellerator/runtime`
- `read`: `src/runtime`

## Dependencies
- `checkpoint`: `CE-GEO-ARCHITECTURE-FROZEN`
<!-- todo-orchestrator:v2-managed:end -->
