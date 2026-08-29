

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-30: Work-window contract

Task revision: `2435`; current project revision is in `todo-status.md`.

## Objective
Implement bounded axis-bound work windows for relation rows, dense columns, and grouped operation instances; the caller chooses membership.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `include/Cellerator/geometry/work_window.hh`
- `exclusive`: `tests/geometry/ce_geo/work_window_test.cc`
- `read`: `include/Cellerator/execution`

## Dependencies
- `checkpoint`: `CE-GEO-ARCHITECTURE-FROZEN`
<!-- todo-orchestrator:v2-managed:end -->
