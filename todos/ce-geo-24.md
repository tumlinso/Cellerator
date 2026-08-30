

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-24: Executable program v2 and v1 wrapper

Task revision: `3016`; current project revision is in `todo-status.md`.

## Objective
Create one v2 program engine without five-entry pointer closure or central projection switch and retain v1 as a compatibility wrapper.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `include/Cellerator/execution/program.hh`
- `exclusive`: `src/execution/program.cc`
- `exclusive`: `tests/ce_geo/catalog/program_v2_test.cu`
- `read`: `include/Cellerator/compute/operation`
- `read`: `src/compute/operation`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
