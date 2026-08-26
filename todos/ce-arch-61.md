

<!-- todo-orchestrator:v2-managed:start -->
# CE-ARCH-61: Legacy CP-Math interface retirement checkpoint

Task revision: `780`; current project revision is in `todo-status.md`.

## Objective
Version the retirement of obsolete CP-Math runtime interfaces and refresh operation-core documentation identity without restoring deleted implementation.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `validated`

## Next Action
Complete: legacy CP-Math backend/runtime interfaces are versioned as retired evidence and the operation-core documentation identity is refreshed.

## Ownership
- `exclusive`: `Cellerator/include/Cellerator/compute/math/MATH_V1_EVIDENCE.md`
- `exclusive`: `Cellerator/include/Cellerator/compute/math/execution_plan.hh`
- `exclusive`: `Cellerator/include/Cellerator/compute/math/operation_core/OPERATION_CORE.md`
- `forbidden`: `Baseplane`
- `forbidden`: `CellShard`

## Dependencies
- `task`: `CE-ARCH-60`
<!-- todo-orchestrator:v2-managed:end -->
