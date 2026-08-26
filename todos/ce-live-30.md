

<!-- todo-orchestrator:v2-managed:start -->
# CE-LIVE-30: Planner-backed executable program API

Task revision: `1852`; current project revision is in `todo-status.md`.

## Objective
Implement one host-side executable_program that enumerates legal activated candidates, prices complete costs, invokes the planner, reserves session-owned persistent state, prepares the winner, binds changing launch state, exposes output order and workspace requirements, and runs without creating a second runtime.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `docs/CE_LIVE_EXECUTABLE_PROGRAM.md`
- `exclusive`: `include/Cellerator/execution/program.hh`
- `exclusive`: `src/execution/program.cc`
- `exclusive`: `tests/execution/program_test.cu`
- `read`: `include/Cellerator/compute/math/operation_core`
- `read`: `include/Cellerator/planner`
- `read`: `include/Cellerator/runtime`

## Dependencies
- `task`: `CE-LIVE-22`
- `task`: `CE-LIVE-23`
- `task`: `CE-LIVE-24`
- `task`: `CE-LIVE-25`
- `task`: `CE-LIVE-26`
- `task`: `CE-LIVE-29`
<!-- todo-orchestrator:v2-managed:end -->
