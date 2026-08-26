<!-- todo-orchestrator:v2-managed:start -->
# CE-ARCH-70B: Fix persistent execution-session allocations

Task revision: `921`; current project revision is in `todo-status.md`.

## Objective
Replace monolithic persistent scratch ownership with fixed-capacity independent stable CUDA allocations while preserving the pre-reserved transient arena and allocation-free sealed launch binding.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
Implement independent recorded persistent allocations, accounting and exhaustion checks, pointer alignment validation, focused CUDA correctness, and Compute Sanitizer if available; do not redesign transient workspace.

## Ownership
- `exclusive`: `Cellerator/include/Cellerator/execution/launch_bindings.hh`
- `exclusive`: `Cellerator/include/Cellerator/runtime`
- `exclusive`: `Cellerator/src/runtime`
- `exclusive`: `Cellerator/tests/runtime/execution_session_test.cu`
- `forbidden`: `Baseplane`
- `forbidden`: `CellShard`

## Dependencies
- `task`: `CE-ARCH-70A`
<!-- todo-orchestrator:v2-managed:end -->
