<!-- todo-orchestrator:v2-managed:start -->
# CE-ARCH-12: Unified Cellerator execution session and runtime ownership

Task revision: `780`; current project revision is in `todo-status.md`.

## Objective
Consolidate runtime ownership so CP-Math and biological operations use one explicit Cellerator execution substrate.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
Extend existing Cellerator::runtime into one session substrate; migrate rather than wrap it with a second context.

## Ownership
- `exclusive`: `Cellerator/include/Cellerator/runtime`
- `exclusive`: `Cellerator/src/runtime`
- `exclusive`: `Cellerator/tests/runtime`
- `forbidden`: `Baseplane`
- `forbidden`: `CellShard`
- `read`: `Cellerator/include/Cellerator/compute/math/runtime.hh`
- `read`: `Cellerator/src/compute/math/backend_registry.cc`
- `read`: `Cellerator/src/compute/math/runtime`

## Dependencies
- `task`: `CE-ARCH-11`
<!-- todo-orchestrator:v2-managed:end -->
