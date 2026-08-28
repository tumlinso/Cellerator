

<!-- todo-orchestrator:v2-managed:start -->
# CE-REMAP-03: Foundational state and execution border inventory

Task revision: `2286`; current project revision is in `todo-status.md`.

## Objective
Inventory and tighten the existing conventional state and execution borders without changing ABI; document the future cellerator.state and cellerator.execution exports but do not add .ccm consumers until native CMake scanning is available.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `integration_exclusive`
- Result: `validated`

## Next Action
_None._

## Ownership
- `exclusive`: `CMakeLists.txt`
- `exclusive`: `docs/REPOSITORY_CONSOLIDATION_PROGRAM.md`
- `exclusive`: `include/Cellerator/execution`
- `exclusive`: `modules/CMakeLists.txt`
- `exclusive`: `modules/execution.ccm`
- `exclusive`: `modules/state.ccm`
- `exclusive`: `tests/modules`
- `read`: `docs`
- `read`: `src/execution`
- `read`: `src/runtime`
- `read`: `tests/execution`
- `read`: `tests/runtime`

## Dependencies
- `task`: `CE-REMAP-02`
<!-- todo-orchestrator:v2-managed:end -->
