

<!-- todo-orchestrator:v2-managed:start -->
# CE-REMAP-07: Make Baseplane and CellShard interoperability explicit

Task revision: `2327`; current project revision is in `todo-status.md`.

## Objective
Move Cellerator-owned Baseplane and CellShard seams into narrow conventional interop homes; document module and GlassHelix deferral rather than freezing speculation.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `integration_exclusive`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `CMakeLists.txt`
- `exclusive`: `docs/REPOSITORY_CONSOLIDATION_PROGRAM.md`
- `exclusive`: `include/Cellerator/compute/sequence`
- `exclusive`: `include/Cellerator/interop`
- `exclusive`: `src/compute/sequence`
- `exclusive`: `src/interop`
- `exclusive`: `src/interop/CMakeLists.txt`
- `exclusive`: `tests/CMakeLists.txt`
- `exclusive`: `tests/cellshard_access_adapter_compile_test.cc`
- `exclusive`: `tests/interop`
- `exclusive`: `tests/sequence`
- `read`: `bench`
- `read`: `components`
- `read`: `docs`
- `read`: `include`
- `read`: `src`
- `read`: `tests`

## Dependencies
- `task`: `CE-REMAP-06`
<!-- todo-orchestrator:v2-managed:end -->
