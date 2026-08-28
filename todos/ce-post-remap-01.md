

<!-- todo-orchestrator:v2-managed:start -->
# CE-POST-REMAP-01: Eliminate the CellShard cycle and finish runtime and seam cleanup

Task revision: `2358`; current project revision is in `todo-status.md`.

## Objective
Consume the independently landed CellShard neutral distributed API, remove the Cellerator-to-CellShard-to-Cellerator runtime cycle, classify every legacy runtime consumer, migrate current core consumers that require only bounded adaptation, and keep genuine compatibility users explicitly scoped.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `project_exclusive`
- Result: `implemented`

## Next Action
Verify the landed CellShard neutral API, classify all runtime and CellShard consumers, then perform only bounded current-core migrations.

## Ownership
- `exclusive`: `ARCHITECTURE_FOLLOWUPS.md`
- `exclusive`: `CMakeLists.txt`
- `exclusive`: `bench`
- `exclusive`: `cellerator-post-remap-plan.json`
- `exclusive`: `components/CelleraTorch`
- `exclusive`: `docs/REPOSITORY_CONSOLIDATION_PROGRAM.md`
- `exclusive`: `examples`
- `exclusive`: `include/Cellerator/compute`
- `exclusive`: `include/Cellerator/interop/cellshard`
- `exclusive`: `include/Cellerator/preprocess`
- `exclusive`: `include/Cellerator/runtime`
- `exclusive`: `src/abi`
- `exclusive`: `src/compute`
- `exclusive`: `src/interop/cellshard`
- `exclusive`: `src/preprocess`
- `exclusive`: `src/runtime`
- `exclusive`: `tests`
- `read`: `AGENTS.md`
- `read`: `cmake`
- `read`: `compat`
- `read`: `docs`
- `read`: `modules`
- `read`: `scope.md`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
