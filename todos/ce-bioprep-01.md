

<!-- todo-orchestrator:v2-managed:start -->
# CE-BIOPREP-01: Extract preprocessing ownership into BioPrep

Task revision: `2370`; current project revision is in `todo-status.md`.

## Objective
Genericize useful sparse row transforms and column moments, remove the preprocessing subsystem and Python ownership, and preserve all unrelated frozen architecture.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `project_exclusive`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `.ctxpp.toml`
- `exclusive`: `CMakeLists.txt`
- `exclusive`: `README.md`
- `exclusive`: `bench`
- `exclusive`: `compat/CMakeLists.txt`
- `exclusive`: `docs`
- `exclusive`: `include/Cellerator`
- `exclusive`: `python`
- `exclusive`: `scope.md`
- `exclusive`: `src`
- `exclusive`: `tests`
- `exclusive`: `tools/CMakeLists.txt`
- `forbidden`: `components/CelleraTorch`
- `forbidden`: `include/CellShard`
- `forbidden`: `src/model`
- `forbidden`: `src/trajectory`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
