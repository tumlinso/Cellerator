

<!-- todo-orchestrator:v2-managed:start -->
# CE-REMAP-06: Collapse runtime and distributed ownership

Task revision: `2366`; current project revision is in `todo-status.md`.

## Objective
Keep one canonical runtime, move GPU/fleet/collective resources under runtime, move hierarchy and communication policy under planner/distributed, and retire duplicate compute/runtime and standalone distributed facades.

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
- `exclusive`: `include/Cellerator/compute/runtime.hh`
- `exclusive`: `include/Cellerator/dist`
- `exclusive`: `include/Cellerator/distributed`
- `exclusive`: `include/Cellerator/planner`
- `exclusive`: `include/Cellerator/runtime`
- `exclusive`: `src/compute/runtime`
- `exclusive`: `src/distributed`
- `exclusive`: `src/planner`
- `exclusive`: `src/planner/CMakeLists.txt`
- `exclusive`: `src/runtime`
- `exclusive`: `src/runtime/CMakeLists.txt`
- `exclusive`: `tests/CMakeLists.txt`
- `exclusive`: `tests/distributed`
- `exclusive`: `tests/planner`
- `exclusive`: `tests/runtime`
- `read`: `bench`
- `read`: `components`
- `read`: `docs`
- `read`: `include`
- `read`: `src`
- `read`: `tests`

## Dependencies
- `task`: `CE-REMAP-05`
<!-- todo-orchestrator:v2-managed:end -->
