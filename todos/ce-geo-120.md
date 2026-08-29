

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-120: Integrate geometry foundation

Task revision: `2379`; current project revision is in `todo-status.md`.

## Objective
Reconcile frozen provider, catalog, semantic geometry, CSG1, CPE2, acquisition, root build, exports, and shared tests without redesigning interfaces or disturbing CE-PTR.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `integration_exclusive`
- Result: `-`

## Next Action
_None._

## Ownership
- `exclusive`: `CMakeLists.txt`
- `exclusive`: `include/Cellerator/Cellerator.hh`
- `exclusive`: `src/CMakeLists.txt`
- `exclusive`: `src/compute/CMakeLists.txt`
- `exclusive`: `src/execution/CMakeLists.txt`
- `exclusive`: `src/geometry/CMakeLists.txt`
- `exclusive`: `tests/CMakeLists.txt`
- `read`: `include/Cellerator/compute/architecture`
- `read`: `include/Cellerator/compute/operation`
- `read`: `include/Cellerator/execution`
- `read`: `include/Cellerator/geometry`
- `read`: `include/Cellerator/runtime`
- `read`: `src/compute/architecture`
- `read`: `src/compute/operation`
- `read`: `src/execution`
- `read`: `src/geometry`
- `read`: `src/runtime`

## Dependencies
- `barrier`: `CE-GEO-FOUNDATION-FANIN`
<!-- todo-orchestrator:v2-managed:end -->
