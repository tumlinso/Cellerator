

<!-- todo-orchestrator:v2-managed:start -->
# CE-REMAP-05: Separate modern compute authority from compatibility evidence

Task revision: `2311`; current project revision is in `todo-status.md`.

## Objective
Promote authoritative operation, projection, candidate, operator, and training implementation; quarantine CP-Math v1 and legacy sparse API evidence without rewriting kernels or algorithms.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `integration_exclusive`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `CMakeLists.txt`
- `exclusive`: `bench/CMakeLists.txt`
- `exclusive`: `bench/architecture_evidence`
- `exclusive`: `bench/math`
- `exclusive`: `compat/CMakeLists.txt`
- `exclusive`: `compat/cp_math_v1`
- `exclusive`: `compat/legacy_sparse`
- `exclusive`: `docs/REPOSITORY_CONSOLIDATION_PROGRAM.md`
- `exclusive`: `include/Cellerator/compute`
- `exclusive`: `src/compute`
- `exclusive`: `src/compute/CMakeLists.txt`
- `exclusive`: `tests/CMakeLists.txt`
- `exclusive`: `tests/live`
- `exclusive`: `tests/math_core`
- `exclusive`: `tests/persistence`
- `read`: `components/CelleraTorch`
- `read`: `docs`
- `read`: `include/Cellerator/execution`
- `read`: `include/Cellerator/planner`
- `read`: `include/Cellerator/runtime`
- `read`: `src/execution`
- `read`: `src/planner`
- `read`: `src/runtime`

## Dependencies
- `task`: `CE-REMAP-04`
<!-- todo-orchestrator:v2-managed:end -->
