<!-- todo-orchestrator:v2-managed:start -->
# CE-ARCH-22: CP-Math recovery into Cellerator core

Task revision: `780`; current project revision is in `todo-status.md`.

## Objective
Transform useful CP-Math experiments into Cellerator core operation, projection, planning, and execution contracts.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
Complete: reusable Cellerator::operation_core target, compact native/vendor/composed candidate registry, direct prepared dispatch, explicit numeric/projection policies, and dynamic launch binding validation are implemented; experimental CP-Math remains quarantined evidence for later migration.

## Ownership
- `exclusive`: `Cellerator/CMakeLists.txt`
- `exclusive`: `Cellerator/include/Cellerator/compute/math`
- `exclusive`: `Cellerator/src/compute/math`
- `exclusive`: `Cellerator/tests/math_core`
- `forbidden`: `CellShard`
- `forbidden`: `Cellerator/components/CelleraTorch`
- `read`: `Cellerator/components/CellPack`
- `read`: `Cellerator/include/Cellerator/runtime`
- `read`: `Cellerator/src/runtime`

## Dependencies
- `task`: `CE-ARCH-12`
- `task`: `CE-ARCH-21`
<!-- todo-orchestrator:v2-managed:end -->
