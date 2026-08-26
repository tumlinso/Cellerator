

<!-- todo-orchestrator:v2-managed:start -->
# CE-ARCH-90: Add hierarchy and scale-out readiness

Task revision: `1412`; current project revision is in `todo-status.md`.

## Objective
Complete Phase 11 with shared-value hierarchy indices, nested partition identity, active-module skipping, execution-order communication planning, and cross-device boundary cost while keeping single-GPU execution simple.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
Implement the minimum nested partition and boundary-edge IR first, then add only hierarchy/runtime mechanisms justified by focused cost and correctness evidence.

## Ownership
- `exclusive`: `Cellerator/CMakeLists.txt`
- `exclusive`: `Cellerator/include/Cellerator/distributed`
- `exclusive`: `Cellerator/include/Cellerator/execution`
- `exclusive`: `Cellerator/include/Cellerator/planner`
- `exclusive`: `Cellerator/src/distributed`
- `exclusive`: `Cellerator/src/execution`
- `exclusive`: `Cellerator/src/planner`
- `exclusive`: `Cellerator/tests/distributed`
- `exclusive`: `Cellerator/tests/execution`
- `exclusive`: `Cellerator/tests/planner`
- `forbidden`: `Baseplane/src`
- `forbidden`: `CellShard`
- `forbidden`: `Cellerator/components/CelleraTorch`
- `read`: `Cellerator/components/CellPack`
- `read`: `Cellerator/include/Cellerator/runtime`

## Dependencies
- `task`: `CE-ARCH-89`
<!-- todo-orchestrator:v2-managed:end -->
