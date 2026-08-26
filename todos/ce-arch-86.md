<!-- todo-orchestrator:v2-managed:start -->
# CE-ARCH-86: Plan connected operations with measured total cost

Task revision: `1412`; current project revision is in `todo-status.md`.

## Objective
Complete Phase 7 by selecting across connected operations with explicit order, conversion, preparation, communication, and reuse costs while retaining bounded empirical autotuning and durable invalidation.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
Generalize the CE-ARCH-73 single-operation planner to a bounded connected-operation plan and prove that order/conversion costs can change the winner.

## Ownership
- `exclusive`: `Cellerator/CMakeLists.txt`
- `exclusive`: `Cellerator/bench/architecture_evidence`
- `exclusive`: `Cellerator/include/Cellerator/planner`
- `exclusive`: `Cellerator/src/planner`
- `exclusive`: `Cellerator/tests/planner`
- `forbidden`: `Baseplane/src`
- `forbidden`: `CellShard`
- `forbidden`: `Cellerator/components/CelleraTorch`
- `read`: `Cellerator/include/Cellerator/compute/math`
- `read`: `Cellerator/include/Cellerator/execution`

## Dependencies
- `task`: `CE-ARCH-85`
<!-- todo-orchestrator:v2-managed:end -->
