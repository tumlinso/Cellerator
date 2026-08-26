

<!-- todo-orchestrator:v2-managed:start -->
# CE-ARCH-31: End-to-end planner, autotuner, and CP-BP objective v2

Task revision: `780`; current project revision is in `todo-status.md`.

## Objective
Select the fastest correct end-to-end strategy and feed measured costs into versioned semantic-geometry optimization.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
Complete: planner v1 ranks complete workflow cost, performs bounded structure-specific measurement, validates factored cache evidence, selects conventional fallbacks without bias, and exposes versioned CP-BP objective v2 without changing v1 semantics; GPU evidence remains background-controlled.

## Ownership
- `exclusive`: `Cellerator/components/CellPack/src/optimization`
- `exclusive`: `Cellerator/include/Cellerator/planner`
- `exclusive`: `Cellerator/src/planner`
- `exclusive`: `Cellerator/tests/planner`
- `forbidden`: `CellShard`
- `read`: `Cellerator/bench/architecture_evidence`
- `read`: `Cellerator/src/compute/math/planner.cc`

## Dependencies
- `task`: `CE-ARCH-20`
- `task`: `CE-ARCH-22`
- `task`: `CE-ARCH-30`
<!-- todo-orchestrator:v2-managed:end -->
