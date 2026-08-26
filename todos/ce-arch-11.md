<!-- todo-orchestrator:v2-managed:start -->
# CE-ARCH-11: Execution-order contract and structure/value/binding separation

Task revision: `780`; current project revision is in `todo-status.md`.

## Objective
Make execution order and data lifetime explicit across Cellerator operations.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
Define RelationStructure, ValuePlane, LaunchBindings, output-order contracts, transform caching, and versioned CPK1 successor semantics without mutating v1.

## Ownership
- `exclusive`: `Cellerator/include/Cellerator/execution`
- `exclusive`: `Cellerator/src/execution`
- `exclusive`: `Cellerator/tests/execution_order`
- `forbidden`: `CellShard`
- `read`: `Cellerator/components/CellPack`
- `read`: `Cellerator/include/Cellerator/compute/math`
- `read`: `Cellerator/src/compute/math`

## Dependencies
- `task`: `CE-ARCH-10`
<!-- todo-orchestrator:v2-managed:end -->
