<!-- todo-orchestrator:v2-managed:start -->
# CE-ARCH-70C: Support bounded multi-structure operation bindings

Task revision: `921`; current project revision is in `todo-status.md`.

## Objective
Make prepared operations and planner candidates explicitly depend on a deterministic bounded set of immutable relation structures and validate each value plane against its own relation and epoch.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
Introduce fixed-size structure requirements and structure-set keys, validate duplicates/missing/stale relations and relation-specific values, then represent coordinate-to-regulatory and regulatory-to-gene as two honest relations in the Baseplane proof.

## Ownership
- `exclusive`: `Cellerator/include/Cellerator/compute/math/operation_core`
- `exclusive`: `Cellerator/include/Cellerator/compute/sequence`
- `exclusive`: `Cellerator/include/Cellerator/execution`
- `exclusive`: `Cellerator/src/compute/math/operation_core`
- `exclusive`: `Cellerator/src/compute/sequence`
- `exclusive`: `Cellerator/tests/biological_abi`
- `exclusive`: `Cellerator/tests/execution_order`
- `exclusive`: `Cellerator/tests/math_core`
- `exclusive`: `Cellerator/tests/sequence`
- `forbidden`: `Baseplane`
- `forbidden`: `CellShard`
- `read`: `Cellerator/include/Cellerator/planner`
- `read`: `Cellerator/src/planner`

## Dependencies
- `task`: `CE-ARCH-70B`
<!-- todo-orchestrator:v2-managed:end -->
