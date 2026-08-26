<!-- todo-orchestrator:v2-managed:start -->
# CE-ARCH-89: Complete Baseplane direct relation and fused planning path

Task revision: `1412`; current project revision is in `todo-status.md`.

## Objective
Complete Phase 10 with common domain/order identities, direct relation-builder output, materialized and fused sequence-to-regulatory execution, no host boundary, and complete-cost planner selection across reused cell states.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
Audit CE-ARCH-40/78 against every Phase 10 exit criterion and implement the missing relation-builder, shared-identity, fused/materialized planner, and repeated-state proof.

## Ownership
- `exclusive`: `Baseplane/CMakeLists.txt`
- `exclusive`: `Baseplane/include/Baseplane/seq`
- `exclusive`: `Baseplane/tests/seq`
- `exclusive`: `Cellerator/CMakeLists.txt`
- `exclusive`: `Cellerator/include/Cellerator/compute/sequence`
- `exclusive`: `Cellerator/include/Cellerator/planner`
- `exclusive`: `Cellerator/src/compute/sequence`
- `exclusive`: `Cellerator/src/planner`
- `exclusive`: `Cellerator/tests/sequence`
- `forbidden`: `CellShard`
- `forbidden`: `Cellerator/components/CelleraTorch`
- `read`: `Baseplane/src/seq`
- `read`: `Cellerator/include/Cellerator/compute/math`
- `read`: `Cellerator/include/Cellerator/execution`

## Dependencies
- `task`: `CE-ARCH-88`
<!-- todo-orchestrator:v2-managed:end -->
