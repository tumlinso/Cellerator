<!-- todo-orchestrator:v2-managed:start -->
# CE-ARCH-70E: Stabilize identity registry and durable planner keys

Task revision: `921`; current project revision is in `todo-status.md`.

## Objective
Implement host-side persistent identity interning and generation-safe resolution, remove runtime handles from durable planner evidence, and make cached projection selection resolve to current runtime candidates.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
Implement the small host registry, deterministic persistent structure-set planning keys, current-handle cache winner resolution, cache-store diagnostics, bounded confidence improvement, and policy fallback when all empirical measurements fail.

## Ownership
- `exclusive`: `Cellerator/CMakeLists.txt`
- `exclusive`: `Cellerator/include/Cellerator/execution`
- `exclusive`: `Cellerator/include/Cellerator/planner`
- `exclusive`: `Cellerator/src/execution`
- `exclusive`: `Cellerator/src/planner`
- `exclusive`: `Cellerator/tests/execution_identity`
- `exclusive`: `Cellerator/tests/planner`
- `forbidden`: `Baseplane`
- `forbidden`: `CellShard`

## Dependencies
- `task`: `CE-ARCH-70D`
<!-- todo-orchestrator:v2-managed:end -->
