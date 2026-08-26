<!-- todo-orchestrator:v2-managed:start -->
# CE-ARCH-82: Close Execution Image v2 runtime and compatibility gaps

Task revision: `1412`; current project revision is in `todo-status.md`.

## Objective
Complete Phase 4 so CPK1 compatibility, sectioned semantic structure, schema-extensible projections, multiple value planes, opaque relocation, and direct prepared execution are one tested path.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
Audit the existing CPE2 implementation against every Phase 4 exit criterion, implement only missing runtime/compatibility pieces, and add direct-execution regression coverage.

## Ownership
- `exclusive`: `Cellerator/CMakeLists.txt`
- `exclusive`: `Cellerator/components/CellPack/include/CellPack/persistence`
- `exclusive`: `Cellerator/components/CellPack/src/persistence`
- `exclusive`: `Cellerator/tests/persistence`
- `forbidden`: `Baseplane/src`
- `forbidden`: `CellShard/src`
- `read`: `CellShard/include`
- `read`: `Cellerator/include/Cellerator/compute/math`
- `read`: `Cellerator/include/Cellerator/execution`

## Dependencies
- `task`: `CE-ARCH-81`
<!-- todo-orchestrator:v2-managed:end -->
