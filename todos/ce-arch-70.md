

<!-- todo-orchestrator:v2-managed:start -->
# CE-ARCH-70: CE-ARCH-70 Foundation Hardening

Task revision: `921`; current project revision is in `todo-status.md`.

## Objective
Correct the existing Cellerator execution architecture's foundational ABI, lifetime, identity, planning-key, and device-prebinding defects without expanding its feature or projection scope.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `validated`

## Next Action
Complete. Start CE-ARCH-71 to register the preserved CP-BP row-masked N=1 kernel as a real operation-core/planner candidate; do not reopen these frozen foundations.

## Ownership
- `forbidden`: `CellShard/include`
- `forbidden`: `CellShard/src`
- `forbidden`: `CellShard/tests`
- `read`: `Baseplane`
- `read`: `CellShard`
- `read`: `Cellerator`

## Dependencies
- `barrier`: `CE-ARCH-70-PHASES-COMPLETE`
<!-- todo-orchestrator:v2-managed:end -->
