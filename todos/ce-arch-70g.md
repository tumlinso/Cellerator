<!-- todo-orchestrator:v2-managed:start -->
# CE-ARCH-70G: Validate, document, and hand off foundation hardening

Task revision: `921`; current project revision is in `todo-status.md`.

## Objective
Reconcile executable invariants, focused host/CUDA/sanitizer evidence, frozen interfaces, paired repository commits, deliberate deferrals, and the exact unchanged downstream path.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `integration_exclusive`
- Result: `validated`

## Next Action
Run the staged validation matrix through cpp-context-compiler and the CUDA controller, freeze the six corrected interfaces, commit the owning repositories separately, record exact evidence and deferrals, and leave CE-ARCH-71 as the first downstream task.

## Ownership
- `exclusive`: `Baseplane/README.md`
- `exclusive`: `Baseplane/include/Baseplane/seq`
- `exclusive`: `Cellerator/AGENTS.md`
- `exclusive`: `Cellerator/components/CellPack/include/CellPack/persistence/EXECUTION_IMAGE_V2.md`
- `exclusive`: `Cellerator/include/Cellerator/compute/math/MATH_V1_EVIDENCE.md`
- `exclusive`: `Cellerator/include/Cellerator/compute/math/operation_core/OPERATION_CORE.md`
- `exclusive`: `Cellerator/include/Cellerator/compute/sequence/BASEPLANE_INTEGRATION.md`
- `exclusive`: `Cellerator/include/Cellerator/execution`
- `exclusive`: `Cellerator/include/Cellerator/planner/PLANNER_V1.md`
- `exclusive`: `Cellerator/include/Cellerator/runtime/SESSION.md`
- `exclusive`: `Cellerator/optimization.md`
- `exclusive`: `Cellerator/planning_strategy.md`
- `exclusive`: `Cellerator/scope.md`
- `forbidden`: `CellShard/include`
- `forbidden`: `CellShard/src`
- `forbidden`: `CellShard/tests`
- `read`: `Baseplane`
- `read`: `CellShard`
- `read`: `Cellerator`

## Dependencies
- `task`: `CE-ARCH-70F`
<!-- todo-orchestrator:v2-managed:end -->
