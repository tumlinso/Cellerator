<!-- todo-orchestrator:v2-managed:start -->
# CP-MATH-01: Math operation contracts

Task revision: `1418`; current project revision is in `todo-status.md`.

## Objective
Implement backend-neutral SpMM MathRequest/OperationSignature, alpha/beta, transpose, dtype/compute, determinism, workspace, reuse, epilogue, stable identity and pointer-free ExecutionPlan metadata separation with zero/trivial semantics.

## State
- Lifecycle: `superseded`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `superseded`

## Next Action
_None._

## Ownership
- `exclusive`: `Cellerator/include/Cellerator/compute/math/execution_plan.hh`
- `exclusive`: `Cellerator/include/Cellerator/compute/math/operation.hh`
- `exclusive`: `Cellerator/src/compute/math/operation.cc`
- `exclusive`: `Cellerator/src/compute/math/operation_signature.cc`
- `exclusive`: `Cellerator/tests/math_operation_contract_test.cc`
- `forbidden`: `Cellerator/components/CellPack`
- `read`: `Cellerator/include/Cellerator/types.cuh`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
