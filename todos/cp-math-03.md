<!-- todo-orchestrator:v2-managed:start -->
# CP-MATH-03: Execution CSR and packed dense operand

Task revision: `1418`; current project revision is in `todo-status.md`.

## Objective
Adapt apply_frozen_plan output to execution-feature CSR, implement order-identity-safe reusable W_packed conversion, prove X_packed W_packed equals canonical math, and design lazy CSR reconstruction from CPK1 without changing CPK1.

## State
- Lifecycle: `superseded`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `superseded`

## Next Action
_None._

## Ownership
- `exclusive`: `Cellerator/include/Cellerator/compute/math/packed_dense_operand.hh`
- `exclusive`: `Cellerator/include/Cellerator/compute/math/physical_csr.hh`
- `exclusive`: `Cellerator/src/compute/math/packed_dense_operand.cu`
- `exclusive`: `Cellerator/src/compute/math/physical_csr.cc`
- `exclusive`: `Cellerator/tests/math_execution_csr_test.cu`
- `forbidden`: `Cellerator/components/CellPack`
- `read`: `Cellerator/components/CellPack/include/CellPack/apply_plan.hh`
- `read`: `Cellerator/components/CellPack/include/CellPack/packing_plan.hh`
- `read`: `Cellerator/components/CellPack/include/CellPack/persistent_packing_payload.hh`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
