

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-99: Advanced Volta operation validation

Task revision: `3281`; current project revision is in `todo-status.md`.

## Objective
Validate transpose, contraction, edge gradients, segment backward, sparse exchange composition, optional fusion dispositions, and exact fallbacks.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `validated`

## Next Action
_None._

## Ownership
- `exclusive`: `tests/tensor_core/sm70/advanced_operations_test.cu`
- `read`: `src/compute/architecture/providers/nvidia/sm70/contract_on_support.cu`
- `read`: `src/compute/architecture/providers/nvidia/sm70/exchange_program.cc`
- `read`: `src/compute/architecture/providers/nvidia/sm70/transpose_relation_apply.cu`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
