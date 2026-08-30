

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-77: Alpha beta epilogue and persistent order

Task revision: `2999`; current project revision is in `todo-status.md`.

## Objective
Prove initialization, alpha, beta, activation, residual, and output order apply exactly once for all width/tail regimes.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `validated`

## Next Action
_None._

## Ownership
- `exclusive`: `tests/tensor_core/sm70/epilogue_order_test.cu`
- `read`: `src/compute/architecture/providers/nvidia/sm70/relation_apply_hybrid.cu`
- `read`: `src/compute/architecture/providers/nvidia/sm70/residual.cu`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
