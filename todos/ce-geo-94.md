

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-94: Logical edge-value gradients

Task revision: `3016`; current project revision is in `todo-status.md`.

## Objective
Implement edge-value gradients preserving stable logical edge identity through arbitrary physical tile and residual order.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `src/compute/architecture/providers/nvidia/sm70/edge_value_gradient.cu`
- `exclusive`: `tests/tensor_core/sm70/edge_value_gradient_test.cu`
- `read`: `src/compute/architecture/providers/nvidia/sm70/contract_on_support.cu`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
