

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-105: Gradient validation

Task revision: `2999`; current project revision is in `todo-status.md`.

## Objective
Validate transpose, logical edge values, support contraction, segment softmax, and composed exchange gradients against independent finite-difference or high-precision references.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `validated`

## Next Action
_None._

## Ownership
- `exclusive`: `tests/ce_geo/validation/gradient_validation_test.cu`
- `read`: `src/compute/architecture/providers/nvidia/sm70/contract_on_support.cu`
- `read`: `src/compute/architecture/providers/nvidia/sm70/edge_value_gradient.cu`
- `read`: `src/compute/architecture/providers/nvidia/sm70/exchange_program.cc`
- `read`: `src/compute/architecture/providers/nvidia/sm70/segment_backward_integration.cu`
- `read`: `src/compute/architecture/providers/nvidia/sm70/transpose_relation_apply.cu`
- `read`: `src/compute/candidate/segment/normalize.cu`

## Dependencies
- `task`: `CE-GEO-91`
- `task`: `CE-GEO-93`
- `task`: `CE-GEO-94`
- `task`: `CE-GEO-95`
- `task`: `CE-GEO-96`
<!-- todo-orchestrator:v2-managed:end -->
