

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-105: Gradient validation

Task revision: `2379`; current project revision is in `todo-status.md`.

## Objective
Validate transpose, logical edge values, support contraction, segment softmax, and composed exchange gradients against independent finite-difference or high-precision references.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `serial`
- Result: `-`

## Next Action
_None._

## Ownership
- `exclusive`: `tests/ce_geo/validation/gradient_validation_test.cu`
- `read`: `src/compute/architecture/providers/nvidia/sm70/edge_value_gradient.cu`
- `read`: `src/compute/architecture/providers/nvidia/sm70/transpose_relation_apply.cu`
- `read`: `src/compute/candidate/segment/normalize.cu`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
