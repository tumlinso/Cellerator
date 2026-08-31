

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-95: Segment backward integration

Task revision: `3281`; current project revision is in `todo-status.md`.

## Objective
Integrate required segment backward primitives into prepared relation programs without generic autograd ownership.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `src/compute/architecture/providers/nvidia/sm70/segment_backward_integration.cu`
- `exclusive`: `tests/tensor_core/sm70/segment_backward_integration_test.cu`
- `read`: `src/compute/candidate/segment/normalize.cu`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
