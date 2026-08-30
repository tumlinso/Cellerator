

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-91: sm_70 transpose relation apply

Task revision: `2999`; current project revision is in `todo-status.md`.

## Objective
Implement and validate dX-style transpose execution with exact residual and separate target cover.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `src/compute/architecture/providers/nvidia/sm70/transpose_relation_apply.cu`
- `exclusive`: `tests/tensor_core/sm70/transpose_relation_apply_test.cu`
- `read`: `src/compute/architecture/providers/nvidia/sm70/transpose_cover.cc`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
