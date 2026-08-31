

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-82: Segment reductions

Task revision: `3281`; current project revision is in `todo-status.md`.

## Objective
Implement sum and maximum with explicit segment/axis identity, FP32 policy, empty and singleton handling, and caller-owned workspace.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `include/Cellerator/compute/candidate/segment/reduce.hh`
- `exclusive`: `src/compute/candidate/segment/reduce.cu`
- `exclusive`: `tests/relation_algebra/segment_reduce_test.cu`
- `read`: `include/Cellerator/execution/operands.hh`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
