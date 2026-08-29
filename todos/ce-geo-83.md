

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-83: Segment normalization

Task revision: `2379`; current project revision is in `todo-status.md`.

## Objective
Implement log-sum-exp and softmax with stable FP32 reduction, empty/singleton behavior, NaN/Inf policy, and required backward primitives.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `serial`
- Result: `-`

## Next Action
_None._

## Ownership
- `exclusive`: `include/Cellerator/compute/candidate/segment/normalize.hh`
- `exclusive`: `src/compute/candidate/segment/normalize.cu`
- `exclusive`: `tests/relation_algebra/segment_normalize_test.cu`
- `read`: `include/Cellerator/compute/candidate/segment/reduce.hh`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
