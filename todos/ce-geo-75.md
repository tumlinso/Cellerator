

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-75: N=32 variants

Task revision: `2379`; current project revision is in `todo-status.md`.

## Objective
Implement and retain empirical-required two-warps/one-group and four-warps/two-compatible-groups candidates until complete-cost evaluation.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `serial`
- Result: `-`

## Next Action
_None._

## Ownership
- `exclusive`: `src/compute/architecture/providers/nvidia/sm70/relation_apply_n32.cu`
- `exclusive`: `tests/tensor_core/sm70/relation_apply_n32_test.cu`
- `read`: `src/compute/architecture/providers/nvidia/sm70/relation_apply_n64.cu`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
