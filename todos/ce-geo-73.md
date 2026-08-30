

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-73: Standalone output-owned N=64 WMMA kernel

Task revision: `2999`; current project revision is in `todo-status.md`.

## Objective
Implement fixture-based four-warp output-owned 16-row by 64-column CTA kernel with resident FP32 accumulators, one final store, and no atomics.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `src/compute/architecture/providers/nvidia/sm70/relation_apply_n64.cu`
- `exclusive`: `src/compute/architecture/providers/nvidia/sm70/relation_apply_n64.cuh`
- `exclusive`: `tests/tensor_core/sm70/relation_apply_n64_test.cu`
- `read`: `src/compute/candidate/tensor_core/v100_dense_fragment_candidate.cu`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
