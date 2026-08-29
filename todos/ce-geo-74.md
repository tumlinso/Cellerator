

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-74: Integrate N=64 hybrid relation apply

Task revision: `2379`; current project revision is in `todo-status.md`.

## Objective
Combine prepared value pack, output-owned MMA, exact row-owned residual, one epilogue, persistent order, planner visibility, and pure-sparse fallback.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `serial`
- Result: `-`

## Next Action
_None._

## Ownership
- `exclusive`: `src/compute/architecture/providers/nvidia/sm70/relation_apply_hybrid.cu`
- `exclusive`: `src/compute/architecture/providers/nvidia/sm70/residual.cu`
- `exclusive`: `tests/tensor_core/sm70/relation_apply_hybrid_test.cu`
- `read`: `include/Cellerator/execution/program.hh`
- `read`: `include/Cellerator/planner/end_to_end_planner.hh`
- `read`: `src/compute/architecture/providers/nvidia/sm70/relation_apply_n64.cu`
- `read`: `src/compute/architecture/providers/nvidia/sm70/value_pack.cu`

## Dependencies
- `checkpoint`: `CE-GEO-FOUNDATION-INTEGRATED`
- `interface`: `cellerator-mma-hybrid-projection-v1`
<!-- todo-orchestrator:v2-managed:end -->
