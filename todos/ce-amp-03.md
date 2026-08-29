

<!-- todo-orchestrator:v2-managed:start -->
# CE-AMP-03: sm_86 staging and PTX MMA kernel

Task revision: `2379`; current project revision is in `todo-status.md`.

## Objective
Implement provider-private cp.async, ldmatrix, and mma.sync staging/execution where measured appropriate, with no private types in public contracts.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `parallel_safe`
- Result: `-`

## Next Action
_None._

## Ownership
- `exclusive`: `src/compute/architecture/providers/nvidia/sm86/relation_apply.cu`
- `exclusive`: `src/compute/architecture/providers/nvidia/sm86/staging.cuh`
- `exclusive`: `tests/tensor_core/sm86/relation_apply_test.cu`
- `read`: `include/Cellerator/compute/architecture/providers/nvidia/sm86_provider.hh`

## Dependencies
- `checkpoint`: `CE-GEO-COMPLETE`
- `decision`: `CE-AMP-PERMISSION`
<!-- todo-orchestrator:v2-managed:end -->
