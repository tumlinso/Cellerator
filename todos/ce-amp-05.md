

<!-- todo-orchestrator:v2-managed:start -->
# CE-AMP-05: Ampere transpose and support contraction

Task revision: `3281`; current project revision is in `todo-status.md`.

## Objective
Implement architecture-comparison parity for transpose relation apply and contract_on_support over the same logical contracts and independent Ampere covers.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `parallel_safe`
- Result: `-`

## Next Action
_None._

## Ownership
- `exclusive`: `src/compute/architecture/providers/nvidia/sm86/contract_on_support.cu`
- `exclusive`: `src/compute/architecture/providers/nvidia/sm86/transpose_relation_apply.cu`
- `exclusive`: `tests/tensor_core/sm86/advanced_ops_test.cu`
- `read`: `include/Cellerator/compute/operation/relation_algebra.hh`

## Dependencies
- `checkpoint`: `CE-GEO-COMPLETE`
- `decision`: `CE-AMP-PERMISSION`
- `task`: `CE-AMP-02`
- `task`: `CE-AMP-03`
- `checkpoint`: `CE-EXOP-COMPLETE`
<!-- todo-orchestrator:v2-managed:end -->
