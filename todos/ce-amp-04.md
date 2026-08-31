

<!-- todo-orchestrator:v2-managed:start -->
# CE-AMP-04: BF16 path

Task revision: `3281`; current project revision is in `todo-status.md`.

## Objective
Add explicit BF16 operand, accumulation, output, rounding, tolerance, and determinism policy plus implementation and referees.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `serial`
- Result: `-`

## Next Action
_None._

## Ownership
- `exclusive`: `src/compute/architecture/providers/nvidia/sm86/relation_apply_bf16.cu`
- `exclusive`: `tests/tensor_core/sm86/relation_apply_bf16_test.cu`
- `read`: `tests/ce_geo/validation/numerical_referee.hh`

## Dependencies
- `task`: `CE-AMP-03`
- `checkpoint`: `CE-EXOP-COMPLETE`
- `decision`: `CE-AMP-PERMISSION`
<!-- todo-orchestrator:v2-managed:end -->
