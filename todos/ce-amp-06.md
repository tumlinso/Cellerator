

<!-- todo-orchestrator:v2-managed:start -->
# CE-AMP-06: Ampere correctness and sanitizer campaign

Task revision: `2379`; current project revision is in `todo-status.md`.

## Objective
Run the same logical fixtures, exact covers, corruptions, generations, graph/runtime rules, numerical policies, gradients, and Compute Sanitizer on live leased sm_86 hardware.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `parallel_safe`
- Result: `-`

## Next Action
_None._

## Ownership
- `exclusive`: `bench/ce_geo/evidence/ampere/validation.json`
- `exclusive`: `tests/tensor_core/sm86/run_validation.py`
- `read`: `tests/ce_geo`
- `read`: `tests/relation_algebra`
- `read`: `tests/tensor_core/sm86`

## Dependencies
- `checkpoint`: `CE-GEO-COMPLETE`
- `decision`: `CE-AMP-PERMISSION`
- `task`: `CE-AMP-04`
- `task`: `CE-AMP-05`
<!-- todo-orchestrator:v2-managed:end -->
