

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-67: Complete marginal-cost selection and local refinement

Task revision: `2379`; current project revision is in `todo-status.md`.

## Objective
Use complete target-calibrated marginal cost and bounded moves, swaps, splits, merges, rectangle toggles, and admissible exchanges; always emit pure sparse, conservative hybrid, and aggressive hybrid covers.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `serial`
- Result: `-`

## Next Action
_None._

## Ownership
- `exclusive`: `src/compute/architecture/providers/nvidia/common/mma_target_refinement.cc`
- `exclusive`: `tests/ce_geo/projection/mma_target_refinement_test.cc`
- `read`: `include/Cellerator/planner/candidate_measurement.hh`
- `read`: `src/compute/architecture/providers/nvidia/common/exact_rectangle_census.cc`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
