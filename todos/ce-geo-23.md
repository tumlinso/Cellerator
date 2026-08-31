

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-23: Erased candidate-owned preparation adapters

Task revision: `3281`; current project revision is in `todo-status.md`.

## Objective
Move typed preparation dispatch behind each catalog entry while preserving existing bridges and eliminating central physical-projection knowledge.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `include/Cellerator/compute/operation/preparation_factory.hh`
- `exclusive`: `src/compute/operation/preparation_factory.cc`
- `exclusive`: `tests/ce_geo/catalog/erased_prepare_test.cu`
- `read`: `include/Cellerator/compute/candidate`
- `read`: `include/Cellerator/execution/projection_activation_v2.hh`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
