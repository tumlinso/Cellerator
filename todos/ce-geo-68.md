

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-68: Provider realization and activation

Task revision: `3016`; current project revision is in `todo-status.md`.

## Objective
Emit CPE2 projection sources, typed capability manifests, schedules, and provider-erased activated views for every complete candidate cover.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `src/compute/architecture/providers/nvidia/common/mma_projection_realization.cc`
- `exclusive`: `tests/ce_geo/projection/mma_provider_realization_test.cu`
- `read`: `include/Cellerator/compute/architecture/provider.hh`
- `read`: `include/Cellerator/execution/projection_activation_v2.hh`
- `read`: `include/Cellerator/geometry/persistence/execution_image_v2.hh`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
