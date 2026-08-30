

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-65: Deterministic source and destination grouping

Task revision: `3016`; current project revision is in `todo-status.md`.

## Objective
Use portable rectangular evidence, disjoint source groups up to 16, and deterministic destination support signatures and groups up to 16.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `src/compute/architecture/providers/nvidia/common/sm70_grouping.cc`
- `exclusive`: `tests/ce_geo/projection/sm70_grouping_test.cc`
- `read`: `include/Cellerator/compute/architecture/target_refinement.hh`
- `read`: `include/Cellerator/geometry/support_atlas.hh`

## Dependencies
- `interface`: `cellerator-rectangular-support-v1`
- `interface`: `cellerator-device-provider-contract-v1`
- `interface`: `cellerator-semantic-geometry-compiler-v1`
<!-- todo-orchestrator:v2-managed:end -->
