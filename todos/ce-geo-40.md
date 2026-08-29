

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-40: Typed CPE2 capability manifest

Task revision: `2379`; current project revision is in `todo-status.md`.

## Objective
Use the existing capability_section hook for a typed device-specific manifest without changing CPE2 record sizes.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `parallel_safe`
- Result: `-`

## Next Action
_None._

## Ownership
- `exclusive`: `include/Cellerator/geometry/persistence/execution_capability_manifest_v1.hh`
- `exclusive`: `tests/geometry/ce_geo/cpe2_capability_manifest_test.cc`
- `read`: `include/Cellerator/geometry/persistence/execution_image_v2.hh`

## Dependencies
- `checkpoint`: `CE-GEO-ARCHITECTURE-FROZEN`
<!-- todo-orchestrator:v2-managed:end -->
