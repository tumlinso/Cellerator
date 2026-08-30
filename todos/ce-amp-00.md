

<!-- todo-orchestrator:v2-managed:start -->
# CE-AMP-00: Permission-gated Ampere extension

Task revision: `3016`; current project revision is in `todo-status.md`.

## Objective
After CE-GEO-COMPLETE and explicit human permission, activate and preflight a subordinate sm_86 extension by revalidating source identity, live A5000 hardware, toolchain, frozen contracts, and identical CSG1 fixtures.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `serial`
- Result: `-`

## Next Action
_None._

## Ownership
- `exclusive`: `bench/ce_geo/evidence/ampere/preflight.json`
- `read`: `.todo-orchestrator`
- `read`: `docs/CE_GEO_PROGRAM.md`
- `read`: `include/Cellerator/compute/architecture`
- `read`: `include/Cellerator/geometry/persistence/semantic_geometry_image_v1.hh`

## Dependencies
- `checkpoint`: `CE-GEO-COMPLETE`
- `decision`: `CE-AMP-PERMISSION`
- `checkpoint`: `CE-EXOP-COMPLETE`
<!-- todo-orchestrator:v2-managed:end -->
