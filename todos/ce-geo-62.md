

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-62: Projection serializer parser and validator

Task revision: `3281`; current project revision is in `todo-status.md`.

## Objective
Implement checked offsets, alignment, identities, counts, index widths, checksums, corruption rejection, and CPE2 embedding.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `src/compute/projection/mma_physical_cover_validation.cc`
- `exclusive`: `src/compute/projection/physical_mma_hybrid.cc`
- `exclusive`: `tests/ce_geo/projection/mma_projection_roundtrip_test.cc`
- `read`: `include/Cellerator/geometry/persistence/execution_image_v2.hh`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
