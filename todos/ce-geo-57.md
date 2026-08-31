

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-57: Portable support validation and determinism

Task revision: `3281`; current project revision is in `todo-status.md`.

## Objective
Prove identical input/seed gives byte-identical evidence, exact rescans own all edges, and architecture/tile widths do not enter portable identity.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `validated`

## Next Action
_None._

## Ownership
- `exclusive`: `tests/geometry/ce_geo/portable_support_property_test.cc`
- `read`: `include/Cellerator/geometry/support_atlas.hh`
- `read`: `src/geometry/persistence/semantic_geometry_support_sections.cc`
- `read`: `src/geometry/strategy`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
