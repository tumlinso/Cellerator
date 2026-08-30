

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-55: Exact relation rescan

Task revision: `3016`; current project revision is in `todo-status.md`.

## Objective
After approximate proposal, perform one exact full-edge pass that decides component membership, occupancy, ownership, and cost.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `src/geometry/strategy/exact_relation_rescan.cc`
- `exclusive`: `tests/geometry/ce_geo/exact_relation_rescan_test.cc`
- `read`: `include/Cellerator/geometry/relation_cover.hh`
- `read`: `include/Cellerator/geometry/strategy/rectangular_affinity.hh`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
