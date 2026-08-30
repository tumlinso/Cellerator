

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-36: Identity strategy and independent validation pipeline

Task revision: `3016`; current project revision is in `todo-status.md`.

## Objective
Implement the full-relation identity strategy and public compile pipeline; prove malformed strategy output cannot certify itself.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `src/geometry/compiler/compile_geometry.cc`
- `exclusive`: `src/geometry/compiler/identity_strategy.cc`
- `exclusive`: `tests/geometry/ce_geo/identity_strategy_test.cc`
- `read`: `include/Cellerator/geometry/compiler`
- `read`: `src/geometry/compiler/relation_cover.cc`
- `read`: `src/geometry/compiler/work_layout.cc`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
