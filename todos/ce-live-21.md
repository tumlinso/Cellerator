

<!-- todo-orchestrator:v2-managed:start -->
# CE-LIVE-21: Built-in operation candidate catalog

Task revision: `2058`; current project revision is in `todo-status.md`.

## Objective
Implement a deterministic host-side catalog over existing operation-core candidates, exposing capability and preparation metadata without changing operation_candidate, owning runtime resources, or introducing virtual dispatch.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `docs/CE_LIVE_CANDIDATE_CATALOG.md`
- `exclusive`: `include/Cellerator/compute/math/operation_core/builtin_catalog.hh`
- `exclusive`: `src/compute/math/operation_core/builtin_catalog.cc`
- `exclusive`: `tests/math_core/builtin_catalog_test.cc`
- `read`: `include/Cellerator/compute/math/operation_core`
- `read`: `src/compute/math/operation_core`

## Dependencies
- `task`: `CE-LIVE-11`
- `task`: `CE-LIVE-13`
- `task`: `CE-LIVE-19`
<!-- todo-orchestrator:v2-managed:end -->
