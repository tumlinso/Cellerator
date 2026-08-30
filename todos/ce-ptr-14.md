

<!-- todo-orchestrator:v2-managed:start -->
# CE-PTR-14: Projection builder scratch cleanup

Task revision: `2509`; current project revision is in `todo-status.md`.

## Objective
Preserve CPK1, FMP1, CTP1, and related pointer-free projection contracts while replacing only generic STL-heavy construction scratch with queried caller-owned prepared workspaces where useful.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
Measure builder scratch and migrate only useful construction ownership while preserving existing pointer-free formats and direct rebound views.

## Ownership
- `exclusive`: `include/Cellerator/compute/projection`
- `exclusive`: `src/compute/projection/physical_feature_major.cc`
- `exclusive`: `src/compute/projection/physical_transpose.cc`
- `read`: `bench`
- `read`: `include/Cellerator/geometry/persistence`
- `read`: `src/geometry/persistence`
- `read`: `tests`

## Dependencies
- `checkpoint`: `CE-PTR-SUBSTRATE-CONTRACT-READY`
<!-- todo-orchestrator:v2-managed:end -->
