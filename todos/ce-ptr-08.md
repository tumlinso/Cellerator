

<!-- todo-orchestrator:v2-managed:start -->
# CE-PTR-08: Statistical validation structures

Task revision: `2376`; current project revision is in `todo-status.md`.

## Objective
Replace nested vectors and hash maps or sets in statistical validation with sorted group-row relations, offsets, row-unit maps, packed edge keys, exact flat membership, and generation-mark workspaces without changing statistical or provenance semantics.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `parallel_safe`
- Result: `-`

## Next Action
Consume CE-PTR-07 identity contracts and migrate validation relations independently of unrelated GPU or runtime work.

## Ownership
- `exclusive`: `src/geometry/record_statistical_validation.cc`
- `exclusive`: `src/geometry/statistical_validation.cc`
- `read`: `bench`
- `read`: `include/Cellerator/geometry`
- `read`: `tests`

## Dependencies
- `task`: `CE-PTR-07`
<!-- todo-orchestrator:v2-managed:end -->
