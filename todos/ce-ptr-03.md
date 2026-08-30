

<!-- todo-orchestrator:v2-managed:start -->
# CE-PTR-03: Frozen and static plan ownership

Task revision: `2503`; current project revision is in `todo-status.md`.

## Objective
Migrate frozen packing and other durable static plans from collections of independent owning arrays toward coherent versioned images and typed views, converging with CPK1 and execution-image precedents where semantics permit.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
Reinspect CE-GEO live scopes before claim; work only on unowned static-plan sections and defer any path currently interlocked by CE-GEO.

## Ownership
- `exclusive`: `include/Cellerator/geometry/packing_plan.hh`
- `exclusive`: `include/Cellerator/geometry/persistence`
- `exclusive`: `src/geometry/packing_plan.cc`
- `exclusive`: `src/geometry/persistence`
- `read`: `compat/cp_math_v1`
- `read`: `include/Cellerator/execution`
- `read`: `src/execution`
- `read`: `tests`

## Dependencies
- `checkpoint`: `CE-PTR-SUBSTRATE-CONTRACT-READY`
<!-- todo-orchestrator:v2-managed:end -->
