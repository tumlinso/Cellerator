

<!-- todo-orchestrator:v2-managed:start -->
# CE-PTR-02: Core memory, image, and workspace substrate

Task revision: `2376`; current project revision is in `todo-status.md`.

## Objective
Establish the smallest shared vocabulary for explicit placement, allocation records, typed non-owning views, caller/session-owned workspace, pointer-free image metadata, alignment, status failures, and safe compiler hints by reusing the existing execution session.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `parallel_safe`
- Result: `-`

## Next Action
Design against live session/image contracts, implement the minimum coherent substrate, and publish CE-PTR-SUBSTRATE-CONTRACT-READY early enough for independent lanes to proceed against the stable interface.

## Ownership
- `exclusive`: `include/Cellerator/memory`
- `exclusive`: `src/runtime/memory`
- `exclusive`: `tests/memory`
- `read`: `include/Cellerator/execution`
- `read`: `include/Cellerator/geometry/persistence`
- `read`: `include/Cellerator/runtime`
- `read`: `src/execution`
- `read`: `src/geometry/persistence`
- `read`: `src/runtime`

## Dependencies
- `checkpoint`: `CE-PTR-POLICY-READY`
<!-- todo-orchestrator:v2-managed:end -->
