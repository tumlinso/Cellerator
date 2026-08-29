

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-81: Relation apply and transpose compatibility semantics

Task revision: `2379`; current project revision is in `todo-status.md`.

## Objective
Map current v1 forward/transpose operations into the typed algebra without silently changing frozen enum or identity semantics.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `serial`
- Result: `-`

## Next Action
_None._

## Ownership
- `exclusive`: `src/compute/operation/relation_apply_compatibility.cc`
- `exclusive`: `tests/relation_algebra/relation_apply_compatibility_test.cu`
- `read`: `include/Cellerator/compute/candidate/transpose_backward_candidate.hh`
- `read`: `include/Cellerator/compute/operation/operation_core.hh`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
