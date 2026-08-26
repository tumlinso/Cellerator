

<!-- todo-orchestrator:v2-managed:start -->
# CE-ARCH-78: Correct Baseplane predicate materialization reuse

Task revision: `1121`; current project revision is in `todo-status.md`.

## Objective
Materialize a predicate once, cache it by sequence generation, predicate identity, and coordinate order, reuse it, and compare total cost against fused execution.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
Add bounded predicate-mask materialization caching and measured fused-versus-reused selection without moving sequence semantics into Cellerator.

## Ownership
_No structured ownership._

## Dependencies
- `task`: `CE-ARCH-77`
<!-- todo-orchestrator:v2-managed:end -->
