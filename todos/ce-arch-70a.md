<!-- todo-orchestrator:v2-managed:start -->
# CE-ARCH-70A: Restore Baseplane and Cellerator sequence ABI coherence

Task revision: `921`; current project revision is in `todo-status.md`.

## Objective
Integrate the existing local predicate-plan work, expose and require one explicit sequence predicate ABI version, make validity authoritative, and fail incompatible sibling checkouts early.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
Audit the user-owned Baseplane predicate-plan implementation, establish the baseline build failure or success, add the minimal ABI/version and validity checks, run focused Baseplane host and Cellerator compile/runtime tests, then freeze the interface.

## Ownership
- `exclusive`: `Baseplane/CMakeLists.txt`
- `exclusive`: `Baseplane/include/Baseplane/seq`
- `exclusive`: `Baseplane/src/seq`
- `exclusive`: `Baseplane/tests/seq`
- `exclusive`: `Cellerator/CMakeLists.txt`
- `exclusive`: `Cellerator/include/Cellerator/compute/sequence`
- `exclusive`: `Cellerator/src/compute/sequence`
- `exclusive`: `Cellerator/tests/biological_abi/baseplane_cpu_consumer`
- `exclusive`: `Cellerator/tests/sequence`
- `forbidden`: `CellShard`
- `read`: `Baseplane/README.md`
- `read`: `Cellerator/README.md`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
