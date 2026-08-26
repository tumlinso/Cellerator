<!-- todo-orchestrator:v2-managed:start -->
# CE-ARCH-70D: Declare output update semantics

Task revision: `921`; current project revision is in `todo-status.md`.

## Objective
Add one compact validated output-effect contract per output and declare the sequence gene-state output as accumulation and predicate-mask output as overwrite.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
Add and validate overwrite, accumulate, affine-accumulate, and partial-write contracts; prove nonzero gene-state initialization yields initial plus contribution without hidden zeroing.

## Ownership
- `exclusive`: `Cellerator/include/Cellerator/compute/math/operation_core`
- `exclusive`: `Cellerator/include/Cellerator/compute/sequence`
- `exclusive`: `Cellerator/include/Cellerator/execution`
- `exclusive`: `Cellerator/src/compute/math/operation_core`
- `exclusive`: `Cellerator/src/compute/sequence`
- `exclusive`: `Cellerator/tests/biological_abi`
- `exclusive`: `Cellerator/tests/math_core`
- `exclusive`: `Cellerator/tests/sequence`
- `forbidden`: `Baseplane`
- `forbidden`: `CellShard`

## Dependencies
- `task`: `CE-ARCH-70C`
<!-- todo-orchestrator:v2-managed:end -->
