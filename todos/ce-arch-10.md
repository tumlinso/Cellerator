

<!-- todo-orchestrator:v2-managed:start -->
# CE-ARCH-10: Biological domain, axis, identity, and operand ABI

Task revision: `780`; current project revision is in `todo-status.md`.

## Objective
Create the minimal Cellerator-owned identity and heterogeneous operand model shared by dense state, sparse relations, and Baseplane sequence structures.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
Design and adversarially review the minimal ABI; freeze only after CPU-only Baseplane and CUDA/POD compatibility evidence.

## Ownership
- `exclusive`: `Cellerator/CMakeLists.txt`
- `exclusive`: `Cellerator/include/Cellerator/execution`
- `exclusive`: `Cellerator/tests/biological_abi`
- `forbidden`: `CellShard`
- `read`: `Baseplane/include/Baseplane/seq`
- `read`: `Cellerator/include/Cellerator/abi.h`
- `read`: `Cellerator/include/Cellerator/parameters.hh`
- `read`: `Cellerator/include/Cellerator/types.cuh`

## Dependencies
- `task`: `CE-ARCH-02`
<!-- todo-orchestrator:v2-managed:end -->
