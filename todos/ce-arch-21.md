

<!-- todo-orchestrator:v2-managed:start -->
# CE-ARCH-21: Relocatable execution image v2 and projection catalog

Task revision: `780`; current project revision is in `todo-status.md`.

## Objective
Evolve CPK1 pointer-free persistence into an execution IR holding one semantic geometry and multiple physical projections.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
Design a new Cellerator payload schema with section/projection directories and CPK1 loader, inside unchanged CPEXEC01 unless an external decision proves otherwise.

## Ownership
- `exclusive`: `Cellerator/components/CellPack/include/CellPack/persistence`
- `exclusive`: `Cellerator/components/CellPack/src/persistence`
- `exclusive`: `Cellerator/components/CellPack/tests/persistence`
- `forbidden`: `CellShard`
- `read`: `CellShard/include/CellShard/io/pack/execution_payload.cuh`
- `read`: `CellShard/src/io/pack/execution_payload.cu`

## Dependencies
- `task`: `CE-ARCH-11`
- `task`: `CE-ARCH-20`
<!-- todo-orchestrator:v2-managed:end -->
