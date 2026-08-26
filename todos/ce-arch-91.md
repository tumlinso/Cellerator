

<!-- todo-orchestrator:v2-managed:start -->
# CE-ARCH-91: Integrate opaque CellShard execution artifact delivery

Task revision: `1412`; current project revision is in `todo-status.md`.

## Objective
Consume the completed CellShard foundation to persist, validate, place, upload, and directly execute opaque Cellerator images without CellShard interpreting execution semantics or Cellerator owning storage transport.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
Wait only on the authoritative CS-FOUND interfaces, then add the narrow cross-repository opaque artifact vertical slice with one-copy caller-stream upload and direct CPE2 execution.

## Ownership
- `exclusive`: `CellShard/CMakeLists.txt`
- `exclusive`: `CellShard/include/CellShard`
- `exclusive`: `CellShard/src`
- `exclusive`: `CellShard/tests`
- `exclusive`: `Cellerator/CMakeLists.txt`
- `exclusive`: `Cellerator/include/Cellerator/execution`
- `exclusive`: `Cellerator/src/execution`
- `exclusive`: `Cellerator/tests/persistence`
- `forbidden`: `Baseplane/src`
- `forbidden`: `Cellerator/components/CelleraTorch`
- `read`: `CellShard/docs`
- `read`: `Cellerator/components/CellPack`

## Dependencies
- `task`: `CE-ARCH-90`
<!-- todo-orchestrator:v2-managed:end -->
