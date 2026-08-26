<!-- todo-orchestrator:v2-managed:start -->
# CE-ARCH-20: CP-BP v1 preservation bridge and semantic geometry boundary

Task revision: `780`; current project revision is in `todo-status.md`.

## Objective
Preserve validated CP-BP v1 behind new identity/lifetime contracts while separating semantic geometry from physical projection.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
Implement read-only adapters for frozen plan/order/records/tiles/CPK1/direct kernel and define semantic geometry/statistics without changing v1 bytes or objective.

## Ownership
- `exclusive`: `Cellerator/components/CellPack/include/CellPack`
- `exclusive`: `Cellerator/components/CellPack/src`
- `exclusive`: `Cellerator/components/CellPack/tests`
- `forbidden`: `CellShard`
- `read`: `Cellerator/include/Cellerator/compute/math/native_tile_view.hh`
- `read`: `Cellerator/src/compute/math/native_tile_view.cc`

## Dependencies
- `task`: `CE-ARCH-10`
- `task`: `CE-ARCH-11`
<!-- todo-orchestrator:v2-managed:end -->
