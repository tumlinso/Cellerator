

<!-- todo-orchestrator:v2-managed:start -->
# CE-ARCH-70F: Complete Execution Image v2 device-relative prebinding

Task revision: `921`; current project revision is in `todo-status.md`.

## Objective
Construct hot projection views from host-validated offsets plus an arbitrary destination image base and prove a CUDA kernel consumes the device-relative payload after one opaque upload.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
Add one relocation-aware host prebind path without directory duplication or CUDA calls, reject wrong destination sizes, and upgrade the device test to read payload through the prebound device pointer in a tiny kernel.

## Ownership
- `exclusive`: `Cellerator/components/CellPack/include/CellPack/persistence`
- `exclusive`: `Cellerator/components/CellPack/src/persistence`
- `exclusive`: `Cellerator/components/CellPack/tests/persistence`
- `forbidden`: `Baseplane`
- `forbidden`: `CellShard/include`
- `forbidden`: `CellShard/src`
- `forbidden`: `CellShard/tests`
- `read`: `CellShard`

## Dependencies
- `task`: `CE-ARCH-70E`
<!-- todo-orchestrator:v2-managed:end -->
