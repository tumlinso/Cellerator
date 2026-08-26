<!-- todo-orchestrator:v2-managed:start -->
# CE-ARCH-84: Add missing forward projection regimes

Task revision: `1412`; current project revision is in `todo-status.md`.

## Objective
Complete the forward half of Phase 6 with measured low-sharing/tail handling and CTA-scale medium-N execution while preserving row-masked, feature-major, and CSR behavior.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
Use CE-ARCH-83 evidence to implement the narrowest truthful low-sharing/tail and CTA medium-N candidates; retain or reject each only by declared correctness and end-to-end evidence.

## Ownership
- `exclusive`: `Cellerator/CMakeLists.txt`
- `exclusive`: `Cellerator/bench/architecture_evidence`
- `exclusive`: `Cellerator/components/CellPack/include/CellPack`
- `exclusive`: `Cellerator/components/CellPack/src`
- `exclusive`: `Cellerator/include/Cellerator/compute/math`
- `exclusive`: `Cellerator/src/compute/math`
- `exclusive`: `Cellerator/tests/math_core`
- `forbidden`: `Baseplane/src`
- `forbidden`: `CellShard`
- `forbidden`: `Cellerator/components/CelleraTorch`
- `read`: `Cellerator/include/Cellerator/execution`
- `read`: `Cellerator/include/Cellerator/planner`

## Dependencies
- `task`: `CE-ARCH-83`
<!-- todo-orchestrator:v2-managed:end -->
