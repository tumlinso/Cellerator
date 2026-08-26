

<!-- todo-orchestrator:v2-managed:start -->
# CE-ARCH-88: Implement native sparse-to-dense training vertical slice

Task revision: `1412`; current project revision is in `todo-status.md`.

## Objective
Complete Phase 9 with a module-local learned projection, fused bias/activation/normalization, module-major dense state, native backward, mutable learned values, and topology-stable training step that beats a fair CSR baseline in its declared regime.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
Build one decisive forward/backward training slice over the frozen logical structure and measure it end to end against CSR plus generic SpMM and separate epilogues.

## Ownership
- `exclusive`: `Cellerator/CMakeLists.txt`
- `exclusive`: `Cellerator/bench/architecture_evidence`
- `exclusive`: `Cellerator/include/Cellerator/compute/math`
- `exclusive`: `Cellerator/src/compute/math`
- `exclusive`: `Cellerator/tests/math_core`
- `forbidden`: `Baseplane/src`
- `forbidden`: `CellShard`
- `forbidden`: `Cellerator/components/CelleraTorch`
- `read`: `Cellerator/components/CellPack`
- `read`: `Cellerator/include/Cellerator/execution`
- `read`: `Cellerator/include/Cellerator/planner`
- `read`: `Cellerator/src/model`

## Dependencies
- `task`: `CE-ARCH-87`
<!-- todo-orchestrator:v2-managed:end -->
