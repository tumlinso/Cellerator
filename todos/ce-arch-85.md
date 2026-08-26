<!-- todo-orchestrator:v2-managed:start -->
# CE-ARCH-85: Implement transpose projection and native backward

Task revision: `1412`; current project revision is in `todo-status.md`.

## Objective
Complete the reverse half of Phase 6 by sharing logical edge identity and mutable values between forward and transpose projections and executing a native backward/propagation operation without topology reconstruction.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
Implement a versioned transpose projection/value map and operation candidate, prove forward/backward edge identity and multi-generation reuse, then collect bounded CUDA parity evidence.

## Ownership
- `exclusive`: `Cellerator/CMakeLists.txt`
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
- `task`: `CE-ARCH-84`
<!-- todo-orchestrator:v2-managed:end -->
