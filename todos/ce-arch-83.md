<!-- todo-orchestrator:v2-managed:start -->
# CE-ARCH-83: Complete instrumentation and representative benchmark corpus

Task revision: `1412`; current project revision is in `todo-status.md`.

## Objective
Complete Phase 5 with real and adversarial structures, full end-to-end phase accounting, forward/transpose observability, reproducible artifact identity, and planner-ready structural features.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
Extend the CE-ARCH-73/76 evidence machinery to the declared real/adversarial corpus and missing forward/transpose and hardware-pressure fields without launching broad tuning.

## Ownership
- `exclusive`: `Cellerator/CMakeLists.txt`
- `exclusive`: `Cellerator/bench/architecture_evidence`
- `exclusive`: `Cellerator/include/Cellerator/planner/instrumentation`
- `exclusive`: `Cellerator/src/planner/instrumentation`
- `exclusive`: `Cellerator/tests/architecture_evidence`
- `forbidden`: `Baseplane/src`
- `forbidden`: `CellShard/src`
- `forbidden`: `Cellerator/data/*.csh5`
- `read`: `Cellerator/components/CellPack`
- `read`: `Cellerator/data/manifests`
- `read`: `Cellerator/include/Cellerator/compute`

## Dependencies
- `task`: `CE-ARCH-82`
<!-- todo-orchestrator:v2-managed:end -->
