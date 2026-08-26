<!-- todo-orchestrator:v2-managed:start -->
# CE-ARCH-87: Complete Objective V2 optimizer integration

Task revision: `1412`; current project revision is in `todo-status.md`.

## Objective
Complete Phase 8 by feeding measured workload-weighted total cost into CP-BP alternating refinement with held-out stability, forward/transpose profiles, activity, and partition-cut terms justified by current evidence.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
Connect the CE-ARCH-77 calibrated predictor to CP-BP refinement, add only measured features, and validate against exact-surrogate, held-out, and bootstrap baselines.

## Ownership
- `exclusive`: `Cellerator/CMakeLists.txt`
- `exclusive`: `Cellerator/bench/architecture_evidence`
- `exclusive`: `Cellerator/components/CellPack/include/CellPack`
- `exclusive`: `Cellerator/components/CellPack/src`
- `exclusive`: `Cellerator/components/CellPack/tests`
- `exclusive`: `Cellerator/include/Cellerator/planner`
- `exclusive`: `Cellerator/src/planner`
- `exclusive`: `Cellerator/tests/planner`
- `forbidden`: `Baseplane/src`
- `forbidden`: `CellShard`
- `forbidden`: `Cellerator/components/CelleraTorch`
- `read`: `Cellerator/data/manifests`
- `read`: `Cellerator/include/Cellerator/compute/math`

## Dependencies
- `task`: `CE-ARCH-86`
<!-- todo-orchestrator:v2-managed:end -->
