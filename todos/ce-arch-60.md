<!-- todo-orchestrator:v2-managed:start -->
# CE-ARCH-60: Migration, cleanup, documentation, and final recovery audit

Task revision: `780`; current project revision is in `todo-status.md`.

## Objective
Complete migration from experimental CP-Math and direct CP-BP v1 coupling into the validated biological execution architecture.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `integration_exclusive`
- Result: `validated`

## Next Action
Complete. Supported targets are consolidated, harmful CP-Math runtime assumptions are retired, compatibility evidence is preserved, and all final host/GPU/audit gates pass.

## Ownership
- `exclusive`: `Cellerator/AGENTS.md`
- `exclusive`: `Cellerator/CMakeLists.txt`
- `exclusive`: `Cellerator/bench/architecture_evidence/benchmark_contract.json`
- `exclusive`: `Cellerator/bench/cuda_background_contract.json`
- `exclusive`: `Cellerator/include/Cellerator/compute/math`
- `exclusive`: `Cellerator/include/Cellerator/runtime/SESSION.md`
- `exclusive`: `Cellerator/optimization.md`
- `exclusive`: `Cellerator/planning_strategy.md`
- `exclusive`: `Cellerator/scope.md`
- `exclusive`: `Cellerator/src/compute/math`
- `exclusive`: `Cellerator/tests/math_cusparse_bell_test.cu`
- `exclusive`: `Cellerator/tests/math_device_runtime_test.cu`
- `exclusive`: `Cellerator/tests/math_planner_test.cc`
- `exclusive`: `Cellerator/tests/math_runtime_test.cu`
- `forbidden`: `CellShard`
- `read`: `Baseplane`
- `read`: `CellShard`

## Dependencies
- `task`: `CE-ARCH-31`
- `task`: `CE-ARCH-40`
- `task`: `CE-ARCH-50`
<!-- todo-orchestrator:v2-managed:end -->
