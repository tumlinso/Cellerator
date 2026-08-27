

<!-- todo-orchestrator:v2-managed:start -->
# CE-LIVE-33: Native training executable integration

Task revision: `2087`; current project revision is in `todo-status.md`.

## Objective
Wrap the validated FMP1 and CTP1 N=16 training slice as an explicit prepared executable path with forward, epilogue, backward, sparse and bias updates, parameter descriptors, readiness transitions, and a fair persistent CSR/cuSPARSE baseline without rebuilding topology.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `bench/ce_live/training`
- `exclusive`: `docs/CE_LIVE_TRAINING_RESULTS.md`
- `exclusive`: `include/Cellerator/execution/training_program.hh`
- `exclusive`: `src/execution/training_program.cu`
- `exclusive`: `tests/execution/training_program_test.cu`
- `read`: `include/Cellerator/execution/program.hh`
- `read`: `include/Cellerator/runtime/value_readiness.cuh`

## Dependencies
- `task`: `CE-LIVE-25`
- `task`: `CE-LIVE-30`
<!-- todo-orchestrator:v2-managed:end -->
