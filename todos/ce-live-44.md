

<!-- todo-orchestrator:v2-managed:start -->
# CE-LIVE-44: Torch quantitative biological smoke and performance validation

Task revision: `2087`; current project revision is in `todo-status.md`.

## Objective
Run the same quantitative fixture through native Cellerator and CelleraTorch forward/autograd paths, prove numerical and identity parity, verify current-stream behavior, and measure adapter overhead.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `serial`
- Result: `-`

## Next Action
_None._

## Ownership
- `exclusive`: `bench/ce_live/celleratorch`
- `exclusive`: `components/CelleraTorch/tests/quantitative_smoke_test.cc`
- `exclusive`: `docs/CE_LIVE_CELLERATORCH_RESULTS.md`
- `read`: `AGENTS.md`
- `read`: `ARCHITECTURE_FOLLOWUPS.md`
- `read`: `CMakeLists.txt`
- `read`: `bench`
- `read`: `components`
- `read`: `data`
- `read`: `docs`
- `read`: `include`
- `read`: `planning_strategy.md`
- `read`: `scope.md`
- `read`: `scripts`
- `read`: `src`
- `read`: `tests`

## Dependencies
- `task`: `CE-LIVE-31`
- `task`: `CE-LIVE-33`
- `task`: `CE-LIVE-43`
<!-- todo-orchestrator:v2-managed:end -->
