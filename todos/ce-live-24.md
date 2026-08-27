

<!-- todo-orchestrator:v2-managed:start -->
# CE-LIVE-24: Native quantitative fixture adapter and independent referee

Task revision: `2185`; current project revision is in `todo-status.md`.

## Objective
Bind the quantitative fixture to exact Cellerator identities, build the forward feature-to-cell relation and mutable generations, generate deterministic dense operands, and compare supported outputs against an independent CPU reference.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `bench/ce_live/runtime_fixture`
- `exclusive`: `docs/CE_LIVE_QUANTITATIVE_EXECUTION.md`
- `exclusive`: `tests/live/quantitative_relation_test.cu`
- `read`: `components/CellPack`
- `read`: `include/Cellerator/execution`

## Dependencies
- `task`: `CE-LIVE-11`
- `task`: `CE-LIVE-12`
- `task`: `CE-LIVE-19`
<!-- todo-orchestrator:v2-managed:end -->
