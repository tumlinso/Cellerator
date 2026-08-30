

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-100: Baseline golden regression

Task revision: `2999`; current project revision is in `todo-status.md`.

## Objective
Independently rerun frozen CPK1, CPE2, session, program, catalog, sparse, transpose, and experimental WMMA baselines and compare exact golden evidence.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `validated`

## Next Action
_None._

## Ownership
- `exclusive`: `tests/ce_geo/validation/baseline_golden_regression.py`
- `read`: `bench/ce_geo/evidence/baseline`
- `read`: `bench/ce_live/tensor_core`
- `read`: `tests`
- `read`: `tests/ce_geo/baseline`

## Dependencies
- `checkpoint`: `CE-GEO-BASELINE-FROZEN`
<!-- todo-orchestrator:v2-managed:end -->
