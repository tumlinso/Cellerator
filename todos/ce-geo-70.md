

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-70: Preserve historical dense-fragment experiment

Task revision: `3281`; current project revision is in `todo-status.md`.

## Objective
Retain the existing V100 dense-fragment code, fixtures, and PBMC3K non-promotion as an untouched reference and negative control outside production mutation.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `validated`

## Next Action
_None._

## Ownership
- `exclusive`: `tests/tensor_core/sm70/historical_dense_fragment_regression.cu`
- `read`: `bench/ce_live/tensor_core`
- `read`: `include/Cellerator/compute/candidate/tensor_core`
- `read`: `src/compute/candidate/tensor_core`

## Dependencies
- `checkpoint`: `CE-GEO-BASELINE-FROZEN`
<!-- todo-orchestrator:v2-managed:end -->
