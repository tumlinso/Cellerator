

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-108: Compute Sanitizer campaign

Task revision: `2999`; current project revision is in `todo-status.md`.

## Objective
Run memcheck and applicable race/init tools over new device views, rebind, value pack, MMA, residual, segments, transpose, contraction, and gradients with exact binary/toolchain evidence.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `validated`

## Next Action
_None._

## Ownership
- `exclusive`: `bench/ce_geo/evidence/sanitizer`
- `exclusive`: `src/compute/candidate/segment/normalize.cu`
- `exclusive`: `tests/ce_geo/validation/run_compute_sanitizer.py`
- `read`: `tests/ce_geo`
- `read`: `tests/relation_algebra`
- `read`: `tests/tensor_core/sm70`

## Dependencies
- `task`: `CE-GEO-107`
<!-- todo-orchestrator:v2-managed:end -->
