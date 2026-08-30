

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-104: Numerical referee and tolerance evidence

Task revision: `2999`; current project revision is in `todo-status.md`.

## Objective
Implement operand-precision and FP32/FP64 logical referees; report max absolute, relative L2/Frobenius, mixed tolerance, degree/depth error, tails, alpha/beta, and NaN/Inf policy.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `validated`

## Next Action
_None._

## Ownership
- `exclusive`: `bench/ce_geo/evidence/numerical_policy.json`
- `exclusive`: `tests/ce_geo/validation/numerical_referee.hh`
- `exclusive`: `tests/ce_geo/validation/numerical_referee_test.cc`
- `read`: `include/Cellerator/compute/math/referee.hh`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
