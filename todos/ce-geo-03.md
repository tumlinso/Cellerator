

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-03: Freeze compatibility and negative-control baseline

Task revision: `2430`; current project revision is in `todo-status.md`.

## Objective
Record CPK1 and CPE2 bytes and fixtures, catalog/program/session behavior, exact regression commands, PBMC3K Tensor Core non-promotion, and the untouched historical dense-fragment path.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `validated`

## Next Action
_None._

## Ownership
- `exclusive`: `bench/ce_geo/evidence/baseline`
- `exclusive`: `tests/ce_geo/baseline`
- `read`: `bench/ce_live/tensor_core`
- `read`: `include/Cellerator/compute/candidate`
- `read`: `include/Cellerator/geometry/persistence`
- `read`: `src/compute/candidate`
- `read`: `src/execution`
- `read`: `src/geometry/persistence`
- `read`: `tests`

## Dependencies
- `task`: `CE-GEO-02`
<!-- todo-orchestrator:v2-managed:end -->
