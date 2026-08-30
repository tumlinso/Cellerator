

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-44: Unified geometry acquisition contract

Task revision: `3016`; current project revision is in `todo-status.md`.

## Objective
Define compile-now, CSG1-load, CPE2-load, and CPK1-adapt routes with explicit incompatible-CPE2 fallback policy.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `include/Cellerator/execution/geometry_acquisition.hh`
- `exclusive`: `tests/ce_geo/persistence/geometry_acquisition_contract_test.cc`
- `read`: `include/Cellerator/execution`
- `read`: `include/Cellerator/geometry`

## Dependencies
- `interface`: `cellerator-device-provider-contract-v1`
- `interface`: `cellerator-candidate-catalog-v2`
- `interface`: `cellerator-semantic-geometry-compiler-v1`
- `interface`: `cellerator-csg1-v1`
<!-- todo-orchestrator:v2-managed:end -->
