

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-86: Operation-core schema and catalog integration

Task revision: `3016`; current project revision is in `todo-status.md`.

## Objective
Add explicit versioned operation kinds and catalog fragments through reviewed transition without reinterpreting frozen v1 meanings.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `integration_exclusive`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `include/Cellerator/compute/operation/relation_algebra_catalog.hh`
- `exclusive`: `src/compute/operation/relation_algebra_catalog.cc`
- `exclusive`: `tests/relation_algebra/catalog_integration_test.cu`
- `read`: `include/Cellerator/compute/operation/operation_core.hh`
- `read`: `src/compute/operation/operation_core.cc`

## Dependencies
- `interface`: `cellerator-candidate-catalog-v2`
<!-- todo-orchestrator:v2-managed:end -->
