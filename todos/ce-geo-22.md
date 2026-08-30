

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-22: Activated projection reference v2

Task revision: `3016`; current project revision is in `todo-status.md`.

## Objective
Implement provider-erased activated projection references carrying provider, view type, ABI, schema, variant, and capability identity.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `include/Cellerator/execution/projection_activation_v2.hh`
- `exclusive`: `src/execution/projection_activation_v2.cc`
- `exclusive`: `tests/ce_geo/catalog/projection_reference_v2_test.cc`
- `read`: `include/Cellerator/execution/projection_activation.hh`

## Dependencies
- `interface`: `cellerator-device-provider-contract-v1`
<!-- todo-orchestrator:v2-managed:end -->
