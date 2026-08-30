

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-43: Provider validation and activation route

Task revision: `2999`; current project revision is in `todo-status.md`.

## Objective
Route architecture-specific projection host validation and device activation through provider descriptors without a new central type switch.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `include/Cellerator/execution/projection_activation.hh`
- `exclusive`: `src/execution/projection_activation.cc`
- `exclusive`: `tests/ce_geo/persistence/provider_activation_test.cu`
- `read`: `include/Cellerator/compute/architecture/provider.hh`

## Dependencies
- `interface`: `cellerator-device-provider-contract-v1`
<!-- todo-orchestrator:v2-managed:end -->
