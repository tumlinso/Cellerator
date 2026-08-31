

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-14: Provider-contract validation and fake provider

Task revision: `3281`; current project revision is in `todo-status.md`.

## Objective
Prove a fake source-linked provider and candidate can be added without central runtime modification; validate descriptor, capability, memory-interface, registry, manifest, and sealed-session contracts.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `tests/ce_geo/hardware/fake_provider.cc`
- `exclusive`: `tests/ce_geo/hardware/provider_contract_test.cu`
- `read`: `include/Cellerator/compute/architecture`
- `read`: `include/Cellerator/runtime/device_descriptor.hh`
- `read`: `src/compute/architecture`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
