

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-12: Source-linked provider registry

Task revision: `3016`; current project revision is in `todo-status.md`.

## Objective
Implement fixed-capacity explicit provider registration, active-device filtering, and a generated compiled-provider manifest with no constructors or dynamic loader.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `cmake/cellerator_provider_manifest.hh.in`
- `exclusive`: `include/Cellerator/compute/architecture/provider.hh`
- `exclusive`: `src/compute/architecture/provider_registry.cc`
- `exclusive`: `tests/ce_geo/hardware/provider_registry_test.cc`
- `read`: `include/Cellerator/compute/architecture/capability.hh`
- `read`: `include/Cellerator/runtime/device_descriptor.hh`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
