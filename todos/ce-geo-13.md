

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-13: Separate build inclusion from tuning policy

Task revision: `2999`; current project revision is in `todo-status.md`.

## Objective
Implement provider inclusion and tuning options, compatibility aliases, and precise provider performance helpers without implicit fast math, cache forcing, register caps, or launch bounds.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `integration_exclusive`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `cmake/CelleratorCudaProviders.cmake`
- `exclusive`: `cmake/cellerator_provider_manifest.hh.in`
- `exclusive`: `src/compute/architecture/provider_performance.cuh`
- `read`: `CMakeLists.txt`
- `read`: `src/CMakeLists.txt`
- `read`: `src/compute/CMakeLists.txt`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
