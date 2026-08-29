

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-123: Integrate complete Volta implementation

Task revision: `2379`; current project revision is in `todo-status.md`.

## Objective
Own final shared CMake manifests, package exports, central catalog assembly, program wiring, benchmark/test registration, and cross-module compatibility after all Volta branches have dispositions.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `integration_exclusive`
- Result: `-`

## Next Action
_None._

## Ownership
- `exclusive`: `CMakeLists.txt`
- `exclusive`: `bench/CMakeLists.txt`
- `exclusive`: `examples/CMakeLists.txt`
- `exclusive`: `include/Cellerator/Cellerator.hh`
- `exclusive`: `src/CMakeLists.txt`
- `exclusive`: `src/compute/CMakeLists.txt`
- `exclusive`: `src/compute/operation/ce_geo_catalog_assembly.cc`
- `exclusive`: `src/execution/CMakeLists.txt`
- `exclusive`: `src/geometry/CMakeLists.txt`
- `exclusive`: `tests/CMakeLists.txt`
- `read`: `bench`
- `read`: `cmake`
- `read`: `examples`
- `read`: `include`
- `read`: `src`
- `read`: `tests`

## Dependencies
- `barrier`: `CE-GEO-VOLTA-IMPLEMENTATION-FANIN`
<!-- todo-orchestrator:v2-managed:end -->
