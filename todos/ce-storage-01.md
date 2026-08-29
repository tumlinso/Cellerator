

<!-- todo-orchestrator:v2-managed:start -->
# CE-STORAGE-01: Remove production storage ownership from Cellerator

Task revision: `2374`; current project revision is in `todo-status.md`.

## Objective
Delete the file-backed dataset subsystem, make deterministic sampling and sampled CSR materialization storage-neutral, invert CellShard execution-payload coupling, remove HDF5 and storage-only CellShard dependencies, add a production no-disk guard, and preserve frozen CPE2 behavior.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `project_exclusive`
- Result: `implemented`

## Next Action
Work from the current dirty post-BioPrep tree, preserve its changes, split only storage-owned APIs, add focused CPU/static tests and the architecture guard, and validate without CUDA execution while X topology is absent.

## Ownership
- `exclusive`: `CMakeLists.txt`
- `exclusive`: `include/Cellerator/Cellerator.hh`
- `exclusive`: `include/Cellerator/compute/dataset.hh`
- `exclusive`: `include/Cellerator/compute/sampling.hh`
- `exclusive`: `include/Cellerator/compute/sampling_materialization.hh`
- `exclusive`: `include/Cellerator/execution/opaque_artifact.hh`
- `exclusive`: `include/Cellerator/interop/cellshard/dataset.hh`
- `exclusive`: `src/compute/current_targets.cmake`
- `exclusive`: `src/compute/dataset`
- `exclusive`: `src/execution/opaque_artifact.cc`
- `exclusive`: `src/geometry/owned_targets.cmake`
- `exclusive`: `tests/CMakeLists.txt`
- `exclusive`: `tests/architecture`
- `exclusive`: `tests/dataset_runtime_test.cc`
- `exclusive`: `tests/persistence/opaque_execution_artifact_test.cu`
- `exclusive`: `tests/sampling_materialization_runtime_test.cc`
- `exclusive`: `tests/sampling_runtime_test.cc`
- `exclusive`: `tools`
- `read`: `bench`
- `read`: `compat`
- `read`: `docs`
- `read`: `include`
- `read`: `src`
- `read`: `tests`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
