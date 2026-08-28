

<!-- todo-orchestrator:v2-managed:start -->
# CE-REMAP-08: Consolidate preprocessing and demote orchestration

Task revision: `2333`; current project revision is in `todo-status.md`.

## Objective
Merge preprocessing islands, move the workbench to tools, and move model and trajectory orchestration to examples after extracting only genuinely reusable native primitives.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `integration_exclusive`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `CMakeLists.txt`
- `exclusive`: `bench/CMakeLists.txt`
- `exclusive`: `bench/preprocess`
- `exclusive`: `docs/REPOSITORY_CONSOLIDATION_PROGRAM.md`
- `exclusive`: `examples/CMakeLists.txt`
- `exclusive`: `examples/models`
- `exclusive`: `examples/trajectory`
- `exclusive`: `include/Cellerator/compute/preprocess`
- `exclusive`: `include/Cellerator/models`
- `exclusive`: `include/Cellerator/preprocess`
- `exclusive`: `include/Cellerator/trajectory`
- `exclusive`: `src/compute/preprocess`
- `exclusive`: `src/models`
- `exclusive`: `src/preprocess`
- `exclusive`: `src/preprocess/CMakeLists.txt`
- `exclusive`: `src/trajectory`
- `exclusive`: `tests/CMakeLists.txt`
- `exclusive`: `tests/preprocess`
- `exclusive`: `tests/trajectory_compile_test.cu`
- `exclusive`: `tests/trajectory_runtime_test.cu`
- `exclusive`: `tools/CMakeLists.txt`
- `exclusive`: `tools/preprocess_workbench`
- `read`: `bench`
- `read`: `components`
- `read`: `docs`
- `read`: `include`
- `read`: `src`
- `read`: `tests`

## Dependencies
- `task`: `CE-REMAP-07`
<!-- todo-orchestrator:v2-managed:end -->
