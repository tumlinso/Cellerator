

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-121: Integrate first sm_70 vertical slice

Task revision: `3281`; current project revision is in `todo-status.md`.

## Objective
Integrate support evidence, target refinement, physical projection, N64 provider/kernel, residual, planner cost, and prepared execution without altering historical experiment.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `integration_exclusive`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `src/compute/CMakeLists.txt`
- `exclusive`: `src/compute/architecture/providers/nvidia/common/CMakeLists.txt`
- `exclusive`: `src/compute/architecture/providers/nvidia/sm70/CMakeLists.txt`
- `exclusive`: `src/compute/architecture/providers/nvidia/sm70/catalog_fragment.cc`
- `exclusive`: `tests/CMakeLists.txt`
- `exclusive`: `tests/ce_geo/run_sm70_vertical_suite.py`
- `exclusive`: `tests/tensor_core/sm70/CMakeLists.txt`
- `read`: `include/Cellerator/compute/projection/physical_mma_hybrid.hh`
- `read`: `include/Cellerator/geometry/support_atlas.hh`
- `read`: `src/compute/architecture/providers/nvidia`
- `read`: `src/compute/projection`
- `read`: `src/planner`

## Dependencies
- `barrier`: `CE-GEO-FIRST-VERTICAL-FANIN`
<!-- todo-orchestrator:v2-managed:end -->
