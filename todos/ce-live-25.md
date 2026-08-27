

<!-- todo-orchestrator:v2-managed:start -->
# CE-LIVE-25: Integrate value readiness with native training and parameters

Task revision: `2212`; current project revision is in `todo-status.md`.

## Objective
Use readiness in the native training slice, preserve topology across updates, expose native parameter descriptors, return explicit next-generation readiness, and prove same-stream and cross-stream correctness without premature publication.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `include/Cellerator/compute/math/native_training_slice.hh`
- `exclusive`: `include/Cellerator/parameters.hh`
- `exclusive`: `src/compute/math/native_training_slice.cu`
- `exclusive`: `tests/math_core/native_training_slice_test.cu`
- `read`: `include/Cellerator/runtime/value_readiness.cuh`

## Dependencies
- `task`: `CE-LIVE-11`
- `task`: `CE-LIVE-14`
- `task`: `CE-LIVE-19`
<!-- todo-orchestrator:v2-managed:end -->
