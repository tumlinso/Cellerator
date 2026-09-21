

<!-- todo-orchestrator:v2-managed:start -->
# CE-NF1A-CORE: Deliver the general-width linked numerical core

Task revision: `7315`; current project revision is in `todo-status.md`.

## Objective
Reuse preserved B/P/V/N implementation and relevant D01/T evidence, coordinate donor ownership, and complete general-width FP32 primitives, prepared execution, independent values and a real separately linked CPU/CUDA consumer.

## State
- Lifecycle: `in_progress`
- Execution: `claimed`
- Parallel policy: `parallel_safe`
- Result: `-`

## Next Action
Read planning/nf1-adaptive-v1/outcomes/CORE.md and relevant source. Choose a short useful implementation loop; preserve actual acceptance.

## Ownership
- `exclusive`: `CMakeLists.txt`
- `exclusive`: `cmake/NativeFoundation.cmake`
- `exclusive`: `cmake/SemanticSpineV1.cmake`
- `exclusive`: `docs/nf1_adaptive/core`
- `exclusive`: `include/Cellerator/compute/operation/native_numeric`
- `exclusive`: `include/Cellerator/compute/operation/prepared_relation.hh`
- `exclusive`: `include/Cellerator/execution/native_value_instance`
- `exclusive`: `include/Cellerator/execution/program`
- `exclusive`: `src/compute/CMakeLists.txt`
- `exclusive`: `src/compute/operation/native_numeric`
- `exclusive`: `src/compute/operation/prepared_relation.cu`
- `exclusive`: `src/execution/CMakeLists.txt`
- `exclusive`: `src/execution/native_value_instance`
- `exclusive`: `src/execution/program`
- `exclusive`: `src/runtime/CMakeLists.txt`
- `exclusive`: `tests/native_foundation/build`
- `exclusive`: `tests/native_foundation/numeric`
- `exclusive`: `tests/native_foundation/program`
- `exclusive`: `tests/native_foundation/values`

## Dependencies
- `task`: `CE-NF1A-ADOPT`
<!-- todo-orchestrator:v2-managed:end -->
