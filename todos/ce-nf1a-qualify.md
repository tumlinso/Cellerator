

<!-- todo-orchestrator:v2-managed:start -->
# CE-NF1A-QUALIFY: Integrate continuously and publish consumer-ready Cellerator

Task revision: `7407`; current project revision is in `todo-status.md`.

## Objective
Integrate CORE first and later producer commits as they arrive, keep central build hooks usable, independently qualify the combined stack and evaluate full-program performance. Publish the pinned consumer-ready receipt.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `validated`

## Next Action
Read planning/nf1-adaptive-v1/outcomes/QUALIFY.md and relevant source. Choose a short useful implementation loop; preserve actual acceptance.

## Ownership
- `exclusive`: `CMakeLists.txt`
- `exclusive`: `bench/native_foundation`
- `exclusive`: `cmake/NativeFoundation.cmake`
- `exclusive`: `cmake/SemanticSpineV1.cmake`
- `exclusive`: `docs/nf1_adaptive/developmental`
- `exclusive`: `docs/nf1_adaptive/performance`
- `exclusive`: `docs/nf1_adaptive/qualification`
- `exclusive`: `include/Cellerator/compute/operation/native_foundation_optimization`
- `exclusive`: `src/compute/CMakeLists.txt`
- `exclusive`: `src/compute/operation/native_foundation_optimization`
- `exclusive`: `src/execution/CMakeLists.txt`
- `exclusive`: `src/runtime/CMakeLists.txt`
- `exclusive`: `tests/native_foundation/conformance`
- `exclusive`: `tests/native_foundation/integration`
- `exclusive`: `tests/native_foundation/reference`
- `exclusive`: `tests/semantic_spine/core/descriptor_test.cc`

## Dependencies
- `task`: `CE-NF1A-CORE`
<!-- todo-orchestrator:v2-managed:end -->
