

<!-- todo-orchestrator:v2-managed:start -->
# CE-NF1A-RESPONSE: Deliver native directional response

Task revision: `7374`; current project revision is in `todo-status.md`.

## Objective
Compose state/parameter JVP/VJP and supported second directions through the real numerical program with correct primal lifetime, precision, branch and invalidation semantics.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
Read planning/nf1-adaptive-v1/outcomes/RESPONSE.md and relevant source. Choose a short useful implementation loop; preserve actual acceptance.

## Ownership
- `exclusive`: `cmake/NativeFoundation.cmake`
- `exclusive`: `docs/nf1_adaptive/response`
- `exclusive`: `include/Cellerator/compute/operation/differential`
- `exclusive`: `include/Cellerator/compute/operation/indexed_mechanism/evaluators.hh`
- `exclusive`: `include/Cellerator/compute/operation/native_foundation_contract.hh`
- `exclusive`: `include/Cellerator/compute/operation/native_numeric/device_linear.hh`
- `exclusive`: `include/Cellerator/execution/program/program_v2.h`
- `exclusive`: `src/compute/CMakeLists.txt`
- `exclusive`: `src/compute/operation/differential`
- `exclusive`: `src/compute/operation/indexed_mechanism/evaluators.cc`
- `exclusive`: `src/compute/operation/indexed_mechanism/evaluators.cu`
- `exclusive`: `src/compute/operation/native_numeric/device_linear.cu`
- `exclusive`: `src/execution/program/program_v2.cc`
- `exclusive`: `tests/native_foundation/differential`
- `exclusive`: `tests/native_foundation/program`

## Dependencies
- `task`: `CE-NF1A-CORE`
- `task`: `CE-NF1A-MECHANISMS`
<!-- todo-orchestrator:v2-managed:end -->
