

<!-- todo-orchestrator:v2-managed:start -->
# CE-SS1-N02: Prepare reusable forward and transpose structure

Task revision: `6834`; current project revision is in `todo-status.md`.

## Objective
Construct one reusable topology asset with both physical views using existing packing/projection builders.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
Use host CSR input only as the cold interchange for this slice. Retain edge identity/order and use the existing geometry/packing-to-FMP1 and CTP1 construction facilities.

## Ownership
- `exclusive`: `src/compute/operation/prepared_relation.cu`
- `exclusive`: `tests/semantic_spine/native/preparation_test.cu`
- `forbidden`: `compat/legacy_sparse`
- `forbidden`: `components`
- `read`: `AGENTS.md`
- `read`: `include/Cellerator/compute/operation/operation_core.hh`
- `read`: `include/Cellerator/compute/operation/operation_core_v2/schema.hh`
- `read`: `include/Cellerator/execution/identity.hh`
- `read`: `planning/semantic-spine-v1/01_SCOPE_AND_DECISIONS.md`
- `read`: `planning/semantic-spine-v1/02_SEMANTIC_AND_NATIVE_CONTRACT.md`
- `read`: `planning/semantic-spine-v1/03_PARALLEL_LANES_AND_INTEGRATION.md`
- `read`: `planning/semantic-spine-v1/04_VALIDATION_AND_DEMO.md`
- `read`: `src/compute/projection/physical_feature_major.cc`
- `read`: `src/compute/projection/physical_transpose.cc`
- `read`: `tests/math_core/transpose_backward_candidate_test.cu`

## Dependencies
- `task`: `CE-SS1-N01`
<!-- todo-orchestrator:v2-managed:end -->
