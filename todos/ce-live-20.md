

<!-- todo-orchestrator:v2-managed:start -->
# CE-LIVE-20: CPE2 typed projection activation

Task revision: `2039`; current project revision is in `todo-status.md`.

## Objective
Resolve validated CPE2 projection entries into typed non-owning device views for CPK1 row-masked, CSR, FMP1, and CTP1 with exact identity, orientation, schema, location, size, and map validation.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
Typed non-owning projection activation completed and validated.

## Ownership
- `exclusive`: `components/CellPack/tests/persistence/execution_image_v2_device_test.cu`
- `exclusive`: `include/Cellerator/execution/projection_activation.hh`
- `exclusive`: `src/execution/projection_activation.cc`
- `exclusive`: `tests/execution/projection_activation_test.cu`
- `read`: `components/CellPack/include/CellPack/feature_weighted_row_reduction.hh`
- `read`: `components/CellPack/include/CellPack/persistence/execution_image_v2.hh`
- `read`: `include/Cellerator/compute/math/operation_core/csr_fallback_candidate.hh`
- `read`: `include/Cellerator/compute/math/operation_core/feature_major_small_n_candidate.hh`
- `read`: `include/Cellerator/compute/math/operation_core/operation_core.hh`
- `read`: `include/Cellerator/compute/math/operation_core/row_masked_n1_candidate.hh`
- `read`: `include/Cellerator/compute/math/operation_core/transpose_backward_candidate.hh`
- `read`: `include/Cellerator/compute/math/physical_csr.hh`
- `read`: `include/Cellerator/compute/math/physical_feature_major.hh`
- `read`: `include/Cellerator/compute/math/physical_transpose.hh`

## Dependencies
- `task`: `CE-LIVE-11`
- `task`: `CE-LIVE-13`
- `task`: `CE-LIVE-19`
<!-- todo-orchestrator:v2-managed:end -->
