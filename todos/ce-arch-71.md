<!-- todo-orchestrator:v2-managed:start -->
# CE-ARCH-71: Register CP-BP row-masked N=1 as an operation-core candidate

Task revision: `976`; current project revision is in `todo-status.md`.

## Objective
Register the existing CP-BP v1 native row-masked N=1 kernel as a real operation-core and planner candidate without changing its projection or kernel semantics.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
Register a truthful N=1 candidate adapter over the preserved CPK1/native-tile/direct-kernel path, add focused capability, binding, order, effect, and parity tests, then run the declared host and CUDA gates.

## Ownership
- `exclusive`: `Cellerator/CMakeLists.txt`
- `exclusive`: `Cellerator/docs/current_implementation.qmd`
- `exclusive`: `Cellerator/include/Cellerator/compute/math/operation_core/row_masked_n1_candidate.hh`
- `exclusive`: `Cellerator/src/compute/math/operation_core/row_masked_n1_candidate.cu`
- `exclusive`: `Cellerator/tests/math_core/row_masked_n1_candidate_test.cu`
- `forbidden`: `Baseplane`
- `forbidden`: `CellShard`
- `forbidden`: `Cellerator/components/CellPack/include/CellPack/warp_tiles.hh`
- `forbidden`: `Cellerator/components/CellPack/src/warp_tiles.cc`
- `forbidden`: `Cellerator/components/CellPack/src/warp_tiles_cuda.cu`
- `read`: `Cellerator/components/CellPack/include/CellPack/feature_weighted_row_reduction.hh`
- `read`: `Cellerator/components/CellPack/include/CellPack/feature_weighted_row_reduction_cuda.hh`
- `read`: `Cellerator/components/CellPack/include/CellPack/persistent_packing_payload.hh`
- `read`: `Cellerator/components/CellPack/src/feature_weighted_row_reduction_cuda.cu`
- `read`: `Cellerator/components/CellPack/tests/feature_weighted_row_reduction_cuda_test.cu`
- `read`: `Cellerator/include/Cellerator/compute/math/native_tile_view.hh`
- `read`: `Cellerator/include/Cellerator/compute/math/operation_core/operation_core.hh`
- `read`: `Cellerator/include/Cellerator/execution`
- `read`: `Cellerator/include/Cellerator/planner/end_to_end_planner.hh`
- `read`: `Cellerator/src/compute/math/native_tile_view.cc`
- `read`: `Cellerator/src/compute/math/operation_core/operation_core.cc`
- `read`: `Cellerator/src/planner/end_to_end_planner.cc`
- `read`: `Cellerator/tests/math_core/operation_core_test.cc`
- `read`: `Cellerator/tests/math_native_tile_adapter_test.cc`
- `read`: `Cellerator/tests/planner/end_to_end_planner_test.cc`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
