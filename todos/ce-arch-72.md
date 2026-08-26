

<!-- todo-orchestrator:v2-managed:start -->
# CE-ARCH-72: Register legal CSR conventional fallback

Task revision: `1016`; current project revision is in `todo-status.md`.

## Objective
Register the existing legal CSR implementation as the first conventional operation-core/planner fallback.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
Expose the existing validated CSR path as a legal fallback candidate with complete binding and output contracts.

## Ownership
- `exclusive`: `Cellerator/CMakeLists.txt`
- `exclusive`: `Cellerator/docs/current_implementation.qmd`
- `exclusive`: `Cellerator/include/Cellerator/compute/math/operation_core/csr_fallback_candidate.hh`
- `exclusive`: `Cellerator/src/compute/math/operation_core/csr_fallback_candidate.cu`
- `exclusive`: `Cellerator/tests/math_core/csr_fallback_candidate_test.cu`
- `forbidden`: `Baseplane`
- `forbidden`: `CellShard`
- `forbidden`: `Cellerator/components/CellPack/include/CellPack/warp_tiles.hh`
- `forbidden`: `Cellerator/components/CellPack/src/warp_tiles.cc`
- `forbidden`: `Cellerator/components/CellPack/src/warp_tiles_cuda.cu`
- `read`: `Cellerator/include/Cellerator/compute/math/operation_core/operation_core.hh`
- `read`: `Cellerator/include/Cellerator/compute/math/operation_core/row_masked_n1_candidate.hh`
- `read`: `Cellerator/include/Cellerator/compute/math/physical_csr.hh`
- `read`: `Cellerator/include/Cellerator/execution`
- `read`: `Cellerator/include/Cellerator/planner/end_to_end_planner.hh`
- `read`: `Cellerator/include/Cellerator/runtime`
- `read`: `Cellerator/src/compute/math/operation_core/operation_core.cc`
- `read`: `Cellerator/src/compute/math/operation_core/row_masked_n1_candidate.cu`
- `read`: `Cellerator/src/compute/math/physical_csr.cc`
- `read`: `Cellerator/src/compute/sparse/ops/kernels/base_sparse.cu`
- `read`: `Cellerator/src/compute/sparse/ops/ops.hh`
- `read`: `Cellerator/src/planner/end_to_end_planner.cc`
- `read`: `Cellerator/src/runtime`
- `read`: `Cellerator/tests/math_core/row_masked_n1_candidate_test.cu`
- `read`: `Cellerator/tests/math_execution_csr_test.cu`
- `read`: `Cellerator/tests/sparse_ops_runtime_test.cu`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
