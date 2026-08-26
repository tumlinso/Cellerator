<!-- todo-orchestrator:v2-managed:start -->
# CE-ARCH-73: Build first real end-to-end planner measurement path

Task revision: `1055`; current project revision is in `todo-status.md`.

## Objective
Measure projection preparation, input ordering, prepared execution, candidate-private output, referee, all total-cost phases, and winner preparation for real registered candidates.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
Implement the bounded real CUDA measurement harness for the registered CP-BP and CSR candidates with candidate-private outputs and a neutral referee.

## Ownership
- `exclusive`: `Cellerator/CMakeLists.txt`
- `exclusive`: `Cellerator/docs/current_implementation.qmd`
- `exclusive`: `Cellerator/include/Cellerator/planner/candidate_measurement.hh`
- `exclusive`: `Cellerator/include/Cellerator/planner/end_to_end_planner.hh`
- `exclusive`: `Cellerator/src/planner/candidate_measurement.cu`
- `exclusive`: `Cellerator/src/planner/end_to_end_planner.cc`
- `exclusive`: `Cellerator/tests/planner/candidate_measurement_test.cu`
- `exclusive`: `Cellerator/tests/planner/end_to_end_planner_test.cc`
- `forbidden`: `Baseplane`
- `forbidden`: `CellShard`
- `forbidden`: `Cellerator/components/CellPack/include/CellPack/warp_tiles.hh`
- `forbidden`: `Cellerator/components/CellPack/src/warp_tiles.cc`
- `forbidden`: `Cellerator/components/CellPack/src/warp_tiles_cuda.cu`
- `forbidden`: `Cellerator/src/compute/sparse/ops/kernels`
- `read`: `Cellerator/bench`
- `read`: `Cellerator/include/Cellerator/compute/math/operation_core/csr_fallback_candidate.hh`
- `read`: `Cellerator/include/Cellerator/compute/math/operation_core/operation_core.hh`
- `read`: `Cellerator/include/Cellerator/compute/math/operation_core/row_masked_n1_candidate.hh`
- `read`: `Cellerator/include/Cellerator/execution`
- `read`: `Cellerator/include/Cellerator/runtime`
- `read`: `Cellerator/src/compute/math/operation_core/csr_fallback_candidate.cu`
- `read`: `Cellerator/src/compute/math/operation_core/operation_core.cc`
- `read`: `Cellerator/src/compute/math/operation_core/row_masked_n1_candidate.cu`
- `read`: `Cellerator/src/runtime`
- `read`: `Cellerator/tests`
- `read`: `Cellerator/tests/math_core/csr_fallback_candidate_test.cu`
- `read`: `Cellerator/tests/math_core/row_masked_n1_candidate_test.cu`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
