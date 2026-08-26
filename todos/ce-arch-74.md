<!-- todo-orchestrator:v2-managed:start -->
# CE-ARCH-74: Prove immutable structure and mutable value separation

Task revision: `1084`; current project revision is in `todo-status.md`.

## Objective
Bind at least two different value generations to one immutable CP-BP structure and projection and prove correct reuse and stale-generation rejection.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
Exercise one immutable CP-BP structure/projection with two valid value generations and adversarial stale-generation, structure-epoch, projection, and pointer-relocation checks.

## Ownership
- `exclusive`: `Cellerator/CMakeLists.txt`
- `exclusive`: `Cellerator/docs/current_implementation.qmd`
- `exclusive`: `Cellerator/include/Cellerator/compute/math/operation_core/csr_fallback_candidate.hh`
- `exclusive`: `Cellerator/include/Cellerator/compute/math/operation_core/operation_core.hh`
- `exclusive`: `Cellerator/include/Cellerator/compute/math/operation_core/row_masked_n1_candidate.hh`
- `exclusive`: `Cellerator/include/Cellerator/execution/launch_bindings.hh`
- `exclusive`: `Cellerator/include/Cellerator/execution/lifetimes.hh`
- `exclusive`: `Cellerator/src/compute/math/operation_core/csr_fallback_candidate.cu`
- `exclusive`: `Cellerator/src/compute/math/operation_core/operation_core.cc`
- `exclusive`: `Cellerator/src/compute/math/operation_core/row_masked_n1_candidate.cu`
- `exclusive`: `Cellerator/tests/math_core/csr_fallback_candidate_test.cu`
- `exclusive`: `Cellerator/tests/math_core/row_masked_n1_candidate_test.cu`
- `exclusive`: `Cellerator/tests/math_core/value_generation_reuse_test.cu`
- `forbidden`: `Baseplane`
- `forbidden`: `CellShard`
- `forbidden`: `Cellerator/components/CellPack/include/CellPack/warp_tiles.hh`
- `forbidden`: `Cellerator/components/CellPack/src/warp_tiles.cc`
- `forbidden`: `Cellerator/components/CellPack/src/warp_tiles_cuda.cu`
- `forbidden`: `Cellerator/src/compute/sparse/ops/kernels`
- `read`: `Cellerator/docs/core_execution_cp_math.qmd`
- `read`: `Cellerator/include/Cellerator/planner`
- `read`: `Cellerator/src/planner`
- `read`: `Cellerator/tests/execution_identity`
- `read`: `Cellerator/tests/planner`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
