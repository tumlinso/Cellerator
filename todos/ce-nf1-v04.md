

<!-- todo-orchestrator:v2-managed:start -->
# CE-NF1-V04: Support f32 authoritative values alongside f16 projections

Task revision: `7260`; current project revision is in `todo-status.md`.

## Objective
_None._

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `include/Cellerator/compute/candidate/sparse/project.hh`
- `exclusive`: `include/Cellerator/compute/operation/prepared_relation.hh`
- `exclusive`: `include/Cellerator/execution/native_value_instance`
- `exclusive`: `src/compute/candidate/sparse/kernels/csr_spmm_fwd_kernel_.cuh`
- `exclusive`: `src/compute/candidate/sparse/project.cu`
- `exclusive`: `src/compute/operation/prepared_relation.cu`
- `exclusive`: `src/execution/native_value_instance`
- `exclusive`: `tests/native_foundation/values`

## Dependencies
- `task`: `CE-NF1-V03`
<!-- todo-orchestrator:v2-managed:end -->
