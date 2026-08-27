

<!-- todo-orchestrator:v2-managed:start -->
# CE-LIVE-22: Strong conventional CSR fallback and hot-path cleanup

Task revision: `2180`; current project revision is in `todo-status.md`.

## Objective
Add a session-integrated cuSPARSE CSR SpMV and SpMM candidate for the live width envelope, remove per-run device selection from the custom CSR path, keep descriptor creation and preprocessing in preparation, and provide a strong fair baseline.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `include/Cellerator/compute/math/operation_core/cusparse_csr_candidate.hh`
- `exclusive`: `src/compute/math/operation_core/csr_fallback_candidate.cu`
- `exclusive`: `src/compute/math/operation_core/cusparse_csr_candidate.cu`
- `exclusive`: `tests/math_core/cusparse_csr_candidate_test.cu`
- `read`: `include/Cellerator/compute/math/physical_csr.hh`
- `read`: `include/Cellerator/runtime`

## Dependencies
- `task`: `CE-LIVE-10`
- `task`: `CE-LIVE-11`
- `task`: `CE-LIVE-20`
<!-- todo-orchestrator:v2-managed:end -->
