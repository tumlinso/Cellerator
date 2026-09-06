<!-- todo-orchestrator:v2-managed:start -->
# CE-SS1-A01: Separate scalar support dot from edge-channel product

Task revision: `6758`; current project revision is in `todo-status.md`.

## Objective
Remove the concrete contraction ambiguity without integrating the entire contraction portfolio.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `parallel_safe`
- Result: `-`

## Next Action
Define scalar dot[e]=sum_k A[s(e),k]B[d(e),k] separately from product[e,k]=A[s(e),k]B[d(e),k]. Prefer an explicit result-shape/reduction tag rather than another arbitrary global enum migration.

## Ownership
- `exclusive`: `include/Cellerator/compute/decomposition/support_embedding_v1.hh`
- `exclusive`: `include/Cellerator/compute/operation/support_product_semantics.hh`
- `exclusive`: `src/compute/decomposition/support_embedding_v1.cc`
- `exclusive`: `tests/semantic_spine/algebra/support_product_test.cc`
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
- `read`: `src/compute/architecture/providers/nvidia/sm70/contract/sparse_contract_v1.cu`

## Dependencies
- `task`: `CE-SS1-I02`
<!-- todo-orchestrator:v2-managed:end -->
