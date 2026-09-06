

<!-- todo-orchestrator:v2-managed:start -->
# CE-SS1-N01: Adapt the canonical descriptor to existing preparation

Task revision: `6812`; current project revision is in `todo-status.md`.

## Objective
Translate canonical semantics into the narrow existing executable candidate contracts without dropping meaning.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
Write one checked adapter for the FMP1 N1 forward and CTP1 N1 transpose path. Compare numeric types separately: f16 weights must not be overwritten by f32 state metadata.

## Ownership
- `exclusive`: `src/compute/operation/prepared_relation.cu`
- `exclusive`: `tests/semantic_spine/native/adapter_test.cu`
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
- `read`: `src/compute/candidate/feature_major_small_n_candidate.cu`
- `read`: `src/compute/candidate/transpose_backward_candidate.cu`
- `read`: `src/compute/operation/builtin_catalog.cc`
- `read`: `src/compute/operation/preparation_factory.cc`

## Dependencies
- `task`: `CE-SS1-I02`
<!-- todo-orchestrator:v2-managed:end -->
