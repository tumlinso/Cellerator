

<!-- todo-orchestrator:v2-managed:start -->
# CE-SS1-V02: Build contract and identity regression probes

Task revision: `6812`; current project revision is in `todo-status.md`.

## Objective
Exercise semantic invariants that a tiny numerical example alone can miss.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
Check high-bit-only identity changes, source/destination axis mistakes at equal extents, stale epochs/order, pointer changes and semantic equality across provenance.

## Ownership
- `exclusive`: `tests/semantic_spine/verification/CMakeLists.txt`
- `exclusive`: `tests/semantic_spine/verification/contract_probes.cc`
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

## Dependencies
- `task`: `CE-SS1-V01`
<!-- todo-orchestrator:v2-managed:end -->
