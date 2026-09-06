

<!-- todo-orchestrator:v2-managed:start -->
# CE-SS1-I03: Retire .ceh files without expanding the language milestone

Task revision: `6834`; current project revision is in `todo-status.md`.

## Objective
Apply the agreed file-format decision while preserving valuable source and recording deferred language infrastructure.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
Inspect all tracked/untracked project-owned .ceh files locally; source-reader failure is not evidence they are empty. Preserve pre-existing user bytes and compare with the I01 inventory.

## Ownership
- `exclusive`: `docs/semantic_spine_v1/ceh_disposition.json`
- `exclusive`: `docs/semantic_spine_v1/cell_units.md`
- `exclusive`: `library`
- `exclusive`: `stdlib`
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
- `task`: `CE-SS1-I02`
<!-- todo-orchestrator:v2-managed:end -->
