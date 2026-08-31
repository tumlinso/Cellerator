

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-20: Candidate catalog v2 contract

Task revision: `3281`; current project revision is in `todo-status.md`.

## Objective
Define a rich cold non-STL POD candidate descriptor and fragment view while retaining the compact hot operation_candidate.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `include/Cellerator/compute/operation/candidate_catalog_v2.hh`
- `exclusive`: `tests/ce_geo/catalog/catalog_contract_test.cc`
- `read`: `include/Cellerator/compute/operation`
- `read`: `include/Cellerator/planner`

## Dependencies
- `checkpoint`: `CE-GEO-ARCHITECTURE-FROZEN`
<!-- todo-orchestrator:v2-managed:end -->
