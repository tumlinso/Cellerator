

<!-- todo-orchestrator:v2-managed:start -->
# CE-PTR-06: Packing optimizer proposals and rollback

Task revision: `2517`; current project revision is in `todo-status.md`.

## Objective
Replace map, set, vector-heavy proposal, shortlist, conflict, blacklist, snapshot, and rollback mechanics with packed keys, flat sort-compact deduplication, direct counters, generation marks, explicit bounded blacklists, and mutation journals.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
Consume CE-PTR-05 state contract, implement proposal relations and mutation journal, then validate exact optimizer behavior and memory reductions.

## Ownership
- `exclusive`: `src/geometry/optimizer.cc`
- `exclusive`: `src/geometry/optimizer_state.hh`
- `read`: `bench`
- `read`: `include/Cellerator/geometry`
- `read`: `tests`

## Dependencies
- `task`: `CE-PTR-05`
<!-- todo-orchestrator:v2-managed:end -->
