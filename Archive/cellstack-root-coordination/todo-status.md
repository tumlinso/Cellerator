# Todo Status

## Summary
Use this file as the quick pickup register for `todos.md` workstreams.
- `ready`: planned work that can be started now.
- `claimed`: currently being written; choose another stream.
- `idle`: unfinished but resumable; safe to pick up.
- `closed`: completed or removed from pickup rotation.

## Workstreams
- `cellerator-sparse-ml-layout-root-coordination` | status: planned | execution: ready | owner: unassigned | file: `todos/cellerator-sparse-ml-layout-root-coordination.md` | next: Pick this up only when root-owned coordination is needed after the Cellerator submodule workstream advances.
- `baseplane-sequence-migration` | status: done | execution: closed | owner: codex | file: `todos/baseplane-sequence-migration.md` | next: No root migration action remains.
- `cellerator-preprocess-rehome-root-coordination` | status: done | execution: closed | owner: codex | file: `todos/cellerator-preprocess-rehome-root-coordination.md` | next: No root migration action remains for the preprocessing rehome.

## Staleness Review
_No staleness review recorded yet._

## Cleanup Status
- Cleanup mode is explicit only.
- Safe to call `todo-cleanup`: no, active workstreams: cellerator-sparse-ml-layout-root-coordination.
- Partial cleanup is available via `todo-cleanup --partial`; include `stale` in `--scope` only when explicitly intended.

