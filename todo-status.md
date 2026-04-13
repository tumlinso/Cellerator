# Todo Status

## Summary
Use this file as the quick pickup register for `todos.md` workstreams.
- `ready`: planned work that can be started now.
- `claimed`: currently being written; choose another stream.
- `idle`: unfinished but resumable; safe to pick up.
- `closed`: completed or removed from pickup rotation.

## Workstreams
- `cellshard-core-partition-rename` | status: done | execution: closed | owner: codex | file: `todos/cellshard-core-partition-rename.md` | next: Completed; optional partial cleanup can remove the finished ledger.
- `dual-cuda-optimization-modes` | status: in_progress | execution: claimed | owner: codex | file: `todos/dual-cuda-optimization-modes.md` | next: Still active per user; next expansion is autograd once the CellShard compile break clears.

## Staleness Review
- Fresh: 1
- Aging: 0
- Stale candidates: 0
- Stale: 0
- Superseded: 0

## Cleanup Status
- Cleanup mode is explicit only.
- Safe to call `todo-cleanup`: no, active workstreams: dual-cuda-optimization-modes.
- Partial cleanup is available for completed workstreams: cellshard-core-partition-rename.
