# Todo Status

## Summary
Use this file as the quick pickup register for `todos.md` workstreams.
- `ready`: planned work that can be started now.
- `claimed`: currently being written; choose another stream.
- `idle`: unfinished but resumable; safe to pick up.
- `closed`: completed or removed from pickup rotation.

## Workstreams
- `dual-cuda-optimization-modes` | status: in_progress | execution: claimed | owner: codex | file: `todos/dual-cuda-optimization-modes.md` | next: Still active per user; next expansion is autograd once the CellShard compile break clears.
- `cellshard-first-stable-release` | status: in_progress | execution: claimed | owner: codex | file: `todos/cellshard-first-stable-release.md` | next: Choose and add the CellShard license file, then re-run the documented release checks if the license text changes packaging metadata or release notes.
- `cellshard-blocked-ell-ingest-runtime` | status: in_progress | execution: claimed | owner: codex | file: `todos/cellshard-blocked-ell-ingest-runtime.md` | next: The first generated `.csh5` codec path is now validated; next is a small explicit sample-file workflow plus better shard ordering and bucket-selection heuristics.
- `cellshard-file-surface-rename` | status: done | execution: closed | owner: codex | file: `todos/cellshard-file-surface-rename.md` | next: Deleted the compatibility headers after confirming nothing still included them; standalone and package-consumer validation still pass.
- `cellshard-runtime-service-contract` | status: in_progress | execution: claimed | owner: codex | file: `todos/cellshard-runtime-service-contract.md` | next: Implement the first owner-node coordinator/runtime surface on top of the new cache tree, then add append staging and publish/cutover state.
- `cellshard-csr-file-codec-removal` | status: planned | execution: ready | owner: unassigned | file: `todos/cellshard-csr-file-codec-removal.md` | next: Start by converting the remaining compressed `.csh5` tests and deciding whether legacy compressed-file read support survives as a temporary compatibility layer.

## Staleness Review
- Fresh: 1
- Aging: 0
- Stale candidates: 0
- Stale: 0
- Superseded: 0

## Cleanup Status
- Cleanup mode is explicit only.
- Safe to call `todo-cleanup`: no, active workstreams: dual-cuda-optimization-modes, cellshard-first-stable-release, cellshard-blocked-ell-ingest-runtime, cellshard-runtime-service-contract, cellshard-csr-file-codec-removal.
- Partial cleanup is available via `todo-cleanup --partial`; include `stale` in `--scope` only when explicitly intended.
