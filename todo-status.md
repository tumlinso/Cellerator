# Todo Status

## Summary
Use this file as the quick pickup register for `todos.md` workstreams.
- `ready`: planned work that can be started now.
- `claimed`: currently being written; choose another stream.
- `idle`: unfinished but resumable; safe to pick up.
- `closed`: completed or removed from pickup rotation.

## Workstreams
- `dual-cuda-optimization-modes` | status: in_progress | execution: claimed | owner: codex | file: `todos/dual-cuda-optimization-modes.md` | next: Once the active CellShard storage rewrite lands or stabilizes enough to compile `cellerator_compute_autograd` again, extend the distinct extreme path into sparse autograd and the model CUDA surfaces that depend on it.
- `cellshard-first-stable-release` | status: in_progress | execution: claimed | owner: codex | file: `todos/cellshard-first-stable-release.md` | next: Choose and add the CellShard license file, then re-run the documented release checks if the license text changes packaging metadata or release notes.
- `cellshard-blocked-ell-ingest-runtime` | status: in_progress | execution: idle | owner: codex | file: `todos/cellshard-blocked-ell-ingest-runtime.md` | next: First-file Blocked-ELL fixture work is complete and validated; remaining work is heuristic tuning and optional inspect-surface expansion.
- `cellshard-file-surface-rename` | status: done | execution: closed | owner: codex | file: `todos/cellshard-file-surface-rename.md` | next: Deleted the compatibility headers after confirming nothing still included them; standalone and package-consumer validation still pass.
- `cellshard-runtime-service-contract` | status: in_progress | execution: claimed | owner: codex | file: `todos/cellshard-runtime-service-contract.md` | next: Implement the first owner-node coordinator/runtime surface on top of the new cache tree, then add append staging and publish/cutover state.
- `cellshard-csr-file-codec-removal` | status: done | execution: closed | owner: codex | file: `todos/cellshard-csr-file-codec-removal.md` | next: Removed the remaining compressed `.csh5` file-codec implementation and tests; only reserved enum ids remain to avoid renumbering persisted codec/execution values.
- `quantized-blocked-ell-codecs` | status: in_progress | execution: idle | owner: codex | file: `todos/quantized-blocked-ell-codecs.md` | next: The persisted/runtime slice is validated through CellShard cache roundtrip and fused SpMM; next add inspect/runtime summary coverage for the new codec family.

## Staleness Review
- Fresh: 1
- Aging: 0
- Stale candidates: 0
- Stale: 0
- Superseded: 0

## Cleanup Status
- Cleanup mode is explicit only.
- Safe to call `todo-cleanup`: no, active workstreams: dual-cuda-optimization-modes, cellshard-first-stable-release, cellshard-blocked-ell-ingest-runtime, cellshard-runtime-service-contract, quantized-blocked-ell-codecs.
- Partial cleanup is available via `todo-cleanup --partial`; include `stale` in `--scope` only when explicitly intended.
