# Todo Status

## Summary
Use this file as the quick pickup register for `todos.md` workstreams.
- `ready`: planned work that can be started now.
- `claimed`: currently being written; choose another stream.
- `idle`: unfinished but resumable; safe to pick up.
- `closed`: completed or removed from pickup rotation.

## Workstreams
- `rewrite-cudabiotypes-semantic-contracts` | status: done | execution: closed | owner: unassigned | file: `todos/rewrite-cudabiotypes-semantic-contracts.md` | next: Leave closed unless the user explicitly asks for cleanup.
- `pointer-first-neighbor-runtime` | status: done | execution: closed | owner: unassigned | file: `todos/pointer-first-neighbor-runtime.md` | next: Leave closed unless the user explicitly asks for cleanup.
- `distributed-time-window-neighbor-runtime` | status: done | execution: closed | owner: unassigned | file: `todos/distributed-time-window-neighbor-runtime.md` | next: Leave closed unless the user explicitly asks for cleanup.
- `developmental-time-cuda-ab` | status: done | execution: closed | owner: codex | file: `todos/developmental-time-cuda-ab.md` | next: Leave closed unless the user explicitly asks for cleanup.
- `make-blocked-ell-csh5-fetch-approach-packfile-performance-as-closely-as-possible-with-shard-packed-payloads-reusable-shard-scratch-and-ssd-only-real-data-fetch-benchmarks` | status: done | execution: closed | owner: unassigned | file: `todos/make-blocked-ell-csh5-fetch-approach-packfile-performance-as-closely-as-possible-with-shard-packed-payloads-reusable-shard-scratch-and-ssd-only-real-data-fetch-benchmarks.md` | next: Leave closed unless the user explicitly asks for cleanup.
- `cellshard-first-class-build-export-python` | status: stale | execution: closed | owner: codex | file: `todos/cellshard-first-class-build-export-python.md` | next: Stale by user decision; do not resume without explicit reactivation.
- `public-omics-shortlist-manuscript-benchmark-seed` | status: stale | execution: closed | owner: codex | file: `todos/public-omics-shortlist-manuscript-benchmark-seed.md` | next: Stale by user decision; do not resume without explicit reactivation.
- `cellshard-core-partition-rename` | status: in_progress | execution: ready | owner: codex | file: `todos/cellshard-core-partition-rename.md` | next: Still active per user; resume the deep partition rename on the cleaned .csh5 backend.
- `cellshard-debug-thread` | status: stale | execution: closed | owner: unassigned | file: `todos/cellshard-debug-thread.md` | next: Stale by user decision; reactivate only if new CellShard crash work appears.
- `cellshard-packfile-cache-rewrite` | status: stale | execution: closed | owner: codex | file: `todos/cellshard-packfile-cache-rewrite.md` | next: Stale by user decision; do not resume without explicit reactivation.
- `dual-cuda-optimization-modes` | status: in_progress | execution: claimed | owner: codex | file: `todos/dual-cuda-optimization-modes.md` | next: Still active per user; next expansion is autograd once the CellShard compile break clears.

## Staleness Review
- Fresh: 2
- Aging: 0
- Stale candidates: 0
- Stale: 4
- Superseded: 0
- `rewrite-cudabiotypes-semantic-contracts` | done | age: 0.0d | threshold: 14d | reason: Terminal workstream.
- `pointer-first-neighbor-runtime` | done | age: 0.0d | threshold: 14d | reason: Terminal workstream.
- `distributed-time-window-neighbor-runtime` | done | age: 0.0d | threshold: 14d | reason: Terminal workstream.
- `developmental-time-cuda-ab` | done | age: 0.0d | threshold: 14d | reason: Terminal workstream.
- `make-blocked-ell-csh5-fetch-approach-packfile-performance-as-closely-as-possible-with-shard-packed-payloads-reusable-shard-scratch-and-ssd-only-real-data-fetch-benchmarks` | done | age: 0.0d | threshold: 14d | reason: Terminal workstream.
- `cellshard-first-class-build-export-python` | stale | age: 0.0d | threshold: 14d | reason: User marked this stream stale; keep it out of pickup rotation until explicitly reactivated.
- `public-omics-shortlist-manuscript-benchmark-seed` | stale | age: 0.0d | threshold: 14d | reason: User marked this stream stale; dataset-root follow-up is deferred until explicitly reactivated.
- `cellshard-debug-thread` | stale | age: 0.0d | threshold: 14d | reason: User marked this stream stale; keep the debug ledger for history but do not pick it up unless a new CellShard crash reactivates it.
- `cellshard-packfile-cache-rewrite` | stale | age: 0.0d | threshold: 14d | reason: User marked this stream stale after partial completion; do not resume unless explicitly reopened.

## Cleanup Status
- Cleanup mode is explicit only.
- Safe to call `todo-cleanup`: no, active workstreams: cellshard-core-partition-rename, dual-cuda-optimization-modes.
- Cleanup still blocked by stale workstreams pending review: cellshard-first-class-build-export-python, public-omics-shortlist-manuscript-benchmark-seed, cellshard-debug-thread, cellshard-packfile-cache-rewrite.
