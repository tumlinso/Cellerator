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
- `cellshard-first-class-build-export-python` | status: in_progress | execution: idle | owner: codex | file: `todos/cellshard-first-class-build-export-python.md` | next: Package-component handling and consumer-test scaffolding are in; next step is rerunning standalone no-GPU package validation after the overlapping series_h5 compile issue clears.
- `public-omics-shortlist-manuscript-benchmark-seed` | status: in_progress | execution: idle | owner: codex | file: `todos/public-omics-shortlist-manuscript-benchmark-seed.md` | next: Get a dataset root from the user, then use the canonical anchor set to build layout plans, manifests, and the first processed-file fetch scope.
- `cellshard-core-partition-rename` | status: in_progress | execution: idle | owner: codex | file: `todos/cellshard-core-partition-rename.md` | next: Do not touch extern/CellShard/src/sharded/{shard_paths,series_h5,sharded_host}* or extern/CellShard/src/sharded/disk* until the packfile-cache rewrite lands; resume rename on top of the new backend after that.
- `cellshard-debug-thread` | status: planned | execution: ready | owner: unassigned | file: `todos/cellshard-debug-thread.md` | next: Use this stream for the next CellShard native crash or regression; start with native-debugging crash capture and record the first conclusive summary.
- `cellshard-packfile-cache-rewrite` | status: in_progress | execution: claimed | owner: codex | file: `todos/cellshard-packfile-cache-rewrite.md` | next: Rewrite extern/CellShard/src/sharded/* packfile and series_h5 storage into the new cache manager, then update focused tests and fetch benchmarks.

## Cleanup Status
- Cleanup mode is explicit only.
- Safe to call `todo-cleanup`: no, waiting on cellshard-first-class-build-export-python, public-omics-shortlist-manuscript-benchmark-seed, cellshard-core-partition-rename, cellshard-debug-thread, cellshard-packfile-cache-rewrite.
