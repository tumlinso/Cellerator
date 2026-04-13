---
slug: "cellshard-packfile-cache-rewrite"
status: "stale"
execution: "closed"
owner: "codex"
created_at: "2026-04-13T14:45:32Z"
last_heartbeat_at: "2026-04-13T14:48:10Z"
last_reviewed_at: "2026-04-13T14:48:10Z"
stale_after_days: 14
objective: "rewrite CellShard packfile into a disposable shard-pack cache for .csh5 with one HDF5 reader thread and explicit predictor overrides"
stale_reason: "User marked this stream stale after partial completion; do not resume unless explicitly reopened."
---

# Current Objective

## Summary
Replace the old packfile persistence path with a shard-pack cache manager for .csh5, including one reader thread, automatic fingerprint invalidation, an access-pattern predictor, and explicit caller cache overrides.

## Quick Start
- Why this stream exists: _Summarize the domain boundary and why it was split out._
- In scope: _List the work this stream owns._
- Out of scope / dependencies: _List handoffs, upstream dependencies, or adjacent streams._
- Required skills: _List the exact repo-local skills to read before starting._
- Required references: _List the exact repo-local references to read before starting._
- Why this stream exists: the old packfile path is currently a primary persistence format plus synchronous fetch path; it needs to become a disposable SSD cache behind .csh5 instead.
- In scope: extern/CellShard/src/sharded/shard_paths.*, sharded_host.cuh, series_h5.*, disk.*, tests/cellshard_blocked_ell_test.cu, and bench/cellshard_fetch_bench.cu.
- Out of scope: the broader partition-vocabulary cleanup can resume after this backend rewrite lands; do not chase unrelated renames inside overlapping files during this stream.
- Required skills: todo-orchestrator for the ledger, cuda-v100 if the rewritten blocked-ELL path needs hot-path validation, native-debugging if the new thread/queue path crashes.
- Required references: AGENTS.md, optimization.md, pointer_migration_plan.md, extern/CellShard/src/sharded/shard_paths.cuh, extern/CellShard/src/sharded/sharded_host.cuh, extern/CellShard/src/sharded/series_h5.cuh, extern/CellShard/src/sharded/series_h5.cc, extern/CellShard/src/sharded/disk.cuh, bench/cellshard_fetch_bench.cu, tests/cellshard_blocked_ell_test.cu.

## Planning Notes
- The new cache unit is one shard pack per shard, not one cache file per partition.
- The default policy is on-demand fill plus explicit preload APIs; predictor decisions must always yield to explicit caller force-cache, evict, and invalidate requests.
- Because other threads are active, this stream owns the overlapping sharded storage files until completion and should keep the ledger current enough for clean pickup.

## Assumptions
- No compatibility shim is required for the old packfile persistence format, APIs, or benchmark mode names.
- The predictor is a deterministic access-pattern heuristic scoped to one bound storage instance, not a learned subsystem or cross-process service.
- Blocked-ELL remains the native fast path; compressed support stays functional but secondary.

## Suggested Skills
- `todo-orchestrator` - Keep the new cache-rewrite ledger current while the backend, benchmark, and tests move together.
- `cuda-v100` - Validate that the rewritten blocked-ELL fetch path keeps the shard-oriented sm_70 cost model intact if hot-path regressions show up.
- `native-debugging` - Use for host-side crashes or deadlocks in the new reader-thread and queue code.

## Useful Reference Files
- `AGENTS.md` - Repo-wide engineering rules, benchmark mutex policy, and Blocked-ELL-first guidance.
- `optimization.md` - Current V100 bottlenecks and explicit sparse-runtime priorities.
- `pointer_migration_plan.md` - Do not add new vector-heavy hot-path interfaces while rewriting the cache manager.
- `extern/CellShard/src/sharded/shard_paths.cuh` - Current shard_storage and backend-open abstraction to replace.
- `extern/CellShard/src/sharded/sharded_host.cuh` - Current fetch_part/fetch_shard dispatch that still treats packfile as a primary backend.
- `extern/CellShard/src/sharded/series_h5.cc` - Current direct HDF5 materialization and per-part cache helpers to replace with the new manager.
- `bench/cellshard_fetch_bench.cu` - Current packfile-vs-csh5 benchmark assumptions that must be rewritten to cold-fill vs warm-cache semantics.
- `tests/cellshard_blocked_ell_test.cu` - Legacy packfile roundtrip coverage to convert into cache behavior coverage.

## Plan
- Introduce cache-manager state, shard-pack manifest/fingerprint logic, and one-reader-thread request handling in the sharded storage backend.
- Rewrite .csh5 fetch_part/fetch_shard and preload paths to route through shard-pack cache files instead of direct worker-thread HDF5 reads.
- Add predictor scoring, cache budget enforcement, and explicit caller force-cache/evict/invalidate controls.
- Remove the old packfile-primary persistence/load/store assumptions and update focused tests and benchmarks to the new cache semantics.
- Build and run the focused CellShard targets serially because the HDF5 tests still share temporary artifacts.

## Tasks
- [x] Audit and replace the current sparse sharded API surface so `.csh5` is primary and legacy packfile fallbacks are quarantined.
- [x] Implement shard-pack manifest, fingerprint invalidation, and atomic shard-file writes.
- [x] Implement one-reader-thread queueing and coalesced shard materialization for .csh5-backed storage.
- [x] Implement predictor state, budgeted keep/evict decisions, and explicit caller override APIs.
- [x] Rewrite focused tests and fetch benchmark contracts to cold-fill and warm-cache behavior.
- [x] Build and run focused CellShard runtime tests and the fetch benchmark compile target.

## Blockers
_None recorded yet._

## Progress Notes
- Planning is complete: shard-pack cache files, auto fingerprint invalidation, on-demand fill plus preload, and caller-overridable predictor behavior are the locked design.
- Initial repo inspection confirmed that the main old-packfile consumers are bench/cellshard_fetch_bench.cu and tests/cellshard_blocked_ell_test.cu, which makes the behavioral rewrite boundary relatively contained.
- `series_h5.*` now owns the active cache-manager path: one reader thread, shard-pack files under a fingerprinted cache directory, shard-level prefetch, explicit pin/evict/invalidate controls, and a bounded access-pattern predictor.
- Focused tests were rewritten away from old packfile artifacts: `cellShardSeriesH5Test`, `cellShardBlockedEllTest`, and `seriesWorkbenchRuntimeTest` build and pass against the new `.csh5` cache path.
- `cellShardFetchBench` no longer benchmarks a packfile-primary mode. It now reports `.csh5` direct fetch against `.csh5` warm-cache behavior and reuses artifacts from `.csh5` only.
- The root build cache had been pinned to `CMAKE_CUDA_ARCHITECTURES=52`; it was reconfigured to `70` so current validation is V100-only instead of compiling for a broad architecture set.
- Sparse sharded `load_header`, `fetch_part`, `fetch_all_parts`, and `fetch_shard` now refuse silent packfile fallback; compressed and blocked-ELL callers go through the `.csh5` series path only.
- The remaining raw packfile helpers were deleted: `extern/CellShard/src/sharded/disk.cu` is gone, `shard_storage` now stores only the source path plus backend state, and `disk.cuh` exposes only the active `.csh5` sharded load/store surface.
- Final validation after deleting the old helpers still passed: `cellShardSeriesH5Test`, `cellShardBlockedEllTest`, `seriesWorkbenchRuntimeTest`, and a synthetic `cellShardFetchBench --impl all` run under `sm_70`.

## Next Actions
- Resume the deep CellShard partition rename on top of this cleaned `.csh5` backend.
- Keep future runtime/docs wording aligned to `shard-pack cache` / `.csh5 source` instead of `packfile`.

## Done Criteria
- CellShard no longer treats packfile as a primary durable storage format; .csh5 is durable and packfile is a disposable shard-pack cache.
- Concurrent worker requests for the same missing shard coalesce onto one HDF5 reader-thread materialization job.
- The predictor can retain or evict shards under a bounded cache budget, but explicit caller commands always override it.
- Focused CellShard tests and benchmark code build and validate the new cold-fill and warm-cache semantics.
