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
- [ ] Audit and replace the current shard_storage backend state and path fields for the new source-path plus cache-root contract.
- [ ] Implement shard-pack manifest, fingerprint invalidation, and atomic shard-file writes.
- [ ] Implement one-reader-thread queueing and coalesced shard materialization for .csh5-backed storage.
- [ ] Implement predictor state, budgeted keep/evict decisions, and explicit caller override APIs.
- [ ] Rewrite focused tests and fetch benchmark contracts to cold-fill and warm-cache behavior.
- [ ] Build and run focused CellShard runtime tests and the fetch benchmark compile target.

## Blockers
_None recorded yet._

## Progress Notes
- Planning is complete: shard-pack cache files, auto fingerprint invalidation, on-demand fill plus preload, and caller-overridable predictor behavior are the locked design.
- Initial repo inspection confirmed that the main old-packfile consumers are bench/cellshard_fetch_bench.cu and tests/cellshard_blocked_ell_test.cu, which makes the behavioral rewrite boundary relatively contained.

## Next Actions
- Patch shard_paths.*, sharded_host.cuh, series_h5.*, and disk.* to establish the new cache-manager data model before touching the benchmark and tests.
- Fork the benchmark/test migration on bench/cellshard_fetch_bench.cu and tests/cellshard_blocked_ell_test.cu once the core cache API shape is stable enough to hand off.

## Done Criteria
- CellShard no longer treats packfile as a primary durable storage format; .csh5 is durable and packfile is a disposable shard-pack cache.
- Concurrent worker requests for the same missing shard coalesce onto one HDF5 reader-thread materialization job.
- The predictor can retain or evict shards under a bounded cache budget, but explicit caller commands always override it.
- Focused CellShard tests and benchmark code build and validate the new cold-fill and warm-cache semantics.
