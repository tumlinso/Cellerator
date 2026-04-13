# Current Objective

## Summary
Implement the first distributed forward-neighbor runtime pass with time-window shard routing, lazy multi-GPU residency, and a reusable executor surface tuned for the real 4x V100 host topology.

## Planning Notes
- Keep the active neighbor API exact-window first and pointer-first.
- Route query blocks by the union of their eligible time windows before any shard search launches.
- Prefer shard-local execution and query reuse per participating device over repeated query upload per shard.
- Preserve current same-embryo policy behavior while extending results with source-shard provenance.

## Assumptions
- Forward neighbors remain the only actively built neighbor surface.
- The runtime should prefer device order `0,2,1,3` when four GPUs are visible.
- The current environment may not expose a CUDA device for runtime validation.

## Suggested Skills
- `cuda-v100` - Keep routing, residency, and merge behavior aligned with the real V100 pair topology.

## Useful Reference Files
- `pointer_migration_plan.md` - Phase 1 places forward neighbors on the critical path for pointer-first runtime cleanup.
- `optimization.md` - Documents current neighbor orchestration overhead and the need to minimize host staging churn.

## Plan
- Extend the public forward-neighbor API with explicit direction, executor configuration, routing plans, shard summaries, and shard provenance in results.
- Change index construction to produce contiguous time-range shards and support lazy device residency instead of mandatory eager upload.
- Route query blocks to overlapping shard windows, group routed shards by device, and search with one query upload per participating device.
- Add a reusable executor surface and update the test target to cover the new routing and result paths.

## Tasks
- [x] Extend the forward-neighbor API with routing, executor, shard-summary, and shard-provenance surfaces.
- [x] Rework shard assignment and build policy around contiguous time-range shards plus lazy device upload.
- [x] Implement query-block routing, per-device grouped execution, and shard-id result propagation through exact and ANN searches.
- [x] Update the forward-neighbor compile/runtime test to cover routing and executor-backed search.
- [x] Add a routed benchmark surface and profile the cross-pair steady-state search on the V100 host.

## Blockers
_None recorded._

## Progress Notes
- Added `ForwardNeighborDirection`, `ForwardNeighborExecutorConfig`, `ForwardNeighborRoutingPlan`, `ForwardNeighborShardSummary`, and `ForwardNeighborSearchExecutor`.
- Added `neighbor_shard_indices` to `ForwardNeighborSearchResult` and propagated shard provenance through both exact and ANN refinement kernels.
- Switched forward-neighbor row ordering to time-first sorting and contiguous shard assignment so shard intervals correspond to routable time windows.
- Added shard summaries and route-planning helpers on `ForwardNeighborIndex`.
- Added lazy shard residency with per-device resident-byte and resident-shard limits.
- Updated the search core to build block routes, group shards by device, upload each query block once per participating device, and merge only routed partials.
- Updated `forwardNeighborsCompileTest` to exercise the new surfaces and to skip cleanly when CUDA devices are unavailable.
- Verified `cmake --build build --target forwardNeighborsCompileTest -j 4` and `./build/forwardNeighborsCompileTest` in the current environment.
- Confirmed the host exposes four Tesla V100-SXM2-16GB devices with fast NVLink pairs `0<->2` and `1<->3`.
- Re-ran `forwardNeighborsCompileTest` on visible GPUs and relaxed the shard-count expectation so it matches contiguous time-range sharding semantics; repeated live GPU runs now pass.
- Added `forwardNeighborsBench` plus a scenario generator that derives single-shard, one-pair, and cross-pair windows from the realized shard summaries rather than assuming the requested shard count is always reached.
- Benchmarked the routed runtime on the V100 host at 262144 rows / 8192 queries / 64 dims and recorded steady-state search times of 310.09 ms for `single-shard`, 285.93 ms for `one-pair`, and 259.59 ms for `cross-pair`.
- Profiled the cross-pair benchmark case with Nsight Systems and Nsight Compute; the run is steady-state valid, `exact_search_kernel_` dominates the timeline, and the sampled kernel counters indicate a mixed limiter rather than a transfer- or allocator-bound path.

## Next Actions
- Decide whether the current grouped-by-device merge is sufficient or whether pair-local explicit merge objects should be added as a second pass.
- Use `forwardNeighborsBench` to compare larger query blocks, pair-local merge, and CUDA Graph capture on the hottest exact path.
- Add metrics or profiling hooks to quantify residency churn and query upload reuse during large bulk searches.

## Done Criteria
- Forward-neighbor callers can inspect shard summaries, plan routes, and reuse an executor across searches.
- Shards are time-range contiguous and lazily resident-uploaded.
- Search touches only routed shards and preserves shard provenance in its outputs.
- The active test target covers the new routing and executor contracts.
- The repo has a benchmark and profiler entrypoint that exercises single-shard, one-pair, and cross-pair routed search on the V100 host.
