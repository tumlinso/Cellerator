---
slug: "gpu-prototype-ingest-blocked-ell"
status: "planned"
execution: "ready"
owner: "codex"
created_at: "2026-04-16T09:18:40Z"
last_heartbeat_at: "2026-04-16T09:19:25Z"
last_reviewed_at: "2026-04-16T09:19:25Z"
stale_after_days: 14
objective: "Prototype GPU-first Blocked-ELL ingest builders, rebucketing, and workbench-adjacent cold materialization paths for speed screening on V100"
---

# Current Objective

## Summary
Prototype GPU-first Blocked-ELL ingest builders, rebucketing, and workbench-adjacent cold materialization paths for speed screening on V100

## Quick Start
- Why this stream exists: _Summarize the domain boundary and why it was split out._
- In scope: _List the work this stream owns._
- Out of scope / dependencies: _List handoffs, upstream dependencies, or adjacent streams._
- Required skills: _List the exact repo-local skills to read before starting._
- Required references: _List the exact repo-local references to read before starting._
- Why this stream exists: cold dataset materialization is still dominated by CPU-side COO-to-Blocked-ELL tuning, shard column remap, and host rebucketing; this stream prototypes a GPU-first replacement so the speedup can be measured on the native V100 host.
- In scope: CUDA prototypes for COO-to-Blocked-ELL conversion, shard column-map planning, bucket-count search, rebucketing, and the workbench-adjacent numeric cold-materialization helpers that currently pay repeated host staging.
- Out of scope / dependencies: MTX text parsing, HDF5 metadata and file assembly semantics, and nonnumeric UI or filesystem discovery remain on CPU; coordinate with the existing cellshard-blocked-ell-ingest-runtime stream rather than duplicating its storage-contract work.
- Required skills: `cuda`, `todo-orchestrator`.
- Required references: `optimization.md`, `todos/cellshard-blocked-ell-ingest-runtime.md`, `src/ingest/dataset/dataset_ingest.cuh`, `src/workbench/dataset_workbench_cuda.cu`, `extern/CellShard/src/disk/csh5.cuh`, `extern/CellShard/src/disk/csh5.cc`.

## Planning Notes
- Keep the prototype benchmark-focused: preserve current CPU planning and file contracts at first, then replace only the numeric builder/rebucket phases behind a switchable path.
- Prefer custom CUDA for irregular sparse build and rebucketing work, but keep library-backed pieces where they map cleanly.

## Assumptions
- The first measurable target is lower cold materialization time, not a perfect production ingest rewrite.
- Blocked-ELL remains the native persisted and execution layout; compressed-only builder work is at most a comparison baseline.

## Suggested Skills
- `cuda` - Use the native V100 path and treat the dominant question as CPU-centric sparse-porting rather than generic CUDA cleanup.
- `todo-orchestrator` - Keep this stream aligned with the adjacent CellShard ingest-runtime contract work.

## Useful Reference Files
- `src/ingest/dataset/dataset_ingest.cuh` - Current host-heavy column remap, bucket planning, and optimized shard assembly path.
- `src/workbench/dataset_workbench_cuda.cu` - Current blocked-ell browse and preprocess helpers with repeated sync and copy boundaries.
- `bench/cellshard_v100_profile.cu` - Existing native benchmark and profiling surface for large sparse runtime paths.

## Plan
- Isolate the current CPU-heavy numeric phases inside dataset ingest and workbench cold paths and define a benchmark contract around them.
- Prototype a CUDA COO-to-Blocked-ELL builder and compare it with the current host-centric conversion plus host shard assembly path.
- Prototype GPU shard-column ordering and bucket-count planning so rebucketing no longer depends on host qsort-style passes.
- Thread the winning prototype behind a switchable path and benchmark cold materialization plus first warmup behavior on V100.

## Tasks
- [ ] Benchmark and document the current cold materialization phases separately from parsing and file I/O.
- [ ] Prototype a native CUDA COO-to-Blocked-ELL builder for dataset ingest windows.
- [ ] Prototype GPU shard column-map planning and bucket-count search.
- [ ] Prototype GPU rebucketing or optimized-shard assembly for persisted execution shards.
- [ ] Prototype reduced-copy workbench numeric helpers for blocked-ell selected-feature and preprocess summaries.
- [ ] Add focused benchmark coverage for the prototype path on V100.

## Blockers
_None recorded yet._

## Progress Notes
_None recorded yet._

## Next Actions
- Start by extracting a benchmarkable cold-materialization slice around dataset ingest blocked builder and optimized shard assembly.

## Done Criteria
- A working CUDA-backed ingest prototype exists for the dominant numeric cold-materialization phases and can be benchmarked against the current path on the V100 host.
- The prototype clearly reports whether the dominant limiter moved from CPU orchestration to PCIe, HBM traffic, or launch/setup overhead.
