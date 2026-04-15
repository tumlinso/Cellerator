---
slug: "cellshard-blocked-ell-ingest-runtime"
status: "in_progress"
execution: "claimed"
owner: "codex"
created_at: "2026-04-13T16:51:58Z"
last_heartbeat_at: "2026-04-15T13:23:53Z"
last_reviewed_at: "2026-04-15T13:23:53Z"
stale_after_days: 14
objective: "implement blocked-ell-first CellShard ingest, explicit machine-local cache warmup, and runtime alignment"
---

# Current Objective

## Summary
Align the repo on Blocked-ELL-first `.csh5` series files by moving runtime consumers off compressed assumptions, separating persisted shard sizing from ingest windows, and adding explicit per-machine cache warmup for low-latency shard-pack serving.

## Quick Start
- Why this stream exists: _Summarize the domain boundary and why it was split out._
- In scope: _List the work this stream owns._
- Out of scope / dependencies: _List handoffs, upstream dependencies, or adjacent streams._
- Required skills: _List the exact repo-local skills to read before starting._
- Required references: _List the exact repo-local references to read before starting._
- Why this stream exists: current ingest writes Blocked-ELL-only `.csh5`, but preprocess and some runtime helpers still reload series as compressed, so the file-format contract and active consumers diverge.
- In scope: Blocked-ELL runtime alignment, explicit cache warmup, runtime-driven persisted shard sizing, and ingest writer/planner changes needed to make those contracts real.
- Out of scope: dual-payload compatibility, keeping compressed preprocess support, or broad model-layer changes unrelated to series ingest/runtime.
- Required references: `extern/CellShard/src/sharded/csh5.cuh`, `extern/CellShard/src/sharded/csh5.cc`, `extern/CellShard/src/sharded/disk.cuh`, `src/ingest/series/series_ingest.cuh`, `src/workbench/series_workbench.cc`, and `src/workbench/series_workbench_cuda.cu`.

## Planning Notes
- Treat `.csh5` as the authoritative bulk container and shard packs as the machine-local serving layer.
- Persisted shard planning must use runtime Blocked-ELL bytes with CUDA `u32` hard caps, not temporary MTX conversion windows.
- Warmup should be explicit and must always allow the caller to select a machine-local cache root instead of defaulting beside the source file.
- 2026-04-13 bucketed Blocked-ELL implementation slice: keep canonical per-part Blocked-ELL payloads in .csh5, extend execution metadata with bucketing intent, build bucketed execution cache artifacts from canonical payloads, and route workbench blocked-ell runtime paths through bucket-aware execution packs while leaving generic fetch/pack compatibility intact.

## Assumptions
- New ingest output is Blocked-ELL-only and active preprocess/runtime code should consume Blocked-ELL `.csh5` directly.
- Compressed series support is not preserved as an active workflow in this migration.
- A local cache root should be treated as required by main serving workflows even though the low-level backend may still keep a legacy default.
- First execution slice uses per-part bucketed Blocked-ELL segments grouped under shard-local execution cache artifacts rather than replacing the canonical .csh5 partition payload contract.

## Suggested Skills
- `todo-orchestrator` - Track the ingest/runtime migration and keep the execution ledger current.
- `cuda-v100` - Use only if Blocked-ELL runtime/preprocess kernel behavior or V100 fit becomes the limiting issue.

## Useful Reference Files
- `extern/CellShard/src/sharded/csh5.cuh` - Public csh5 and cache-management surface.
- `extern/CellShard/src/sharded/csh5.cc` - Actual csh5 writer, cache pack builder, and shard fetch behavior.
- `extern/CellShard/src/sharded/disk.cuh` - Type-directed `.csh5` header loading entrypoints.
- `src/ingest/series/series_ingest.cuh` - Current MTX-to-csh5 writer path that always emits Blocked-ELL.
- `src/workbench/series_workbench.cc` - Current ingest planner and shard/execution metadata logic.
- `src/workbench/series_workbench_cuda.cu` - Current conversion driver, browse-cache builder, and preprocess runtime assumptions.

## Plan
- Add explicit Blocked-ELL cache warmup and cache-root enforcement in the active runtime/workbench flow.
- Move preprocess and related runtime helpers to Blocked-ELL header/load paths and Blocked-ELL partition views.
- Split persisted shard sizing from MTX conversion windows in ingest policy and planning.
- Update the ingest writer so persisted shard offsets and execution metadata come from the same Blocked-ELL runtime plan.
- Rebuild focused tests around Blocked-ELL-only series output and explicit cache warmup.

## Tasks
- [ ] Add explicit warmup API and wire it into the workbench/runtime flow.
- [ ] Port preprocess/runtime consumers from compressed series loading to Blocked-ELL loading.
- [ ] Split ingest policy into conversion-window and runtime-shard byte controls.
- [ ] Make ingest writer honor runtime shard planning instead of conflating it with conversion windows.
- [ ] Update tests for Blocked-ELL-only ingest and cache warmup behavior.
- [ ] Plan and persist bucket-aware execution metadata for Blocked-ELL partitions/shards.
- [ ] Build bucketed execution-cache materialization and fetch helpers alongside the existing plain pack path.
- [ ] Switch workbench blocked-ell browse/preprocess runtime paths to the bucket-aware execution cache.
- [ ] Add focused tests for bucket-aware execution metadata and blocked-ell execution-cache roundtrips.

## Blockers
_None recorded yet._

## Progress Notes
- Created the ingest/runtime migration workstream and recorded the implementation order: runtime alignment, cache warmup, shard-sizing split, writer alignment, then tests.
- Implementation started for bucketed Blocked-ELL execution packs driven by .csh5 execution metadata.
- Retargeted the storage/backend work to the renamed `extern/CellShard/src/sharded/csh5.*` surface and kept the compatibility forwarding headers out of the active implementation path.
- Extended `.csh5` execution metadata with bucket-aware partition and shard fields, added shard-local `shard.<id>.exec.pack` materialization/fetch helpers, and routed blocked-ell browse/preprocess runtime paths through `fetch_series_blocked_ell_h5_execution_partition(...)`.
- Fixed the preprocess persistence path so it closes the read-only `series_h5` backend before reopening the same `.csh5` read-write for `append_series_preprocess_h5(...)`; this removed the HDF5 "file is already open for read-only" failure.
- Rebuilt and passed `./build/seriesWorkbenchRuntimeTest` and `./build/cellShardSeriesH5Test` after the bucketed execution-pack integration.
- Reworked the Blocked-ELL storage path again so `.csh5` can persist a shard-scoped optimized codec with shard-local column maps and bucketed execution partitions instead of only flat per-part Blocked-ELL payloads.
- Switched ingest to emit persisted optimized shard blobs, taught `csh5` to materialize both canonical `.pack` and execution `.exec.pack` caches from that stored codec, and updated blocked-ell browse/preprocess hot paths to remap shard-local columns correctly.
- Added a focused optimized-codec roundtrip inside `cellShardSeriesH5Test` and reran `./build/cellShardSeriesH5Test`, `./build/seriesWorkbenchRuntimeTest`, `./build/cellShardSeriesH5Test`, and `./build/seriesWorkbenchRuntimeTest` successfully.
- Promoted the persisted optimized shard codec to the canonical `series_codec_family_blocked_ell` slot, moved the previous flat payload tests onto the transitional `series_codec_family_sliced_ell` slot, and added a separate CUDA blocked-ell bucket planner header for ingest while keeping the existing host bucketing code in `csh5.cc` as the fallback/reference path.
- Tightened the generated-file validation path: `series_workbench` summaries now expose top-level `matrix_format`, `payload_layout`, execution `preferred_base_format`, and runtime-service metadata, and `seriesWorkbenchRuntimeTest` now verifies that a freshly converted `.csh5` advertises the optimized bucketed blocked-ell payload layout and owner-hosted runtime-service contract.
- Added an end-to-end codec assertion on the first generated file by reopening the converted `.csh5`, fetching execution metadata and the first execution partition, and confirming the persisted shard-local column remap is the expected non-identity permutation rather than only smoke-testing that the file exists.
- Aligned workbench-written execution metadata with the actual persisted optimized codec so the file now reports `bucketed_blocked_ell` as the preferred execution base instead of leaking the planner's pre-persist preference.
- Added a dedicated  target that writes a tiny optimized bucketed Blocked-ELL .csh5, warms cache and execution cache, summarizes the file, and reopens it to validate the frozen non-identity execution column remap.

## Next Actions
- Tighten the shard column-order heuristic and bucket-count search so persisted optimized shards are chosen by measured occupancy/runtime gain rather than the current simple signature sort plus byte-minimizing bucket sweep.
- Add a small explicit sample-file generation workflow or fixture around the now-validated converted `.csh5` so first-file regression tests do not depend only on the larger workbench runtime binary.
- Extend inspect surfaces further if needed so shard-level codec details can be reported without reopening the runtime fetch path in tests.

## Done Criteria
- Freshly converted `.csh5` series can be preprocessed and browsed through Blocked-ELL runtime paths without compressed fallbacks.
- Explicit warmup can populate shard packs into a caller-selected local cache root for a source file on local or remote storage.
- Persisted shard offsets are driven by runtime Blocked-ELL bytes plus CUDA `u32` caps, not ingest windows.
- Focused CellShard and workbench tests cover Blocked-ELL-only output, explicit cache warmup, and runtime reload/fetch behavior.
