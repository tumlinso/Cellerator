---
slug: "cellshard-runtime-service-contract"
status: "in_progress"
execution: "claimed"
owner: "codex"
created_at: "2026-04-14T10:42:32Z"
last_heartbeat_at: "2026-04-15T00:00:00Z"
last_reviewed_at: "2026-04-15T00:00:00Z"
stale_after_days: 14
objective: "reset CellShard around owner-hosted pack delivery, append-only canonical generations, and a Cellerator immutable-emission boundary"
---

# Current Objective

## Summary
Reset the CellShard/Cellerator boundary so Cellerator emits one-pass immutable canonical sparse matrices while CellShard owns owner-hosted pack delivery, append-only canonical generations, and distributed runtime service metadata.

## Quick Start
- Why this stream exists: the current docs and csh5 execution metadata describe local cache materialization, but the intended contract is an owner-hosted csh5 service that prepares and delivers pack artifacts to executors.
- In scope: public CellShard format/runtime docs, csh5 schema metadata for runtime service and generation identity, and minimal public inspection accessors so code can observe the new contract.
- Out of scope / dependencies: no full network transport stack in this first slice, no destructive migration of existing payload layout, and no new preblocked direct-transfer ingest path yet.
- Required skills: todo-orchestrator for the ledger and cuda-v100 only if runtime staging changes become GPU-fit sensitive.
- Required references: AGENTS.md, extern/CellShard/README.md, extern/CellShard/src/disk/csh5.cuh, extern/CellShard/src/disk/csh5.cc, tests/cellshard_series_h5_test.cu, and src/ingest/series/series_ingest.cuh.

## Planning Notes
- Treat pack as the primary execution artifact and csh5 as the durable canonical source plus append target.
- Keep the first implementation slice backward-compatible by adding optional metadata and inspect accessors instead of forcing a full reader/writer rewrite at once.

## Assumptions
- The first implementation wave should make the owner-service contract explicit in schema/docs even if the full networked runtime is staged later.
- Existing local cache behavior should map to the new owner-service model without changing current fetch semantics.

## Suggested Skills
- `todo-orchestrator` - Track the architecture reset as a standalone resumable stream.
- `cuda-v100` - Use only if the runtime delivery or staging changes expose new Volta bottlenecks.

## Useful Reference Files
- `extern/CellShard/README.md` - Current public ownership split and csh5/pack wording that needs to be tightened.
- `extern/CellShard/src/disk/csh5.cuh` - Public csh5 schema structs and append/load APIs.
- `extern/CellShard/src/disk/csh5.cc` - Current execution metadata persistence, cache runtime, and pack materialization code.
- `tests/cellshard_series_h5_test.cu` - Focused roundtrip/runtime test surface for new csh5 metadata.

## Plan
- Update the new workstream ledger and root todos to reflect the owner-hosted runtime-service contract.
- Extend the public csh5 schema surface with generation/runtime-service metadata and inspect accessors.
- Rewrite the CellShard README and related ingest wording so Cellerator vs CellShard ownership matches the new contract.
- Add focused tests covering runtime-service metadata roundtrip and inspection from a loaded csh5 backend.

## Tasks
- [x] Create the runtime-service workstream and record the owner-hosted pack-delivery contract.
- [x] Add csh5 runtime-service and generation metadata to the public CellShard surface.
- [x] Expose inspect accessors for loaded execution/runtime-service metadata.
- [x] Update CellShard and Cellerator docs to describe the immutable-emission boundary and owner-hosted pack service.
- [x] Run focused csh5/runtime tests after the metadata changes.

## Blockers
_None recorded yet._

## Progress Notes
- Created a dedicated workstream for the CellShard runtime-service contract reset so it does not blur with the older blocked-ell ingest/runtime stream.
- Extended `extern/CellShard/src/disk/csh5.cuh` and `csh5.cc` with optional runtime-service metadata, generation identity fields, shard owner node/rank arrays, and inspect accessors for loaded series backends.
- Updated the workbench writer to append default owner-hosted runtime-service metadata when it records execution metadata for a new series.
- Rewrote the CellShard and Cellerator ingest/runtime docs so pack is described as the primary execution artifact and Cellerator is limited to immutable canonical emission before CellShard layout/build work.
- Built `cellShardSeriesH5Test` successfully and ran `./build/cellShardSeriesH5Test` successfully with the new runtime-service metadata roundtrip coverage.
- Expanded the docs to spell out the owner-hosted runtime model in single-machine and distributed operation, including coordinator, master reader, pack-prep, executor, and append-staging roles.
- Reworked the on-disk cache layout into a legible `instances/<fingerprint>/metadata` plus `packs/canonical` and `packs/execution` tree, wrote the new paths into the cache manifest, and added runtime coverage that checks those directories and files exist after cache warmup.
- Fixed the canonical blocked-ell execution-pack builder so serialized execution partitions always carry explicit identity column maps; this restored green execution-pack warmup coverage under the new cache layout.
- Qualified execution-pack cache paths by `execution_plan_generation`, `pack_generation`, and `service_epoch`, updated pack-delivery descriptors to advertise the same generation-aware relative path, and documented the new execution-pack directory shape in `extern/CellShard/README.md`.
- Hardened `.csh5` open/append paths to require `schema_version`, replaced raw HDF5 1D reads with extent-checked reads, and reject inconsistent top-level totals, non-boundary shard tables, and unknown partition codec ids before any payload load.
- Added focused runtime coverage for generation-qualified execution-pack cache warmup plus rejection tests for unsupported schema versions, inconsistent header totals, and malformed matrix-table extents; `cellShardDatasetH5Test`, `cellShardExportRuntimeTest`, and `cellShardFirstFileFixtureTest` all pass with the hardened loader.

## Next Actions
- Implement the first concrete owner-node coordinator/runtime surface on top of the now generation-qualified cache tree and runtime-service metadata.
- Add explicit append staging and publish/cutover state handling so packs can rebuild and swap generations without mutating active payloads in place.
- Decide whether executor-side delivered packs should mirror the owner-side `plan.<execution_plan_generation>-pack.<pack_generation>-epoch.<service_epoch>` directory shape exactly once remote delivery is implemented.

## Done Criteria
- The public CellShard docs describe pack as the primary execution artifact served from the owner-hosted csh5 source.
- csh5 can persist and reload runtime-service and generation metadata needed for the owner-service contract.
- A focused test verifies the new metadata roundtrips through append/load and inspect accessors.
