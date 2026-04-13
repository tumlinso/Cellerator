---
slug: "cellshard-core-partition-rename"
status: "in_progress"
execution: "ready"
owner: "codex"
created_at: "2026-04-13T14:45:32Z"
last_heartbeat_at: "2026-04-13T14:47:06Z"
last_reviewed_at: "2026-04-13T14:47:06Z"
stale_after_days: 14
objective: "rename CellShard core storage runtime and schema terminology from part to partition without compatibility shims"
---

# Current Objective

## Summary
Complete the deep CellShard rename by moving the remaining storage, runtime, and csh5 schema surfaces from part terminology to partition terminology.

## Quick Start
- Why this stream exists: the public export, Python, and workbench summaries now say partition, but the core CellShard storage and schema layers still use part names.
- In scope: extern/CellShard/src/sharded, related disk and HDF5 schema code, affected Cellerator call sites, tests, benchmarks, and docs.
- Out of scope and dependencies: unrelated performance tuning stays separate unless the rename uncovers a correctness issue or a required schema migration detail.
- Required skills: todo-orchestrator for the ledger and cuda-v100 if a hot runtime path needs careful validation after the rename.
- Required references: AGENTS.md, extern/CellShard/README.md, extern/CellShard/src/sharded/sharded.cuh, extern/CellShard/src/sharded/sharded_host.cuh, extern/CellShard/src/sharded/series_h5.cuh, extern/CellShard/src/sharded/series_h5.cc, extern/CellShard/src/sharded/disk.cuh, extern/CellShard/src/sharded/disk.cu, src/ingest/series/series_ingest.cuh, src/workbench/series_workbench.cc, tests/cellshard_series_h5_test.cu.

## Planning Notes
- The user explicitly removed compatibility requirements, so the internal API and on-disk schema can be renamed directly rather than carried through a compatibility layer.
- This stream should preserve compileability and restore the affected runtime tests as the rename proceeds, not defer all validation until the end.

## Assumptions
- No backward compatibility shim is required for older part-named APIs or part-named csh5 schema keys.
- The export and workbench surface rename that is already landed becomes the target vocabulary for the remaining core layers.
- The separate CellShard debug stream should hold crash triage so the rename ledger stays focused on the structural rename.

## Suggested Skills
- `todo-orchestrator` - Keep the deep rename ledger current while the refactor spans multiple layers.
- `cuda-v100` - Validate any touched hot sparse runtime surfaces on the real sm_70 assumptions if the rename interacts with execution code.

## Useful Reference Files
- `AGENTS.md` - Repo-wide engineering and performance rules.
- `extern/CellShard/README.md` - Defines the intended CellShard scope and the current Blocked-ELL-first posture.
- `extern/CellShard/src/sharded/sharded.cuh` - Core sharded matrix type and helper surface that still uses part names.
- `extern/CellShard/src/sharded/sharded_host.cuh` - Host-side sharded builders and fetch helpers that still use part terminology.
- `extern/CellShard/src/sharded/series_h5.cuh` - Series HDF5 public constants and view types.
- `extern/CellShard/src/sharded/series_h5.cc` - Series HDF5 writer, reader, and fetch implementation.
- `extern/CellShard/src/sharded/disk.cuh` - Packfile-style disk schema and loader surface.
- `src/ingest/series/series_ingest.cuh` - Main Cellerator series writer path that consumes CellShard layout names.
- `tests/cellshard_series_h5_test.cu` - Existing HDF5 schema and roundtrip coverage that will need coordinated rename updates.

## Plan
- Build a concrete inventory of remaining part-named symbols, file schema keys, and public or semi-public APIs in CellShard core and dependent Cellerator code.
- Rename the core sharded matrix structs, helper functions, and fetch or drop APIs from part terminology to partition terminology.
- Rename csh5 schema fields such as num_parts and part_* to num_partitions and partition_* in the writer, reader, and validation paths.
- Update Cellerator ingest, workbench, tests, and benchmarks to the renamed core and schema vocabulary.
- Add or update focused coverage so the renamed partition schema roundtrips cleanly and the build catches stale part terminology in the touched surfaces.

## Tasks
- [x] Completed the source-facing partition rename for export, Python, and workbench summaries.
- [~] Build a full inventory of part-named symbols and schema fields across extern/CellShard/src and the dependent Cellerator call sites.
- [ ] Rename the core sharded data structures and helper APIs from part to partition.
- [ ] Rename the csh5 schema attrs and datasets from num_parts and part_* to num_partitions and partition_*.
- [ ] Update ingest, workbench, tests, and benchmark code to the renamed core schema.
- [ ] Add targeted schema and runtime coverage for the renamed partition surface.

## Blockers
- Avoid overlapping edits in extern/CellShard/src/sharded/{shard_paths,series_h5,sharded_host}* and extern/CellShard/src/sharded/disk* until the packfile-cache rewrite stream finishes.

## Progress Notes
- Completed the source-facing partition rename in export, Python, and workbench summaries, which now defines the target vocabulary for the remaining core layers.
- The separate debug stream owns the current workbench runtime crash so this ledger can stay focused on the structural rename and schema sweep.
- Inventory pass found roughly 1877 part-related matches concentrated in extern/CellShard/src/sharded/*, extern/CellShard/src/sharded/disk.*, extern/CellShard/src/sharded/series_h5.*, and extern/CellShard/src/convert/blocked_ell_from_compressed.cuh.
- Paused overlapping storage-file edits while the dedicated packfile-cache rewrite stream owns shard_paths, sharded_host, series_h5, and disk surfaces.

## Next Actions
- Run a targeted ripgrep inventory over extern/CellShard/src/sharded and extern/CellShard/src/sharded/disk* to establish the rename order.
- Start the first code patch in the core sharded type and helper headers before moving on to the HDF5 schema keys.
- Rename order should start with sharded.cuh and sharded_host.cuh, then disk.cuh and disk.cu, then series_h5.cuh and series_h5.cc, and only then the dependent Cellerator surfaces and tests.
- Resume the deep rename after the packfile-cache rewrite lands, starting from the rewritten storage backend instead of the old packfile code.

## Done Criteria
- Core CellShard sharded structs, helpers, and fetch surfaces use partition terminology instead of part terminology.
- The csh5 on-disk schema uses num_partitions and partition_* consistently in the primary storage metadata.
- Cellerator builds and the focused CellShard runtime tests pass against the renamed partition schema.
- The remaining active CellShard ledgers are about debugging or new features, not lingering part terminology.
