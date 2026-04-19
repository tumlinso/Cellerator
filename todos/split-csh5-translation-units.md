---
slug: "split-csh5-translation-units"
status: "done"
execution: "closed"
owner: "codex"
created_at: "2026-04-18T12:44:31Z"
last_heartbeat_at: "2026-04-18T12:58:00Z"
last_reviewed_at: "2026-04-18T12:58:00Z"
stale_after_days: 14
objective: "Split extern/CellShard/src/disk/csh5.cc into coherent translation units without changing behavior"
---

# Current Objective

## Summary
Split extern/CellShard/src/disk/csh5.cc into coherent translation units without changing behavior

## Quick Start
- Why this stream exists: `extern/CellShard/src/disk/csh5.cc` had grown into a mixed 8k+ line translation unit that bundled HDF5 schema helpers, dataset create/append/finalize logic, cache and pack materialization, header loading, fetch paths, and sliced execution-device caching in one file.
- In scope: split the monolithic translation unit without changing the public `.csh5` API or runtime behavior, keep the build graph coherent, and revalidate the focused CellShard runtime binaries.
- Out of scope / dependencies: no schema redesign, no symbol renames, no deeper extraction of the private helper surface yet; that remains a follow-up if the private shared include should also be subdivided.
- Required skills: `todo-orchestrator`
- Required references: `AGENTS.md`, `todos.md`, `todo-status.md`, `extern/CellShard/CMakeLists.txt`, `extern/CellShard/src/disk/csh5.cuh`, `extern/CellShard/src/disk/csh5.cc`, `tests/cellshard_dataset_h5_test.cu`, `tests/dataset_workbench_runtime_test.cc`, `tests/cellshard_first_file_fixture_test.cu`

## Planning Notes
- `csh5.cc` was not just a writer. The file owned the canonical `.csh5` container contract, pack/cache materialization, runtime fetch entrypoints, and the sliced execution-device cache.
- The lowest-risk split was a physical translation-unit split first, not a semantic redesign: keep all private helpers shared via one private include, then separate write/schema APIs from runtime/cache/fetch APIs.

## Assumptions
- Behavior preservation mattered more than reducing total private helper LOC in one pass.
- A private shared implementation include under `src/disk/` is acceptable for stage 1 as long as it is not accidentally installed as part of the public header surface.

## Suggested Skills
- `todo-orchestrator`

## Useful Reference Files
- `extern/CellShard/CMakeLists.txt`
- `extern/CellShard/src/disk/csh5.cuh`
- `extern/CellShard/src/disk/csh5.cc`
- `tests/cellshard_dataset_h5_test.cu`
- `tests/dataset_workbench_runtime_test.cc`
- `tests/cellshard_first_file_fixture_test.cu`

## Plan
- Extract the shared private implementation and state from the top of `csh5.cc` into a non-installed internal include local to `src/disk/`.
- Keep create/append/finalize APIs in `extern/CellShard/src/disk/csh5.cc`.
- Move bind/load/cache/fetch/warm/execution-device-cache APIs into `extern/CellShard/src/disk/csh5_runtime.cc`.
- Update `extern/CellShard/CMakeLists.txt` so `cellshard_inspect` builds both translation units.
- Rebuild the focused CellShard targets and run the runtime binaries that cover the moved paths.

## Tasks
- [x] Map the real responsibility boundaries inside `csh5.cc`.
- [x] Extract the shared private implementation into a local non-installed include.
- [x] Split public write/schema entrypoints from runtime/cache/fetch entrypoints.
- [x] Update the CellShard inspect build graph.
- [x] Rebuild and run focused runtime validation binaries.

## Blockers
_None recorded yet._

## Progress Notes
- Extracted the shared implementation into `extern/CellShard/src/disk/csh5/internal.hh` so the split keeps the shared implementation private without leaving the old monolith in place.
- Reduced `extern/CellShard/src/disk/csh5.cc` from 8363 lines to 2006 lines and moved runtime/cache/fetch behavior into the new `extern/CellShard/src/disk/csh5/runtime.cc` at 1322 lines.
- Updated `extern/CellShard/CMakeLists.txt` so `cellshard_inspect` now builds the split `src/disk/csh5/*.cc` translation units instead of one monolithic `csh5.cc`.
- Rebuilt `cellShardDatasetH5Test`, `datasetWorkbenchRuntimeTest`, and `cellShardFirstFileFixtureTest` successfully.
- Ran `./build/cellShardDatasetH5Test`, `./build/datasetWorkbenchRuntimeTest`, and `./build/cellShardFirstFileFixtureTest` successfully after the split. `cellShardDatasetH5Test` still emitted its expected negative-path stderr but exited `0`.
- Followed up by giving `csh5` its own directory: the canonical public API now lives at `extern/CellShard/src/disk/csh5/api.cuh`, implementation lives under `extern/CellShard/src/disk/csh5/`, and `extern/CellShard/src/disk/csh5.cuh` is now a forwarding header for compatibility.
- Started the write-side cleanup by renaming the private shared include to `extern/CellShard/src/disk/csh5/internal.hh`, moving dataset creation into the new `extern/CellShard/src/disk/csh5/create.cc`, and extracting common create-time schema writers so the blocked, quantized-blocked, and sliced create paths no longer each hand-inline the same dataset/provenance/codec scaffolding.
- Split the remaining write-side responsibilities again: metadata appenders now live in `extern/CellShard/src/disk/csh5/metadata.cc`, preprocess refinalization lives in `extern/CellShard/src/disk/csh5/finalize_preprocess.cc`, and `extern/CellShard/src/disk/csh5/write.cc` is now limited to payload writers.
- Rebuilt `cellShardDatasetH5Test`, `datasetWorkbenchRuntimeTest`, and `cellShardFirstFileFixtureTest` successfully after the metadata/finalize split, then reran all three binaries successfully. `cellShardDatasetH5Test` still emitted its expected negative-path stderr but exited `0`.
- Split the private `extern/CellShard/src/disk/csh5/internal.hh` monolith into ordered private part headers: `internal_base_part.hh`, `preprocess_helpers_part.hh`, `execution_builders_part.hh`, and `execution_runtime_part.hh`, with `internal.hh` reduced to a thin umbrella that preserves the original namespace behavior for the `.cc` files.
- Rebuilt `cellShardDatasetH5Test`, `datasetWorkbenchRuntimeTest`, and `cellShardFirstFileFixtureTest` successfully after the private-header split, then reran all three binaries successfully. `cellShardDatasetH5Test` still emitted its expected negative-path stderr but exited `0`.

## Next Actions
- Optional follow-up only: if the new private part headers are still too broad, split `preprocess_helpers_part.hh` and `execution_builders_part.hh` further by domain (`schema_io`, `payload_codecs`, `cache_runtime`, `execution_fetch`) without changing the public API.

## Done Criteria
- `extern/CellShard/src/disk/csh5.cc` no longer carries the full `.csh5` backend alone.
- The runtime/cache/fetch surface lives in its own translation unit.
- Metadata appenders, preprocess finalization, and payload writers no longer share one `write.cc`.
- The shared implementation stays private to `src/disk/` and is not installed as part of the public API.
- The private shared implementation is no longer one 5k-line header; it is split into smaller ordered private parts behind `internal.hh`.
- Focused CellShard runtime build and run validation passes after the split.
