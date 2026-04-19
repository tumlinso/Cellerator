---
slug: "cellerator-hierarchy-reset"
status: "in_progress"
execution: "claimed"
owner: "codex"
created_at: "2026-04-18T00:00:00Z"
last_heartbeat_at: "2026-04-18T00:00:00Z"
last_reviewed_at: "2026-04-18T00:00:00Z"
stale_after_days: 14
objective: "Reorganize Cellerator around a curated include/Cellerator facade tree and clearer src ownership without hiding hot paths"
---

# Current Objective

## Summary
Reset Cellerator around a canonical `include/Cellerator/` facade tree, migrate in-repo callers onto that surface, and move mixed-responsibility `src/` areas toward clearer subsystem ownership while keeping hot GPU code explicit.

## Quick Start
- Why this stream exists: `src/` is still the de facto public include surface, root-level helper headers remain flat, and several ingest/runtime files still mix multiple responsibilities in one place.
- In scope: canonical public facade tree, in-repo include migration, root helper rehome, explicit app entrypoint placement, focused ingest/runtime file splits, CMake/docs updates, and focused validation.
- Out of scope: namespace redesign, algorithm rewrites, kernel decomposition, or abstraction-heavy wrappers over hot paths.
- Required skills: `todo-orchestrator`.
- Required references: `AGENTS.md`, `CMakeLists.txt`, `src/workbench/dataset_workbench.hh`, `src/workbench/runtime/pipeline.cu`, `src/ingest/h5ad/h5ad_reader.cuh`, `src/ingest/dataset/dataset_ingest.cuh`.

## Planning Notes
- The new `include/Cellerator/` tree is canonical immediately; tests and benches should stop including `../src/...`.
- Preserve existing major facades such as workbench, model umbrellas, autograd, quantized, and forward-neighbor surfaces.
- Split orchestration-heavy files by behavior or helper domain, but do not bury cost-model-relevant kernels.

## Assumptions
- The repo is not trying to ship a standalone installable Cellerator package yet; this stream only makes `include/Cellerator/` the canonical in-repo public surface.
- The existing dirty worktree reflects active user work and must be preserved.

## Suggested Skills
- `todo-orchestrator` - Keep this repo-wide refactor resumable and pickup-safe.

## Useful Reference Files
- `CMakeLists.txt` - Current target include posture and app target paths.
- `src/workbench/runtime/pipeline.cu` - Largest shared runtime implementation surface.
- `src/ingest/h5ad/h5ad_reader.cuh` - Mixed HDF5 reader, dataframe, and conversion helpers.
- `src/ingest/dataset/dataset_ingest.cuh` - Mixed dataset planning, conversion, and write path.
- `todos/cellerator-hierarchy-reset-inventory.md` - Current concrete change inventory for this stream.

## Plan
- Add `include/Cellerator/` facades and migrate public callers to them.
- Rehome flat root helper headers into `src/support/` and move app entrypoints under `src/apps/`.
- Split the large ingest/runtime files behind thinner public entry headers.
- Update docs and focused targets to reflect the new hierarchy.

## Tasks
- [x] Add the canonical `include/Cellerator/` tree and migrate in-repo includes.
- [x] Rehome root helper headers and app entrypoints.
- [ ] Split the remaining `h5ad_reader.cuh`, `dataset_ingest.cuh`, and `pipeline.cu` helper bands into clearer internal substructures.
- [x] Re-run the focused runtime matrix after the current `datasetWorkbenchRuntimeTest` failure is understood.

## Blockers
_None recorded right now._

## Progress Notes
- Created the Cellerator hierarchy-reset workstream to track the repo-wide source-tree reset separately from the active CellShard streams.
- Added the first canonical `include/Cellerator/` facade tree, migrated tests, benches, and Python bindings onto that public surface, moved flat helper headers out of `src/` root, and rehomed the ncurses entrypoint under `src/apps/workbench/`.
- Rebuilt the focused compile/runtime matrix and passed `datasetWorkbenchRuntimeTest`, `cellShardFirstFileFixtureTest`, `forwardNeighborsCompileTest`, `computeAutogradRuntimeTest`, `trajectoryRuntimeTest`, `developmentalTimeCudaRuntimeTest`, and `quantizeModelTest` against the new include surface.
- Split the H5AD dataframe/feature-table helper band into `src/ingest/h5ad/internal/dataframe_part.hh` and replaced the body in `src/ingest/h5ad/h5ad_reader.cuh` with a private include.
- Split `src/ingest/dataset/dataset_ingest.cuh` twice: source-loading helpers now live in `src/ingest/dataset/internal/source_load_part.hh`, and the optimized layout/spool helpers now live in `src/ingest/dataset/internal/layout_build_part.hh`.
- Split the metadata-normalization and browse-cache support block out of `src/workbench/runtime/pipeline.cu` into `src/workbench/runtime/internal/metadata_support_part.hh`.
- Rebuilt `datasetIngestCompileTest`, `cellerator_workbench`, `datasetWorkbenchRuntimeTest`, and `cellShardFirstFileFixtureTest` successfully after the private-helper extraction.
- Added durable diagnostics to the silent preprocess/finalize section of `tests/dataset_workbench_runtime_test.cc`, then used them to trace the failure sequence instead of leaving the test as a black-box exit code.
- Fixed two stale runtime expectations in `tests/dataset_workbench_runtime_test.cc`: the blocked-finalize subcase now writes its own blocked-ELL fixture instead of assuming the sliced conversion path produces blocked payloads, and the sliced conversion subcase now expects browse-cache generation to wait until blocked finalize.
- Fixed a latent embedded-metadata fixture bug in `tests/dataset_workbench_runtime_test.cc` where the manual embedded metadata table declared `cols = 2` while carrying 3 columns.
- Wired `append_execution_layout_metadata()` into `convert_plan_to_dataset_csh5()` so converted datasets persist `/execution` metadata again instead of exposing only runtime-service metadata.
- Rebuilt and passed `datasetWorkbenchRuntimeTest` again after the execution-metadata fix and test-fixture refresh, and re-ran `cellShardFirstFileFixtureTest` successfully.
- Split the browse-cache build path out of `src/workbench/runtime/pipeline.cu` into `src/workbench/runtime/internal/browse_cache_build_part.hh`, leaving the remaining runtime translation unit focused on conversion/orchestration instead of multi-GPU browse assembly details.
- Rebuilt `cellerator_workbench` and `datasetWorkbenchRuntimeTest` after the browse-cache extraction and reran `datasetWorkbenchRuntimeTest` plus `cellShardFirstFileFixtureTest` successfully.
- Split the preprocess analysis/persist/finalize tail out of `src/workbench/runtime/pipeline.cu` into `src/workbench/runtime/internal/preprocess_runtime_part.hh`, so the main runtime translation unit now stops at conversion and defers the preprocess workflow to a private implementation part.
- The first preprocess extraction pass exposed a real public-symbol boundary for `analyze_dataset_preprocess`, `persist_preprocess_analysis`, and `run_preprocess_pass`; restored those as non-`inline` definitions inside the private part and reran the focused workbench build/tests successfully.
- Split the remaining `convert_manifest_dataset_to_hdf5()` body out of `src/ingest/dataset/dataset_ingest.cuh` into `src/ingest/dataset/internal/dataset_convert_part.hh`, leaving the public ingest header to own the stable conversion surface while the long offline conversion workflow lives in a private part.
- Rebuilt `datasetIngestCompileTest`, `cellerator_workbench`, and `datasetWorkbenchRuntimeTest` after the dataset-convert extraction, then reran `datasetWorkbenchRuntimeTest` and `cellShardFirstFileFixtureTest` successfully.
- Recorded the landed hierarchy-reset scope in `todos/cellerator-hierarchy-reset-inventory.md` so the file-level change inventory is durable and separate from the resumable workstream notes.

## Next Actions
- The largest hierarchy-reset objective is now mostly complete; remaining cleanup is optional second-pass shaping of the smaller helper bands still left inline in `src/workbench/runtime/pipeline.cu` and `src/ingest/dataset/dataset_ingest.cuh`.
- Keep the refreshed `datasetWorkbenchRuntimeTest` assertions in sync while finishing the remaining runtime splits so future hierarchy-reset passes fail with concrete messages instead of silent exits.

## Done Criteria
- Public callers use `#include <Cellerator/...>` instead of `../src/...`.
- `src/` root no longer owns legacy helper wrappers or app entrypoints that belong under clearer subsystem paths.
- The major mixed-responsibility ingest/runtime files are split into narrower internal units without changing hot-path behavior.
- Focused compile/runtime validation remains green after the migration.
