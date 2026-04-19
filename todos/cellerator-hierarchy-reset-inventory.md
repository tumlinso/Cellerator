# Cellerator Hierarchy Reset Inventory

Last updated: 2026-04-19

## Scope

This file tracks the concrete code and doc changes landed under the active
`cellerator-hierarchy-reset` workstream.

## Public Surface And Build

- Added the canonical public include facade tree under `include/Cellerator/`.
- Added facade wrappers for compute, ingest, models, quantized, support, torch,
  trajectory, and workbench entry surfaces.
- Updated `CMakeLists.txt` so repo targets consume the `include/Cellerator/`
  tree as the canonical public include surface.
- Migrated in-repo callers off `../src/...` includes across tests, benches, and
  `python/module.cc`.
- Rehomed flat support headers under `src/support/`.
- Moved the ncurses app entrypoint under `src/apps/workbench/`.
- Removed the orphaned embryo MTX shard converter, the stale `matrixTest` surface, and the last live quarantine-tree build path after confirming they still depended on a deleted matrix-era subsystem.
- Split the last preprocess orchestration helper band into `src/workbench/runtime/internal/orchestration_support_part.hh` and the remaining blocked-ELL layout heuristics into `src/ingest/dataset/internal/convert_layout_support_part.hh`.
- Replaced the main ingest planning/layout numeric arrays with `owned_buffer` storage in `src/ingest/dataset/dataset_ingest.cuh` and `src/ingest/dataset/internal/dataset_convert_part.hh`, and updated the sliced payload append path to emit the current bucketed sliced partition type.

## Runtime Cleanup

- Split metadata helper support from `src/workbench/runtime/pipeline.cu` into
  `src/workbench/runtime/internal/metadata_support_part.hh`.
- Split metadata read/write and rewrite helpers from
  `src/workbench/runtime/pipeline.cu` into
  `src/workbench/runtime/internal/metadata_io_part.hh`.
- Split browse-cache construction from `src/workbench/runtime/pipeline.cu` into
  `src/workbench/runtime/internal/browse_cache_build_part.hh`.
- Split preprocess analysis, persistence, and finalize-after-preprocess logic
  from `src/workbench/runtime/pipeline.cu` into
  `src/workbench/runtime/internal/preprocess_runtime_part.hh`.

## Ingest Cleanup

- Split H5AD dataframe and metadata-table helpers from
  `src/ingest/h5ad/h5ad_reader.cuh` into
  `src/ingest/h5ad/internal/dataframe_part.hh`.
- Split source-loading helpers from `src/ingest/dataset/dataset_ingest.cuh`
  into `src/ingest/dataset/internal/source_load_part.hh`.
- Split optimized layout and spool helpers from
  `src/ingest/dataset/dataset_ingest.cuh` into
  `src/ingest/dataset/internal/layout_build_part.hh`.
- Split the long `convert_manifest_dataset_to_hdf5()` workflow from
  `src/ingest/dataset/dataset_ingest.cuh` into
  `src/ingest/dataset/internal/dataset_convert_part.hh`.

## Behavior And Test Fixes

- Restored converted-dataset `/execution` metadata persistence by wiring
  `append_execution_layout_metadata()` back into the conversion path.
- Added durable diagnostics in `tests/dataset_workbench_runtime_test.cc` so
  runtime failures no longer collapse into silent exit codes.
- Fixed a stale blocked-finalize runtime test assumption by writing a dedicated
  blocked-ELL fixture for that subcase.
- Fixed the sliced browse-cache expectation in
  `tests/dataset_workbench_runtime_test.cc` so it matches the current runtime
  contract.
- Fixed an embedded-metadata fixture bug where the manual table shape declared
  two columns while carrying three.
- During the preprocess extraction, kept `analyze_dataset_preprocess`,
  `persist_preprocess_analysis`, and `run_preprocess_pass` as emitted symbols so
  downstream link targets remain valid.

## Docs And Tracking

- Updated `README.md`.
- Updated `AGENTS.md`.
- Updated `docs/architecture.qmd`.
- Added and maintained `todos/cellerator-hierarchy-reset.md`.
- Updated `todo-status.md` as the stream progressed.

## Validation Run During This Stream

- `cmake --build build -j 4 --target datasetIngestCompileTest`
- `cmake --build build -j 4 --target cellerator_workbench`
- `cmake --build build -j 4 --target datasetWorkbenchRuntimeTest`
- `./build/datasetWorkbenchRuntimeTest`
- `./build/cellShardFirstFileFixtureTest`
- Earlier focused coverage also passed during the include-surface migration:
  `forwardNeighborsCompileTest`, `computeAutogradRuntimeTest`,
  `trajectoryRuntimeTest`, `developmentalTimeCudaRuntimeTest`, and
  `quantizeModelTest`.
