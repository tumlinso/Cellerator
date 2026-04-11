# ncurses UI Notes

## Why

The most promising terminal UI surfaces in this repo are:

- ingest planning and conversion into `series.csh5`
- inspection of existing `series.csh5` files plus preprocess execution

This should live in `Cellerator`, not `CellShard`.

Reason:

- `CellShard` should define storage layout, codecs, and persisted matrix/container rules
- `Cellerator` should own source-format ingest, dataset/provenance handling, planning, validation, and workflow orchestration

## Good First UI

Build a dataset ingest planner first.

Core flow:

1. choose or load a manifest / set of source files
2. classify inputs:
   - matrix path
   - feature path
   - barcode path
   - metadata path
   - source format
3. preview detected rows / cols / nnz
4. assign dataset ids and output grouping
5. choose part and shard planning knobs
6. preview estimated part counts, shard counts, and output bytes
7. run conversion with progress and validation
8. inspect the written `series.csh5`

Useful panes:

- left: datasets / manifests / parts / shards
- center: details for selected item
- bottom: progress log, warnings, and validation failures

## Second UI

Add a `series.csh5` inspect + preprocess console.

Core flow:

1. open an existing `series.csh5`
2. inspect datasets, provenance, parts, shards, rows, cols, nnz
3. configure preprocess thresholds
4. run preprocess over selected shards / parts
5. show kept-cell / kept-gene summaries and timing

## Suggested Boundaries

Keep the UI thin.

- ncurses layer: forms, navigation, progress, summaries
- Cellerator app layer: ingest planning, validation, preprocess orchestration
- CellShard layer: read/write/fetch/cache/codec primitives only

Do not make the UI depend on legacy monoliths like `src/load_embryo_mtx_shards.cu`.
Prefer wrapping the modular ingest and preprocess paths as they become first-class targets.

## Near-Term Backend Targets

Useful future Cellerator entrypoints for a UI:

- `inspect_manifest(...)`
- `plan_series_ingest(...)`
- `convert_manifest_to_series_csh5(...)`
- `verify_series_csh5(...)`
- `summarize_series_csh5(...)`
- `run_preprocess_pass(...)`

Each should return structured status/progress data, not just print to stdout.
