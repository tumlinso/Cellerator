# Todo Status

## Summary
Use this file as the quick pickup register for `todos.md` workstreams.
- `ready`: planned work that can be started now.
- `claimed`: currently being written; choose another stream.
- `idle`: unfinished but resumable; safe to pick up.
- `closed`: completed or removed from pickup rotation.

## Workstreams
- `cellerator-sparse-ml-layout`: idle. Source moves and compatibility wrappers are checkpointed; resume with a focused wiring review and build/test pass.
- `cellshard-preprocess-gpu-biology-backbone`: closed. CellShardPreprocess owns native preprocessing APIs and benchmarks; Cellerator preprocessing APIs and root benchmark targets have been removed.
- `cellshard-multi-assay-archive`: closed. Multi-assay archive descriptors, row-map helpers, the biology semantics package validation, docs, and tests are in place; CSPACK payloads remain single-assay.

## Staleness Review
_No staleness review recorded yet._

## Cleanup Status
- Cleanup mode is explicit only.
- Safe to call `todo-cleanup`: no, `cellerator-sparse-ml-layout` is active.
