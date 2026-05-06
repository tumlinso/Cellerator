# Todo Status

## Summary
Use this file as the quick pickup register for `todos.md` workstreams.
- `ready`: planned work that can be started now.
- `claimed`: currently being written; choose another stream.
- `idle`: unfinished but resumable; safe to pick up.
- `closed`: completed or removed from pickup rotation.

## Workstreams
- `sequence-bits-dna2`: closed. Added and validated the first SequenceBits dna2 primitive with packed word64 storage, planes32 warp compute representation, correctness kernels, tests, benchmark, and docs.
- `cellerator-sparse-ml-layout`: idle. Contract-first CelleratorCore matrix/runtime/quantized layout, compute-owned matrix conversion and CUDA primitives, and CellShard shims are checkpointed and build-tested; remaining pickup is deciding whether the CellShard mask-groups exit-14 expectation belongs here or a separate CellShard test fix.
- `cellerator-python-preprocess-runtime`: closed. Added Cellerator Python packaging plus GPU-native preprocessing runtime delegation and validated direct build, source smoke, wheel build, and installed-wheel import smoke.
- `cellerator-preprocess-rehome`: closed. Cellerator now owns preprocessing through split compute and pipeline targets; CellShard and CellStack no longer install or track the old preprocessing package/submodule.
- `cellshard-preprocess-gpu-biology-backbone`: closed. CellShardPreprocess owns native preprocessing APIs and benchmarks; Cellerator preprocessing APIs and root benchmark targets have been removed.
- `cellshard-multi-assay-archive`: closed. Multi-assay archive descriptors, row-map helpers, the biology semantics package validation, docs, and tests are in place; CSPACK payloads remain single-assay.

## Staleness Review
_No staleness review recorded yet._

## Cleanup Status
- Cleanup mode is explicit only.
- Safe to call `todo-cleanup`: no, `cellerator-sparse-ml-layout` is active.
