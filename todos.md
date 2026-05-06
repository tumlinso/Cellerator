# Active Objectives

## Summary
Use this file as the canonical index for substantial multi-step work.

## Shared Assumptions
- Cellerator owns the accelerated scRNA preprocessing backbone, native workbench implementation, and preprocessing benchmarks.
- Native Cellerator GPU policy remains V100 `sm_70` and Blocked-ELL-first for Cellerator execution. Cellerator preprocessing treats Blocked-ELL and Sliced-ELL as first-class preprocessing layouts, with compressed / CSR as fallback.

## Suggested Skills
- `todo-orchestrator`: maintain the resumable migration ledger while implementing the supplied plan.
- `bio-experiments`: preserve scRNA raw-count, QC, normalization, and double-processing semantics.
- `cuda`: keep the implementation aligned with native V100 sparse bio-data layout and hot-path constraints.

## Useful Reference Files
- `AGENTS.md`: repository structure, testing, Blocked-ELL, and Volta policy.
- `optimization.md`: current V100 bottleneck and sparse preprocessing guidance.
- `docs/pipeline/preprocess/README.md`: existing preprocessing documentation to update if behavior moves.
- `docs/pipeline/README.md`: pipeline-level documentation surface to keep in sync.

## Workstreams
- `sequence-bits-dna2`: done / closed - Added the first narrow SequenceBits primitive with packed word64 and planes32 DNA encoding, CUDA proof kernels, CPU/CUDA tests, benchmark target, and docs.
- `cellerator-sparse-ml-layout`: in_progress / idle - Refactored Cellerator Core into contract-first matrix/runtime/quantized/interop layers, moved conversion and CUDA compute primitives under `src/compute`, updated CellShard shims, and kept `src/compute` as the sparse math/operator layer. Needs a decision on the standalone CellShard mask-groups exit-14 expectation.
- `cellerator-python-preprocess-runtime`: done / closed - Added the Cellerator-owned Python package, pybind module, `cellerator.pp` facade, and GPU-native preprocessing session delegation for `.csh5`/CellShard-backed scRNA preprocessing.
- `cellerator-preprocess-rehome`: done / closed - Moved the standalone preprocessing project back into Cellerator with split `Cellerator::compute_preprocess` and `Cellerator::preprocess` targets, removed CellShard's nested preprocessing package path, and removed the root submodule entry.
- `cellshard-preprocess-gpu-biology-backbone`: done / closed - CellShardPreprocess owns native preprocessing APIs and benchmarks; Cellerator preprocessing API/implementation/benchmark targets have been removed.
- `cellshard-multi-assay-archive`: done / closed - Added the multi-assay archive foundation, pointer-first row-map validation, the biology semantics package-backed semantic checks, cshard POD descriptors, docs, and compile/runtime coverage while leaving CSPACK payloads single-assay.

## Global Blockers
_None recorded yet._

## Progress Notes
- Started `sequence-bits-dna2`: requested scope is a new `include/Cellerator/seq/dna2.cuh` GPU-native DNA 2-bit primitive with packed storage words, warp-compute bitplanes, correctness kernels, tests, primitive benchmark, and docs. This work is separate from the existing `include/Cellerator/core/sequence/` port material.
- Finished `sequence-bits-dna2`: configured Cellerator, built `sequenceDna2Test`, `sequenceDna2CudaTest`, and `sequenceDna2Bench`, ran both focused tests, and ran the primitive benchmark with `./build/sequenceDna2Bench 1048576 16 1 10`.
- Extended `sequence-bits-dna2` validation: added `tests/seq/dna2_test_helpers.hh` for deterministic random sequence generation, changed the benchmark to use random DNA input with independent representation selection, and ran a packed-word64 vs warp-planes32 comparison matrix plus Nsight Systems profiles.
- Finished the first CPU SIMD backend pass: Highway now uses SIMD mask extraction/materialization for full 32-base ASCII pack/unpack, the CPU benchmark reports the active backend, and both Highway-enabled and scalar-only builds pass `sequenceDna2Test` plus `sequenceDna2CpuBench`.
- Added the first CelleratorCore ownership slice: `Cellerator::core` now exposes
  sparse layout primitives/device views under `include/Cellerator/core`, and
  CellShard layout headers are compatibility shims over those types.
- Verified the CelleratorCore/CellShard wiring with CellShard package-consumer
  checks, Cellerator sparse/quantized tests, and a CellShardPreprocess build.
- Started `cellerator-sparse-ml-layout` from the supplied source layout plan. The intended first pass is behavior-preserving except for moving forward-neighbor policy/API ownership into the new the external neighbor-caller package.
- Checkpointed `cellerator-sparse-ml-layout`: moved sparse-operator/model-op code under `src/compute/ml`, moved shared host buffering to `src/compute/core`, moved cuVS/KNN scoring helpers under `src/compute/neighbors/scoring`, and removed Cellerator-owned forward-neighbor compatibility wrappers.
- Implemented `cellshard-multi-assay-archive`: CellShard now has measurement-agnostic assay descriptors and row-map helpers, Cellerator validates those semantics against the biology semantics package, and docs state that multiome execution uses coordinated single-assay CSPACK artifacts.
- Ran `todo-cleanup --partial` and cleared workstreams: dual-cuda-optimization-modes, cellshard-first-stable-release, cellshard-blocked-ell-ingest-runtime, cellshard-runtime-service-contract, quantized-blocked-ell-codecs, cellshard-user-metadata-annotations, gpu-prototype-ingest-blocked-ell, gpu-prototype-model-sparse-boundaries, gpu-prototype-neighbors-trajectory, blocked-ell-optimization-study, gpu-benchmark-sliced-preprocess-campaign, cellshard-hierarchy-reset, implement-derived-subset-and-reorder-materialization-for-cellshard-and-workbench.
- Started `cellshard-preprocess-gpu-biology-backbone` from the supplied implementation plan.
- Finished `cellshard-preprocess-gpu-biology-backbone`: moved Blocked-ELL/Sliced-ELL native preprocessing and CSR fallback ownership into CellShardPreprocess, moved preprocessing benchmarks there, and removed Cellerator preprocessing APIs and root targets.
- Checkpointed CelleratorCore layout migration: old `core/sparse` and `quantized` public paths were removed, Core owns matrix representation/runtime substrate/quantized packing under `include/Cellerator/core`, compute owns conversion and CUDA primitives, CellShard shims were updated, and focused Cellerator runtime checks passed. Standalone CellShard `cellShardMaskGroupsRuntimeTest` builds but exits 14 on its row-keep expectation.
- Checkpointed the matrix/compute boundary cleanup: `core/format` became `core/matrix`, conversion and bucket moved to `compute/matrix/convert`, `warp_reduce.cuh` moved under `src/compute/core/primitives`, and `cmake --build build -j 4` plus `./build/coreSparseLayoutRuntimeTest` passed.
- Started `cellerator-preprocess-rehome`: requested scope is a hard cutover that removes the CellShardPreprocess submodule/package and splits the moved implementation into Cellerator compute preprocessing kernels plus the Cellerator preprocessing policy/runtime API.
- Finished `cellerator-preprocess-rehome`: Cellerator now builds preprocessing compute and pipeline targets, ported preprocessing tests and benchmarks, CellShard no longer fetches or installs the old package, and the root no longer tracks the preprocessing submodule. Validation passed for the Cellerator build, focused preprocessing tests, adjacent Cellerator smoke tests, CellShard build, and CellShard package-consumer install check. `cellShardMaskGroupsRuntimeTest` still exits 14 as previously recorded under `cellerator-sparse-ml-layout`.
- Started `cellerator-python-preprocess-runtime`: Python should expose Scanpy-like Cellerator preprocessing entry points, but all omics preprocessing compute must delegate to GPU-native Cellerator/CellShard runtime paths.
- Finished `cellerator-python-preprocess-runtime`: direct `_cellerator` build, source smoke test, wheel build, and installed-wheel import smoke passed. A fixture-backed GPU session test remains a future follow-up because no stable tiny `.csh5` fixture is currently checked in.

## Next Actions
- Decide whether CellShard `cellShardMaskGroupsRuntimeTest` exit 14 is an existing CellShard behavior issue or should be fixed under `cellerator-sparse-ml-layout`.

## Done Criteria
- Every active workstream in `todos/` is reflected here with a current status.

## Historical Summary
- Recent completed work included Blocked-ELL persistence, real-data sparse replay benchmarking, quantize sparse operator kernels, workbench browse-cache updates, semantic the biology semantics package cleanup, and the initial pointer-first neighbor workspace refactor.
- Detailed historical workstream notes are preserved in git history rather than as active repo ledgers.
