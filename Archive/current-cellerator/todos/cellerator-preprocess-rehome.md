# Cellerator Preprocess Rehome

Last updated: 2026-05-04

## Quick Start

Move the standalone preprocessing project back into Cellerator with a hard
cutover. Keep reusable CUDA preprocessing math under `src/compute/preprocess/`
and biology-facing preprocessing policy/runtime/workbench under
`src/preprocess/`.

## Status

- Status: done
- Execution: closed
- Owner: codex

## Assumptions

- Assay default is scRNA-seq with explicit raw-count state before QC,
  normalize-total, and log1p.
- Blocked-ELL and Sliced-ELL are first-class preprocessing layouts; compressed /
  CSR remains fallback.
- Target CUDA posture is native V100 `sm_70`.
- This is a hard cutover: no `CellShardPreprocess` submodule, package, target,
  or compatibility include remains.

## Suggested Skills

- `todo-orchestrator`: maintain resumable migration state.
- `bio-experiments`: preserve scRNA preprocessing semantics.
- `cuda`: keep preprocessing kernels aligned with V100 sparse bio-data guidance.

## Useful Reference Files

- `AGENTS.md`
- `scope.md`
- `optimization.md`
- `docs/architecture.qmd`
- `../CellShard/AGENTS.md`

## Plan

- [x] Move preprocessing headers, CUDA source, runtime policy, workbench, tests,
  and benchmarks into Cellerator-owned paths.
- [x] Split CMake targets into `Cellerator::compute_preprocess` and
  `Cellerator::preprocess`.
- [x] Remove CellShard's nested CellShardPreprocess package/install path.
- [x] Remove the CellStack `CellShardPreprocess` submodule entry.
- [x] Update documentation and ledgers to describe Cellerator ownership.
- [x] Configure, build, and run focused validation where the local environment
  permits.

## Progress Notes

- Created `include/Cellerator/compute/preprocess/`,
  `src/compute/preprocess/`, `include/Cellerator/preprocess/`, and
  `src/preprocess/`.
- Ported native preprocessing tests and benchmarks into Cellerator target names.
- Removed CellShard's `CELLSHARD_INSTALL_PREPROCESS` CMake branch and package
  consumer.
- Verified Cellerator configure/build, all focused preprocessing tests, adjacent
  Cellerator smoke tests, CellShard configure/build, and CellShard package
  consumer install check.
- `cellShardMaskGroupsRuntimeTest` still exits 14, matching the existing
  `cellerator-sparse-ml-layout` follow-up note.

## Validation Commands

- `cmake -S Cellerator -B Cellerator/build`
- `cmake --build Cellerator/build -j 4`
- `./Cellerator/build/celleratorPreprocessAdapterStagingTest`
- `./Cellerator/build/celleratorPreprocessDoublePreprocessRejectionTest`
- `./Cellerator/build/celleratorPreprocessRuntimeTest`
- `./Cellerator/build/celleratorPreprocessQCMetricsEquivalenceTest`
- `./Cellerator/build/celleratorPreprocessQCMaskGroupsTest`
- `./Cellerator/build/celleratorPreprocessQCFleetTest`
- `./Cellerator/build/coreSparseLayoutRuntimeTest`
- `./Cellerator/build/quantizedMatrixTest`
- `./Cellerator/build/exactSearchRuntimeTest`
- `./Cellerator/build/sparseOpsRuntimeTest`
- `cmake -S CellShard -B CellShard/build`
- `cmake --build CellShard/build -j 4`
- `cmake --build CellShard/build --target cellShardInspectPackageTest -j 1`

## Next Actions

- No migration action remains; the remaining mask-groups exit-14 decision stays
  with `cellerator-sparse-ml-layout`.

## Done Criteria

- Cellerator builds the moved preprocessing targets.
- CellShard no longer fetches, installs, or tests `CellShardPreprocess`.
- Root no longer tracks the `CellShardPreprocess` submodule.
- Public docs describe Cellerator as the preprocessing owner.
