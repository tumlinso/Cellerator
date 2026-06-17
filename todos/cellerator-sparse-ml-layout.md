---
status: done
execution: closed
owner: codex
---

# Cellerator Sparse ML Layout

## Quick Start

Refactor Cellerator so `src/compute` is organized around sparse ML math over
CellShard matrices. The former core split is dissolved: matrix,
runtime, quantized, interop, types, and parameter contracts live as direct
Cellerator domains, while matrix conversion, bucket planning, and CUDA compute
primitives live under `compute`. Forward-neighbor caller policy is external to
Cellerator.

## Skills And References

- `todo-orchestrator`: keep this ledger current during the multi-step refactor.
- `cuda`: keep CUDA source organization aligned with sparse hot-path tuning.
- `AGENTS.md`, `scope.md`, `optimization.md`, and `src/compute/README.md`.

## Tasks

- [x] Move compute files into sparse ML math folders.
- [x] Add backend-folder structure for library and custom paths.
- [x] Move forward-neighbor policy into the neighbor-caller sibling package.
- [x] Replace old Core public include paths with the new unpublished Core layout.
- [x] Update CellShard compatibility shims to the new Core format paths.
- [x] Move CellShard-free CUDA runtime substrate into the Cellerator runtime domain.
- [x] Move matrix conversion, bucket planning, and CUDA warp reduction primitives out of the representation/runtime substrate.
- [x] Update CMake targets and aliases.
- [x] Run focused Cellerator build/test pass after Core layout migration.
- [x] Decide whether the pre-existing CellShard mask-groups expectation failure belongs in this stream or a separate CellShard test fix.
- [x] Hard-cut the `former core` identity into first-class Cellerator domains with no compatibility headers or `Cellerator::core` alias.
- [x] Run focused Cellerator build/test pass after the hard cut.
- [x] Verify sibling CellShard shims still build against the new include layout.

## Assumptions

- This is a layout/refactor pass, not a new math-feature implementation pass.
- Library-backed NVIDIA paths remain the default unless an existing custom path
  already owns the operation.
- CellShard is data handling only; Cellerator owns preprocessing policy and compute.
- Cellerator is unpublished, so old Core includes were removed rather than kept as wrappers.
- Cellerator owns substrate and format mechanics directly; `src/compute` owns sparse math such as SpMM, matmul, ML reductions, and training operators.
- The CellShard mask-groups exit-14 behavior is separate unless this hard cut changes its failure mode.

## Progress Notes

- Workstream opened from the accepted plan.
- Moved `compute/sparse/ops` to `compute/sparse/ops`.
- Moved `compute/model_ops` to `compute/ml/model_ops`.
- Moved shared `host_buffer` to `compute/core`.
- Moved cuVS/KNN scoring helpers to `compute/neighbors/scoring`.
- Removed Cellerator-owned forward-neighbor compatibility wrappers; caller policy now lives outside this package.
- Rebuilt the former core split into `core/matrix`, `core/runtime`, `core/quantized`, and `core/interop`, with conversion and compute primitives owned by `compute`.
- Moved quantized format/pack/decode headers under `include/Cellerator/quantized` and kept quantized SpMM use in `src/compute`.
- Updated CellShard matrix, conversion, bucket, and device-view shims to include `Cellerator/matrix/...` for representation and `Cellerator/compute/matrix/convert/...` for compute-owned conversion.
- Moved conversion/bucket sources into `src/compute/matrix/convert`, added the `cellerator_compute_matrix_convert` target, and moved `warp_reduce.cuh` into `src/compute/core/primitives`.
- Verified the mechanical ownership move with `cmake --build build -j 4` and `./build/coreSparseLayoutRuntimeTest`.
- Verified Cellerator targets: `coreSparseLayoutRuntimeTest`, `sparseOpsRuntimeTest`, `quantizedMatrixTest`, `quantizePrimitiveTest`, and `abiRuntimeTest`.
- Ran full Cellerator build with `cmake --build build -j 4`.
- Verified standalone CellShard configure/build for `cellShardMaskGroupsRuntimeTest`; the binary still exits 14 on the row-keep expectation without diagnostics.
- Hard-cut the former core split into direct Cellerator domains: `include/Cellerator/{matrix,runtime,quantized,interop}`, top-level `matrix.cuh`, `types.cuh`, and `parameters.hh`, plus `src/runtime` and `src/config.cuh.in`.
- Removed the `Cellerator::core` target identity; the runtime substrate now exports as `Cellerator::runtime`.
- Verified no stale hard-cut references outside build output with repository scans for `Cellerator/core`, `cellerator::core`, `Cellerator::core`, `cellerator_core`, `include/Cellerator/core`, `src/core`, `CelleratorCore`, and `CORE_REAL`.
- Verified Cellerator configure and build: `cmake -S . -B build` and `cmake --build build -j 4`.
- Verified focused Cellerator tests: `./build/coreSparseLayoutRuntimeTest`, `./build/quantizedMatrixTest`, `./build/sparseOpsRuntimeTest`, `./build/exactSearchRuntimeTest`, `./build/abiRuntimeTest`, `./build/developmentalTimeRuntimeTest`, `./build/developmentalTimeTrajectoryRuntimeTest`, `./build/stateReduceRuntimeTest`, and `./build/celleratorPreprocessRuntimeTest`.
- Updated sibling CellShard Cellerator shims and package config to the direct include layout and `Cellerator::runtime`.
- Verified sibling CellShard configure and build: `cmake -S . -B build` and `cmake --build build -j 4`.

## Next Actions

_None recorded yet._
