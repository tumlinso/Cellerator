---
status: in_progress
execution: claimed
owner: codex
---

# Cellerator Sparse ML Layout

## Quick Start

Refactor Cellerator so `src/compute` is organized around sparse ML math over
CellShard matrices. CelleratorCore is now contract-first: `core/matrix`,
`core/runtime`, `core/quantized`, and `core/interop`; matrix conversion,
bucket planning, and CUDA compute primitives live under `compute`. Forward-neighbor caller policy is external to Cellerator.

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
- [x] Move CellShard-free CUDA runtime substrate into CelleratorCore.
- [x] Move matrix conversion, bucket planning, and CUDA warp reduction primitives out of CelleratorCore.
- [x] Update CMake targets and aliases.
- [x] Run focused Cellerator build/test pass after Core layout migration.
- [ ] Decide whether the pre-existing CellShard mask-groups expectation failure belongs in this stream or a separate CellShard test fix.

## Assumptions

- This is a layout/refactor pass, not a new math-feature implementation pass.
- Library-backed NVIDIA paths remain the default unless an existing custom path
  already owns the operation.
- CellShard is data handling only; CellShardPreprocess is bio policy only.
- Cellerator is unpublished, so old Core includes were removed rather than kept as wrappers.
- Core owns substrate and format mechanics; `src/compute` owns sparse math such as SpMM, matmul, ML reductions, and training operators.

## Progress Notes

- Workstream opened from the accepted plan.
- Moved `compute/sparse/ops` to `compute/sparse/ops`.
- Moved `compute/model_ops` to `compute/ml/model_ops`.
- Moved shared `host_buffer` to `compute/core`.
- Moved cuVS/KNN scoring helpers to `compute/neighbors/scoring`.
- Removed Cellerator-owned forward-neighbor compatibility wrappers; caller policy now lives outside this package.
- Rebuilt CelleratorCore into `core/matrix`, `core/runtime`, `core/quantized`, and `core/interop`, with conversion and compute primitives owned by `compute`.
- Moved quantized format/pack/decode headers under `include/Cellerator/core/quantized` and kept quantized SpMM use in `src/compute`.
- Updated CellShard matrix, conversion, bucket, and device-view shims to include `Cellerator/core/matrix/...` for representation and `Cellerator/compute/matrix/convert/...` for compute-owned conversion.
- Moved conversion/bucket sources into `src/compute/matrix/convert`, added the `cellerator_compute_matrix_convert` target, and moved `warp_reduce.cuh` into `src/compute/core/primitives`.
- Verified the mechanical ownership move with `cmake --build build -j 4` and `./build/coreSparseLayoutRuntimeTest`.
- Verified Cellerator targets: `coreSparseLayoutRuntimeTest`, `sparseOpsRuntimeTest`, `quantizedMatrixTest`, `quantizePrimitiveTest`, and `abiRuntimeTest`.
- Ran full Cellerator build with `cmake --build build -j 4`.
- Verified standalone CellShard configure/build for `cellShardMaskGroupsRuntimeTest`; the binary still exits 14 on the row-keep expectation without diagnostics.

## Next Actions

- Triage `cellShardMaskGroupsRuntimeTest` exit 14 separately unless this workstream is extended to cover CellShard mask-group behavior.
