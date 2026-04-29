---
status: in_progress
execution: claimed
owner: codex
---

# Cellerator Sparse ML Layout

## Quick Start

Refactor Cellerator so `src/compute` is organized around sparse ML math over
CellShard matrices. Keep old public include paths as compatibility wrappers.
Forward-neighbor caller policy is external to Cellerator.

## Skills And References

- `todo-orchestrator`: keep this ledger current during the multi-step refactor.
- `cuda`: keep CUDA source organization aligned with sparse hot-path tuning.
- `AGENTS.md`, `scope.md`, `optimization.md`, and `src/compute/README.md`.

## Tasks

- [x] Move compute files into sparse ML math folders.
- [x] Add backend-folder structure for library and custom paths.
- [x] Move forward-neighbor policy into the neighbor-caller sibling package.
- [x] Preserve old public include paths as compatibility wrappers.
- [x] Update CMake targets and aliases.
- [ ] Run final focused build/test pass after checkpoint review.

## Assumptions

- This is a layout/refactor pass, not a new math-feature implementation pass.
- Library-backed NVIDIA paths remain the default unless an existing custom path
  already owns the operation.
- CellShard is data handling only; CellShardPreprocess is bio policy only.

## Progress Notes

- Workstream opened from the accepted plan.
- Moved `compute/autograd` to `compute/ml/autograd`.
- Moved `compute/model_ops` to `compute/ml/model_ops`.
- Moved shared `host_buffer` to `compute/core`.
- Moved cuVS/KNN scoring helpers to `compute/neighbors/scoring`.
- Removed Cellerator-owned forward-neighbor compatibility wrappers; caller policy now lives outside this package.

## Next Actions

- Review CMake/include wiring, then run configure/build and focused tests.
