---
slug: "cellerator-umbrella-restructure"
status: "done"
execution: "closed"
owner: "codex"
---

# Cellerator Umbrella Restructure

## Quick Start
- Cellerator is becoming the umbrella repository that contains Baseplane and
  CellShard as components.
- CellStack is being deprecated to a thin wrapper with only the Cellerator
  submodule.
- The old Cellerator implementation is preserved under
  `Archive/current-cellerator/` and is not part of the default build graph.

## Assumptions
- Hard cuts are preferred over compatibility aliases for old Cellerator
  preprocessing/model/Torch targets.
- Baseplane and CellShard keep their own implementation ownership.
- The foundation slice stays small: core layout/runtime, dist, and matrix
  conversion needed by CellShard.
- Ignored large local fixture payloads are not moved into Git.

## Tasks
- [x] Archive current Cellerator implementation.
- [x] Restore required Cellerator foundation files in the new root.
- [x] Add Baseplane and CellShard under `components/`.
- [x] Add umbrella docs, agent guide, fixture scaffold, and root ledger.
- [x] Validate default Cellerator configure/build and focused targets.
- [x] Reduce CellStack to a deprecated one-submodule wrapper.
- [x] Validate wrapper submodule status.

## Progress Notes
- Archived the tracked pre-umbrella Cellerator tree under
  `Archive/current-cellerator/`.
- Added nested component submodules for Baseplane and CellShard.
- Added the first umbrella CMake root with default foundation targets and
  source-only archive exposure behind `CELLERATOR_ENABLE_ARCHIVE=ON`.
- Validation passed:
  `cmake -S Cellerator -B Cellerator/build`
- Validation passed:
  `cmake --build Cellerator/build -j 4`
- Validation passed:
  `cmake --build Cellerator/build --target cellshard_inspect cellshard_runtime celleratorFoundationRuntimeTest -j 4`
- Validation passed:
  `./Cellerator/build/components/Baseplane/baseplaneDna2Test`
- Validation passed:
  `./Cellerator/build/celleratorFoundationRuntimeTest`
- Validation passed:
  `cmake -S Cellerator -B Cellerator/build-archive -DCELLERATOR_ENABLE_ARCHIVE=ON`
- CellStack wrapper validation passed: `git submodule status` lists only
  `Cellerator`.

## Next Actions
_None._

## Done Criteria
- `cmake -S Cellerator -B Cellerator/build` succeeds from the deprecated
  CellStack wrapper.
- Default build does not build old archived Cellerator implementation targets.
- Focused Baseplane, CellShard, and Cellerator foundation targets build.
- CellStack root tracks only `Cellerator/`, minimal README/AGENTS, and
  `.gitmodules`.
