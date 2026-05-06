# Cellerator Python Preprocess Runtime

Last updated: 2026-05-04

## Quick Start

Add a Cellerator-owned Python package that exposes Scanpy-like preprocessing
entry points while delegating actual work to CellShard/Cellerator native GPU
runtime paths. Keep AnnData as adapter metadata only; `.csh5`/CellShard-native
data is the execution substrate.

## Status

- Status: done
- Execution: closed
- Owner: codex

## Assumptions

- v1 target is scRNA-seq, observations by features, raw-count input.
- GPU execution is central: Python must not loop over cells/features or run
  hidden SciPy transforms for the preprocessing hot path.
- Blocked-ELL is preferred, Sliced-ELL is supported, CSR/compressed remains
  fallback-only for future work.
- `PreprocessSession.publish(...)` may delegate to CellShard finalize/publish
  helpers; CellShard remains the storage owner.

## Suggested Skills

- `bio-experiments`
- `cuda`
- `todo-orchestrator`

## Useful Reference Files

- `AGENTS.md`
- `scope.md`
- `include/Cellerator/preprocess/runtime.hh`
- `include/Cellerator/compute/preprocess/preprocess.cuh`
- `../CellShard/include/CellShard/io/csh5/api.cuh`
- `../CellShard/include/CellShard/runtime/device/sharded_device.cuh`

## Plan

- [x] Record the Python runtime workstream.
- [x] Add `pyproject.toml`, Python package files, and optional CMake Python
  extension target.
- [x] Add native pybind wrappers for preprocessing policy helpers and GPU
  dataset preprocessing sessions.
- [x] Add Python facade functions under `cellerator.pp`.
- [x] Add smoke tests for import, policy helpers, and optional dataset-session
  API.
- [x] Build and run focused validation.

## Progress Notes

- Started from the user-approved direction: all omics preprocessing compute
  should be GPU/native-runtime centered from this abstraction level.
- Added `cellerator` Python packaging, `_cellerator` pybind module,
  `cellerator.pp`, and a `PreprocessSession` that dispatches CellShard `.csh5`
  datasets through Cellerator GPU preprocessing kernels.
- Verified direct CMake extension build, Python smoke test, wheel build, and
  installed-wheel import smoke.

## Next Actions

- No immediate action remains. A future follow-up should add a real `.csh5`
  fixture-driven Python session test once a stable checked-in tiny dataset
  fixture exists.

## Done Criteria

- `python -m pip wheel ./Cellerator --no-deps` builds the Python package.
- `import cellerator; import cellerator.pp` succeeds from an installed wheel.
- Python can validate raw state, compile QC feature masks, plan CellShard
  staging, and run a GPU preprocessing session for a CellShard `.csh5` when a
  dataset path is supplied.
