---
workstream: cellerator-runtime-autotune
status: done
execution: closed
owner: codex
created: 2026-05-06
updated: 2026-05-06
---

# Cellerator Runtime Autotune

## Quick Start
- Scope: add an optional Cellerator runtime optimizer surface and wire the first lightweight provider into preprocessing.
- Required skills: `todo-orchestrator`, `cuda`, `bio-experiments`.
- Read first: `AGENTS.md`, `style_hint.md`, `optimization.md`, `include/Cellerator/compute/preprocess/preprocess.cuh`, `python/module.cc`, `python/cellerator/pp.py`.
- Do not move biological preprocessing policy out of Cellerator's preprocessing layer, and do not put reusable CUDA math in workflow/session files.

## Objective
Implement optional runtime autotuning so Python users can ask Cellerator to make close-enough performance choices automatically while C++ callers keep the default explicit path unless they opt in.

## Assumptions
- Autotune must stay bounded and may skip itself on tiny datasets where calibration overhead would dominate the real run.
- The first provider may focus on preprocessing execution plans and should preserve existing QC, normalization, and feature-stat semantics.
- The default path remains the fused preprocessing workflow unless the measured sample shows a better path.

## Tasks
- [x] Add a small optimizer options/result surface that can be reused by later Cellerator subsystems.
- [x] Add a preprocessing execution-plan switch for fused versus separate primitive workflows.
- [x] Expose Python `autotune=True` plus session metrics describing the selected plan and timing.
- [x] Keep C++ mode opt-in by exposing the plan vocabulary without changing default behavior.
- [x] Build and run focused validation.

## Blockers
_None recorded yet._

## Progress Notes
- Started implementation from the user request to keep autotune light, optional, and close-enough rather than exhaustive.
- Added `include/Cellerator/optimize/runtime_optimizer.hh` with reusable light optimizer options/results and a callback-based chooser for future subsystems.
- Added plan-aware preprocessing compute entry points for fused versus separate primitive workflows without changing the existing default API.
- Added Python `autotune=True` support and `metrics()["autotune"]` diagnostics; PBMC3K `.csh5` smoke selected the separate Sliced-ELL primitive path from a two-trial sample.
- Validation passed for focused preprocessing build targets, runtime/API tests, QC tests, and Python PBMC autotune smoke. The 64 GB generated `.csh5` fixture was intentionally not used as the Python smoke sample.

## Next Actions
_None; implementation complete._

## Done Criteria
- `cellerator.pp.preprocess(..., autotune=True)` runs a bounded calibration or cleanly skips it when overhead would be counterproductive.
- Default Python and C++ preprocessing behavior remains unchanged when autotune is disabled.
- Session metrics report whether autotune ran and which plan was selected.
- Focused build/test commands are recorded.
