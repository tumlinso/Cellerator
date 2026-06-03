# Active Objectives

## Summary
Root coordination ledger for the Cellerator umbrella workspace. Cellerator now
owns the cross-component checkout that previously lived in CellStack.

## Shared Assumptions
- `Cellerator.git` is the umbrella repository for the core genomics GPU stack.
- `components/Baseplane/` owns sequence primitives.
- `components/CellShard/` owns storage, metadata, pack generation, and runtime
  delivery.
- `include/Cellerator/` and `src/foundation/` own only the foundation slice
  needed by components.
- `Archive/current-cellerator/` preserves the previous implementation for later
  subproject classification and is not a default build surface.
- The deprecated CellStack root should remain a wrapper with only the
  `Cellerator/` submodule.

## Suggested Skills
- `todo-orchestrator`: maintain this umbrella ledger for substantial
  cross-component work.
- `cuda`: use for CUDA-sensitive foundation, Baseplane, or CellShard runtime
  work.

## Useful Reference Files
- `AGENTS.md`: umbrella ownership, component boundaries, and git hygiene.
- `components/Baseplane/AGENTS.md`: Baseplane-local implementation rules.
- `components/CellShard/AGENTS.md`: CellShard-local implementation rules.
- `Archive/current-cellerator/`: archived implementation material.

## Workstreams
- `cellerator-umbrella-restructure` | status: done | owner: codex | file:
  `todos/cellerator-umbrella-restructure.md` | objective: Make Cellerator the
  umbrella workspace and deprecate CellStack to a one-submodule wrapper.

## Global Blockers
_None recorded yet._

## Progress Notes
- Completed the umbrella migration from the user-approved restructuring plan.
- Default Cellerator configure/build passed from the deprecated CellStack
  wrapper.
- Focused Baseplane, CellShard, Cellerator foundation, and archive opt-in
  validation passed.

## Next Actions
_None._

## Done Criteria
- Cellerator contains Baseplane and CellShard under `components/`.
- The previous Cellerator implementation is preserved under
  `Archive/current-cellerator/`.
- Default CMake builds only the umbrella foundation and components.
- CellStack is reduced to a thin `Cellerator/` wrapper.
