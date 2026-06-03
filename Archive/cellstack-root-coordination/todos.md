# Active Objectives

## Summary
Root coordination ledger for CellStack cross-repo work. Submodule implementation
details stay in the owning submodule ledgers.

## Shared Assumptions
- CellStack root owns workspace layout, submodule pointers, cross-repo
  coordination metadata, and helper scripts.
- Implementation work belongs inside the owning submodule; root work should not
  hide uncommitted submodule changes.
- Before a root pointer update, check both root status and the affected
  submodule status.

## Suggested Skills
- `todo-orchestrator`: maintain this root coordination ledger for substantial
  cross-repo work.

## Useful Reference Files
- `AGENTS.md`: root scope, submodule ownership boundaries, and git hygiene.
- `Cellerator/todos.md`: current Cellerator-owned active workstream index.
- `Cellerator/todo-status.md`: current Cellerator pickup register.

## Workstreams
- `cellerator-sparse-ml-layout-root-coordination` | status: planned | owner: unassigned | file: `todos/cellerator-sparse-ml-layout-root-coordination.md` | objective: Coordinate any CellStack root follow-up for the Cellerator sparse ML layout workstream.
- `baseplane-sequence-migration` | status: done | owner: codex | file: `todos/baseplane-sequence-migration.md` | objective: Set up Baseplane as the sequence bit primitive owner and hard-cut Cellerator sequence ownership.
- `cellerator-preprocess-rehome-root-coordination` | status: done | owner: codex | file: `todos/cellerator-preprocess-rehome-root-coordination.md` | objective: Remove the root CellShardPreprocess submodule entry and update root docs for the Cellerator preprocessing rehome.

## Global Blockers
_None recorded yet._

## Progress Notes
- Initialized the root CellStack ledger after finding active Cellerator submodule
  todo state and no prior root-level `todos.md`.
- Completed root coordination for the Cellerator preprocessing rehome: removed
  the `CellShardPreprocess` submodule mapping and updated root docs to describe
  Cellerator as the preprocessing owner.
- Started `baseplane-sequence-migration`: Baseplane owns the migrated `dna2`
  packed-word, bitplane, CPU, and CUDA sequence primitives; Cellerator will
  consume Baseplane and drop its local sequence target and headers.
- Finished `baseplane-sequence-migration`: Baseplane exports `Baseplane::seq`
  with CUDA-preferred and CPU-only validation, Cellerator consumes sibling
  Baseplane, and Cellerator no longer owns sequence bit source, headers, tests,
  benches, docs, or CMake targets.

## Next Actions
- When Cellerator sparse ML layout work resumes or finishes, update the root
  coordination workstream only for root-owned actions such as submodule pointer
  updates, cross-repo notes, or validation summaries.
- No root coordination action remains for the Baseplane sequence migration.

## Done Criteria
- Every active workstream in `todos/` is reflected here with a current status.

