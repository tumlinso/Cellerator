---
slug: "cellpack-bp10-alternating-refinement"
status: "blocked"
execution: "closed"
owner: "unassigned"
created_at: "2026-08-14T13:00:00Z"
last_heartbeat_at: "2026-08-16T14:38:44Z"
last_reviewed_at: "2026-08-16T14:38:44Z"
stale_after_days: 7
objective: "CP-BP-10: Refine global gene blocks and local cell packing against measured held-out tile costs."
---

# Current Objective

## Summary

Alternate frozen gene packing and local cell/tile packing, accepting deterministic refinements only while held-out encoded/runtime cost meaningfully improves.

## Quick Start

- Why this stream exists: first-pass gene blocks can be improved using actual tile duplication and execution costs.
- In scope: gene packing -> cell packing -> gene refinement -> cell refinement, stopping criteria, held-out objective, reproducibility, and rollback to best plan.
- Out of scope / dependencies: endless training, neural optimizer, in-minibatch repacking, and train-only acceptance.
- Required skills: `todo-orchestrator`, `cuda`, `bio-experiments` for held-out design.
- Required references: CP-BP-00, CP-BP-04, CP-BP-07 through CP-BP-09, CP-BP-11, and CP-BP-12.

## Planning Notes

- The refined objective can include duplicated per-cell/block metadata, duplicated tile references/masks, and measured/predicted runtime cost.
- Preserve the best accepted plan and make iteration/seed/tolerance configuration reproducible.

## Assumptions

- Gene plan changes trigger explicit recompilation of affected packed data; they never occur in ordinary runtime batches.
- Held-out improvement, not training-set improvement alone, governs acceptance.

## Suggested Skills

- `todo-orchestrator`
- `cuda`
- `bio-experiments`

## Useful Reference Files

- `components/CellPack/AGENTS.md`
- `todos/cellpack-bp04-packing-plan-optimizer.md`
- `todos/cellpack-bp11-statistical-validation.md`

## Plan

1. Define iteration state, best-plan checkpoint, held-out objective, and deterministic stopping rules.
2. Connect CP-BP-04 refinement to CP-BP-07/08 rebuilt cell tiles.
3. Incorporate CP-BP-12 execution predictions after storage-only correctness is stable.
4. Test monotonic accepted cost, rollback, convergence cap, and reproducibility.

## Tasks

- [!] Wait for working plan, tile, consumer, and held-out cost surfaces.
- [ ] Implement bounded alternating controller and best-plan rollback.
- [ ] Validate deterministic convergence and held-out improvement.
- [ ] Record preprocessing cost versus resulting runtime/storage benefit.

## Blockers

- CP-BP-04 is complete. This remains blocked on CP-BP-07, CP-BP-08,
  CP-BP-09, and foundational CP-BP-11 metrics.
- Hardware-aware refinement terms additionally wait on CP-BP-12.

## Progress Notes

- 2026-08-14: Added as a missing blocked workstream; no implementation evidence was found.
- 2026-08-16: Reconciliation found no alternating controller, held-out
  acceptance loop, or tile-cost refinement. CP-BP-04's bounded feature
  move/swap search is the completed first-pass optimizer, not this downstream
  alternating gene/cell refinement.

## Next Actions

- Reactivate after a complete first-pass plan -> tile -> consumer loop is measurable on held-out cells.

## Done Criteria

- Fixed inputs/configuration reproduce accepted iterations and final cost.
- Each accepted iteration improves the configured held-out objective beyond tolerance; regressions roll back.
- Iteration/time caps terminate reliably and report train versus held-out metrics.
- No ordinary preprocessing/minibatch path silently relearns the plan.
