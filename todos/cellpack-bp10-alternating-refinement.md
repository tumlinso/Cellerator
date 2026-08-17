---
slug: "cellpack-bp10-alternating-refinement"
status: "done"
execution: "closed"
owner: "codex-cp-bp10-11-serial"
created_at: "2026-08-14T13:00:00Z"
last_heartbeat_at: "2026-08-17T14:17:23Z"
last_reviewed_at: "2026-08-17T14:17:23Z"
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
- V1 may optimize storage plus the measured CP-BP-09 runtime surface. Predicted
  CP-BP-12 hardware terms remain a later optional extension and do not justify
  fabricating a model before CP-BP-12 is ready.

## CP-BP-06→11 Fork Interlock

- Read `todos/cellpack-bp06-11-parallel-execution.md`. Do not claim until
  `CP10_READY` is published. If assigned earlier, remain read-only and report
  the exact missing gate.
- Claim under `/tmp/cellerator-cp-bp06-11-shared.lock`, use
  `build-cp-bp10-11-serial`, consume CP-BP-04/07/08/09/11 public APIs, and prefer new
  alternating-controller files. Editing optimizer or representation internals
  requires a transferred shared-seam lease.
- Implement only offline bounded acceptance/rollback; no minibatch repacking,
  no neural optimizer, no train-only acceptance, and no CP-BP-12/13 scope. Do
  not perform git operations.

## File Lease

Claimed together with CP-BP-11 Phase F by `codex-cp-bp10-11-serial` from pushed
base `3fc28c4e7e6e780edd7207fd7b87895797cc0ab3`. The combined serial lease is:

- new `components/CellPack/include/CellPack/alternating_refinement.hh`;
- new `components/CellPack/src/alternating_refinement.cc`;
- new `components/CellPack/tests/alternating_refinement_test.cc`;
- only the labelled `CP-BP-10 Phase F target insertion point` block in root
  `CMakeLists.txt`;
- CP-BP-10 entries in this ledger, the coordinator, parent roadmap,
  `todos.md`, and `todo-status.md` while holding the shared lock.

and, after CP-BP-10 is validated:

- new `components/CellPack/include/CellPack/runtime_statistical_validation.hh`;
- new `components/CellPack/src/runtime_statistical_validation.cc`;
- new `components/CellPack/tests/runtime_statistical_validation_test.cc`;
- new `components/CellPack/bench/runtime_statistical_validation_bench.cu`;
- only the labelled `CP-BP-11 Phase F target insertion point` block in root
  `CMakeLists.txt`;
- CP-BP-11 entries in its ledger and the shared coordination ledgers.

Existing optimizer, plan, record, order,
tile, runtime, and statistical-validation implementations are frozen read-only
inputs. A demonstrated frozen-input defect must be recorded and the stream must
stop; it does not authorize crossing the lease.

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

- [x] Wait for working plan, tile, consumer, and held-out cost surfaces.
- [x] Implement bounded alternating controller and best-plan rollback.
- [x] Validate deterministic convergence and held-out improvement.
- [x] Record preprocessing cost versus resulting runtime/storage benefit.

## Blockers

- No v1 implementation blocker remains. `CP10_READY` is published from
  `BARRIER_E_INTEGRATED`; CP-BP-04/07/08/09 and the required CP-BP-11 held-out
  surfaces are available in pushed source.
- CP-BP-12 hardware prediction remains an optional later extension and is not
  part of Phase F acceptance.

## Progress Notes

- 2026-08-17: Closed at `BARRIER_F_INTEGRATED`; the pushed implementation
  checkpoint is `2cfa5c8d26f0c973dfef4659d72ea5f635201835`. Combined host,
  downstream CUDA, sanitizer, benchmark, TODO, staleness, and diff gates passed.
- 2026-08-17: Released the combined Phase F implementation lease after focused
  and combined validation. `CP10_REFINEMENT_READY` remains published; the
  appointed serial owner now acts only as Barrier F integrator.
- 2026-08-17: `CP10_REFINEMENT_READY` implementation gate passed in
  `build-cp-bp10-11-serial`. The versioned pointer-first host controller binds
  plan/domain/split/metric identities, requires exact correctness evidence,
  preserves objective denominators, alternates gene and cell phases, accepts
  only held-out improvement beyond tolerance, and restores the best plan after
  regression or evaluation error. Focused tests cover reproducibility, strict
  tolerance, training-only improvement rejection, caps, empty input, tampering,
  and best-plan restoration. CP-BP-11 serial implementation may now begin.
- 2026-08-17: `BARRIER_E_INTEGRATED` at source checkpoint
  `0334f954b1b9e04366f2e2ce191e098c1d476597` published `CP10_READY`.
  Phase F is unclaimed and fork-ready under the exact lease above. It may run
  in parallel with CP-BP-11 Phase F; both stop at distinct gates without git.
- 2026-08-17: Serial combined owner `codex-cp-bp10-11-serial` claimed CP-BP-10
  then CP-BP-11 from pushed coordinator `3fc28c4`. CP-BP-10 must pass before
  CP-BP-11 consumes its actual relearned plans. This owner is also the appointed
  Barrier F integrator and may commit/push only after joint acceptance.
- 2026-08-14: Added as a missing blocked workstream; no implementation evidence was found.
- 2026-08-16: Reconciliation found no alternating controller, held-out
  acceptance loop, or tile-cost refinement. CP-BP-04's bounded feature
  move/swap search is the completed first-pass optimizer, not this downstream
  alternating gene/cell refinement.

## Next Actions

- Complete and closed. CP-BP-12 may consume the measured objective/controller
  seam but must not reopen CP-BP-10 or move hardware policy into this v1 API.

## Done Criteria

- Fixed inputs/configuration reproduce accepted iterations and final cost.
- Each accepted iteration improves the configured held-out objective beyond tolerance; regressions roll back.
- Iteration/time caps terminate reliably and report train versus held-out metrics.
- No ordinary preprocessing/minibatch path silently relearns the plan.
