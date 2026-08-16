---
slug: "cellpack-bp07-local-cell-ordering"
status: "blocked"
execution: "closed"
owner: "unassigned"
created_at: "2026-08-14T13:00:00Z"
last_heartbeat_at: "2026-08-14T13:00:00Z"
last_reviewed_at: "2026-08-16T19:45:16Z"
stale_after_days: 7
objective: "CP-BP-07: Infer cheap local cell ordering from active gene-block signatures for warp-friendly groups."
---

# Current Objective

## Summary

Represent each transformed cell by its sorted active-block set, compute a compact signature, and order only local chunks/windows so similar cells share warp/slice groups.

## Quick Start

- Why this stream exists: local row order should reduce tile block unions and metadata without globally optimizing millions of cells.
- In scope: `cell_block_minhash` or equivalent, active-block counts, deterministic local sort/grouping, configurable chunk sizes, permutation/inverse maps, and baseline metrics.
- Out of scope / dependencies: global cell clustering, manual cell labels, biological semantics, global dataset reorder, and gene-plan relearning.
- Required skills: `cuda`, `todo-orchestrator`.
- Required references: CP-BP-00, CP-BP-06, `components/CellPack/AGENTS.md`, and the completed PackingPlan evaluator's row-permutation conventions.

## Planning Notes

- Windows should remain partition/chunk oriented, plausibly hundreds to a few thousand rows, and preserve explicit reversible row identity.
- Compare against original/random order and simple row-NNZ sorting; neither is an intentionally weak sole baseline.
- CP-BP-07 emits bounded local permutation and inverse arrays. It does not
  physically rewrite CP-BP-06 record payloads; CP-BP-08 consumes the records
  through these maps.

## CP-BP-06→11 Fork Interlock

- Read `todos/cellpack-bp06-11-parallel-execution.md`. Do not claim before
  `CP06_HOST_ABI_READY` and Barrier A are recorded. If assigned too early,
  remain read-only and report the missing gate without inventing an ABI.
- Claim/lease under `/tmp/cellerator-cp-bp06-11-shared.lock`, use
  `build-cp-bp07`, consume CP-BP-06 files read-only, publish
  `CP07_ORDER_ABI_READY` and `CP07_DEVICE_READY`, then release/close without git
  operations.

## File Lease

_Blocked and unclaimed._ Record exact intended paths atomically after the gate.

## Assumptions

- CP-BP-06 provides sorted active block IDs per row without decoding values.
- Signature approximation chooses local order only; measured tile cost judges usefulness.

## Suggested Skills

- `cuda`
- `todo-orchestrator`

## Useful Reference Files

- `todos/cellpack-packing-plan-evaluator.md`
- `components/CellPack/include/CellPack/evaluator.hh`
- `components/CellPack/AGENTS.md`

## Plan

1. Define local-window, signature, and reversible row-order contracts.
2. Add deterministic reference ordering and baseline metrics.
3. Implement GPU signature/grouping with library sort primitives.
4. Measure block-union and metadata reduction over multiple chunk sizes.

## Tasks

- [!] Wait for CP-BP-06 active-block record/view contract.
- [ ] Implement signature and local grouping.
- [ ] Preserve row permutation and inverse maps.
- [ ] Benchmark against original/random and row-length ordering.

## Blockers

- Blocked on CP-BP-06's active-block set and offset semantics.

## Progress Notes

- 2026-08-14: Added as a missing blocked workstream; no implementation evidence was found.

## Next Actions

- Reactivate after CP-BP-06 exposes a stable active-block view per cell.

## Done Criteria

- Ordering is deterministic, local to explicit bounded chunks, and fully reversible.
- No global million-cell optimization or label dependency is introduced.
- Held fixtures report per-warp/tile block-union size and metadata cost against original/random and row-length order.
- CPU/reference and GPU signature/order semantics agree.
