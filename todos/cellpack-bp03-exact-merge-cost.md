---
slug: "cellpack-bp03-exact-merge-cost"
status: "planned"
execution: "ready"
owner: "unassigned"
created_at: "2026-08-14T13:00:00Z"
last_heartbeat_at: "2026-08-16T14:57:37Z"
last_reviewed_at: "2026-08-16T14:57:37Z"
stale_after_days: 3
objective: "CP-BP-03: Score candidate merges with exact structural overlap and a replaceable target-layout cost model."
---

# Current Objective

## Summary

Implement exact wordwise support overlap and `estimate_merge_gain`/layout-cost contracts so approximate similarity never decides the final packing.

## Quick Start

- Why this stream exists: candidate similarity is only a filter; exact target-representation cost must decide merges.
- In scope: CPU/reference cost, `score_gene_merges`, bitset AND/OR/popcount, overflow-safe accounting, and a replaceable policy covering identifiers, masks, alignment, padding, values, offsets, and metadata.
- Out of scope / dependencies: candidate discovery, global merge scheduling, hard-coded Jaccard optimization, and final hardware model calibration.
- Required skills: `cuda`, `todo-orchestrator`.
- Required references: CP-BP-00, CP-BP-01 support contract, CP-BP-02 candidate contract, completed `cellpack-packing-plan-evaluator`, and CellPack layout-metrics sources.

## Planning Notes

- The completed evaluator already separates exact occupancy from hypothetical cost policy; reuse or generalize that seam instead of creating a conflicting evaluator.
- Initial storage policy may be provisional, but every assumed byte/offset/alignment term must be explicit. CP-BP-12 later supplies hardware execution cost through the same replaceable interface.
- CP-BP-04 planning makes sampled active-block-reference reduction the optimizer-owned structural proxy. CP-BP-03 exact pair gain remains valuable candidate evidence and a same-policy tie/filter term, but it must not be reused as though a singleton-pair score were an exact score for a later merged block pair.

## Assumptions

- A candidate pair references canonical genes or immutable block handles; canonical identity remains recoverable.
- Integer accounting detects overflow and matches a CPU reference exactly.

## Single-Worktree Interlock

- If assigned CP-BP-03, first acquire the shared lock named in `todos.md`,
  reread the CP-BP-03 and CP-BP-05 ledgers, change only this stream to
  `in_progress/claimed` with a unique owner, and list every intended path under
  `File Lease` before source edits. Synchronize both pickup registers before
  releasing the lock.
- Own new exact-cost policy/scorer files, tests, and benchmarks. Consume
  CP-BP-01 support, CP-BP-02 `gene_candidate_pair_view`, and the completed
  evaluator seam. Do not edit sampling, support extraction, candidate
  discovery, `optimizer.*`, `optimizer_state.hh`, `packing_plan.*`, or any
  CP-BP-05 remap/order file.
- `evaluator.*`, `layout_metrics.*`, `candidate_relation.*`, and either CMake
  file are shared seams: edit one only after leasing its exact path under the
  lock. CP-BP-03 emits exact scored candidate evidence; it must not schedule
  global optimization, apply a frozen plan, or define physical records.
- Build only in `build-cp-bp03`. Do not perform git state-changing operations;
  the final integrator owns combined validation and commit/push after both
  streams release their claims.

## File Lease

- Unclaimed. The assigned fork must replace this line with exact paths before
  editing source.

## Suggested Skills

- `cuda`
- `todo-orchestrator`

## Useful Reference Files

- `todos/cellpack-packing-plan-evaluator.md`
- `components/CellPack/include/CellPack/evaluator.hh`
- `components/CellPack/include/CellPack/layout_metrics.hh`
- `components/CellPack/src/evaluator.cc`
- `include/Cellerator/compute/gene_candidate_discovery.hh`

## Plan

1. Define exact merge-cost inputs/outputs and a versioned/replaceable cost policy.
2. Implement CPU/reference support union and byte accounting.
3. Implement GPU candidate scoring with wordwise OR/AND/popcount where beneficial.
4. Cross-check all scores and gains exactly, including unprofitable and overflow cases.

## Tasks

- [ ] Reconcile the new policy with the completed evaluator cost seam.
- [ ] Implement CPU/reference `cost` and `merge_gain`.
- [ ] Implement and test GPU candidate scoring.
- [ ] Document provisional versus physically validated cost terms.

## Blockers

- No candidate-input blocker remains: CP-BP-01 support and CP-BP-02's immutable
  canonical `gene_candidate_pair_view` are complete in the current worktree.
- Final codec-byte terms remain provisional until CP-BP-06/08 settle the physical record and tile ABI.

## Progress Notes

- 2026-08-14: Added as a missing ready workstream and linked to the completed exact plan evaluator; no merge scorer implementation was found.
- 2026-08-16: Reconciliation found reusable but non-superseding pieces:
  CP-BP-04 normalizes `candidate_relation`, computes exact sampled-support
  intersection proxies for its private mutable blocks, and checkpoints through
  the exact whole-plan evaluator. No public CPU/GPU pair/block codec-cost or
  `merge_gain` scorer satisfying this ledger exists. Status remains
  `planned/ready`, unassigned.
- 2026-08-16: Recorded the conditional one-worktree ownership and shared-seam
  lease rules for a future CP-BP-03 fork; no claim or implementation began.

## Next Actions

- Start with the evaluator-compatible CPU policy, consume
  `gene_candidate_pair_view`, and emit exact `candidate_relation` evidence for
  the completed optimizer without treating its private support proxy as the
  codec-cost API. Keep physical-format assumptions explicit and replaceable.

## Done Criteria

- CPU and GPU score every candidate identically for the supported integer cost contract.
- `merge_gain = cost(A) + cost(B) - cost(A union B)` is verified on empty, identical, disjoint, overlapping, tail-word, maximum-width, unprofitable, and overflow cases.
- Cost terms cover or explicitly defer every required metadata/value/offset/alignment component.
- No acceptance claim treats Jaccard or sketch similarity as the final objective.
