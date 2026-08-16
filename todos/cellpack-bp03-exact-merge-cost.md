---
slug: "cellpack-bp03-exact-merge-cost"
status: "done"
execution: "closed"
owner: "codex-cp-bp-03-fork"
created_at: "2026-08-14T13:00:00Z"
last_heartbeat_at: "2026-08-16T15:27:39Z"
last_reviewed_at: "2026-08-16T15:27:39Z"
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

- Released for final integration on 2026-08-16. The completed `merge_cost` files
  remain CP-BP-03-owned and must not be modified by the active CP-BP-05 stream;
  the shared `components/CellPack/CMakeLists.txt` lease is released.

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

- [x] Reconcile the new policy with the completed evaluator cost seam.
- [x] Implement CPU/reference `cost` and `merge_gain`.
- [x] Implement and test GPU candidate scoring.
- [x] Document provisional versus physically validated cost terms.

## Blockers

_None; this stream is complete and closed._ Final codec calibration remains a
documented CP-BP-06/08/12 responsibility, not an acceptance blocker for the
versioned provisional policy.

## Progress Notes

- 2026-08-16: Completed a versioned, replaceable exact byte policy covering
  block metadata, canonical feature identifiers, block/active-row offsets,
  masks, compact-or-dense value slots, and three explicit alignment stages.
  Zero byte terms deliberately defer components; no policy is represented as a
  finalized CP-BP-06 physical ABI.
- 2026-08-16: Added exact CPU `estimate_exact_block_cost`,
  `estimate_merge_gain`, and `score_gene_merges_cpu`, plus a host-staged native
  V100 scorer. Both paths consume immutable CP-BP-01 support and CP-BP-02
  candidate provenance and emit optimizer-valid `exact_merge_gain`
  `candidate_relation` evidence without changing discovery or optimization.
- 2026-08-16: `cellPackMergeCostTest` passed CPU/CUDA exact agreement for
  empty/zero-row, identical, overlapping, disjoint, tail-word, maximum-width,
  unprofitable, deferred-term, invalid-provenance/count, determinism, and
  overflow cases. Adjacent evaluator, optimizer, and candidate-discovery tests
  also passed from `build-cp-bp03`.
- 2026-08-16: Mutex benchmark on Tesla V100 (`65,536` cells, `30,000` genes,
  `2,048` words/gene, `105,000` candidates) measured `308.250 ms` CPU and
  `77.924/78.895 ms` CUDA minimum/median including allocation, 245,760,000-byte
  support staging, kernel, and D2H output. All integer fields matched exactly;
  persistent device support remains an explicit future optimization.
- 2026-08-16: Claimed as `codex-cp-bp-03-fork` under the single-worktree
  interlock. Leased only the exact scorer files above and the CellPack CMake
  integration seam; CP-BP-05 remains independently ready and untouched.
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

- No CP-BP-03 implementation remains. CP-BP-12 may later replace/calibrate the
  policy after CP-BP-08/09 provide measurable physical consumers; singleton
  pair scores must still not be misused as exact later-block merge scores.

## Done Criteria

- CPU and GPU score every candidate identically for the supported integer cost contract.
- `merge_gain = cost(A) + cost(B) - cost(A union B)` is verified on empty, identical, disjoint, overlapping, tail-word, maximum-width, unprofitable, and overflow cases.
- Cost terms cover or explicitly defer every required metadata/value/offset/alignment component.
- No acceptance claim treats Jaccard or sketch similarity as the final objective.
