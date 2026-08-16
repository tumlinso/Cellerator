---
slug: "cellpack-bp05-apply-frozen-plan"
status: "done"
execution: "closed"
owner: "codex-cp-bp-05-fork"
created_at: "2026-08-14T13:00:00Z"
last_heartbeat_at: "2026-08-16T15:34:00Z"
last_reviewed_at: "2026-08-16T15:34:00Z"
stale_after_days: 7
objective: "CP-BP-05: Apply a frozen PackingPlan to arbitrary/full sparse partitions in packed gene-coordinate order."
---

# Current Objective

## Summary

Map canonical gene IDs to stable `(block_id, local_gene)` coordinates and reorder each row without relearning the global plan.

## Quick Start

- Why this stream exists: ordinary preprocessing must cheaply apply a compiled plan to full partitions.
- In scope: `remap_gene_indices`, segmented CUB/radix sorting, explicit canonical identity metadata, pointer-first scratch, and exact round-trip tests.
- Out of scope / dependencies: plan inference, per-minibatch repacking, compact block-run emission, custom short-row sort before evidence, and CellShard persistence.
- Required skills: `cuda`, `todo-orchestrator`.
- Required references: CP-BP-00, CP-BP-04, `AGENTS.md`, `style_hint.md`, and existing Cellerator matrix-conversion contracts.

## Planning Notes

- Prefer established segmented radix/sort primitives for the first correct path. Specialized row sorts are a measured future optimization.
- Transformation must accept a frozen versioned plan and preserve inverse/canonical identity metadata.
- CP-BP-04 planning requires the frozen semantic object to expose canonical/execution maps, feature block/local lookup, boundaries, feature-axis fingerprint, row-domain identity, and sample-versus-full evaluation scope. CP-BP-05 must reject incompatible feature or row domains rather than silently project a sampled plan.

## Assumptions

- Input is canonical sparse row data with validated gene IDs.
- Output remains an intermediate ordered sparse representation until CP-BP-06 emits the compact block records.

## Single-Worktree Interlock

- If assigned CP-BP-05, first acquire the shared lock named in `todos.md`,
  reread the CP-BP-03 and CP-BP-05 ledgers, change only this stream to
  `in_progress/claimed` with a unique owner, and list every intended path under
  `File Lease` before source edits. Synchronize both pickup registers before
  releasing the lock.
- Own new frozen-plan application, canonical-remap, segmented-ordering files,
  tests, and benchmarks. Consume `frozen_packing_plan` read-only. Do not edit
  candidate discovery/scoring, `evaluator.*` cost policy, `optimizer.*`, or
  `optimizer_state.hh`; do not implement CP-BP-06 records, tiles, runtime
  kernels, or persistence.
- `packing_plan.*`, `pack.*`, and either CMake file are shared seams: prefer new
  CP-BP-05-specific files and edit a shared seam only after leasing its exact
  path under the lock. If the frozen plan is insufficient, record the concrete
  compatibility blocker instead of expanding CP-BP-04 silently.
- Build only in `build-cp-bp05`. Do not perform git state-changing operations;
  the final integrator owns combined validation and commit/push after both
  streams release their claims.

## File Lease

- Released for final integration on 2026-08-16. The completed `apply_plan`
  files remain CP-BP-05-owned and must not be changed without a new claim.
- Historical implementation lease: `CMakeLists.txt`,
  `components/CellPack/include/CellPack/apply_plan.hh`,
  `components/CellPack/src/apply_plan.cc`,
  `components/CellPack/src/apply_plan_cuda.cu`,
  `components/CellPack/tests/apply_plan_test.cu`, and
  `components/CellPack/bench/cellpack_apply_plan_bench.cu`.
- The coordination-only lease on
  `todos/cellpack-bp06-cell-block-records.md` is released by this handoff.

## Suggested Skills

- `cuda`
- `todo-orchestrator`

## Useful Reference Files

- `components/CellPack/include/CellPack/planner.hh`
- `components/CellPack/include/CellPack/pack.hh`
- `src/compute/matrix/`

## Plan

1. Freeze CP-BP-04 plan-view and mapping/inverse contracts.
2. Define explicit input/output, residency, stream, scratch, and overflow behavior.
3. Implement remap plus segmented ordering with library primitives.
4. Round-trip to canonical coordinates across arbitrary partition boundaries.

## Tasks

- [x] Confirm CP-BP-04's stable semantic `frozen_packing_plan` mapping/version/compatibility contract.
- [x] Implement remap and segmented packed-coordinate ordering.
- [x] Add exact reconstruction, determinism, and memory-bound tests.
- [x] Compare library path before considering specialized sorts.

## Blockers

_None; this stream is complete and closed._ Production candidate quality is
upstream of optimization and does not block applying an already-frozen plan.

## Progress Notes

- 2026-08-16: Completed pointer-first host and CUDA application contracts for
  arbitrary contiguous partitions. The implementation validates full-domain
  plan compatibility, maps every canonical feature to `(block_id, local)`,
  preserves row order and exact value bytes, and emits explicit canonical IDs
  alongside packed coordinates. Sample-scoped plans are rejected rather than
  silently projected onto the full dataset.
- 2026-08-16: The regular CUDA path uses small map/gather kernels plus CUB
  segmented radix sort with caller-owned device maps, output, scratch, stream,
  and no internal synchronization. CPU and CUDA outputs match exactly for
  empty, short, long, partial-partition, invalid-domain, invalid-CSR,
  insufficient-buffer, and index-overflow cases; compute-sanitizer memcheck
  reported zero errors.
- 2026-08-16: Serialized Tesla V100 `sm_70` benchmark at 65,536 rows, 30,000
  features, 2,097,152 nonzeros, and width 16 measured 34.461 ms CUDA minimum
  and 35.592 ms mean versus 51.442 ms CPU. CUB scratch was 25,166,079 bytes;
  transfers were excluded and the API performed no synchronization. A custom
  short-row sort remains deferred because the library path is already correct
  and no evidence yet justifies replacing it.
- 2026-08-16: Claimed by `codex-cp-bp-05-fork` under the single-worktree
  interlock. CP-BP-03 owns `components/CellPack/CMakeLists.txt` and its
  merge-cost files; CP-BP-05 will integrate only through the root CMake file
  and the exact new paths in `File Lease`.
- 2026-08-14: Added as a missing blocked workstream; no implementation evidence was found.
- 2026-08-14: Reactivated as `planned/ready` after CP-BP-04 implemented `frozen_packing_plan`, lifetime-bound evaluator view, canonical feature permutation/inverse, feature block/local lookups, fixed identity-row groups, schema/configuration fields, exact summaries, and feature/row compatibility validation. No CP-BP-05 implementation was performed.
- 2026-08-16: Reconciliation confirmed `frozen_packing_plan` has no consumer
  outside optimizer/tests and the older `build_packed_coordinate_plan()` accepts
  `static_plan`, not the inferred frozen plan. No remap/segmented-ordering API or
  test was found. CP-BP-05 remains genuinely unassigned/ready and is the primary
  continuation frontier.
- 2026-08-16: Recorded the conditional one-worktree ownership and shared-seam
  lease rules for a future CP-BP-05 fork; no claim or implementation began.

## Next Actions

- CP-BP-06 may consume `ordered_plan_partition_view` to define compact
  per-cell block records. It must not move record emission, tile construction,
  runtime kernels, or persistence back into CP-BP-05.

## Done Criteria

- Applying a frozen plan never invokes discovery or optimization.
- Every input nonzero maps once, rows are ordered by packed coordinate, and exact inverse reconstruction restores canonical row/gene/value tuples.
- Empty/short/long rows, invalid plan IDs, partition offsets, index-width overflow, and deterministic output are covered.
- Scratch/residency and synchronization behavior are explicit.
