---
slug: "cellpack-bp08-warp-tiles"
status: "planned"
execution: "ready"
owner: "unassigned"
created_at: "2026-08-14T13:00:00Z"
last_heartbeat_at: "2026-08-17T08:17:44Z"
last_reviewed_at: "2026-08-17T08:17:44Z"
stale_after_days: 7
objective: "CP-BP-08: Build compact 32-cell warp tiles with shared block dictionaries, cell masks, gene masks, and exact payload offsets."
---

# Current Objective

## Summary

Combine up to 32 genes per global block with 32 locally ordered cells per tile, sharing each global block ID once and representing participating cells with `uint32_t cell_mask`.

## Quick Start

- Why this stream exists: create the distinctive logical sparse 32-gene by 32-cell tile without storing zeros.
- In scope: `build_warp_tiles`, tile block union, ballot/popcount occupancy, `WarpBlock`-like descriptors, `emit_tile_payloads`, complete offsets, decode, and metadata/storage metrics.
- Out of scope / dependencies: generic BELL conversion, padded dense tiles, final runtime dispatch, persistence publication, and unsupported access semantics.
- Required skills: `cuda`, `todo-orchestrator`.
- Required references: CP-BP-00, CP-BP-06, CP-BP-07, `components/CellPack/AGENTS.md`, `style_hint.md`, and `optimization.md`.

## Planning Notes

- Logical masks may use `uint32_t gene_mask` and `uint32_t cell_mask`, but values remain compact. Tail tiles and narrower configured gene blocks require explicit validity semantics.
- Ballot/popcount are natural tools, not mandatory proof of speed; library/custom choices need correctness and benchmark evidence.
- Tile identity must bind the exact feature-block geometry and the CP-BP-07 row
  permutation configuration; dataset identity or schema version alone is not
  sufficient for safe decode.

## CP-BP-06→11 Fork Interlock

- Read `todos/cellpack-bp06-11-parallel-execution.md`. Do not claim until
  CP-BP-06 is closed and both CP-BP-07 handoff gates are recorded.
- Claim Phase C under `/tmp/cellerator-cp-bp06-11-shared.lock`, use
  `build-cp-bp08`, and stop/release at `CP08_HOST_ABI_READY`. Resume CUDA Phase
  D only after Barrier C; publish `CP08_DEVICE_READY` and close without git
  operations.
- Consume CP-BP-06/07 contracts read-only. Do not implement CP-BP-09 runtime or
  CP-BP-13 persistence.

## File Lease

_Ready and unclaimed for Phase C host ABI/reference only._ On assignment,
atomically lease exactly:

- new `components/CellPack/include/CellPack/warp_tiles.hh`;
- new `components/CellPack/src/warp_tiles.cc`;
- new `components/CellPack/tests/warp_tiles_test.cc`;
- only clearly labelled CP-BP-08 target blocks in
  `components/CellPack/CMakeLists.txt`;
- this ledger and CP-BP-08 entries in the coordinator, `todos.md`,
  `todo-status.md`, and parent roadmap while holding the shared lock.

Record the unique owner and full current pushed `origin/main` claim hash under
the shared lock before editing. `cell_block_records.*`, `local_cell_ordering.*`,
`packing_plan.*`, `apply_plan.*`, all statistical-validation files, and root
`CMakeLists.txt` are read-only. CUDA Phase D remains a separate later claim
after `BARRIER_C_INTEGRATED`.

## Assumptions

- Each tile has at most 32 rows after local ordering and a sorted union of global gene blocks.
- Rank rules define exactly where each cell/block's compact values begin.

## Suggested Skills

- `cuda`
- `todo-orchestrator`

## Useful Reference Files

- `components/CellPack/include/CellPack/format.hh`
- `components/CellPack/include/CellPack/pack.hh`
- `bench/benchmark_mutex.hh`

## Plan

1. Phase C: specify versionable pointer-first tile descriptor, dictionary,
   mask, identity, payload, capacity, and checked offset/rank contracts.
2. Phase C: build deterministic CPU/reference construction, validator, exact
   canonical decoder, and adversarial tests; publish `CP08_HOST_ABI_READY`,
   release, and stop.
3. After Barrier C only, Phase D: implement GPU union/mask/payload emission with
   explicit caller stream/scratch ownership.
4. Phase D: compare construction throughput and storage/metadata with relevant
   current Cellerator layouts, then publish `CP08_DEVICE_READY`.

## Tasks

- [x] Wait for CP-BP-06 record and CP-BP-07 ordering contracts.
- [ ] Define complete warp-tile logical ABI.
- [ ] Implement reference/GPU construction and decode.
- [ ] Measure bytes/NNZ, metadata/NNZ, union size, and build throughput.

## Blockers

- No blocker for Phase C host ABI/reference: Barrier B source checkpoint
  `eeb8c39` closes CP-BP-06/07 with stable record and local-order contracts.
- CUDA tile construction remains blocked until `CP08_HOST_ABI_READY` is
  integrated at Barrier C.

## Progress Notes

- 2026-08-17: Phase C's exact unclaimed lease and acceptance boundary are
  frozen against the current pushed coordinator base. The host contract must be pointer-first and
  device-ready; represent at most 32 ordered rows, a sorted unique feature-block
  dictionary, block `cell_mask`, participating row/block `gene_mask`, compact
  real value bytes, checked rank/offsets/capacities, and exact
  plan/order/row-domain identity. Phase C must not add CUDA files or consume
  CP-BP-09/11 scope.
- 2026-08-17: Reactivated as `planned/ready` for Phase C after
  `BARRIER_B_INTEGRATED` pushed exact CP-BP-06 records and CP-BP-07 bounded
  order maps in source checkpoint `eeb8c39`. This phase is host ABI/reference,
  exact decoder, adversarial tests, and identity propagation only; it must not
  start CUDA construction, runtime consumers, or persistence.
- 2026-08-14: Added as a missing blocked workstream; existing CellPack coordinate spans and layout labels do not implement this representation.

## Next Actions

- If explicitly assigned, claim only the recorded Phase C lease under the
  shared lock. Define the pointer-first tile dictionary/mask/payload/rank ABI,
  CPU builder/validator/decoder, exact identity propagation, and tests for
  empty/tail rows, shared/disjoint blocks, full/sparse masks including bit 31,
  arbitrary value bytes/widths, deterministic rebuild, overflow, and tampering.
  Validate from `build-cp-bp08` with CP-BP-06/07 and plan/evaluator regressions;
  publish `CP08_HOST_ABI_READY`, release to idle, and stop without git for
  Barrier C. Do not begin merely because this setup is committed; wait for user
  assignment.

## Done Criteria

- Tile dictionaries, cell/gene masks, compact payloads, and all offsets decode exactly to canonical data.
- Tail cells, empty tiles, full/empty masks, repeated blocks, multiple value widths, offset overflow, and deterministic construction are covered.
- No zeros are materialized as BELL/Blocked-ELL padding.
- Benchmarks report construction throughput and fair metadata/storage deltas against relevant Cellerator layouts.
