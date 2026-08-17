---
slug: "cellpack-bp08-warp-tiles"
status: "in_progress"
execution: "idle"
owner: "unassigned"
created_at: "2026-08-14T13:00:00Z"
last_heartbeat_at: "2026-08-17T08:39:21Z"
last_reviewed_at: "2026-08-17T08:39:21Z"
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

_Released at `CP08_HOST_ABI_READY` by `codex-cp-bp08-phase-c` on 2026-08-17._
The completed Phase C lease was exactly:

- new `components/CellPack/include/CellPack/warp_tiles.hh`;
- new `components/CellPack/src/warp_tiles.cc`;
- new `components/CellPack/tests/warp_tiles_test.cc`;
- only clearly labelled CP-BP-08 target blocks in
  `components/CellPack/CMakeLists.txt`;
- this ledger and CP-BP-08 entries in the coordinator, `todos.md`,
  `todo-status.md`, and parent roadmap while holding the shared lock.

No implementation lease is active. `cell_block_records.*`,
`local_cell_ordering.*`, `packing_plan.*`, `apply_plan.*`, all
statistical-validation files, and root `CMakeLists.txt` remain read-only. CUDA
Phase D is a separate later claim after `BARRIER_C_INTEGRATED`.

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
- [x] Define complete versioned pointer-first warp-tile logical ABI.
- [x] Implement deterministic host construction, exact validation/decode, and
  metadata/storage count metrics.
- [ ] Implement CUDA construction and measure bytes/NNZ, metadata/NNZ, union
  size, and build throughput in Phase D.

## Blockers

- Phase C host ABI/reference is complete at `CP08_HOST_ABI_READY`.
- CUDA tile construction remains blocked until the host gate is integrated at
  Barrier C.

## Progress Notes

- 2026-08-17: Published `CP08_HOST_ABI_READY`, released every Phase C lease,
  and returned idle without git operations. Added a versioned trivially-copyable
  pointer/count ABI over caller-owned tile dictionary, cell masks, compact
  row-block gene masks, terminal rank/value offsets, and arbitrary value bytes.
  Tile identity binds exact feature-block geometry, CP-BP-07 ordering, partition
  row domain, feature axis, tile width, and value width.
- 2026-08-17: Deterministic allocation-free host query/build, exact source-bound
  validator, canonical-row decoder, and metadata/storage metrics cover empty
  partitions/tiles, 32-lane and tail tiles, shared/disjoint blocks, full/empty
  cell masks, gene/cell bit 31, non-identity row order, 1/3/8-byte values,
  insufficient capacities, offset/identity/mask/value tampering, and repeat
  determinism. No zeros are materialized.
- 2026-08-17: Fresh `build-cp-bp08` with CUDA 12.9.86, GNU 13.3.0, and
  `sm_70` built and passed `cellPackWarpTilesTest`,
  `cellPackCellBlockRecordsTest`, `cellPackReconstructionTest`,
  `cellPackLocalCellOrderingTest`, `cellPackPlannerTest`,
  `cellPackEvaluatorTest`, `cellPackOptimizerTest`, and
  `cellPackInferredPackingPipelineTest`. The existing CP-BP-07 CPU/CUDA test
  ran under the shared GPU lock; `git diff --check` passed.
- 2026-08-17: `codex-cp-bp08-phase-c` claimed the exact Phase C lease at
  pushed base `3925c155de1dab89dd506dd229c97acb96de27a7`. CP-BP-11 remains
  independently idle/unassigned with its disjoint new-file and root-CMake
  lease. This claim owns only host `warp_tiles` ABI/reference/test files,
  CP-BP-08 component-CMake blocks, and locked coordination entries; it will
  publish `CP08_HOST_ABI_READY`, release, and stop without git operations.
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

- Await the appointed Barrier C integrator. Do not claim CUDA Phase D or begin
  CP-BP-09 until the combined CP-BP-08/11 tree is freshly validated, committed,
  pushed, and `BARRIER_C_INTEGRATED` is recorded.

## Done Criteria

- Tile dictionaries, cell/gene masks, compact payloads, and all offsets decode exactly to canonical data.
- Tail cells, empty tiles, full/empty masks, repeated blocks, multiple value widths, offset overflow, and deterministic construction are covered.
- No zeros are materialized as BELL/Blocked-ELL padding.
- Benchmarks report construction throughput and fair metadata/storage deltas against relevant Cellerator layouts.
