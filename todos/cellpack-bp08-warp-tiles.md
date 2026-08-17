---
slug: "cellpack-bp08-warp-tiles"
status: "done"
execution: "closed"
owner: "codex-cp-bp08-phase-d"
created_at: "2026-08-14T13:00:00Z"
last_heartbeat_at: "2026-08-17T10:14:49Z"
last_reviewed_at: "2026-08-17T10:14:49Z"
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
- Barrier C is integrated at source checkpoint `ebe0509`. On explicit Phase D
  assignment, claim the exact CUDA lease under the shared lock, use
  `build-cp-bp08`, publish `CP08_DEVICE_READY`, release, and stop without git.
- Consume CP-BP-06/07 contracts read-only. Do not implement CP-BP-09 runtime or
  CP-BP-13 persistence.

## File Lease

_Closed by Barrier D at pushed source checkpoint `0bf9acf` on 2026-08-17._
Released at `CP08_DEVICE_READY` by `codex-cp-bp08-phase-d` on 2026-08-17.
The completed Phase D lease at pushed base
`fe095fb6d6592a0194b0a86f13f0421e23081cd0` was exactly:

- new `components/CellPack/include/CellPack/warp_tiles_cuda.hh`;
- new `components/CellPack/src/warp_tiles_cuda.cu`;
- new `components/CellPack/tests/warp_tiles_cuda_test.cu`;
- new `components/CellPack/bench/warp_tiles_bench.cu`;
- only clearly labelled CP-BP-08 Phase D target blocks in
  `components/CellPack/CMakeLists.txt`;
- this ledger and CP-BP-08 entries in the coordinator, `todos.md`,
  `todo-status.md`, and parent roadmap while holding the shared lock.

No implementation lease is active. `warp_tiles.hh/.cc`,
`cell_block_records.*`, `local_cell_ordering.*`, `packing_plan.*`,
`apply_plan.*`, all CP-BP-09/statistical-validation files, and root
`CMakeLists.txt` remained read-only.

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
- [x] Implement CUDA construction and measure bytes/NNZ, metadata/NNZ, union
  size, and build throughput in Phase D.

## Blockers

- None. CP-BP-08 satisfies its recorded acceptance criteria and is closed.

## Progress Notes

- 2026-08-17: `BARRIER_D_INTEGRATED` accepted and pushed CP-BP-08 at source
  checkpoint `0bf9acf`. Fresh CUDA 12.9.86/GNU 13.3.0 `sm_70` validation passed
  exact host/CUDA tile agreement, memcheck with zero errors, racecheck with zero
  hazards, and the required upstream/downstream regressions. CP-BP-08 is now
  complete and closed; CP-BP-09 Phase E consumes its ABI read-only.
- 2026-08-17: Published `CP08_DEVICE_READY`, released every Phase D lease, and
  returned idle without git operations. Added a versioned asynchronous CUDA
  constructor over caller-owned device scratch/output and caller stream. Narrow
  `sm_70` kernels perform warp-local sorted tile unions, descriptor/mask/row
  emission, source-record mapping, and arbitrary-byte payload copies; CUB
  exclusive scans build the three offset domains. The frozen host ABI and all
  CP-BP-09 files remained unchanged.
- 2026-08-17: Exact CPU/CUDA byte agreement passed for deterministic random row
  order, zero-row and nonzero empty tiles, 34-row tail tiles, bit-31 cell/gene
  masks, 1/3/5/8-byte values, inconsistent counts, null/undersized buffers,
  alias rejection, and identity errors. CUDA 12.9 Compute Sanitizer memcheck
  reported zero errors and racecheck zero hazards; record, ordering, host-tile,
  reconstruction, planner, apply-plan, evaluator, optimizer, and inferred-
  pipeline regressions passed.
- 2026-08-17: Serialized Tesla V100 `sm_70` construction benchmark used 65,536
  rows, 30,000 features, 2,097,152 NNZ, 1,048,576 source records, 2,048 tiles,
  32,768 tile blocks, width-32 tiles, width-16 feature blocks, and 4-byte
  values. Transfers were excluded. Exact CUDA construction measured 0.756 ms
  min/median/mean after two warmups across seven repeats versus 31.954 ms CPU,
  2.775 GNNZ/s, 8,664,075 scratch bytes, and 4.191 tile metadata bytes/NNZ
  versus 6.125 source-record and 4.125 canonical-CSR metadata bytes/NNZ.
- 2026-08-17: `codex-cp-bp08-phase-d` claimed the exact CUDA tile lease at
  pushed base `fe095fb6d6592a0194b0a86f13f0421e23081cd0`. Concurrent CP-BP-09
  owner `codex-cp-bp09-phase-d` retains its disjoint host
  `feature_weighted_row_reduction` and root-CMake lease. This stream owns only
  new `warp_tiles_cuda` API/source/test/benchmark files, labelled component-
  CMake blocks, and locked CP-BP-08 coordination entries; it must publish
  `CP08_DEVICE_READY`, release, and stop without git operations.
- 2026-08-17: Barrier C integrated the host ABI/reference at source checkpoint
  `ebe0509` after fresh combined validation. Phase D is now fork-ready but
  unclaimed: add only the asynchronous caller-stream/device-scratch CUDA
  constructor, exact CPU/CUDA tests, and serialized V100 construction benchmark
  under the recorded lease. Preserve the frozen host ABI byte-for-byte; publish
  `CP08_DEVICE_READY`, release, and stop for Barrier D.
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

- Complete and closed. Do not reopen this stream to fit CP-BP-09 runtime or
  CP-BP-13 persistence; consumers must use the pushed ABI.

## Done Criteria

- Tile dictionaries, cell/gene masks, compact payloads, and all offsets decode exactly to canonical data.
- Tail cells, empty tiles, full/empty masks, repeated blocks, multiple value widths, offset overflow, and deterministic construction are covered.
- No zeros are materialized as BELL/Blocked-ELL padding.
- Benchmarks report construction throughput and fair metadata/storage deltas against relevant Cellerator layouts.
