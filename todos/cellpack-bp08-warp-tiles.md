---
slug: "cellpack-bp08-warp-tiles"
status: "blocked"
execution: "closed"
owner: "unassigned"
created_at: "2026-08-14T13:00:00Z"
last_heartbeat_at: "2026-08-14T13:00:00Z"
last_reviewed_at: "2026-08-16T19:45:16Z"
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

_Blocked and unclaimed._ Record exact intended paths atomically after the gate.

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

1. Specify versionable tile descriptor, dictionary, mask, payload, and offset/rank contracts.
2. Build CPU/reference tile construction and exact decoder.
3. Implement GPU union/mask/payload emission with explicit scratch/stream ownership.
4. Compare storage/metadata with relevant current Cellerator layouts.

## Tasks

- [!] Wait for CP-BP-06 record and CP-BP-07 ordering contracts.
- [ ] Define complete warp-tile logical ABI.
- [ ] Implement reference/GPU construction and decode.
- [ ] Measure bytes/NNZ, metadata/NNZ, union size, and build throughput.

## Blockers

- Blocked on CP-BP-06 compact record semantics and CP-BP-07 local row grouping.

## Progress Notes

- 2026-08-14: Added as a missing blocked workstream; existing CellPack coordinate spans and layout labels do not implement this representation.

## Next Actions

- Reactivate only after per-cell value-rank semantics and local 32-row grouping are stable.

## Done Criteria

- Tile dictionaries, cell/gene masks, compact payloads, and all offsets decode exactly to canonical data.
- Tail cells, empty tiles, full/empty masks, repeated blocks, multiple value widths, offset overflow, and deterministic construction are covered.
- No zeros are materialized as BELL/Blocked-ELL padding.
- Benchmarks report construction throughput and fair metadata/storage deltas against relevant Cellerator layouts.
