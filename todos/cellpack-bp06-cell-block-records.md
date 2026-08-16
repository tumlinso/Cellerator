---
slug: "cellpack-bp06-cell-block-records"
status: "blocked"
execution: "closed"
owner: "unassigned"
created_at: "2026-08-14T13:00:00Z"
last_heartbeat_at: "2026-08-16T14:38:44Z"
last_reviewed_at: "2026-08-16T14:38:44Z"
stale_after_days: 7
objective: "CP-BP-06: Collapse ordered row entries into compact per-cell gene-block records with complete offset metadata."
---

# Current Objective

## Summary

Detect adjacent block runs and emit `block_id`, within-block `gene_mask`, compact real values, and sufficient row/block/value offsets for unambiguous access and exact decode.

## Quick Start

- Why this stream exists: convert ordered sparse entries into the first compact block grammar without dense BELL padding.
- In scope: `detect_block_runs`, `emit_cell_blocks`, offset/index invariants, compact value order, pointer-free views, and exact decode.
- Out of scope / dependencies: zeros/padded slots, warp-tile sharing, persistence ABI, and native compute kernels.
- Required skills: `cuda`, `todo-orchestrator`.
- Required references: CP-BP-00, CP-BP-04, CP-BP-05, `components/CellPack/AGENTS.md`, and `style_hint.md`.

## Planning Notes

- Only real values are stored. This representation is not Blocked-ELL/BELL.
- A consumer must be able to locate each row, block record, and variable-length value payload through explicit offsets/rank rules; underspecified masks plus values are unacceptable.

## Assumptions

- Initial block widths are bounded by CP-BP-04 and may use `uint32_t gene_mask` at width 32.
- Payload order and rank semantics are deterministic and versionable.

## Suggested Skills

- `cuda`
- `todo-orchestrator`

## Useful Reference Files

- `components/CellPack/include/CellPack/format.hh`
- `components/CellPack/include/CellPack/pack.hh`
- `components/CellPack/AGENTS.md`

## Plan

1. Resolve logical record fields, offset/index widths, and value-rank semantics.
2. Implement CPU/reference run detection and emission.
3. Implement GPU detect/scan/emit using established scan primitives where appropriate.
4. Validate exact decode and every monotonicity/range/terminal-offset invariant.

## Tasks

- [!] Wait for CP-BP-05 ordered-row/value contract; CP-BP-04 plan semantics are complete.
- [ ] Specify complete logical record and access rules.
- [ ] Implement CPU/GPU run detection and compact emission.
- [ ] Add exact reconstruction and adversarial offset tests.

## Blockers

- Blocked only on CP-BP-05 ordered-row/value output; CP-BP-04 already supplies the required plan width and canonical block/local mappings.
- Final persistence field widths/versioning are coordinated later in CP-BP-13, but runtime logical access must be settled here.

## Progress Notes

- 2026-08-14: Added as a missing blocked workstream; existing CellPack packed-coordinate scaffolding is not accepted as this compact physical record.
- 2026-08-16: Reconciliation found no compact per-cell block record, gene-mask,
  variable-payload offset, or exact decoder implementation. The older
  `packed_coordinate_plan` remains coordinate scaffolding and does not satisfy
  this acceptance contract.

## Next Actions

- Reactivate after CP-BP-05 settles ordered rows and values, then resolve the logical row/block/value offset contract.

## Done Criteria

- Exact decode reproduces every canonical coordinate and value.
- Masks contain only real values; no dense block padding is introduced.
- Row, record, payload, and terminal offsets are monotone, bounded, self-consistent, and sufficient for the documented access complexity.
- CPU/reference and GPU outputs match exactly for empty, sparse, dense-within-block, multi-block, and maximum-width cases.
