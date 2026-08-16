---
slug: "cellpack-bp06-cell-block-records"
status: "planned"
execution: "ready"
owner: "unassigned"
created_at: "2026-08-14T13:00:00Z"
last_heartbeat_at: "2026-08-16T19:45:16Z"
last_reviewed_at: "2026-08-16T19:45:16Z"
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
- Compact mask decode depends on the exact feature-block geometry, not only the
  dataset/row identity. Phase A must add or consume a versioned fingerprint of
  the frozen plan's authoritative block offsets/permutation.
- V1 uses exactly one `uint32_t` mask and rejects plans whose maximum feature
  block width exceeds 32; multiword masks are future scope.

## CP-BP-06→11 Fork Interlock

- Read `todos/cellpack-bp06-11-parallel-execution.md` before claiming. Phase A
  may run now in parallel only with CP-BP-11 Phase A.
- If assigned CP-BP-06, follow the coordinator's CP-BP-06 conditional section
  exactly. Claim and lease under `/tmp/cellerator-cp-bp06-11-shared.lock`, use
  `build-cp-bp06`, stop/release at `CP06_HOST_ABI_READY`, and perform no git
  operation. Resume CUDA work only after Barrier A.
- Prefer new `cell_block_records` API/source/test/benchmark files. Any edit to
  `packing_plan.*`, `apply_plan.*`, either CMake file, or common format files is
  a shared seam requiring an explicit lease under the lock.

## File Lease

_Unclaimed._ Record exact intended paths here atomically at claim time.

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

1. Phase A: resolve plan-geometry identity, checked width-32 record fields,
   offset/index widths, capacities, and value-rank semantics.
2. Phase A: implement CPU/reference run detection, emission, validation, and
   exact decode; publish `CP06_HOST_ABI_READY` and stop at Barrier A.
3. Phase B: implement GPU detect/scan/emit using established scan primitives
   with caller-owned scratch/stream semantics.
4. Validate exact decode and every monotonicity/range/terminal-offset invariant;
   publish `CP06_DEVICE_READY` and close at Barrier B.

## Tasks

- [x] Consume CP-BP-05 ordered-row/value contract; CP-BP-04 plan semantics are complete.
- [ ] Specify complete logical record and access rules.
- [ ] Implement CPU/GPU run detection and compact emission.
- [ ] Add exact reconstruction and adversarial offset tests.

## Blockers

_None; this stream is ready and unclaimed._ CP-BP-05 now supplies ordered row
offsets, block/local coordinates, canonical feature IDs, and exact value bytes
for arbitrary full-domain partitions. Final persistence field widths/versioning
remain CP-BP-13 work and do not block the runtime logical record contract here.

## Progress Notes

- 2026-08-16: Reactivated as `planned/ready` after CP-BP-05 completed
  `ordered_plan_partition_view` and exact host/CUDA application of full-domain
  frozen plans. CP-BP-06 can consume that view read-only; no record emission
  was implemented by CP-BP-05.
- 2026-08-14: Added as a missing blocked workstream; existing CellPack packed-coordinate scaffolding is not accepted as this compact physical record.
- 2026-08-16: Reconciliation found no compact per-cell block record, gene-mask,
  variable-payload offset, or exact decoder implementation. The older
  `packed_coordinate_plan` remains coordinate scaffolding and does not satisfy
  this acceptance contract.

## Next Actions

- Define the pointer-first row/block/value offset and mask-rank contract over
  `ordered_plan_partition_view`, then implement exact CPU/CUDA run detection
  and emission without absorbing CP-BP-07 ordering, CP-BP-08 tiles, runtime
  kernels, or persistence.

## Done Criteria

- Exact decode reproduces every canonical coordinate and value.
- Masks contain only real values; no dense block padding is introduced.
- Row, record, payload, and terminal offsets are monotone, bounded, self-consistent, and sufficient for the documented access complexity.
- CPU/reference and GPU outputs match exactly for empty, sparse, dense-within-block, multi-block, and maximum-width cases.
