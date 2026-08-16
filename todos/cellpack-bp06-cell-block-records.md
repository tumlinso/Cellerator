---
slug: "cellpack-bp06-cell-block-records"
status: "in_progress"
execution: "idle"
owner: "unassigned"
created_at: "2026-08-14T13:00:00Z"
last_heartbeat_at: "2026-08-16T20:14:53Z"
last_reviewed_at: "2026-08-16T20:14:53Z"
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

- Read `todos/cellpack-bp06-11-parallel-execution.md` before claiming. Barrier A
  is integrated; Phase B may now run in parallel only with CP-BP-07.
- If assigned CP-BP-06, follow the coordinator's CP-BP-06 conditional section
  exactly. Claim and lease under `/tmp/cellerator-cp-bp06-11-shared.lock`, use
  `build-cp-bp06`, stop/release at `CP06_DEVICE_READY`, and perform no git
  operation.
- Prefer new `cell_block_records` API/source/test/benchmark files. Any edit to
  `packing_plan.*`, `apply_plan.*`, either CMake file, or common format files is
  a shared seam requiring an explicit lease under the lock.

## File Lease

_No active lease._ Phase A owner `codex-cp-bp06-phase-a` released the host
record, plan-geometry, component-CMake, and coordination leases at
`CP06_HOST_ABI_READY`; Barrier A integrated them. Phase B must record its new
CUDA/API/test/benchmark lease before editing, while CP-BP-07 consumes the host
record contract read-only.

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
- [x] Specify complete logical record and access rules.
- [x] Implement CPU run detection, compact emission, validation, and decoder.
- [x] Add exact reconstruction and adversarial offset tests for Phase A.
- [ ] Implement GPU run detection and compact emission in Phase B.

## Blockers

_None for Phase B._ Barrier A integrated and pushed the host contract. Final
persistence field widths/versioning remain CP-BP-13 work and do not block the
runtime logical record contract here.

## Progress Notes

- 2026-08-16: Barrier A jointly validated and integrated the host record ABI
  with CP-BP-11 foundations. CP-BP-06 remains unclaimed; Phase B may now add
  only the CUDA detect/scan/emit implementation required by the coordinator.
- 2026-08-16: Published `CP06_HOST_ABI_READY`. Phase A added a versioned
  feature-block geometry fingerprint; checked pointer-first width-32 compact
  records; CPU requirements/build/validate/decode; and adversarial exact
  reconstruction tests covering empty rows, dense/multi-block rows, arbitrary
  value bytes, maximum bit 31, width rejection, offset/rank/capacity failures,
  and incompatible geometry. The focused test plus planner, evaluator,
  optimizer, exact merge-cost CPU/CUDA, and inferred-pipeline regressions pass
  in `build-cp-bp06`; `git diff --check` passes. All Phase A leases were
  released without a git operation.
- 2026-08-16: `codex-cp-bp06-phase-a` claimed Phase A at pushed base
  `8773f87`. The lease is limited to new cell-block record host files/tests,
  the exact plan-geometry identity seam, CellPack CMake target blocks, and
  coordination ledgers. CP-BP-11 may proceed through disjoint files/root CMake.
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

- Claim Phase B under the shared lock, lease only CP-BP-06 CUDA/API/test/bench
  seams, and implement caller-owned stream/scratch CUB-backed detect/scan/emit.
  Prove exact CPU agreement, sanitizer coverage, and serialized V100 benchmark
  evidence; do not absorb CP-BP-07 row ordering or later tile/runtime work.

## Done Criteria

- Exact decode reproduces every canonical coordinate and value.
- Masks contain only real values; no dense block padding is introduced.
- Row, record, payload, and terminal offsets are monotone, bounded, self-consistent, and sufficient for the documented access complexity.
- CPU/reference and GPU outputs match exactly for empty, sparse, dense-within-block, multi-block, and maximum-width cases.
