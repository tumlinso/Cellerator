---
slug: "cellpack-bp07-local-cell-ordering"
status: "done"
execution: "closed"
owner: "codex-cp-bp07"
created_at: "2026-08-14T13:00:00Z"
last_heartbeat_at: "2026-08-17T08:04:27Z"
last_reviewed_at: "2026-08-17T08:04:27Z"
stale_after_days: 7
objective: "CP-BP-07: Infer cheap local cell ordering from active gene-block signatures for warp-friendly groups."
---

# Current Objective

## Summary

Represent each transformed cell by its sorted active-block set, compute a compact signature, and order only local chunks/windows so similar cells share warp/slice groups.

## Quick Start

- Why this stream exists: local row order should reduce tile block unions and metadata without globally optimizing millions of cells.
- In scope: `cell_block_minhash` or equivalent, active-block counts, deterministic local sort/grouping, configurable chunk sizes, permutation/inverse maps, and baseline metrics.
- Out of scope / dependencies: global cell clustering, manual cell labels, biological semantics, global dataset reorder, and gene-plan relearning.
- Required skills: `cuda`, `todo-orchestrator`.
- Required references: CP-BP-00, CP-BP-06, `components/CellPack/AGENTS.md`, and the completed PackingPlan evaluator's row-permutation conventions.

## Planning Notes

- Windows should remain partition/chunk oriented, plausibly hundreds to a few thousand rows, and preserve explicit reversible row identity.
- Compare against original/random order and simple row-NNZ sorting; neither is an intentionally weak sole baseline.
- CP-BP-07 emits bounded local permutation and inverse arrays. It does not
  physically rewrite CP-BP-06 record payloads; CP-BP-08 consumes the records
  through these maps.

## CP-BP-06→11 Fork Interlock

- Read `todos/cellpack-bp06-11-parallel-execution.md`. Do not claim before
  `CP06_HOST_ABI_READY` and Barrier A are recorded. If assigned too early,
  remain read-only and report the missing gate without inventing an ABI.
- Claim/lease under `/tmp/cellerator-cp-bp06-11-shared.lock`, use
  `build-cp-bp07`, consume CP-BP-06 files read-only, publish
  `CP07_ORDER_ABI_READY` and `CP07_DEVICE_READY`, then release/close without git
  operations.

## File Lease

No active CP-BP-07 lease. Owner `codex-cp-bp07` released the new local-order
host/CUDA API, source, focused test, benchmark, root-CMake target blocks, and
coordination ledgers at both CP-BP-07 gates. The Barrier B integrator accepted
the combined CP-BP-06/07 tree from pushed base `1e25e11`; no implementation
work remains in this stream.

## Assumptions

- CP-BP-06 provides sorted active block IDs per row without decoding values.
- Signature approximation chooses local order only; measured tile cost judges usefulness.

## Suggested Skills

- `cuda`
- `todo-orchestrator`

## Useful Reference Files

- `todos/cellpack-packing-plan-evaluator.md`
- `components/CellPack/include/CellPack/evaluator.hh`
- `components/CellPack/AGENTS.md`

## Plan

1. Define local-window, signature, and reversible row-order contracts.
2. Add deterministic reference ordering and baseline metrics.
3. Implement GPU signature/grouping with library sort primitives.
4. Measure block-union and metadata reduction over multiple chunk sizes.

## Tasks

- [x] Wait for CP-BP-06 active-block record/view contract.
- [x] Implement signature and local grouping.
- [x] Preserve row permutation and inverse maps.
- [x] Benchmark against original/random and row-length ordering.

## Blockers

_None; implementation and validation are complete._ CP-BP-08 remains gated
only until the coordinator records the pushed Barrier B checkpoint.

## Progress Notes

- 2026-08-17: Barrier B integrator rebuilt CP-BP-06 and CP-BP-07 together from
  fresh `build-cp-bp-barrier-b` with CUDA 12.9.86, GNU 13.3.0, and V100
  `sm_70`. Exact device maps, record/apply-plan/merge-cost CUDA tests, all host
  regressions, CUDA 12.9 memcheck, and racecheck passed. The serialized
  65,536-row benchmark reproduced exact agreement at 0.23344 ms CUDA median
  versus 22.2067 ms CPU, with inferred metadata 131,072 bytes versus 4,194,304
  original/row-NNZ and 2,701,568 random bytes. CP-BP-07 is closed; the source
  checkpoint is committed and pushed before `BARRIER_B_INTEGRATED` is opened.
- 2026-08-17: Published `CP07_ORDER_ABI_READY` and `CP07_DEVICE_READY`, then
  released every lease without a git operation. Added a versioned pointer-first
  bounded-window order view, deterministic four-lane active-block MinHash,
  original/random/row-NNZ baselines, reversible local/global row identity, and
  exact group-union/metadata metrics. The asynchronous CUDA path uses two
  stable CUB segmented radix sorts plus narrow signature/index kernels with
  caller-owned stream/scratch and exact CPU agreement for every order kind.
  `cellPackLocalCellOrderingTest`, CUDA 12.9 compute-sanitizer memcheck and
  racecheck, record, evaluator, optimizer, and CUDA apply-plan regressions
  passed. On Tesla V100
  `sm_70`, 65,536 rows, 16 active blocks/row, 1,024-row windows, and 32-row
  groups measured CPU 22.7307 ms and CUDA median 0.233472 ms with transfers
  excluded and 2,359,811 temporary bytes. Inferred block-id metadata was
  131,072 bytes versus 4,194,304 original/row-NNZ and 2,701,568 deterministic
  random bytes. Additional 256/4,096-row windows preserved exact agreement:
  CUDA medians were 0.439296/0.326624 ms and inferred metadata was
  524,288/131,072 bytes versus 4,194,304 original/row-NNZ bytes and
  2,780,032/2,677,824 random bytes. The default HPC SDK sanitizer wrapper
  incorrectly targeted an
  absent CUDA 13.1 install; the matching installed CUDA 12.9 sanitizer reported
  zero memory errors and zero race hazards.
- 2026-08-17: `codex-cp-bp07` claimed CP-BP-07 at pushed base `1e25e11` with
  new local-order host/CUDA API/source/test/benchmark files, root-CMake target
  blocks, and coordination ledgers. The route is native V100 `sm_70` with CUB
  device sorting plus narrow signature/index kernels; sparse signature ordering
  is not Tensor Core eligible. CP-BP-06 host records are read-only and its
  component-CMake/CUDA record lease remains untouched.
- 2026-08-16: Reactivated as `planned/ready` after Barrier A integrated the
  versioned CP-BP-06 host record ABI. CP-BP-07 may consume row-to-record
  offsets and sorted block IDs read-only while CP-BP-06 independently adds its
  CUDA emitter.
- 2026-08-14: Added as a missing blocked workstream; no implementation evidence was found.

## Next Actions

- Complete and closed. CP-BP-08 may consume the integrated order contract only
  after the coordinator records the pushed Barrier B checkpoint; no CP-BP-07
  implementation remains.

## Done Criteria

- Ordering is deterministic, local to explicit bounded chunks, and fully reversible.
- No global million-cell optimization or label dependency is introduced.
- Held fixtures report per-warp/tile block-union size and metadata cost against original/random and row-length order.
- CPU/reference and GPU signature/order semantics agree.
