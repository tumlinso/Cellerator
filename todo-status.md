# Todo Status

## Summary
Use this file as the quick pickup register for `todos.md` workstreams.
- `ready`: planned work that can be started now.
- `claimed`: currently being written; choose another stream.
- `idle`: unfinished but resumable; safe to pick up.
- `closed`: completed or removed from pickup rotation.

## Workstreams
- `cellpack-bp00-05-integration-audit` | status: done | execution: closed | owner: codex-cp-bp00-05-integration | file: `todos/cellpack-bp00-05-integration-audit.md` | next: Complete; the real sampling→support→candidate→score→optimize→apply chain is covered and CP-BP-06 remains untouched.
- `cellpack-bp06-11-parallel-execution` | status: in_progress | execution: claimed | owner: coordination | file: `todos/cellpack-bp06-11-parallel-execution.md` | next: Barrier C source checkpoint `ebe0509` is integrated; await explicit, independent CP-BP-08 Phase D and CP-BP-09 Phase D assignments under the published leases.
- `cellpack-data-inferred-block-packing-roadmap` | status: in_progress | execution: claimed | owner: coordination | file: `todos/cellpack-data-inferred-block-packing-roadmap.md` | next: Do not implement from this parent. Phase D CP-BP-08 CUDA construction and CP-BP-09 reference/API are the unclaimed parallel frontier.
- `cellpack-bp01-support-extraction` | status: done | execution: closed | owner: parallel-agent-step-1 | file: `todos/cellpack-bp01-support-extraction.md` | next: Complete: deterministic sampled CSR now flows through exact CPU/CUDA gene-major support bitsets with counts, provenance, and global-row mapping.
- `cellpack-bp02-candidate-discovery` | status: done | execution: closed | owner: codex-cp-bp-02 | file: `todos/cellpack-bp02-candidate-discovery.md` | next: Complete; CP-BP-03 consumes the frozen host pair view and provenance through its exact scorer.
- `cellpack-bp03-exact-merge-cost` | status: done | execution: closed | owner: codex-cp-bp-03-fork | file: `todos/cellpack-bp03-exact-merge-cost.md` | next: Complete and closed; no CP-BP-03 implementation remains.
- `cellpack-bp04-packing-plan-optimizer` | status: done | execution: closed | owner: codex-cp-bp-04 | file: `todos/cellpack-bp04-packing-plan-optimizer.md` | next: Complete supplied-candidate optimizer and frozen semantic plan; CP-BP-03 now emits optimizer-valid exact scored relations.
- `cellpack-bp05-apply-frozen-plan` | status: done | execution: closed | owner: codex-cp-bp-05-fork | file: `todos/cellpack-bp05-apply-frozen-plan.md` | next: Complete and closed; CP-BP-06 may consume the ordered partition view, but no CP-BP-05 implementation remains.
- `cellpack-packing-plan-cuda-evaluator` | status: planned | execution: ready | owner: unassigned | file: `todos/cellpack-packing-plan-cuda-evaluator.md` | next: Profile representative larger oracle workloads, then define persistent device source and caller-owned CUB scratch; not a CP-BP-05 prerequisite.
- `cellpack-bp06-cell-block-records` | status: done | execution: closed | owner: codex-cp-bp06-phase-b | file: `todos/cellpack-bp06-cell-block-records.md` | next: Complete and closed; consumers use the pushed Barrier B checkpoint.
- `cellpack-bp07-local-cell-ordering` | status: done | execution: closed | owner: codex-cp-bp07 | file: `todos/cellpack-bp07-local-cell-ordering.md` | next: Complete and closed; consumers use the pushed Barrier B checkpoint.
- `cellpack-bp08-warp-tiles` | status: in_progress | execution: idle | owner: unassigned | file: `todos/cellpack-bp08-warp-tiles.md` | next: Await explicit CP-BP-08 Phase D assignment; claim only CUDA construction/test/benchmark files, publish `CP08_DEVICE_READY`, release, and stop for Barrier D.
- `cellpack-bp09-native-runtime-consumers` | status: planned | execution: ready | owner: unassigned | file: `todos/cellpack-bp09-native-runtime-consumers.md` | next: Await explicit CP-BP-09 Phase D assignment; claim only the CPU/reference and pointer-first consumer API lease, publish `CP09_REFERENCE_READY`, release, and stop before CUDA runtime work.
- `cellpack-bp10-alternating-refinement` | status: blocked | execution: closed | owner: unassigned | file: `todos/cellpack-bp10-alternating-refinement.md` | next: Do not claim until `CP10_READY`: CP-BP-07/08 closed, CP-BP-09 measurable, and CP-BP-11 held-out gate published.
- `cellpack-bp11-statistical-validation` | status: in_progress | execution: idle | owner: unassigned | file: `todos/cellpack-bp11-statistical-validation.md` | next: `CP11_HELDOUT_READY` is published; remain idle until the coordinator opens the later tile/bootstrap Phase E extension.
- `cellpack-bp12-hardware-cost-autotune` | status: blocked | execution: closed | owner: unassigned | file: `todos/cellpack-bp12-hardware-cost-autotune.md` | next: CP-BP-03's replaceable policy seam is complete; reactivate after correct measured CP-BP-08/09 paths exist.
- `cellpack-bp13-persistence-integration` | status: blocked | execution: closed | owner: unassigned | file: `todos/cellpack-bp13-persistence-integration.md` | next: Reactivate after stable plan, record, tile, and direct-runtime contracts.

## Staleness Review
- Fresh: 17
- Aging: 0
- Stale candidates: 0
- Stale: 0
- Superseded: 0

## Cleanup Status
- Cleanup mode is explicit only.
- Safe to call `todo-cleanup`: no, active workstreams: cellpack-bp06-11-parallel-execution, cellpack-data-inferred-block-packing-roadmap, cellpack-packing-plan-cuda-evaluator, cellpack-bp08-warp-tiles, cellpack-bp09-native-runtime-consumers, cellpack-bp10-alternating-refinement, cellpack-bp11-statistical-validation, cellpack-bp12-hardware-cost-autotune, cellpack-bp13-persistence-integration.
- Partial cleanup is available via `todo-cleanup --partial`; include `stale` in `--scope` only when explicitly intended.
