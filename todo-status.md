# Todo Status

## Summary
Use this file as the quick pickup register for `todos.md` workstreams.
- `ready`: planned work that can be started now.
- `claimed`: currently being written; choose another stream.
- `idle`: unfinished but resumable; safe to pick up.
- `closed`: completed or removed from pickup rotation.

## Workstreams
- `cellpack-bp00-05-integration-audit` | status: done | execution: closed | owner: codex-cp-bp00-05-integration | file: `todos/cellpack-bp00-05-integration-audit.md` | next: Complete; the real sampling→support→candidate→score→optimize→apply chain is covered and CP-BP-06 remains untouched.
- `cellpack-bp06-11-parallel-execution` | status: in_progress | execution: claimed | owner: coordination | file: `todos/cellpack-bp06-11-parallel-execution.md` | next: Fork-ready Phase E: assign CP-BP-09 Phase E and CP-BP-11 Phase E exactly; both are idle with disjoint leases and need no addendum.
- `cellpack-data-inferred-block-packing-roadmap` | status: in_progress | execution: claimed | owner: coordination | file: `todos/cellpack-data-inferred-block-packing-roadmap.md` | next: Do not implement from this parent. Phase E CP-BP-09/11 is fork-ready; CP-BP-10/12/13 remain gated.
- `cellpack-bp01-support-extraction` | status: done | execution: closed | owner: parallel-agent-step-1 | file: `todos/cellpack-bp01-support-extraction.md` | next: Complete: deterministic sampled CSR now flows through exact CPU/CUDA gene-major support bitsets with counts, provenance, and global-row mapping.
- `cellpack-bp02-candidate-discovery` | status: done | execution: closed | owner: codex-cp-bp-02 | file: `todos/cellpack-bp02-candidate-discovery.md` | next: Complete; CP-BP-03 consumes the frozen host pair view and provenance through its exact scorer.
- `cellpack-bp03-exact-merge-cost` | status: done | execution: closed | owner: codex-cp-bp-03-fork | file: `todos/cellpack-bp03-exact-merge-cost.md` | next: Complete and closed; no CP-BP-03 implementation remains.
- `cellpack-bp04-packing-plan-optimizer` | status: done | execution: closed | owner: codex-cp-bp-04 | file: `todos/cellpack-bp04-packing-plan-optimizer.md` | next: Complete supplied-candidate optimizer and frozen semantic plan; CP-BP-03 now emits optimizer-valid exact scored relations.
- `cellpack-bp05-apply-frozen-plan` | status: done | execution: closed | owner: codex-cp-bp-05-fork | file: `todos/cellpack-bp05-apply-frozen-plan.md` | next: Complete and closed; CP-BP-06 may consume the ordered partition view, but no CP-BP-05 implementation remains.
- `cellpack-packing-plan-cuda-evaluator` | status: planned | execution: ready | owner: unassigned | file: `todos/cellpack-packing-plan-cuda-evaluator.md` | next: Profile representative larger oracle workloads, then define persistent device source and caller-owned CUB scratch; not a CP-BP-05 prerequisite.
- `cellpack-bp06-cell-block-records` | status: done | execution: closed | owner: codex-cp-bp06-phase-b | file: `todos/cellpack-bp06-cell-block-records.md` | next: Complete and closed; consumers use the pushed Barrier B checkpoint.
- `cellpack-bp07-local-cell-ordering` | status: done | execution: closed | owner: codex-cp-bp07 | file: `todos/cellpack-bp07-local-cell-ordering.md` | next: Complete and closed; consumers use the pushed Barrier B checkpoint.
- `cellpack-bp08-warp-tiles` | status: done | execution: closed | owner: codex-cp-bp08-phase-d | file: `todos/cellpack-bp08-warp-tiles.md` | next: Complete and closed at pushed Barrier D checkpoint `0bf9acf`; consumers use its frozen ABI.
- `cellpack-bp09-native-runtime-consumers` | status: in_progress | execution: idle | owner: unassigned | file: `todos/cellpack-bp09-native-runtime-consumers.md` | next: Assign exactly “You are assigned CP-BP-09 Phase E”; claim only the direct CUDA consumer/test/benchmark and component-CMake lease.
- `cellpack-bp10-alternating-refinement` | status: blocked | execution: closed | owner: unassigned | file: `todos/cellpack-bp10-alternating-refinement.md` | next: Do not claim until `CP10_READY`: CP-BP-07/08 closed, CP-BP-09 measurable, and CP-BP-11 held-out gate published.
- `cellpack-bp11-statistical-validation` | status: in_progress | execution: idle | owner: unassigned | file: `todos/cellpack-bp11-statistical-validation.md` | next: Assign exactly “You are assigned CP-BP-11 Phase E”; claim only the host tile-validation files and root-CMake lease.
- `cellpack-bp12-hardware-cost-autotune` | status: blocked | execution: closed | owner: unassigned | file: `todos/cellpack-bp12-hardware-cost-autotune.md` | next: CP-BP-03 and measured CP-BP-08 construction are complete; reactivate after measured CP-BP-09 runtime is integrated at Barrier E.
- `cellpack-bp13-persistence-integration` | status: blocked | execution: closed | owner: unassigned | file: `todos/cellpack-bp13-persistence-integration.md` | next: Reactivate after stable plan, record, tile, and direct-runtime contracts.

## Staleness Review
- Fresh: 17
- Aging: 0
- Stale candidates: 0
- Stale: 0
- Superseded: 0

## Cleanup Status
- Cleanup mode is explicit only.
- Safe to call `todo-cleanup`: no, active workstreams: cellpack-bp06-11-parallel-execution, cellpack-data-inferred-block-packing-roadmap, cellpack-packing-plan-cuda-evaluator, cellpack-bp09-native-runtime-consumers, cellpack-bp10-alternating-refinement, cellpack-bp11-statistical-validation, cellpack-bp12-hardware-cost-autotune, cellpack-bp13-persistence-integration.
- Partial cleanup is available via `todo-cleanup --partial`; include `stale` in `--scope` only when explicitly intended.
