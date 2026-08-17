# Todo Status

## Summary
Use this file as the quick pickup register for `todos.md` workstreams.
- `ready`: planned work that can be started now.
- `claimed`: currently being written; choose another stream.
- `idle`: unfinished but resumable; safe to pick up.
- `closed`: completed or removed from pickup rotation.

## Workstreams
- `cellpack-bp00-05-integration-audit` | status: done | execution: closed | owner: codex-cp-bp00-05-integration | file: `todos/cellpack-bp00-05-integration-audit.md` | next: Complete; the real sampling→support→candidate→score→optimize→apply chain is covered and CP-BP-06 remains untouched.
- `cellpack-bp06-11-parallel-execution` | status: in_progress | execution: claimed | owner: coordination | file: `todos/cellpack-bp06-11-parallel-execution.md` | next: Barrier E is integrated; fork the exact unclaimed CP-BP-10/11 Phase F pair and stop each child at its named gate without git.
- `cellpack-data-inferred-block-packing-roadmap` | status: in_progress | execution: claimed | owner: coordination | file: `todos/cellpack-data-inferred-block-packing-roadmap.md` | next: Do not implement from this parent. CP-BP-10/11 Phase F are the next disjoint implementation pair.
- `cellpack-bp01-support-extraction` | status: done | execution: closed | owner: parallel-agent-step-1 | file: `todos/cellpack-bp01-support-extraction.md` | next: Complete: deterministic sampled CSR now flows through exact CPU/CUDA gene-major support bitsets with counts, provenance, and global-row mapping.
- `cellpack-bp02-candidate-discovery` | status: done | execution: closed | owner: codex-cp-bp-02 | file: `todos/cellpack-bp02-candidate-discovery.md` | next: Complete; CP-BP-03 consumes the frozen host pair view and provenance through its exact scorer.
- `cellpack-bp03-exact-merge-cost` | status: done | execution: closed | owner: codex-cp-bp-03-fork | file: `todos/cellpack-bp03-exact-merge-cost.md` | next: Complete and closed; no CP-BP-03 implementation remains.
- `cellpack-bp04-packing-plan-optimizer` | status: done | execution: closed | owner: codex-cp-bp-04 | file: `todos/cellpack-bp04-packing-plan-optimizer.md` | next: Complete supplied-candidate optimizer and frozen semantic plan; CP-BP-03 now emits optimizer-valid exact scored relations.
- `cellpack-bp05-apply-frozen-plan` | status: done | execution: closed | owner: codex-cp-bp-05-fork | file: `todos/cellpack-bp05-apply-frozen-plan.md` | next: Complete and closed; CP-BP-06 may consume the ordered partition view, but no CP-BP-05 implementation remains.
- `cellpack-packing-plan-cuda-evaluator` | status: planned | execution: ready | owner: unassigned | file: `todos/cellpack-packing-plan-cuda-evaluator.md` | next: Profile representative larger oracle workloads, then define persistent device source and caller-owned CUB scratch; not a CP-BP-05 prerequisite.
- `cellpack-bp06-cell-block-records` | status: done | execution: closed | owner: codex-cp-bp06-phase-b | file: `todos/cellpack-bp06-cell-block-records.md` | next: Complete and closed; consumers use the pushed Barrier B checkpoint.
- `cellpack-bp07-local-cell-ordering` | status: done | execution: closed | owner: codex-cp-bp07 | file: `todos/cellpack-bp07-local-cell-ordering.md` | next: Complete and closed; consumers use the pushed Barrier B checkpoint.
- `cellpack-bp08-warp-tiles` | status: done | execution: closed | owner: codex-cp-bp08-phase-d | file: `todos/cellpack-bp08-warp-tiles.md` | next: Complete and closed at pushed Barrier D checkpoint `0bf9acf`; consumers use its frozen ABI.
- `cellpack-bp09-native-runtime-consumers` | status: done | execution: closed | owner: codex-cp-bp09-phase-e | file: `todos/cellpack-bp09-native-runtime-consumers.md` | next: Complete at pushed Barrier E source checkpoint `0334f95`; downstream consumers use the frozen v1 API.
- `cellpack-bp10-alternating-refinement` | status: planned | execution: ready | owner: unassigned | file: `todos/cellpack-bp10-alternating-refinement.md` | next: Assign exactly CP-BP-10 Phase F; claim the recorded host-controller lease, publish `CP10_REFINEMENT_READY`, release, and stop without git.
- `cellpack-bp11-statistical-validation` | status: in_progress | execution: idle | owner: unassigned | file: `todos/cellpack-bp11-statistical-validation.md` | next: Assign exactly CP-BP-11 Phase F; claim final mapping/runtime-stability lease, publish `CP11_FINAL_VALIDATION_READY`, release, and stop without git.
- `cellpack-bp12-hardware-cost-autotune` | status: blocked | execution: closed | owner: unassigned | file: `todos/cellpack-bp12-hardware-cost-autotune.md` | next: Technical inputs are integrated; reactivate after Barrier F to avoid a competing GPU benchmark/integration campaign.
- `cellpack-bp13-persistence-integration` | status: blocked | execution: closed | owner: unassigned | file: `todos/cellpack-bp13-persistence-integration.md` | next: V1 Cellerator contracts are integrated; after Barrier F start with a read-only Cellerator/CellShard compatibility audit.

## Staleness Review
- Fresh: 17
- Aging: 0
- Stale candidates: 0
- Stale: 0
- Superseded: 0

## Cleanup Status
- Cleanup mode is explicit only.
- Safe to call `todo-cleanup`: no, active workstreams: cellpack-bp06-11-parallel-execution, cellpack-data-inferred-block-packing-roadmap, cellpack-packing-plan-cuda-evaluator, cellpack-bp10-alternating-refinement, cellpack-bp11-statistical-validation, cellpack-bp12-hardware-cost-autotune, cellpack-bp13-persistence-integration.
- Partial cleanup is available via `todo-cleanup --partial`; include `stale` in `--scope` only when explicitly intended.
