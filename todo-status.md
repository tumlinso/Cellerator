# Todo Status

## Summary
Use this file as the quick pickup register for `todos.md` workstreams.
- `ready`: planned work that can be started now.
- `claimed`: currently being written; choose another stream.
- `idle`: unfinished but resumable; safe to pick up.
- `closed`: completed or removed from pickup rotation.

## Workstreams
- `cellpack-data-inferred-block-packing-roadmap` | status: in_progress | execution: claimed | owner: coordination | file: `todos/cellpack-data-inferred-block-packing-roadmap.md` | next: Do not implement from this parent. CP-BP-03/05 are complete and closed; CP-BP-06 is ready and unclaimed.
- `cellpack-bp01-support-extraction` | status: done | execution: closed | owner: parallel-agent-step-1 | file: `todos/cellpack-bp01-support-extraction.md` | next: Complete: deterministic sampled CSR now flows through exact CPU/CUDA gene-major support bitsets with counts, provenance, and global-row mapping.
- `cellpack-bp02-candidate-discovery` | status: done | execution: closed | owner: codex-cp-bp-02 | file: `todos/cellpack-bp02-candidate-discovery.md` | next: Complete; CP-BP-03 consumes the frozen host pair view and provenance through its exact scorer.
- `cellpack-bp03-exact-merge-cost` | status: done | execution: closed | owner: codex-cp-bp-03-fork | file: `todos/cellpack-bp03-exact-merge-cost.md` | next: Complete and closed; no CP-BP-03 implementation remains.
- `cellpack-bp04-packing-plan-optimizer` | status: done | execution: closed | owner: codex-cp-bp-04 | file: `todos/cellpack-bp04-packing-plan-optimizer.md` | next: Complete supplied-candidate optimizer and frozen semantic plan; CP-BP-03 now emits optimizer-valid exact scored relations.
- `cellpack-bp05-apply-frozen-plan` | status: done | execution: closed | owner: codex-cp-bp-05-fork | file: `todos/cellpack-bp05-apply-frozen-plan.md` | next: Complete and closed; CP-BP-06 may consume the ordered partition view, but no CP-BP-05 implementation remains.
- `cellpack-packing-plan-cuda-evaluator` | status: planned | execution: ready | owner: unassigned | file: `todos/cellpack-packing-plan-cuda-evaluator.md` | next: Profile representative larger oracle workloads, then define persistent device source and caller-owned CUB scratch; not a CP-BP-05 prerequisite.
- `cellpack-bp06-cell-block-records` | status: planned | execution: ready | owner: unassigned | file: `todos/cellpack-bp06-cell-block-records.md` | next: Define and implement compact per-cell records over CP-BP-05's ordered partition view; do not absorb CP-BP-07/08, runtime, or persistence scope.
- `cellpack-bp07-local-cell-ordering` | status: blocked | execution: closed | owner: unassigned | file: `todos/cellpack-bp07-local-cell-ordering.md` | next: Reactivate after CP-BP-06 exposes sorted active-block sets per cell.
- `cellpack-bp08-warp-tiles` | status: blocked | execution: closed | owner: unassigned | file: `todos/cellpack-bp08-warp-tiles.md` | next: Reactivate after CP-BP-06/07 settle value-rank and local-group contracts.
- `cellpack-bp09-native-runtime-consumers` | status: blocked | execution: closed | owner: unassigned | file: `todos/cellpack-bp09-native-runtime-consumers.md` | next: Reactivate after CP-BP-08 stable device tile views and exact decode.
- `cellpack-bp10-alternating-refinement` | status: blocked | execution: closed | owner: unassigned | file: `todos/cellpack-bp10-alternating-refinement.md` | next: Reactivate after a measurable first-pass plan/tile/runtime loop exists.
- `cellpack-bp11-statistical-validation` | status: planned | execution: ready | owner: unassigned | file: `todos/cellpack-bp11-statistical-validation.md` | next: Build isolated metric/provenance and degree-preserving null references; integrate stages later.
- `cellpack-bp12-hardware-cost-autotune` | status: blocked | execution: closed | owner: unassigned | file: `todos/cellpack-bp12-hardware-cost-autotune.md` | next: CP-BP-03's replaceable policy seam is complete; reactivate after correct measured CP-BP-08/09 paths exist.
- `cellpack-bp13-persistence-integration` | status: blocked | execution: closed | owner: unassigned | file: `todos/cellpack-bp13-persistence-integration.md` | next: Reactivate after stable plan, record, tile, and direct-runtime contracts.

## Staleness Review
- Fresh: 15
- Aging: 0
- Stale candidates: 0
- Stale: 0
- Superseded: 0

## Cleanup Status
- Cleanup mode is explicit only.
- Safe to call `todo-cleanup`: no, active workstreams: cellpack-data-inferred-block-packing-roadmap, cellpack-packing-plan-cuda-evaluator, cellpack-bp06-cell-block-records, cellpack-bp07-local-cell-ordering, cellpack-bp08-warp-tiles, cellpack-bp09-native-runtime-consumers, cellpack-bp10-alternating-refinement, cellpack-bp11-statistical-validation, cellpack-bp12-hardware-cost-autotune, cellpack-bp13-persistence-integration.
- Partial cleanup is available via `todo-cleanup --partial`; include `stale` in `--scope` only when explicitly intended.
