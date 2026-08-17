# Todo Status

## Summary
Use this file as the quick pickup register for `todos.md` workstreams.
- `ready`: planned work that can be started now.
- `claimed`: currently being written; choose another stream.
- `idle`: unfinished but resumable; safe to pick up.
- `closed`: completed or removed from pickup rotation.

## Workstreams
- `cellpack-bp00-05-integration-audit` | status: done | execution: closed | owner: codex-cp-bp00-05-integration | file: `todos/cellpack-bp00-05-integration-audit.md` | next: Complete; the real sampling→support→candidate→score→optimize→apply chain is covered and CP-BP-06 remains untouched.
- `cellpack-bp06-11-parallel-execution` | status: done | execution: closed | owner: codex-cp-bp10-11-serial | file: `todos/cellpack-bp06-11-parallel-execution.md` | next: Complete at pushed Barrier F source checkpoint `2cfa5c8`; historical interlocks remain for audit only.
- `cellpack-data-inferred-block-packing-roadmap` | status: in_progress | execution: claimed | owner: coordination | file: `todos/cellpack-data-inferred-block-packing-roadmap.md` | next: Do not implement from this parent. CP-BP-13 is the next ready unclaimed child and begins with a read-only audit.
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
- `cellpack-bp10-alternating-refinement` | status: done | execution: closed | owner: codex-cp-bp10-11-serial | file: `todos/cellpack-bp10-alternating-refinement.md` | next: Complete at pushed Barrier F checkpoint `2cfa5c8`; CP-BP-12 may consume the measured objective seam.
- `cellpack-bp11-statistical-validation` | status: done | execution: closed | owner: codex-cp-bp10-11-serial | file: `todos/cellpack-bp11-statistical-validation.md` | next: Complete at pushed Barrier F checkpoint `2cfa5c8`; final mapping/runtime stability is available.
- `cellpack-bp12-hardware-cost-autotune` | status: done | execution: closed | owner: codex-cp-bp12 | file: `todos/cellpack-bp12-hardware-cost-autotune.md` | next: Complete; CP-BP-13 may consume the stable v1 plan-selection policy.
- `cellpack-bp13-persistence-integration` | status: planned | execution: ready | owner: unassigned | file: `todos/cellpack-bp13-persistence-integration.md` | next: Claim from pushed Cellerator/CellStack, then perform the read-only compatibility/ownership audit before serialization edits.

## Staleness Review
- Fresh: 4
- Aging: 0
- Stale candidates: 0
- Stale: 0
- Superseded: 0
- `cellpack-bp00-05-integration-audit` | done | age: 0.9d | threshold: 3d | reason: Terminal workstream.
- `cellpack-bp06-11-parallel-execution` | done | age: 0.0d | threshold: 7d | reason: Terminal workstream.
- `cellpack-bp01-support-extraction` | done | age: 1.0d | threshold: 3d | reason: Terminal workstream.
- `cellpack-bp02-candidate-discovery` | done | age: 1.0d | threshold: 3d | reason: Terminal workstream.
- `cellpack-bp03-exact-merge-cost` | done | age: 1.0d | threshold: 3d | reason: Terminal workstream.
- `cellpack-bp04-packing-plan-optimizer` | done | age: 1.0d | threshold: 3d | reason: Terminal workstream.
- `cellpack-bp05-apply-frozen-plan` | done | age: 0.9d | threshold: 7d | reason: Terminal workstream.
- `cellpack-bp06-cell-block-records` | done | age: 0.3d | threshold: 7d | reason: Terminal workstream.
- `cellpack-bp07-local-cell-ordering` | done | age: 0.3d | threshold: 7d | reason: Terminal workstream.
- `cellpack-bp08-warp-tiles` | done | age: 0.2d | threshold: 7d | reason: Terminal workstream.
- `cellpack-bp09-native-runtime-consumers` | done | age: 0.0d | threshold: 7d | reason: Terminal workstream.
- `cellpack-bp10-alternating-refinement` | done | age: 0.0d | threshold: 7d | reason: Terminal workstream.
- `cellpack-bp11-statistical-validation` | done | age: 0.0d | threshold: 3d | reason: Terminal workstream.

## Cleanup Status
- Cleanup mode is explicit only.
- Safe to call `todo-cleanup`: no, active workstreams: cellpack-data-inferred-block-packing-roadmap, cellpack-packing-plan-cuda-evaluator, cellpack-bp13-persistence-integration.
- Partial cleanup is available via `todo-cleanup --partial`; include `stale` in `--scope` only when explicitly intended.
