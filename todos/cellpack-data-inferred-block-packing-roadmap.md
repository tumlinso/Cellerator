---
slug: "cellpack-data-inferred-block-packing-roadmap"
status: "in_progress"
execution: "claimed"
owner: "coordination"
created_at: "2026-08-14T13:00:00Z"
last_heartbeat_at: "2026-08-17T08:39:21Z"
last_reviewed_at: "2026-08-17T08:39:21Z"
stale_after_days: 7
objective: "CP-BP-00 parent coordination epic for the complete offline compiler, compact tile format, native runtime, validation, autotuning, and persistence roadmap; do not implement from the parent."
---

# Current Objective

## Summary
Track the parent roadmap for a two-layer system: an offline compiler learns a reusable global gene-block grammar and durable `PackingPlan`; a GPU-oriented conversion/runtime path applies the frozen plan, orders cells locally, emits compact 32-gene by 32-cell tiles, and consumes those tiles without unpacking to CSR or BELL.

## Quick Start
- Why this stream exists: preserve shared architecture, ordering, ownership, and concurrency context across CP-BP-01 through CP-BP-13.
- In scope: coordination, cross-step contracts, dependency reconciliation, and durable design constraints.
- Out of scope: implementation; pick an executable child workstream instead.
- Required skills: `todo-orchestrator`; implementation children generally require `cuda`, and CP-BP-11 also requires `bio-experiments`.
- Required references: `AGENTS.md`, `components/CellPack/AGENTS.md`, `style_hint.md`, this file, `todos.md`, and `todo-status.md`.
- Parallel CP-BP-06→11 execution additionally requires
  `todos/cellpack-bp06-11-parallel-execution.md`; its gates and integration
  barriers are authoritative over optimistic downstream pickup.

## Planning Notes
- Parent ID: CP-BP-00. Child IDs CP-BP-01 through CP-BP-13 map one-to-one to roadmap Steps 1 through 13.
- The compiler objective is the actual target codec/layout cost, not correlation, PCA, biological clustering, Jaccard, or a neural surrogate.
- Structural input begins as `A[cell,gene] = 1 iff X[cell,gene] != 0`; expression magnitudes are not required for the first optimizer.
- Conceptual exact primitive: `merge_gain(A,B) = cost(A) + cost(B) - cost(A union B)`, with `cost` evolving from exact encoded bytes to a replaceable storage-plus-execution model.
- The gene organization is a reusable block grammar, not merely a permutation. Canonical biological gene IDs and inverse mappings remain explicit.
- This is not Blocked-ELL/BELL with a smarter ordering. It is a compact variable-payload tile representation whose metadata and access rules must be fully specified.
- Offline combinatorial optimization may be host-side. Repeated full-dataset transformation and runtime consumption should be GPU-oriented and use CUB/library primitives for generic sort/scan work unless measurements justify custom kernels.

## Assumptions
- Cellerator owns discovery semantics, plan inference, transformation, CUDA representation, and native consumption.
- CellShard will eventually own durable `.cspack` serialization, validation, fetch, and upload integration, but not plan discovery or semantic definition.
- The completed `cellpack-packing-plan-evaluator` is reusable evidence and infrastructure, not CP-BP-04 completion and not a physical codec.
- CP-BP-01 through CP-BP-05 are complete and closed. Barrier A integrated
  CP-BP-06's host record contract and CP-BP-11's validation foundations;
  CP-BP-06 and CP-BP-07 are complete and closed in pushed Barrier B source
  checkpoint `eeb8c39`. Both Phase C gates are published and await Barrier C
  integration.
- Every performance claim follows a CPU/reference correctness test and a relevant existing-layout baseline; benchmarks/profilers use the repository mutex.

## Suggested Skills
- `todo-orchestrator`: maintain ownership, pickup state, and dependencies.
- `cuda`: implementation and validation of GPU-facing children on V100 `sm_70`.
- `bio-experiments`: CP-BP-11 statistical and omics validation.

## Useful Reference Files
- `components/CellPack/AGENTS.md`
- `todos/cellpack-packing-plan-evaluator.md`
- `include/Cellerator/compute/sampling.hh`
- `src/compute/dataset/sampling.cc`
- `tests/sampling_runtime_test.cc`
- `style_hint.md`
- `optimization.md`
- `scope.md`

## Plan
1. CP-BP-01 builds sampled structural support.
2. CP-BP-02 proposes candidate gene pairs/blocks; CP-BP-03 scores them exactly.
3. CP-BP-04 forms a durable constrained global plan.
4. CP-BP-05 and CP-BP-06 apply that plan and emit compact per-cell records.
5. CP-BP-07 locally orders cells; CP-BP-08 emits warp tiles; CP-BP-09 consumes them natively.
6. CP-BP-10 alternates gene/cell refinement; CP-BP-11 proves generalization; CP-BP-12 learns hardware cost; CP-BP-13 integrates persistence/execution ownership.

## Tasks
- [x] CP-BP-01 representative sparse-support extraction.
- [x] CP-BP-02 candidate gene-block discovery.
- [x] CP-BP-03 exact packing-cost and merge-gain scoring.
- [x] CP-BP-04 global gene-block optimization and durable semantic `PackingPlan`.
- [x] CP-BP-05 apply a frozen plan to full partitions.
- [x] CP-BP-06 emit compact per-cell block records.
- [x] CP-BP-07 infer local cell order from active-block signatures.
- [ ] CP-BP-08 build warp-oriented 32-cell tiles.
- [ ] CP-BP-09 implement native packed-runtime consumers.
- [ ] CP-BP-10 alternating/refined packing optimization.
- [ ] CP-BP-11 statistical legitimacy and anti-overfitting validation.
- [ ] CP-BP-12 hardware-aware cost model and autotuning.
- [ ] CP-BP-13 persistence and execution integration.

## Blockers
- CP-BP-08/11 Phase C implementations are gate-complete and idle; Barrier C
  integration is the sole next action. CP-BP-09/10/12/13 retain their later
  tile/runtime dependencies.
- CP-BP-12 needs measured CP-BP-08/09 kernels and cannot select a hardware objective yet.

## Progress Notes
- 2026-08-17: CP-BP-08 published `CP08_HOST_ABI_READY`, released its host-tile
  lease, and returned idle with a tested pointer-first compact tile contract.
  CP-BP-11 is already idle at `CP11_HELDOUT_READY`; Barrier C must integrate
  both before Phase D or CP-BP-09 opens.
- 2026-08-17: CP-BP-11 published `CP11_HELDOUT_READY`, released its Phase C
  lease, and returned idle with exact record-level held-out/null validation.
  CP-BP-08 remains actively claimed; Barrier C waits for its host gate before
  integrating both streams. CP-BP-11's later tile/bootstrap/runtime acceptance
  remains open and was not absorbed into Phase C.
- 2026-08-17: CP-BP-11 Phase C is actively claimed by
  `codex-cp-bp11-phase-c` at base `3925c15` through its exact new
  record-statistical-validation and root-CMake lease. CP-BP-08 remains actively
  claimed on disjoint host-tile files/component-CMake blocks; both children
  stop without git at their respective Phase C gates for Barrier C.
- 2026-08-17: CP-BP-08 Phase C is actively claimed by
  `codex-cp-bp08-phase-c` at pushed base `3925c15` under the exact host-only
  lease. CP-BP-11 remains idle and independently claimable through its disjoint
  files/CMake seam. No Phase D, CP-BP-09, or git integration is authorized.
- 2026-08-17: Phase C is fully fork-specified but not started. CP-BP-08 host
  tiles and CP-BP-11 record-level held-out adapters have exact disjoint new-file,
  CMake-block, build, read-only-input, validation, publication, release, and
  Barrier C rules from the same current pushed coordinator base. Both remain unassigned pending the
  user's explicit branch assignments.
- 2026-08-17: `BARRIER_B_INTEGRATED` records Cellerator source checkpoint
  `eeb8c39`. CP-BP-06/07 are closed; CP-BP-08 host tiles and CP-BP-11 record-
  level held-out adapters are the next unclaimed parallel pair.
- 2026-08-17: Combined Barrier B validation accepted the exact asynchronous
  CUDA record builder and bounded host/CUDA local-order contracts. Both child
  streams are closed; the checkpoint is pushed and recorded before CP-BP-08 is
  reactivated.
- 2026-08-16: Barrier A integrated the tested versioned CP-BP-06 width-32 host
  record ABI/reference with CP-BP-11 metric, split/bootstrap provenance, and
  exact degree-preserving null foundations. The next legal fork pair is
  CP-BP-06 Phase B plus CP-BP-07; neither is claimed by this integration.
- 2026-08-16: Added a checkpointed one-worktree execution schedule for
  CP-BP-06 through CP-BP-11. Phase A parallelizes CP-BP-06's exact record
  ABI/CPU reference with CP-BP-11's metrics, group-aware split provenance, and
  exact degree-preserving null reference. Later CP-BP-07/08/09/10 claims are
  tied to tested host/device handoff gates rather than calendar order.
- 2026-08-16: A serial CP-BP-00→05 audit proved the real public APIs compose in
  one deterministic test from sampled support through exact scored candidates,
  full-domain optimization, frozen-plan application, and exact canonical/value
  reconstruction. The CP-BP-01 sample/mapping identity now reaches CP-BP-04,
  and full-domain plans reject partial evaluator sources. CP-BP-06 scope was
  not entered.
- 2026-08-16: CP-BP-05 completed pointer-first CPU/CUDA application of
  full-domain frozen plans, exact canonical reconstruction, CUB segmented
  ordering, focused validation, and a serialized V100 benchmark. CP-BP-06 is
  now ready; compact record emission remains deliberately unimplemented.
- 2026-08-16: Repository reconciliation verified CP-BP-01, CP-BP-02, and
  CP-BP-04 against focused CPU/CUDA tests and serialized V100 benchmarks. They
  satisfied their child acceptance criteria in the dirty worktree based at
  `1ebb734`; their implementation was checkpointed as `597a3eb`. Older
  supplied-plan coordinate/evaluator/gating scaffolding
  remains distinct from inferred-plan application and compact physical packing.
- 2026-08-16: CP-BP-03 completed a reusable versioned CPU/CUDA exact candidate
  merge-cost API and optimizer-valid exact relation output. CP-BP-04's private
  support proxies and whole-plan evaluator remain distinct consumers/seams;
  singleton pair gain is not an exact score for later merged blocks. CP-BP-11
  can reuse deterministic disjoint sampling/replay
  provenance and frozen-plan/evaluator contracts, but null generation,
  bootstrap stability, and held-out reporting are not implemented.
- 2026-08-14: Inspected Cellerator/root guidance, current root and Cellerator ledgers, active worktree state, the completed exact plan evaluator, CellPack guidance, and new sampling source/tests.
- 2026-08-14: Reconciled CP-BP-01 with existing untracked deterministic sampling work and CP-BP-04 with the completed evaluator plus user-confirmed parallel optimizer work; neither was marked complete.
- 2026-08-14: Added child ledgers for the complete roadmap and encoded safe parallel seams, prerequisite blockers, architectural invariants, and acceptance criteria.
- 2026-08-14: Completed CP-BP-01 with deterministic sampled CSR, exact CPU/CUDA gene-major support bitsets, detected-cell counts, immutable provenance/global-row mapping, boundary/overflow tests, and a V100 full-size allocation smoke.
- 2026-08-14: Completed CP-BP-04 v1 supplied-candidate optimization and immutable semantic `frozen_packing_plan`; CP-BP-05 was reactivated. The later CP-BP-03 implementation and serial integration audit supplied the production exact-relation adapter without altering the optimizer/frozen-plan boundary.
- 2026-08-14: CP-BP-02 completed deterministic SplitMix64-v1 global-row MinHash, configurable LSH, bounded oversized-bucket handling, CUB grouping/deduplication, canonical host candidate pairs, CPU/GPU exact tests, and a serialized V100 smoke benchmark. CP-BP-03 was not started.

## Next Actions
- Do not implement from this parent. Appoint the Barrier C integrator for the
  combined CP-BP-08/11 checkpoint. Do not begin
  CP-BP-08 CUDA Phase D or CP-BP-09 early.

## Done Criteria
- Every child is `done/closed`, exact reconstruction and numerical equivalence pass, held-out/null/stability validation is recorded, hardware-aware benchmarks justify the selected layout, and CellShard/Cellerator persistence ownership is integrated without per-minibatch repacking.
