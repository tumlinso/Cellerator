---
slug: "cellpack-bp11-statistical-validation"
status: "in_progress"
execution: "idle"
owner: "unassigned"
created_at: "2026-08-14T13:00:00Z"
last_heartbeat_at: "2026-08-17T08:17:44Z"
last_reviewed_at: "2026-08-17T08:17:44Z"
stale_after_days: 3
objective: "CP-BP-11: Prove held-out generalization, null separation, and bootstrap stability of inferred packing."
---

# Current Objective

## Summary

Build statistical validation into the feature: held-out cells, degree-preserving null incidence matrices, and bootstrap relearning/stability with correctness and performance metrics.

## Quick Start

- Why this stream exists: demonstrate reusable sparse structure rather than memorization of one row order/sample.
- In scope: deterministic train/held-out splits, frozen-plan evaluation, degree-preserving null generation, bootstrap runs, uncertainty summaries, metric definitions, and correctness gates.
- Out of scope / dependencies: using cell labels to train packing, paper-only analysis, or claiming completed plan/runtime performance before child implementations exist.
- Required skills: `bio-experiments`, `todo-orchestrator`; `cuda` only when GPU kernels/benchmarks are exercised.
- Required references: CP-BP-00, CP-BP-01 sampling provenance, completed plan evaluator, `AGENTS.md`, and benchmark mutex guidance.

## Planning Notes

- Foundational split/null/metric reference work can proceed now in isolated
  validation files. CP-BP-04 and the exact evaluator are complete;
  end-to-end physical/runtime acceptance naturally waits on CP-BP-06/08/09.
- Stability should prioritize achievable cost/performance distributions because exact block memberships may be non-unique.
- Degree-preserving nulls should approximately preserve cell NNZ and gene detection frequency; record deviation rather than hiding it.
- Phase A's reference null uses duplicate-rejecting bipartite double-edge swaps
  and preserves row and column degrees exactly. Provenance records seed,
  requested/attempted/accepted swaps and conservation results; later approximate
  GPU generators, if any, require separately reported deviations.
- Caller-supplied donor/sample/study groups govern splits when available. A
  cell-level split without those identities is explicitly cell-level structural
  validation and cannot claim donor- or study-level generalization.

## CP-BP-06→11 Fork Interlock

- Read `todos/cellpack-bp06-11-parallel-execution.md`. Phase A is integrated;
  CP-BP-11 remains idle throughout Phase B and resumes only when the coordinator
  explicitly opens Phase C.
- If later assigned CP-BP-11 Phase C, claim under
  `/tmp/cellerator-cp-bp06-11-shared.lock`, use `build-cp-bp11`, and own only
  new statistical-validation/metric/null/provenance files and tests. Consume
  CP-BP-01/04 APIs read-only; do not edit sampling, optimizer, record, tile, or
  runtime implementations.
- Consume the integrated foundations without redefining them, publish the
  coordinator-named Phase C gate, release all leases, and become idle. Do not
  remain claimed while waiting on CP-BP-08/09. No git operations.

## File Lease

- Released at `CP11_FOUNDATIONS_READY` by `codex-cp-bp11-phase-a` on
  2026-08-16. Phase C is ready and unclaimed. On assignment, atomically lease
  exactly:

  - new
    `components/CellPack/include/CellPack/record_statistical_validation.hh`;
  - new `components/CellPack/src/record_statistical_validation.cc`;
  - new
    `components/CellPack/tests/record_statistical_validation_test.cc`;
  - only clearly labelled CP-BP-11 Phase C target blocks in root
    `CMakeLists.txt`;
  - this ledger and CP-BP-11 entries in the coordinator, `todos.md`,
    `todo-status.md`, and parent roadmap while holding the shared lock.

  Record the unique owner and full current pushed `origin/main` claim hash under
  the shared lock before editing. Existing `statistical_validation.*`, plan/evaluator/optimizer,
  CP-BP-06/07 representation/ordering files, all `warp_tiles.*`, and component
  `CMakeLists.txt` are read-only.

## Assumptions

- Train/held-out identity and sampling provenance are immutable and auditable.
- Metrics are reported with denominators and dataset/partition/sample context.

## Suggested Skills

- `bio-experiments`
- `todo-orchestrator`
- `cuda`

## Useful Reference Files

- `include/Cellerator/compute/sampling.hh`
- `tests/sampling_runtime_test.cc`
- `todos/cellpack-packing-plan-evaluator.md`
- `bench/benchmark_mutex.hh`

## Plan

1. Phase A: define metric schema and immutable group-aware
   split/bootstrap/null provenance.
2. Phase A: implement/reference-test exact degree-preserving sparse-incidence
   randomization and conservation/leakage checks; publish
   `CP11_FOUNDATIONS_READY` and stop at Barrier A.
3. Phase C: add a record-level adapter that evaluates one frozen plan on
   immutable held-out rows, reports denominator-preserving CP-BP-06 record
   metrics and exact reconstruction, and compares real versus the existing
   degree-preserving null without relearning; publish `CP11_HELDOUT_READY`,
   release, and stop.
4. Phase E: extend the same identities/schema with CP-BP-08 tile and bootstrap
   evidence without depending on final runtime measurements.
5. Phase F: relearn across bootstrap samples and summarize cost/performance
   stability plus mapping variability.

## Tasks

- [x] Define metrics: bytes/NNZ, metadata/NNZ, blocks/cell, tile-union size, padding/NNZ where relevant, compression, preprocessing throughput, runtime throughput/bandwidth, and correctness.
- [x] Implement deterministic held-out and bootstrap provenance without modifying CP-BP-01-owned files.
- [x] Add degree-preserving null reference generator and conservation tests.
- [ ] Integrate completed plan/tile/runtime stages as they land.

## Blockers

- No blocker for Phase C frozen-plan/CP-BP-06 record-level held-out adapters:
  Barrier B source checkpoint `eeb8c39` is pushed and both input contracts are
  stable. Full tile/runtime/bootstrap acceptance still waits on later phases.

## Progress Notes

- 2026-08-17: Phase C's exact unclaimed lease and scientific boundary are
  frozen against the current pushed coordinator base. New record-level adapters must consume the
  Phase A metric/split/null contract read-only, never relearn on held-out/null
  inputs, preserve sparse binary rows=cells and canonical-gene columns, retain
  immutable identities, and report only currently measurable record metrics.
  Tile, runtime, hardware, and final bootstrap-stability evidence remain later
  phases.
- 2026-08-17: Barrier B opened Phase C from pushed source checkpoint `eeb8c39`.
  CP-BP-11 may now add frozen-plan and CP-BP-06 record-level held-out/null
  metric adapters in files disjoint from CP-BP-08, publish
  `CP11_HELDOUT_READY`, release, and stop. Tile/runtime metrics remain later
  scope and no phase is claimed by the integrator.
- 2026-08-16: Barrier A jointly validated and integrated these foundations with
  the CP-BP-06 host record ABI. CP-BP-11 stays idle; it is not part of the next
  Phase B fork and resumes only for the coordinator-named Phase C.
- 2026-08-16: Published `CP11_FOUNDATIONS_READY`. Added the pointer-first
  `statistical_validation` CPU/reference library with versioned raw metric
  denominators, separate preprocessing/runtime repeat counts, deterministic
  exact-count row/group held-out splits, group-aware bootstrap multiplicities,
  immutable identities, and validators that reject leakage or tampering.
- 2026-08-16: Added the deterministic duplicate-rejecting bipartite double-edge
  swap null reference. It preserves row and canonical-feature degrees exactly,
  records source/output identities plus requested/attempted/accepted swaps, and
  reports an unreached mixing target instead of relaxing conservation.
- 2026-08-16: Fresh `build-cp-bp11` configuration and target build passed with
  CUDA 12.9.86, GNU 13.3.0, and `sm_70`. Passed
  `cellPackStatisticalValidationTest`, `cellPackEvaluatorTest`,
  `cellPackOptimizerTest`, `samplingMaterializationRuntimeTest`, and
  `git diff --check`. No GPU validation or benchmark was required for this
  CPU/reference phase.
- 2026-08-16: Phase A claimed by `codex-cp-bp11-phase-a` after rereading the
  concurrently claimed CP-BP-06 lease. CP-BP-11 owns isolated statistical
  validation files and root-CMake target blocks only; CP-BP-06 owns the
  component CMake and packing-plan/record seams.
- 2026-08-14: Added as a missing ready workstream; existing sampling provenance and exact plan evaluator are reusable inputs, but no statistical validation implementation was found.
- 2026-08-16: Reconciliation credits partial foundations without claiming this
  stream complete: CP-BP-01 provides deterministic disjoint hash-quantile
  splits, replayable immutable provenance, density-stratified sampling, sampled
  CSR, and feature support; CP-BP-04 freezes a compatibility-checked semantic
  plan and the supplied-plan evaluator measures exact occupancy/cost. No
  degree-preserving null generator, bootstrap relearning/stability summary,
  held-out orchestration/report, or CP-BP-11-specific metric schema was found.

## Next Actions

- If explicitly assigned, claim only the recorded Phase C lease under the
  shared lock. Add frozen-plan/CP-BP-06 record-level held-out and null metric
  adapters with split/null/source/plan identity, bytes/NNZ, metadata/NNZ,
  blocks/cell, exact reconstruction, and tests for disjoint/group-aware splits,
  cell-level-only labeling, zero denominators, empty inputs, exact arithmetic,
  real/null comparison, overlap/tamper rejection, and absence of relearning.
  Validate from `build-cp-bp11` with Phase A, record, and plan/evaluator
  regressions; publish `CP11_HELDOUT_READY`, release to idle, and stop without
  git for Barrier C. Do not begin merely because this setup is committed; wait
  for user assignment.

## Done Criteria

- A plan learned on one subset is frozen and evaluated on unseen cells with exact reconstruction.
- Null matrices preserve documented row/column degree tolerances and establish a real-versus-null packing comparison.
- Multiple representative samples report cost/performance stability and membership variability.
- All metrics include denominators, uncertainty/repeats where applicable, exact commands, and relevant existing-layout baselines.
