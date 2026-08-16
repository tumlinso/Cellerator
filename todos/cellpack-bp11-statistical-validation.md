---
slug: "cellpack-bp11-statistical-validation"
status: "planned"
execution: "ready"
owner: "unassigned"
created_at: "2026-08-14T13:00:00Z"
last_heartbeat_at: "2026-08-16T19:45:16Z"
last_reviewed_at: "2026-08-16T19:45:16Z"
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

- Read `todos/cellpack-bp06-11-parallel-execution.md`. Phase A may run now in
  parallel only with CP-BP-06 Phase A.
- If assigned CP-BP-11 now, claim under
  `/tmp/cellerator-cp-bp06-11-shared.lock`, use `build-cp-bp11`, and own only
  new statistical-validation/metric/null/provenance files and tests. Consume
  CP-BP-01/04 APIs read-only; do not edit sampling, optimizer, record, tile, or
  runtime implementations.
- Publish `CP11_FOUNDATIONS_READY`, release all leases, and become
  `in_progress/idle` at Barrier A. Resume only for the coordinator-named phase;
  do not remain claimed while waiting on CP-BP-06/08/09. No git operations.

## File Lease

_Unclaimed._ Record exact intended paths here atomically at claim time.

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
3. Phases C/E: evaluate frozen plans and available record/tile/runtime stages on
   held-out cells and compare real versus null achievable packing.
4. Phase F: relearn across bootstrap samples and summarize cost/performance
   stability plus mapping variability.

## Tasks

- [ ] Define metrics: bytes/NNZ, metadata/NNZ, blocks/cell, tile-union size, padding/NNZ where relevant, compression, preprocessing throughput, runtime throughput/bandwidth, and correctness.
- [ ] Implement deterministic held-out and bootstrap provenance without modifying CP-BP-01-owned files.
- [ ] Add degree-preserving null reference generator and conservation tests.
- [ ] Integrate completed plan/tile/runtime stages as they land.

## Blockers

- No blocker for foundational validation contracts/reference generators.
- Full acceptance waits on CP-BP-06, CP-BP-08, and CP-BP-09 outputs; the
  frozen-plan/evaluator prerequisite is complete.

## Progress Notes

- 2026-08-14: Added as a missing ready workstream; existing sampling provenance and exact plan evaluator are reusable inputs, but no statistical validation implementation was found.
- 2026-08-16: Reconciliation credits partial foundations without claiming this
  stream complete: CP-BP-01 provides deterministic disjoint hash-quantile
  splits, replayable immutable provenance, density-stratified sampling, sampled
  CSR, and feature support; CP-BP-04 freezes a compatibility-checked semantic
  plan and the supplied-plan evaluator measures exact occupancy/cost. No
  degree-preserving null generator, bootstrap relearning/stability summary,
  held-out orchestration/report, or CP-BP-11-specific metric schema was found.

## Next Actions

- Begin with metric/provenance contracts and CPU/reference null conservation in isolated CellPack validation files; coordinate read-only use of CP-BP-01 sampling contracts.

## Done Criteria

- A plan learned on one subset is frozen and evaluated on unseen cells with exact reconstruction.
- Null matrices preserve documented row/column degree tolerances and establish a real-versus-null packing comparison.
- Multiple representative samples report cost/performance stability and membership variability.
- All metrics include denominators, uncertainty/repeats where applicable, exact commands, and relevant existing-layout baselines.
