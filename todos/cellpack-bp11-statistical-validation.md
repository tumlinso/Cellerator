---
slug: "cellpack-bp11-statistical-validation"
status: "in_progress"
execution: "idle"
owner: "unassigned"
created_at: "2026-08-14T13:00:00Z"
last_heartbeat_at: "2026-08-17T10:50:48Z"
last_reviewed_at: "2026-08-17T10:50:48Z"
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
- Barrier D is integrated at pushed source checkpoint `0bf9acf`. Phase E may
  consume the frozen CP-BP-08 tile ABI and existing validation foundations
  read-only, publish `CP11_TILE_BOOTSTRAP_READY`, release, and stop without git.

## File Lease

- Released by `codex-cp-bp11-phase-c` at `CP11_HELDOUT_READY` on 2026-08-17.
  The integrated Phase C input is exactly new
  `record_statistical_validation.hh/.cc/_test.cc`, labelled CP-BP-11 Phase C
  root-CMake blocks, and these coordination notes. Existing Phase A, plan,
  evaluator/optimizer, CP-BP-06/07, every `warp_tiles.*`, and component CMake
  file remain unmodified by this stream.

Released by `codex-cp-bp11-phase-e` at `CP11_TILE_BOOTSTRAP_READY` on
2026-08-17. The completed Phase E lease is:

- new `components/CellPack/include/CellPack/tile_statistical_validation.hh`;
- new `components/CellPack/src/tile_statistical_validation.cc`;
- new `components/CellPack/tests/tile_statistical_validation_test.cc`;
- only clearly labelled CP-BP-11 Phase E blocks in root `CMakeLists.txt`;
- CP-BP-11 entries in this ledger, the coordinator, parent roadmap, `todos.md`,
  and `todo-status.md` while holding the shared lock.

The existing statistical/record-validation APIs, every CP-BP-08 tile file,
CP-BP-09 host/CUDA/runtime file, component `CMakeLists.txt`, plan/record/order
files, sampling, optimizer, and evaluator are read-only. CP-BP-09 exclusively
owns its new CUDA consumer files and component-CMake blocks in parallel.

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
- `components/CellPack/include/CellPack/statistical_validation.hh`
- `components/CellPack/include/CellPack/record_statistical_validation.hh`
- `components/CellPack/include/CellPack/warp_tiles.hh`
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
- [x] Add frozen-plan CP-BP-06 record held-out/null adapters and publish
  `CP11_HELDOUT_READY`.
- [x] Phase E: add frozen-plan tile held-out/null metrics and bootstrap physical-
  layout stability summaries; publish `CP11_TILE_BOOTSTRAP_READY`.
- [ ] Phase F: add relearned-plan membership and final runtime stability.

## Blockers

- Phase E is complete at its released gate. Phase F remains closed until
  Barrier E integrates the tile-validation and native-runtime streams; its
  relearned-plan mapping/runtime stability must not be absorbed early.

## Progress Notes

- 2026-08-17: Published `CP11_TILE_BOOTSTRAP_READY`, released every Phase E
  lease, and returned idle without git. Added a versioned pointer-first,
  allocation-free host adapter over one const plan, immutable split/bootstrap/
  null provenance, canonical rows, records, frozen local order, and caller-built
  CP-BP-08 tiles. Held-out and degree-null paths preserve exact canonical row,
  feature, and arbitrary value bytes; report raw encoded/metadata bytes, rows,
  NNZ, tiles, tile-block union, active block references, zero padding, and
  correctness denominators; retain explicit group-versus-cell scope; and reject
  a changed real/null row permutation rather than silently relearning order.
- 2026-08-17: Bootstrap validation binds explicit repeated-row realizations to
  Phase A multiplicities and the frozen execution order, preserves each raw
  replicate packet, and derives deterministic repeat/min/mean/max/sample-SD
  summaries. Zero-NNZ replicates retain raw bytes/counts while rate summaries
  expose zero observations instead of fabricating denominators. Plan mapping
  variability, relearning, runtime summaries, and CUDA implementation remain
  Phase F/out of scope.
- 2026-08-17: Fresh `build-cp-bp11` validation used CUDA 12.9, GNU 13.3.0,
  V100 `sm_70`. Passed `cellPackTileStatisticalValidationTest`,
  `cellPackStatisticalValidationTest`, `cellPackRecordStatisticalValidationTest`,
  `cellPackWarpTilesTest`, `cellPackCellBlockRecordsTest`,
  `cellPackLocalCellOrderingTest`, `cellPackPlannerTest`,
  `cellPackApplyPlanTest`, `cellPackEvaluatorTest`, `cellPackOptimizerTest`, and
  `cellPackInferredPackingPipelineTest`. Under the shared GPU lock,
  `cellPackWarpTilesCudaTest`, `cellPackCellBlockRecordsCudaTest`,
  `cellPackLocalCellOrderingTest`, and `cellPackApplyPlanTest` passed. TODO
  summary, staleness dry-run, and `git diff --check` passed. No CP-BP-11 CUDA
  kernel, sanitizer, runtime benchmark, or performance claim was added.
- 2026-08-17: `codex-cp-bp11-phase-e` claimed the exact host-only Phase E lease
  at pushed coordinator `b76a861a5c21a908b1ed9368fa1f4961dbf42c3b`.
  Concurrent owner `codex-cp-bp09-phase-e` retains the disjoint CUDA consumer,
  component-CMake, and benchmark lease. CP-BP-11 owns only new
  `tile_statistical_validation` host files, labelled root-CMake blocks, and
  locked CP-BP-11 coordination entries; it must publish
  `CP11_TILE_BOOTSTRAP_READY`, release, and stop without git.
- 2026-08-17: `BARRIER_D_INTEGRATED` pushed source checkpoint `0bf9acf` with
  the exact CP-BP-08 host/CUDA tile contract and CP-BP-09 reference API. Phase E
  is now fork-ready for a host-only tile held-out/null/bootstrap adapter in new
  validation files. It must preserve binary sparse incidence, immutable donor/
  sample/study-aware identities, and the existing no-relearning boundary.
- 2026-08-17: Barrier C integrated `CP11_HELDOUT_READY` with the CP-BP-08 host
  tile ABI in pushed source checkpoint `ebe0509` after fresh combined
  validation. CP-BP-11 remains idle and unclaimed; Phase D does not lease any
  validation file, and this stream resumes only for the coordinator-named
  Phase E tile/bootstrap extension.
- 2026-08-17: Published `CP11_HELDOUT_READY` and released every Phase C lease.
  Added a versioned pointer-first record-validation API that accepts one const
  frozen plan, immutable split/training identities, canonical partition CSR and
  CP-BP-06 records. It validates exact canonical feature and arbitrary value-byte
  reconstruction on unseen rows, reports denominator-preserving projected
  bytes/NNZ, metadata/NNZ, blocks/cell, exact correctness and group scope, and
  compares real support against the existing exactly degree-preserving null.
  Zero-NNZ rows preserve raw byte counts without fabricating a storage rate;
  tile/runtime/hardware metrics remain unavailable by construction.
- 2026-08-17: Fresh `build-cp-bp11` configuration used CUDA 12.9.86, GNU 13.3.0,
  Torch models disabled, and `sm_70`. Passed
  `cellPackRecordStatisticalValidationTest`,
  `cellPackStatisticalValidationTest`, `cellPackCellBlockRecordsTest`,
  `cellPackEvaluatorTest`, `cellPackOptimizerTest`, and
  `cellPackInferredPackingPipelineTest`, plus `git diff --check`. Tests cover
  deterministic group/cell splits, nonzero global partitions, empty/zero-NNZ
  rows, exact projection arithmetic, stable frozen-plan/split identities,
  real/null structural separation, exact degree conservation, leakage, source
  identity, value-byte and null-provenance tampering. No GPU runtime or
  benchmark was required or executed.
- 2026-08-17: `codex-cp-bp11-phase-c` claimed the exact record-validation lease
  at pushed base `3925c155de1dab89dd506dd229c97acb96de27a7`. Concurrent
  CP-BP-08 owner `codex-cp-bp08-phase-c` retains its disjoint `warp_tiles` and
  component-CMake lease. This stream owns only new
  `record_statistical_validation` files, labelled root-CMake blocks, and locked
  CP-BP-11 coordination entries; it will publish `CP11_HELDOUT_READY`, release
  to idle, and stop without git operations.
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

- Do not resume CP-BP-11. Barrier E must jointly validate and integrate
  `CP09_RUNTIME_READY` plus `CP11_TILE_BOOTSTRAP_READY`; Phase F remains closed
  until the coordinator explicitly opens it after that checkpoint.

## Phase E Acceptance Boundary

- Assume sparse scRNA binary structural incidence with rows=cells and
  columns=canonical genes. Expression bytes are preserved for reconstruction
  only; do not normalize, log-transform, densify, change feature order, use
  labels to learn packing, or claim donor/study generalization without supplied
  grouping identities.
- Add a versioned allocation-free adapter that accepts one const frozen plan,
  immutable Phase A split/bootstrap/null provenance, canonical source rows, and
  caller-built CP-BP-08 tile views. Validate plan/feature-axis/row-domain/order/
  split/bootstrap/source identities before computing any result.
- Held-out and real-versus-degree-null evaluation must exactly reconstruct
  canonical row/feature/value bytes and report raw denominator-preserving tile
  metrics: encoded and metadata bytes, NNZ, rows, tile count, tile-block union,
  active block references, zero padding, correctness items/mismatches, and
  explicit group-versus-cell-level scope. Do not report runtime throughput.
- Bootstrap evidence evaluates the same frozen plan across caller-materialized
  bootstrap tile realizations bound to existing row multiplicities; it does not
  relearn blocks or order on held-out/null inputs. Preserve every per-replicate
  raw metric packet as authoritative and derive repeat count plus deterministic
  min/mean/max and sample-standard-deviation summaries with explicit zero-
  denominator handling. Mapping variability and relearned-plan stability stay
  in Phase F.
- Tests must cover grouped and cell-level splits, repeated bootstrap rows,
  empty/zero-NNZ/tail tiles, bit 31, non-identity local order, real/null exact
  degree conservation, canonical/value-byte reconstruction, zero denominators,
  deterministic repeat summaries, and rejection of overlapping/tampered plan,
  split, bootstrap, null, source, row-domain, feature-axis, and tile identities.
- Run the Phase A and record-validation tests plus tile host/CUDA, records,
  ordering, plan/apply/evaluator/optimizer, inferred-pipeline, TODO, staleness,
  and diff checks. GPU regressions use the shared GPU lock; no CP-BP-11 CUDA
  kernel or performance benchmark is required. Publish
  `CP11_TILE_BOOTSTRAP_READY`, release, and stop; do not edit CP-BP-08/09,
  implement Phase F relearning/runtime summaries, or perform git operations.

## Done Criteria

- A plan learned on one subset is frozen and evaluated on unseen cells with exact reconstruction.
- Null matrices preserve documented row/column degree tolerances and establish a real-versus-null packing comparison.
- Multiple representative samples report cost/performance stability and membership variability.
- All metrics include denominators, uncertainty/repeats where applicable, exact commands, and relevant existing-layout baselines.
