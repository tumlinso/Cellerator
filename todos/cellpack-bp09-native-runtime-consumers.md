---
slug: "cellpack-bp09-native-runtime-consumers"
status: "in_progress"
execution: "idle"
owner: "unassigned"
created_at: "2026-08-14T13:00:00Z"
last_heartbeat_at: "2026-08-17T10:14:49Z"
last_reviewed_at: "2026-08-17T10:14:49Z"
stale_after_days: 7
objective: "CP-BP-09: Execute directly from compact warp tiles without unpacking to CSR or BELL."
---

# Current Objective

## Summary

Implement benchmark-driven native consumers that use shared block IDs, cell/gene masks, rank/popcount, and predication directly over CP-BP-08 tiles.

## Quick Start

- Why this stream exists: the packed format is useful only if kernels consume it without conversion back to existing sparse formats.
- In scope: exact reference operation(s), direct packed consumers, occupancy-based dispatch experiments, numerical equivalence, throughput/bandwidth, and relevant baselines.
- Out of scope / dependencies: hidden unpacking, a speculative universal dispatch policy, per-cell launches, and unbenchmarked custom-kernel claims.
- Required skills: `cuda`, `todo-orchestrator`.
- Required references: CP-BP-00, CP-BP-08, `AGENTS.md`, `components/CellPack/AGENTS.md`, `optimization.md`, and benchmark mutex guidance.

## Planning Notes

- Candidate execution paths include lane-per-cell, compact active lanes, subwarp cooperation, and dense/sparse occupancy dispatch. Select only with measurements on supported V100 `sm_70` hardware.
- Compare against canonical/reference math and relevant CSR/current Cellerator sparse layouts, not only an intentionally weak baseline.
- The first and only v1 operation is canonical feature-weighted row reduction,
  `y[row] = sum(value * weight[canonical_feature])`. Broader SpMM/operator
  coverage is future scope after this direct path is proven.
- V1 numeric semantics use configured `cellerator::real::storage_t` tile/CSR
  values, configured `real::compute_t` canonical-feature weights, and configured
  `real::accum_t` accumulation/output. A tile whose `value_size_bytes` differs
  from `sizeof(real::storage_t)` is rejected; CP-BP-09 does not weaken CP-BP-08's
  arbitrary-byte storage contract.

## CP-BP-06→11 Fork Interlock

- Read `todos/cellpack-bp06-11-parallel-execution.md`. Barrier D source
  checkpoint `0bf9acf` integrates `CP08_DEVICE_READY` and
  `CP09_REFERENCE_READY`; Phase E is fork-ready but unclaimed.
- Phase D may claim only the reference/API portion, use `build-cp-bp09`, publish
  `CP09_REFERENCE_READY`, release, and become idle. Device implementation waits
  for `CP08_DEVICE_READY` and Barrier D.
- Phase E implements one direct V100 consumer without CSR/BELL reconstruction,
  extra operations, per-cell launches, or universal dispatch. Publish
  `CP09_RUNTIME_READY`, release/close, and perform no git operation.

## File Lease

_Released at `CP09_REFERENCE_READY` by `codex-cp-bp09-phase-d` on 2026-08-17._
The completed Phase D lease was exactly:

- new
  `components/CellPack/include/CellPack/feature_weighted_row_reduction.hh`;
- new `components/CellPack/src/feature_weighted_row_reduction.cc`;
- new
  `components/CellPack/tests/feature_weighted_row_reduction_test.cc`;
- only clearly labelled CP-BP-09 Phase D target blocks in root `CMakeLists.txt`;
- this ledger and CP-BP-09 entries in the coordinator, `todos.md`,
  `todo-status.md`, and parent roadmap while holding the shared lock.

No implementation lease is active. On exact assignment, the available Phase E
lease is:

- new
  `components/CellPack/include/CellPack/feature_weighted_row_reduction_cuda.hh`;
- new
  `components/CellPack/src/feature_weighted_row_reduction_cuda.cu`;
- new
  `components/CellPack/tests/feature_weighted_row_reduction_cuda_test.cu`;
- new
  `components/CellPack/bench/feature_weighted_row_reduction_bench.cu`;
- only clearly labelled CP-BP-09 Phase E blocks in
  `components/CellPack/CMakeLists.txt`;
- CP-BP-09 entries in this ledger, the coordinator, parent roadmap, `todos.md`,
  and `todo-status.md` while holding the shared lock.

Root `CMakeLists.txt`, the frozen host `feature_weighted_row_reduction.hh/.cc`,
all `warp_tiles*`, plan/record/order files, and every CP-BP-11 validation file
are read-only. CP-BP-11 exclusively owns its new tile-validation files and root-
CMake blocks in the parallel branch.

## Assumptions

- CP-BP-08 exposes an integrated pointer-first host tile view with complete
  bounds and rank semantics; Phase D must not assume CUDA construction details.
- The first operation should be narrow enough to establish native consumption and numerical equivalence before broader operator coverage.

## Suggested Skills

- `cuda`
- `todo-orchestrator`

## Useful Reference Files

- `components/CellPack/include/CellPack/gating_cuda.cuh`
- `components/CellPack/src/gating_cuda.cu`
- `components/CellPack/include/CellPack/feature_weighted_row_reduction.hh`
- `components/CellPack/include/CellPack/warp_tiles_cuda.hh`
- `src/compute/sparse/ops/`
- `bench/benchmark_mutex.hh`

## Plan

1. Define the canonical feature-weighted row-reduction reference and packed runtime view/scratch contracts.
2. Implement direct correctness path with no unpacking.
3. Benchmark occupancy-dependent execution alternatives under the mutex.
4. Document selected dispatch conditions, limiter, tolerance, and reference fallback.

## Tasks

- [x] Wait for the stable CP-BP-08 host tile ABI and Barrier C.
- [x] Phase D: freeze the direct packed consumer contract and CPU/canonical
  feature-weighted row-reduction reference.
- [ ] Phase E: implement the direct CUDA consumer and runtime benchmark from
  pushed Barrier D checkpoint `0bf9acf`.
- [ ] Explore and benchmark occupancy dispatch paths.
- [ ] Record header benchmark justification for custom GPU math.

## Blockers

- None for the exact Phase E lease. The stream is idle/unclaimed and may begin
  only after explicit assignment.

## Progress Notes

- 2026-08-17: `BARRIER_D_INTEGRATED` pushed source checkpoint `0bf9acf` after a
  fresh combined CUDA 12.9.86/GNU 13.3.0 `sm_70` build. Phase E is now
  fork-ready: implement exactly one direct configured-precision weighted-row-
  reduction CUDA consumer, focused exact/tolerance tests, sanitizer coverage,
  and a serialized V100 benchmark. CP-BP-08 and the Phase D host contract are
  frozen read-only inputs.
- 2026-08-17: Published `CP09_REFERENCE_READY`, released every Phase D lease,
  and returned idle without git operations. Added a versioned trivially-copyable
  pointer-first plan/tile/weight contract with exact plan/tile/weight-generation,
  feature-axis, row-domain, configured storage/compute/accumulator, capacity,
  alias, and result identities. Outputs are always canonical partition-local
  rows despite CP-BP-07 execution order.
- 2026-08-17: Added allocation-free canonical CSR, compact-record, and direct
  tile host evaluators for the sole v1 operation
  `y[row] = sum(value * weight[canonical_feature])`. Direct tile evaluation
  traverses dictionaries, cell/gene masks, rank-ordered compact values, and plan
  mappings without decode or CSR/BELL materialization. Canonical and packed
  accumulation orders use the versioned absolute-plus-relative comparison rule.
- 2026-08-17: Fresh `build-cp-bp09` used CUDA 12.9.86, GNU 13.3.0, Torch models
  disabled, and `sm_70`. Passed `cellPackFeatureWeightedRowReductionTest`,
  `cellPackWarpTilesTest`, `cellPackCellBlockRecordsTest`,
  `cellPackReconstructionTest`, `cellPackPlannerTest`, `cellPackEvaluatorTest`,
  `cellPackOptimizerTest`, `cellPackApplyPlanTest`, and
  `cellPackInferredPackingPipelineTest`. The last two relevant CUDA regressions
  ran under the shared GPU lock; warning-clean host syntax checks and
  `git diff --check` passed. No new GPU runtime, sanitizer, or benchmark was
  required or executed.
- 2026-08-17: `codex-cp-bp09-phase-d` claimed the exact reference/API lease at
  pushed base `fe095fb6d6592a0194b0a86f13f0421e23081cd0`. CP-BP-08 remains
  independently idle/unassigned with its disjoint CUDA tile and component-CMake
  lease. This stream owns only new `feature_weighted_row_reduction` host files,
  labelled root-CMake blocks, and locked CP-BP-09 coordination entries; it must
  publish `CP09_REFERENCE_READY`, release, and stop without git operations.
- 2026-08-17: Barrier C checkpoint `ebe0509` integrates the exact compact tile
  host ABI. Published a disjoint Phase D lease for one allocation-free
  CPU/canonical reference and pointer-first direct-consumer contract for
  `y[row] = sum(value * weight[canonical_feature])`. This phase must preserve
  canonical feature IDs and canonical output row identity, use configured
  storage/compute/accumulator types, define deterministic accumulation,
  tolerance, capacity, identity, and error semantics, publish
  `CP09_REFERENCE_READY`, release, and stop without CUDA runtime code or
  performance claims.
- 2026-08-14: Added as a missing blocked workstream; existing coordinate-based oracle-gating CUDA is scaffolding/reference, not this packed runtime.

## Next Actions

- Await exact assignment “You are assigned CP-BP-09 Phase E.” Then claim only
  the published Phase E lease at current pushed `origin/main`, use
  `build-cp-bp09`, publish `CP09_RUNTIME_READY`, release to idle, and stop
  without git operations for Barrier E.

## Phase E Acceptance Boundary

- Provide an asynchronous caller-stream API over device-resident CP-BP-08
  tiles, canonical-feature weights, and caller-owned canonical-row output. No
  allocation, transfer, device-wide synchronization, CSR/BELL reconstruction,
  per-cell launch, or runtime feature lookup is allowed in the hot path.
- Preserve the Phase D configured storage/compute/accumulator types, immutable
  identities, canonical output order, capacity/alias checks, CUDA error
  propagation, and versioned absolute-plus-relative comparison rule.
- Start with one lane-per-cell/warp-per-tile regular custom kernel. At most one
  additional precompiled low-occupancy specialization may be retained, and only
  if its serialized repeated median improves its declared regime by at least
  5% while preserving the same numerical contract. Otherwise publish one path
  and explicitly record that dispatch was rejected. No universal dispatcher.
- This irregular single-RHS sparse reduction is not Tensor Core eligible. Keep
  the binary `sm_70`-specific for this phase. Benchmark direct packed execution
  against the existing Cellerator CSR SpMV path and cuSPARSE where configured
  types permit, using identical resident inputs/outputs, excluded transfers and
  setup, fixed warmups/repeats, occupancy regimes, bytes, NNZ/s, bandwidth,
  launch count, and scratch. Do not create a weaker duplicate CSR baseline.
- Tests must cover empty/tail tiles, sparse/full masks, non-identity row order,
  canonical feature recovery, zero NNZ, configured numeric values, tampered
  identities/capacities/aliases, repeat determinism within the comparison rule,
  and direct agreement with both canonical and host-tile references.
- Run CUDA 12.9 memcheck and racecheck plus host-reference, tile host/CUDA,
  record, order, apply-plan, evaluator, optimizer, inferred-pipeline, and the
  CP-BP-11 Phase E test if visible. The benchmark uses both shared GPU and
  repository benchmark locks. Publish `CP09_RUNTIME_READY`, release, and stop;
  do not implement CP-BP-10/11/12, extra operators, persistence, or git.

## Done Criteria

- Consumers operate directly on packed tiles and never reconstruct CSR/BELL in the execution path.
- Numerical output matches canonical input/reference within documented tolerances.
- Benchmarks report input shapes, hardware/toolchain, commands, throughput/bandwidth, and comparisons to relevant existing formats.
- Dispatch choices are evidence-backed and retain a correctness/reference path.
