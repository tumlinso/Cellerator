---
slug: "cellpack-bp09-native-runtime-consumers"
status: "planned"
execution: "ready"
owner: "unassigned"
created_at: "2026-08-14T13:00:00Z"
last_heartbeat_at: "2026-08-17T09:00:11Z"
last_reviewed_at: "2026-08-17T09:00:11Z"
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

- Read `todos/cellpack-bp06-11-parallel-execution.md`. Barrier C source
  checkpoint `ebe0509` integrates `CP08_HOST_ABI_READY`; Phase D reference/API
  work is now ready but unclaimed.
- Phase D may claim only the reference/API portion, use `build-cp-bp09`, publish
  `CP09_REFERENCE_READY`, release, and become idle. Device implementation waits
  for `CP08_DEVICE_READY` and Barrier D.
- Phase E implements one direct V100 consumer without CSR/BELL reconstruction,
  extra operations, per-cell launches, or universal dispatch. Publish
  `CP09_RUNTIME_READY`, release/close, and perform no git operation.

## File Lease

_Phase D is ready and unclaimed._ On explicit assignment atomically lease
exactly:

- new
  `components/CellPack/include/CellPack/feature_weighted_row_reduction.hh`;
- new `components/CellPack/src/feature_weighted_row_reduction.cc`;
- new
  `components/CellPack/tests/feature_weighted_row_reduction_test.cc`;
- only clearly labelled CP-BP-09 Phase D target blocks in root `CMakeLists.txt`;
- this ledger and CP-BP-09 entries in the coordinator, `todos.md`,
  `todo-status.md`, and parent roadmap while holding the shared lock.

Record a unique owner and full current pushed `origin/main` hash before editing.
`warp_tiles.hh/.cc`, every `warp_tiles_cuda.*`, component `CMakeLists.txt`,
plan/record/order files, and all statistical-validation files are read-only.

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
- `src/compute/sparse/ops/`
- `bench/benchmark_mutex.hh`

## Plan

1. Define the canonical feature-weighted row-reduction reference and packed runtime view/scratch contracts.
2. Implement direct correctness path with no unpacking.
3. Benchmark occupancy-dependent execution alternatives under the mutex.
4. Document selected dispatch conditions, limiter, tolerance, and reference fallback.

## Tasks

- [x] Wait for the stable CP-BP-08 host tile ABI and Barrier C.
- [ ] Phase D: freeze the direct packed consumer contract and CPU/canonical
  feature-weighted row-reduction reference.
- [!] Phase E: wait for `CP08_DEVICE_READY` and Barrier D before the direct CUDA
  consumer and runtime benchmark.
- [ ] Explore and benchmark occupancy dispatch paths.
- [ ] Record header benchmark justification for custom GPU math.

## Blockers

- No blocker for Phase D reference/API work. Device implementation remains
  blocked on `CP08_DEVICE_READY` and Barrier D.

## Progress Notes

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

- Await explicit assignment text “You are assigned CP-BP-09 Phase D”. Claim
  only the exact reference/API lease, use `build-cp-bp09`, publish
  `CP09_REFERENCE_READY`, release to idle, and stop without git. Do not begin
  the Phase E CUDA consumer before Barrier D.

## Done Criteria

- Consumers operate directly on packed tiles and never reconstruct CSR/BELL in the execution path.
- Numerical output matches canonical input/reference within documented tolerances.
- Benchmarks report input shapes, hardware/toolchain, commands, throughput/bandwidth, and comparisons to relevant existing formats.
- Dispatch choices are evidence-backed and retain a correctness/reference path.
