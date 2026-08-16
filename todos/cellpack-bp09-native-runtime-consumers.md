---
slug: "cellpack-bp09-native-runtime-consumers"
status: "blocked"
execution: "closed"
owner: "unassigned"
created_at: "2026-08-14T13:00:00Z"
last_heartbeat_at: "2026-08-14T13:00:00Z"
last_reviewed_at: "2026-08-16T14:38:44Z"
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

## Assumptions

- CP-BP-08 exposes a device-resident pointer-first tile view with complete bounds and rank semantics.
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

1. Select one relevant operation/reference and define packed runtime view/scratch contracts.
2. Implement direct correctness path with no unpacking.
3. Benchmark occupancy-dependent execution alternatives under the mutex.
4. Document selected dispatch conditions, limiter, tolerance, and reference fallback.

## Tasks

- [!] Wait for CP-BP-08 stable device tile view.
- [ ] Implement direct packed consumer and CPU/canonical reference comparison.
- [ ] Explore and benchmark occupancy dispatch paths.
- [ ] Record header benchmark justification for custom GPU math.

## Blockers

- Blocked on CP-BP-08 physical tile ABI and payload rank/offset semantics.

## Progress Notes

- 2026-08-14: Added as a missing blocked workstream; existing coordinate-based oracle-gating CUDA is scaffolding/reference, not this packed runtime.

## Next Actions

- Reactivate after CP-BP-08 exact decode and stable device views pass.

## Done Criteria

- Consumers operate directly on packed tiles and never reconstruct CSR/BELL in the execution path.
- Numerical output matches canonical input/reference within documented tolerances.
- Benchmarks report input shapes, hardware/toolchain, commands, throughput/bandwidth, and comparisons to relevant existing formats.
- Dispatch choices are evidence-backed and retain a correctness/reference path.
