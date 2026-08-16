---
slug: "cellpack-m3-layout-selection"
status: "done"
execution: "closed"
owner: "codex"
created_at: "2026-06-17T00:00:00Z"
last_heartbeat_at: "2026-06-17T00:00:00Z"
last_reviewed_at: "2026-06-17T00:00:00Z"
stale_after_days: 3
objective: "Add CellPack M3 host-side layout metrics, hybrid selection, and synthetic benchmark summaries."
---

# Current Objective

## Summary

Build M3 as a GPU-first CellPack layout compiler milestone without adding new CellPack GPU kernels. The implemented surface scores packed regions for expected native V100 behavior, selects static hybrid layouts, and benchmarks synthetic plans against CSR and current Blocked-ELL estimate baselines.

## Quick Start

- Why this stream exists: CellPack M2 can reconstruct static packed coordinates, but M3 needed cost metrics and deterministic format selection before runtime kernels.
- In scope: host-side metrics, deterministic selector, synthetic benchmark summaries, and focused validation.
- Out of scope / dependencies: new GPU kernels, learned gating, runtime dynamic assembly, route masks, and real CellShard-root benchmarking.
- Required skills: `compare-benchmarks`, `cuda`, `todo-orchestrator`.
- Required references: `components/CellPack/AGENTS.md`, `AGENTS.md`, `optimization.md`, compare-benchmarks comparison contract, CUDA native Volta benchmark route, CUDA sparse bio-layout route, CUDA fusion route, CUDA compute-library routing.

## Planning Notes

- Native route assumes Tesla V100 `sm_70`; first limiters are HBM/index/value bytes, padding, launch groups, and host-device conversion cost.
- Tensor Core eligibility is metadata only in M3 and applies only to dense/tile candidates whose dimensions satisfy the configured tile multiple.
- Structural equivalence for this milestone means same source matrix coverage, stable region mapping, and reconstruction-compatible descriptors rather than SpMM numeric equivalence.

## Assumptions

- M2 CSR/COO views, packed-coordinate metadata, and reconstruction tests are available.
- Synthetic benchmarking is the first proof surface; real-data sweeps are a later evaluation pass.
- Selector output remains static descriptors and metrics; no runtime module lookup or dynamic assembly is introduced.

## Suggested Skills

- `compare-benchmarks`: keep CSR, current Blocked-ELL, and hybrid estimates under one scenario/summary contract.
- `cuda`: keep the cost model aligned with native V100 sparse bio-data layout, fusion, and library-routing constraints.
- `todo-orchestrator`: preserve pickup context for the M3 workstream.

## Useful Reference Files

- `AGENTS.md`
- `components/CellPack/AGENTS.md`
- `optimization.md`
- `components/CellPack/include/CellPack/layout_metrics.hh`
- `components/CellPack/include/CellPack/layout_selector.hh`
- `components/CellPack/bench/cellpack_layout_bench.cc`

## Plan

1. Add per-region and per-plan layout metric APIs over static plans and packed coordinate plans.
2. Add a deterministic selector for residual CSR, Blocked-ELL, Sliced-ELL, and dense/tile candidates.
3. Add focused tests for exact metric math, selector choices, determinism, and invalid configuration.
4. Add a mutex-serialized synthetic benchmark with stable phase labels and compare-style summary outputs.
5. Build and run focused CellPack validation.

## Tasks

- [x] Add CellPack layout metrics headers and implementation.
- [x] Add deterministic hybrid layout selector headers and implementation.
- [x] Add `cellPackLayoutMetricsTest` and `cellPackLayoutSelectorTest`.
- [x] Add `cellPackLayoutBench` and synthetic scenarios.
- [x] Honor the compare-benchmarks mutex environment in the shared benchmark guard.
- [x] Build focused CellPack targets and run focused tests.
- [x] Run a benchmark smoke that emits `summary.txt` and `summary.json`.

## Blockers

_None._

## Progress Notes

- Added metrics for row-width distributions, padded slots, fill ratios, CSR/Blocked-ELL/Sliced-ELL/dense estimated bytes, output bytes, residual fractions, and launch group keys.
- Added selector thresholds for rare/scattered residual CSR fallback, Blocked-ELL fill/byte preference, Sliced-ELL grouped-variable rows, and dense/tile Tensor Core candidate metadata.
- Added `cellPackLayoutBench` phases: `generate`, `plan`, `select_layout`, `estimate_runtime`, and `baseline_reference`.
- Verified `cellPackLayoutBench` acquires the benchmark mutex and writes compare-style summaries.

## Next Actions

_None recorded yet._

## Done Criteria

- Focused CellPack targets build.
- Existing CellPack M0-M2 tests pass.
- New metrics and selector tests pass.
- Synthetic layout benchmark smoke produces stable summary text and JSON.
