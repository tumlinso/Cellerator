---
slug: "cellpack-m4-oracle-gating"
status: "done"
execution: "closed"
owner: "codex"
created_at: "2026-06-17T00:00:00Z"
last_heartbeat_at: "2026-06-17T00:00:00Z"
last_reviewed_at: "2026-06-17T00:00:00Z"
stale_after_days: 3
objective: "Add CellPack M4 static-oracle route gating over precompiled regions with a narrow CUDA forward/backward replay prototype."
---

# Current Objective

## Summary

Build M4 as an optional gating experiment over existing CellPack M2/M3 static plans. The implemented surface should prove that selected precompiled regions can be skipped and replayed for backward through compact route masks and route tapes, without runtime module assembly.

## Quick Start

- Why this stream exists: M3 can select static hybrid layouts, but CellPack still needs a proof that optional gating can skip precompiled regions and replay the same route for backward.
- In scope: route masks, route tapes, deterministic static oracle scenarios, coordinate-span CUDA forward, transpose replay, correctness validation, and a small mutex-serialized benchmark.
- Out of scope: learned gates, optimizer lazy updates, real CellShard data sweeps, dynamic region construction, per-cell routing, and production Blocked-ELL/Sliced-ELL CUDA kernels.
- Required skills: `cuda`.
- Required references: `AGENTS.md`, `components/CellPack/AGENTS.md`, `style_hint.md`, `optimization.md`.

## Planning Notes

- Native route assumes Tesla V100 `sm_70`.
- This is regular custom CUDA. It is not Tensor Core eligible because the proof kernel is sparse coordinate SpMV plus transpose replay.
- First expected limiters are HBM traffic, atomic accumulation, and launch count.
- Runtime selection is allowed only over static region ids. No dynamic module lookup, coordinate rebuilding, or per-cell module assembly belongs in M4.

## Assumptions

- M2 packed coordinate plans and M3 layout selection are available in the current dirty tree.
- Route masks are compact active region-id lists for one microbatch.
- Route tapes record the exact active region-id order from forward and must validate against the expected forward route before backward replay.
- Coordinate-plan execution is a proof surface, not the final CellPack packed-layout kernel ABI.

## Plan

1. Add host-side route-mask, route-tape, and static-oracle gating descriptors.
2. Add validation for unknown regions, duplicates, discarded regions, oracle mismatch, and wrong backward tapes.
3. Add a separate CUDA target that compiles packed coordinates into region-sorted arrays and static region spans.
4. Add CUDA forward and transpose-replay kernels that consume only active route ids.
5. Add focused host and CUDA tests.
6. Add `cellPackGatingBench` with no-gating versus oracle-gating summaries.
7. Build and run focused validation.

## Tasks

- [x] Add host route-mask, route-tape, and oracle active-set contracts.
- [x] Add coordinate-span CUDA runtime target.
- [x] Add `cellPackGatingTest`.
- [x] Add `cellPackGatingCudaTest`.
- [x] Add `cellPackGatingBench`.
- [x] Build focused CellPack targets.
- [x] Run focused host and CUDA tests.
- [x] Run a benchmark smoke that emits `summary.txt` and `summary.json`.

## Blockers

_None._

## Progress Notes

- Added static oracle scenarios: all regions, alternating modules, conditional only, dense tile only, and high residual skip.
- Added a coordinate-based CUDA proof runtime as a separate CellPack CUDA target, leaving the host `cellpack` library CPU-only.
- Added a benchmark harness that reports `generate`, `plan_select`, `compile_runtime`, `forward`, `backward_replay`, and `validate` phases.
- Built focused CellPack M0-M4 targets, ran `cellPackGatingTest` and `cellPackGatingCudaTest`, and smoke-ran `cellPackGatingBench` for `alternating_modules`.

## Next Actions

_None recorded yet._

## Done Criteria

- Focused CellPack M0-M4 targets build.
- `cellPackGatingTest` passes.
- `cellPackGatingCudaTest` passes.
- `cellPackGatingBench` writes `summary.txt` and `summary.json` for a small mutex-serialized smoke scenario.
