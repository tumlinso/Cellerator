---
slug: "cellpack-bp12-hardware-cost-autotune"
status: "blocked"
execution: "closed"
owner: "unassigned"
created_at: "2026-08-14T13:00:00Z"
last_heartbeat_at: "2026-08-14T13:00:00Z"
last_reviewed_at: "2026-08-16T14:38:44Z"
stale_after_days: 7
objective: "CP-BP-12: Fit a replaceable hardware-aware execution-cost model and autotune storage/runtime tradeoffs."
---

# Current Objective

## Summary

Benchmark supported packed kernels/layouts and predict execution cost from block width, occupancy, payload/index/alignment, memory transactions, and kernel path so optimization can minimize `storage_cost + lambda * predicted_execution_cost`.

## Quick Start

- Why this stream exists: smallest encoded size is not necessarily fastest on the target GPU.
- In scope: serialized benchmark matrix, empirical model/lookup table, calibration/holdout error, configurable lambda, replaceable hardware model interface, and hardware/toolchain provenance.
- Out of scope / dependencies: infecting logical format contracts with V100-specific policy, assuming one width is universally optimal, or fitting before correct kernels exist.
- Required skills: `cuda`, `todo-orchestrator`.
- Required references: CP-BP-03, CP-BP-08, CP-BP-09, `optimization.md`, `AGENTS.md`, and `bench/benchmark_mutex.hh`.

## Planning Notes

- Initial target is the repository's Tesla V100 `sm_70` environment. Hardware-specific tables sit behind a stable logical cost interface.
- Benchmark dimensions include gene width, tile occupancy, active cells, blocks/cell/tile, payload size, index width, alignment, transactions, and kernel path.

## Assumptions

- CP-BP-03 exposes a replaceable cost-policy seam.
- CP-BP-08/09 expose correct representative construction and execution paths before timing begins.

## Suggested Skills

- `cuda`
- `todo-orchestrator`

## Useful Reference Files

- `todos/cellpack-bp03-exact-merge-cost.md`
- `todos/cellpack-bp08-warp-tiles.md`
- `todos/cellpack-bp09-native-runtime-consumers.md`
- `optimization.md`
- `bench/benchmark_mutex.hh`

## Plan

1. Freeze benchmark dimensions and serialize runs through the mutex.
2. Collect storage and execution measurements with exact hardware/toolchain context.
3. Fit/construct a held-out-validated model or lookup table behind CP-BP-03's policy seam.
4. Compare selected plans across lambda/width choices and record prediction error.

## Tasks

- [!] Wait for correct CP-BP-08/09 kernels and stable CP-BP-03 policy interface.
- [ ] Build benchmark campaign and data-quality checks.
- [ ] Fit and validate replaceable execution-cost model.
- [ ] Integrate configurable storage-plus-runtime objective.

## Blockers

- Blocked on CP-BP-03 interface and measured CP-BP-08/09 implementation.

## Progress Notes

- 2026-08-14: Added as a missing blocked workstream; existing layout-estimate benchmarks do not constitute this packed-tile hardware model.

## Next Actions

- Reactivate only after correct native tile consumers exist and can be benchmarked without unpacking.

## Done Criteria

- Benchmark commands, V100/toolchain context, shapes/layouts, repeats, tolerances, and mutex use are recorded.
- Model/lookup predictions are evaluated on held-out configurations with error bounds.
- Optimizer can vary lambda and supported widths without changing logical representation semantics.
- Selected plans are compared for both bytes and measured execution rather than storage alone.
