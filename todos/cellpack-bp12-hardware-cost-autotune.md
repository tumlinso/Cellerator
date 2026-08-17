---
slug: "cellpack-bp12-hardware-cost-autotune"
status: "done"
execution: "closed"
owner: "codex-cp-bp12"
created_at: "2026-08-14T13:00:00Z"
last_heartbeat_at: "2026-08-17T14:44:32Z"
last_reviewed_at: "2026-08-17T14:44:32Z"
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
- This pass calibrates and selects; it does not aggressively optimize kernels.
  It reuses the existing CP-BP-09 direct tile consumer and maintained CSR
  fallback under one resident-I/O comparison contract. The later aggressive
  end-to-end optimization pass remains explicitly separate.
- Frozen v1 benchmark matrix: rows `{8192, 32768}`, feature-block widths
  `{8, 16, 32}`, blocks/row `{1, 2}`, and sharing groups/tile
  `{1, 4, 8, 16, 32}`. Every configuration measures both direct packed tiles
  and the existing configured-type CSR path with 3 warmups and 11 repeats.
  A deterministic configuration hash assigns held-out cases before timing.

## Claim And File Lease

Claimed by `codex-cp-bp12` from pushed Cellerator
`46f648f8c12f6e31981df6a301fdebae80e156f2` under
`/tmp/cellerator-cp-bp-shared.lock`. The exact implementation lease is:

- new `components/CellPack/include/CellPack/hardware_cost_model.hh`;
- new `components/CellPack/src/hardware_cost_model.cc`;
- new `components/CellPack/tests/hardware_cost_model_test.cc`;
- new `components/CellPack/bench/hardware_cost_autotune_bench.cu`;
- only clearly labelled CP-BP-12 host target blocks in root `CMakeLists.txt`
  and CUDA benchmark target blocks in `components/CellPack/CMakeLists.txt`;
- CP-BP-12 entries in this ledger, `todos.md`, `todo-status.md`, and the parent
  roadmap while holding the shared lock.

All CP-BP-03/04/08/09/10/11 source and logical plan/record/tile/runtime ABIs are
read-only inputs. A demonstrated frozen-input defect is a stop condition, not
authority to widen this lease. Build in `build-cp-bp12`; serialize every GPU
run through `/tmp/cellerator-cp-bp12-gpu.lock` and the repository benchmark
mutex. No CP-BP-13, persistence, kernel tuning, profiler campaign, or git
integration occurs before acceptance is complete.

## Assumptions

- CP-BP-03 exposes a replaceable cost-policy seam.
- CP-BP-08 exposes correct measured construction and CP-BP-09 exposes correct
  measured execution before model fitting begins.

## Suggested Skills

- `cuda`
- `todo-orchestrator`
- `compare-benchmarks`

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

- [x] Wait for the correct measured CP-BP-09 runtime; CP-BP-03 and measured
  CP-BP-08 construction are complete.
- [x] Build benchmark campaign and data-quality checks.
- [x] Fit and validate replaceable execution-cost model.
- [x] Integrate configurable storage-plus-runtime objective.

## Blockers

- None. Barrier F closed CP-BP-10/11 and released the shared GPU/integration
  wave; CP-BP-03's replaceable cost-policy seam and CP-BP-08/09 measured tile
  construction/runtime inputs are pushed and stable.

## Progress Notes

- 2026-08-17: Completed the versioned pointer-first raw-observation, fixed-size
  log-linear path-model, held-out error-report, and deterministic
  storage-plus-runtime plan-selection API. Model identity binds the campaign,
  hardware, toolchain, operation, supported widths, fit policy, and fitted
  parameters; candidate results retain the replaceable storage-cost policy.
- 2026-08-17: Serialized V100 `sm_70` campaign command
  `flock /tmp/cellerator-cp-bp12-gpu.lock ./build-cp-bp12/cellPackHardwareCostAutotuneBench --output-dir build-cp-bp12/cp-bp12-compare`
  measured all 60 frozen configurations and 120 paired resident one-launch
  observations with 3 warmups/11 repeats. All outputs matched the versioned
  CP-BP-09 numerical rule. The deterministic holdout contained 9 observations
  per path; MAPE was 5.15105% for direct tiles and 5.87580% for CSR. With
  storage-byte and runtime-nanosecond weights both 1, paired selection chose
  direct tiles 57 times and CSR 3 times. Those selected paths totaled
  105,495,452 representation bytes and 1,795,072 measured ns versus
  105,447,336 bytes and 1,860,608 measured ns for storage-only selection.
  Index width, alignment, and campaign-defined estimated transaction counts are
  explicit raw features; no profiler-counter claim is made.
- 2026-08-17: Focused model/adversarial tests and CP-BP-03/04/06/08/09/11 plus
  inferred-pipeline regressions passed in `build-cp-bp12`; warning-clean host
  syntax, artifact data-quality checks, and `git diff --check` passed. No
  kernel, logical packing ABI, persistence surface, or aggressive optimization
  was changed. Acceptance is complete and all leases are released.
- 2026-08-17: Claimed as `codex-cp-bp12` from clean pushed Barrier F ledger
  `46f648f8`. The native route is V100 `sm_70`, Volta benchmark micro-router,
  device-resident direct tiles versus the exact configured-type CSR fallback.
  This irregular single-RHS sparse workload is not Tensor Core eligible. The
  model/autotuner is host-side and pointer-first; no new CUDA kernel is leased.
- 2026-08-17: Reactivated as `planned/ready` after Barrier F pushed
  `2cfa5c8d26f0c973dfef4659d72ea5f635201835` and closed CP-BP-10/11. This is
  the single recommended continuation package; it is not claimed by the
  integrator.
- 2026-08-17: Barrier E integrated measured CP-BP-09 direct runtime at
  `0334f954b1b9e04366f2e2ce191e098c1d476597`. The technical input is now
  available. Coordination deliberately keeps CP-BP-12 closed during the
  CP-BP-10/11 Phase F fork; it is the primary post-Barrier-F continuation.
- 2026-08-17: Barrier D integrated exact measured CP-BP-08 tile construction at
  `0bf9acf`; that dependency is satisfied. CP-BP-12 remains blocked only on the
  Phase E direct CP-BP-09 runtime and its fair CSR/current-layout measurements.
- 2026-08-14: Added as a missing blocked workstream; existing layout-estimate benchmarks do not constitute this packed-tile hardware model.
- 2026-08-16: CP-BP-03 completed the versioned replaceable storage-cost seam;
  CP-BP-12 remains blocked because no correct CP-BP-08/09 tile consumer exists
  to supply execution measurements.

## Next Actions

- Complete/closed. CP-BP-13 is now unblocked and should begin with its recorded
  read-only cross-repository compatibility/ownership audit. The later
  aggressive optimization pass remains separate from both workstreams.

## Done Criteria

- Benchmark commands, V100/toolchain context, shapes/layouts, repeats, tolerances, and mutex use are recorded.
- Model/lookup predictions are evaluated on held-out configurations with error bounds.
- Optimizer can vary lambda and supported widths without changing logical representation semantics.
- Selected plans are compared for both bytes and measured execution rather than storage alone.
