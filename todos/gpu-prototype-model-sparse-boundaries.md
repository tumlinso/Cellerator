---
slug: "gpu-prototype-model-sparse-boundaries"
status: "planned"
execution: "ready"
owner: "codex"
created_at: "2026-04-16T09:18:45Z"
last_heartbeat_at: "2026-04-16T09:19:46Z"
last_reviewed_at: "2026-04-16T09:19:46Z"
stale_after_days: 14
objective: "Prototype GPU-first sparse projection, sparse-aware loss, and quantizer supervision boundaries across model codepaths"
---

# Current Objective

## Summary
Prototype GPU-first sparse projection, sparse-aware loss, and quantizer supervision boundaries across model codepaths

## Quick Start
- Why this stream exists: _Summarize the domain boundary and why it was split out._
- In scope: _List the work this stream owns._
- Out of scope / dependencies: _List handoffs, upstream dependencies, or adjacent streams._
- Required skills: _List the exact repo-local skills to read before starting._
- Required references: _List the exact repo-local references to read before starting._
- Why this stream exists: the main model codepaths still pay CPU or Torch-managed sparse-layout churn around GPU math, especially CSR-to-COO conversion, densification, and CPU-side supervision assembly.
- In scope: CUDA prototypes for sparse projection, sparse-aware losses, quantizer supervision assembly, and any custom Torch-op or compute-autograd boundary needed to keep sparse batches resident on device.
- Out of scope / dependencies: the explicit CPU torch export bridge remains an interop boundary, and model-family decisions are out of scope because this stream only changes execution boundaries for existing models.
- Required skills: `cuda`, `todo-orchestrator`.
- Required references: `optimization.md`, `custom_torch_ops.md`, `src/compute/autograd/autograd.hh`, `src/compute/model_ops/model_ops.cu`, `src/models/developmental_time/dT_model.hh`, `src/models/developmental_time_cuda/dtc_model.cu`, `src/models/dense_reduce/dR_model.hh`, `src/models/quantize/quantize.hh`.

## Planning Notes
- Favor extending `compute/autograd` and `compute/model_ops` over adding more libtorch-side conversions or wrappers.
- Treat DenseReduce reconstruction and quantizer supervision as the first sparse-aware loss prototypes because they currently densify or assemble targets on CPU.

## Assumptions
- A prototype can be model-local and narrow so long as it exposes a clean timing comparison against the existing baseline.
- The native sparse execution layout should stay Blocked-ELL-first where possible, with CSR retained only where the current runtime still needs row-compressed semantics.

## Suggested Skills
- `cuda` - Use the Volta Torch-extension and CPU-porting paths; these prototypes are mostly about boundary relocation and sparse glue, not dense GEMM tuning.
- `todo-orchestrator` - Keep the model-boundary work distinct from ingest and neighbor streams so benchmarks stay interpretable.

## Useful Reference Files
- `custom_torch_ops.md` - Current custom-op registry and the preferred extension seams.
- `src/compute/autograd/autograd.hh` - Pointer-first sparse runtime and distributed topology helpers that should absorb new sparse building blocks.
- `src/models/dense_reduce/dR_model.hh` - Current CSR-to-COO churn and reconstruction densification hotspot.
- `src/models/quantize/quantize.hh` - Current dense future-target and range-loss assembly path.

## Plan
- Prototype GPU-first sparse projection for developmental_time and dense_reduce without CSR-to-COO churn on every step.
- Prototype a sparse-aware DenseReduce reconstruction loss to replace `to_dense()` benchmarking-wise.
- Prototype GPU future-neighbor target assembly and sparse quantizer loss flow so supervision no longer depends on CPU row lookup and dense host staging.
- Benchmark each prototype against the existing Torch-managed baseline and keep only the ones that show a clear residency or throughput win.

## Tasks
- [ ] Prototype a compute-autograd-backed sparse projection path for developmental_time baseline and CUDA sibling codepaths.
- [ ] Prototype a sparse-aware DenseReduce reconstruction loss without full target densification.
- [ ] Prototype GPU-side corruption or masking for sparse DenseReduce input if it remains part of the training contract.
- [ ] Prototype GPU future-neighbor target assembly and weighting for quantize supervision.
- [ ] Prototype quantizer sparse loss variants that avoid unnecessary dense target materialization.
- [ ] Add focused compile or runtime benchmarks for the prototype model boundaries on V100.

## Blockers
_None recorded yet._

## Progress Notes
_None recorded yet._

## Next Actions
- Start with DenseReduce because it has both obvious sparse layout churn and the largest avoidable `to_dense()` boundary in current model code.

## Done Criteria
- At least one working GPU prototype exists for each major model-boundary class: sparse projection, sparse-aware loss, and quantizer supervision assembly.
- Each prototype can be benchmarked against the existing model path and makes the dominant limiter explicit.
