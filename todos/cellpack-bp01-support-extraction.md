---
slug: "cellpack-bp01-support-extraction"
status: "done"
execution: "closed"
owner: "parallel-agent-step-1"
created_at: "2026-08-14T13:00:00Z"
last_heartbeat_at: "2026-08-14T13:21:10Z"
last_reviewed_at: "2026-08-16T14:38:44Z"
stale_after_days: 3
objective: "CP-BP-01 representative sampling, binary support, per-gene bitsets, counts, provenance, and exact reconstruction."
---

# Current Objective

## Summary
Build representative/random row selection, binary nonzero support extraction, per-gene sampled-cell bitsets, detection counts, and provenance suitable for downstream candidate discovery and exact scoring.

## Quick Start
- Why this stream exists: downstream search must avoid combinatorial work over every cell while preserving exact sampled support.
- In scope: representative sampling, sampled CSR validation/materialization boundary, `build_support_bitsets`, counts, memory bounds, reproducibility, and exact reference tests.
- Out of scope / dependencies: candidate generation, merge scoring, plan optimization, physical packing, and runtime kernels.
- Required skills: `cuda`, `todo-orchestrator`.
- Required references: parent CP-BP-00, `AGENTS.md`, `style_hint.md`, `include/Cellerator/compute/sampling.hh`, `src/compute/dataset/sampling.cc`, and `tests/sampling_runtime_test.cc`.

## Planning Notes
- Existing worktree evidence already implements deterministic/reproducible sampling contracts and tests, including exact samples up to 65,536 rows and density-stratified sampling, plus separate sampled-CSR materialization header/source/tests. No support-bitset implementation was found during coordination inspection.
- Dense bitsets over tens of thousands of sampled cells are acceptable initially because the gene axis is comparatively small; memory sizing must remain explicit and overflow-safe.
- Preserve canonical sampled-row and gene identities. Validate against original sparse coordinates, not only a generated bitset fixture.

## Assumptions
- The current thread owns the existing `parallel-agent-step-1` claim under the user's explicit continuation instruction; no other live agent owns this stream.
- Sampling, sampled-CSR materialization, and support extraction are
  checkpointed in `597a3eb`; their focused runtime tests passed during the
  reconciliation recorded below.

## Suggested Skills
- `cuda`
- `todo-orchestrator`

## Useful Reference Files
- `include/Cellerator/compute/sampling.hh`
- `src/compute/dataset/sampling.cc`
- `tests/sampling_runtime_test.cc`
- `include/Cellerator/compute/sampling_materialization.hh`
- `src/compute/dataset/sampling_materialization.cc`
- `tests/sampling_materialization_runtime_test.cc`
- `components/CellPack/AGENTS.md`

## Plan
1. Finalize deterministic representative sampling and provenance.
2. Define pointer-first sampled support-bitset/count views with explicit host/device residency and sizes.
3. Implement CPU/reference extraction, then GPU `build_support_bitsets` if appropriate.
4. Validate exact support reconstruction and bounded allocation across empty, dense, skewed, duplicate-invalid, and maximum-sample cases.

## Tasks
- [x] Reconcile and finish current representative/density sampling work.
- [x] Add binary per-gene support bitsets and detection counts.
- [x] Add exact reconstruction/reference and memory-bound tests.
- [x] Record build/runtime validation; use benchmark mutex for any measurements.

## Blockers
_None; this stream is complete and closed._

## Progress Notes
- 2026-08-16: Reconciled against the dirty worktree at `HEAD` `1ebb734`.
  `samplingRuntimeTest` (separately rebuilt/linked because an unrelated dirty
  NCCL header breaks its normal dependency build),
  `samplingMaterializationRuntimeTest`, and `geneSupportBitsetRuntimeTest`
  passed on the V100 host. The acceptance-complete implementation was
  checkpointed in `597a3eb` after this inspection.
- 2026-08-14: Coordination inspection found new sampling and sampled-CSR materialization header/source/runtime test files with deterministic hash, stable-ID, exact-size, density-stratified, provenance replay, and complete-row materialization coverage. Support-bitset code was not found.
- 2026-08-14: Reconciled as CP-BP-01 and marked claimed from the user's explicit parallel-work notice; no implementation status beyond repository evidence is asserted.
- 2026-08-14: Current thread resumed CP-BP-01 from the completed deterministic sampler and structural sampled-CSR bundle; next implementation is the pointer-first CPU/CUDA gene-support bitset primitive.
- 2026-08-14: Added include/Cellerator/compute/gene_support_bitset.hh and src/compute/dataset/gene_support_bitset.cu. The pointer-first host bundle retains immutable sampling provenance and sampled-position/global-row mapping; the CUDA builder uploads only CSR row pointers and column indices, constructs gene-major u32 support words with atomicOr, and counts unique detected cells despite duplicate entries.
- 2026-08-14: Added tests/gene_support_bitset_runtime_test.cu and normal CMake targets. Passed samplingRuntimeTest, datasetRuntimeTest, samplingMaterializationRuntimeTest, geneSupportBitsetRuntimeTest, and the opt-in geneSupportBitsetRuntimeTest --full-size-gpu smoke on Tesla V100.
- 2026-08-14: For 65,536 cells and 30,000 genes, words_per_gene=2,048 and support_bytes=245,760,000. The current CUDA convenience builder holds host and device support concurrently; peak structural storage is 2*support_bytes + 2*(genes*4) + cells*8 + (cells+1)*4 + nnz*4 + 4 bytes, excluding small provenance allocations.

## Next Actions
- CP-BP-02 may consume the frozen gene-major host support view and detected-cell counts; persistent device-resident ownership is a later optimization, not part of CP-BP-01 correctness.

## Done Criteria
- Fixed input/seed produces reproducible sampled identities and supports.
- Per-gene bitsets and detection counts exactly reconstruct sampled CSR nonzero incidence.
- Empty genes/rows, tail words, invalid indices, overflow, and the documented maximum sample size are covered.
- Peak and formula-based memory use are recorded and bounded for the supported sample/gene dimensions.
