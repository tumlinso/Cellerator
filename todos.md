# Active Objectives

## Summary
Use this file as the canonical index for substantial multi-step work.

## Shared Assumptions
- Cellerator owns the accelerated scRNA preprocessing backbone, native workbench implementation, and preprocessing benchmarks.
- Native Cellerator GPU policy remains V100 `sm_70` and Blocked-ELL-first for Cellerator execution. Cellerator preprocessing treats Blocked-ELL and Sliced-ELL as first-class preprocessing layouts, with compressed / CSR as fallback.
- The CP-BP data-inferred packing roadmap is a distinct compact block grammar,
  not ordinary Blocked-ELL/BELL with a smarter permutation. Its codec cost,
  variable-payload offsets, canonical identity maps, and direct runtime path
  must be explicit.
- Cellerator owns packing discovery, plan semantics, transformation, CUDA
  representation, and native consumption. CellShard may later own durable
  `.cspack` serialization/validation/fetch/upload integration without learning
  or defining the plan.
- CP-BP-01, CP-BP-02, and CP-BP-04 satisfy their recorded acceptance criteria
  and are closed. Their sources, build wiring, tests, and benchmarks are
  checkpointed in commit `597a3eb`; CP-BP-03 still owns exact candidate scoring
  and CP-BP-05 is ready against the frozen semantic plan contract.
- CP-BP-03 and CP-BP-05 may run concurrently in one worktree only under this
  cooperative protocol. Before claiming, changing a lease, or editing a shared
  file, atomically acquire `/tmp/cellerator-cp-bp-shared.lock` with `mkdir`.
  While holding it, reread both child ledgers, record the claim and exact file
  lease in the assigned ledger, and synchronize this index and
  `todo-status.md`; then release it with `rmdir`. If acquisition fails, do not
  edit ledgers or shared files. Never remove an unexplained lock without the
  coordinating user's approval.
- Shared files are `CMakeLists.txt`, `components/CellPack/CMakeLists.txt`,
  `todos.md`, and `todo-status.md`, plus any pre-existing path either child has
  leased. New files belong exclusively to the first recorded lease. A stream
  may not touch the other stream's lease; transfer it through both ledgers
  while holding the lock. Use `build-cp-bp03` and `build-cp-bp05`, respectively,
  and serialize all benchmarks with the repository benchmark mutex.
- Neither implementation thread may commit, push, stash, reset, switch branch,
  or rewrite another thread's changes while either child is claimed. The final
  integrator waits until both ledgers are idle or done, acquires the lock,
  rereads leases and diffs, validates the combined tree, and alone performs
  integration git operations.

## Suggested Skills
- `todo-orchestrator`: maintain the resumable migration ledger while implementing the supplied plan.
- `bio-experiments`: preserve scRNA raw-count, QC, normalization, and double-processing semantics.
- `cuda`: keep the implementation aligned with native V100 sparse bio-data layout and hot-path constraints.

## Useful Reference Files
- `AGENTS.md`: repository structure, testing, Blocked-ELL, and Volta policy.
- `optimization.md`: current V100 bottleneck and sparse preprocessing guidance.
- `docs/pipeline/preprocess/README.md`: existing preprocessing documentation to update if behavior moves.
- `docs/pipeline/README.md`: pipeline-level documentation surface to keep in sync.
- `todos/cellpack-data-inferred-block-packing-roadmap.md`: CP-BP-00 parent
  architecture, dependency graph, invariants, and child IDs.
- `todos/cellpack-packing-plan-evaluator.md`: completed exact supplied-plan
  evaluator to reuse; it is not plan inference or a physical codec.
- `components/CellPack/AGENTS.md`: CellPack compiler/runtime boundaries and
  static-packing rules.

## Workstreams
- `cellpack-data-inferred-block-packing-roadmap` | status: in_progress | owner: coordination | file: `todos/cellpack-data-inferred-block-packing-roadmap.md` | objective: CP-BP-00 parent coordination epic for the complete offline compiler, compact tile format, native runtime, validation, autotuning, and persistence roadmap; do not implement from the parent.
- `cellpack-bp01-support-extraction` | status: done | owner: parallel-agent-step-1 | file: `todos/cellpack-bp01-support-extraction.md` | objective: CP-BP-01 representative sampling, binary support, per-gene bitsets, counts, provenance, and exact reconstruction.
- `cellpack-bp02-candidate-discovery` | status: done | owner: codex-cp-bp-02 | file: `todos/cellpack-bp02-candidate-discovery.md` | objective: CP-BP-02 deterministic sketch/LSH candidate generation and deduplication; approximate similarity proposes only.
- `cellpack-bp03-exact-merge-cost` | status: planned | owner: unassigned | file: `todos/cellpack-bp03-exact-merge-cost.md` | objective: CP-BP-03 exact bitset overlap and replaceable codec-cost/merge-gain scoring, reconciled with the completed evaluator.
- `cellpack-bp04-packing-plan-optimizer` | status: done | owner: codex-cp-bp-04 | file: `todos/cellpack-bp04-packing-plan-optimizer.md` | objective: CP-BP-04 supplied-candidate deterministic optimizer, exact-oracle rollback, and immutable semantic PackingPlan are complete.
- `cellpack-bp05-apply-frozen-plan` | status: planned | owner: unassigned | file: `todos/cellpack-bp05-apply-frozen-plan.md` | objective: CP-BP-05 GPU-oriented remap and segmented packed-coordinate ordering; ready against `frozen_packing_plan`.
- `cellpack-packing-plan-cuda-evaluator` | status: planned | owner: unassigned | file: `todos/cellpack-packing-plan-cuda-evaluator.md` | objective: deferred native V100 CUB acceleration of exact PackingPlan evaluation; opened by measured oracle share and not prerequisite to CP-BP-05.
- `cellpack-bp06-cell-block-records` | status: blocked | owner: unassigned | file: `todos/cellpack-bp06-cell-block-records.md` | objective: CP-BP-06 compact per-cell block records and complete variable-payload offsets; the CP-BP-04 plan contract is complete and this now waits only on CP-BP-05 ordered rows.
- `cellpack-bp07-local-cell-ordering` | status: blocked | owner: unassigned | file: `todos/cellpack-bp07-local-cell-ordering.md` | objective: CP-BP-07 bounded local active-block-signature ordering; waits on CP-BP-06.
- `cellpack-bp08-warp-tiles` | status: blocked | owner: unassigned | file: `todos/cellpack-bp08-warp-tiles.md` | objective: CP-BP-08 compact 32-cell tile dictionary, cell/gene masks, payloads, and offsets; waits on CP-BP-06/07.
- `cellpack-bp09-native-runtime-consumers` | status: blocked | owner: unassigned | file: `todos/cellpack-bp09-native-runtime-consumers.md` | objective: CP-BP-09 direct packed-tile kernels with no CSR/BELL unpack; waits on CP-BP-08.
- `cellpack-bp10-alternating-refinement` | status: blocked | owner: unassigned | file: `todos/cellpack-bp10-alternating-refinement.md` | objective: CP-BP-10 bounded held-out gene/cell alternating refinement; waits on the first complete plan/tile/runtime loop.
- `cellpack-bp11-statistical-validation` | status: planned | owner: unassigned | file: `todos/cellpack-bp11-statistical-validation.md` | objective: CP-BP-11 held-out, degree-preserving null, bootstrap/stability, metric, and correctness infrastructure.
- `cellpack-bp12-hardware-cost-autotune` | status: blocked | owner: unassigned | file: `todos/cellpack-bp12-hardware-cost-autotune.md` | objective: CP-BP-12 replaceable measured execution-cost model; waits on CP-BP-03 and measured CP-BP-08/09 paths.
- `cellpack-bp13-persistence-integration` | status: blocked | owner: unassigned | file: `todos/cellpack-bp13-persistence-integration.md` | objective: CP-BP-13 Cellerator/CellShard .cspack lifecycle integration; waits on stable plan/record/tile/runtime contracts.

## Global Blockers
- CP-BP-06 through CP-BP-10 and CP-BP-13 remain intentionally blocked by the
  unresolved per-cell record, warp-tile, or direct-consumer APIs identified in
  their child ledgers. CP-BP-05 is no longer blocked on PackingPlan semantics.
- CP-BP-12 cannot fit a hardware model until correct CP-BP-08/09 kernels exist.

## Progress Notes
- 2026-08-16: Added the single-worktree CP-BP-03/05 claim, file-lease, build,
  benchmark, and git-integration interlock. The streams remain unclaimed until
  a fork records one specific assignment under the cooperative lock.
- 2026-08-16: Reconciled CP-BP-00 through CP-BP-13 against source, tests,
  CMake wiring, V100 benchmarks, git history, and the dirty worktree based at
  `1ebb734`; the acceptance-complete CP-BP-01/02/04 implementation was then
  checkpointed as `597a3eb`. CP-BP-03 and CP-BP-05 are genuinely
  unassigned/ready; CP-BP-05 is
  the primary continuation frontier because it unlocks CP-BP-06 through the
  tile/runtime/persistence chain. CP-BP-11 has reusable deterministic split and
  frozen-plan/evaluator foundations but no null/bootstrap validation harness.
- 2026-08-16: Focused rebuilds and runtime tests passed for sampled CSR
  materialization, CPU/CUDA gene support, CPU/CUDA candidate discovery, exact
  plan evaluation, optimizer, and reconstruction. Serialized V100 candidate
  and optimizer benchmarks passed. After checkpointing, a fresh
  `cmake -S . -B build-checkpoint-validation -DCMAKE_CUDA_ARCHITECTURES=70`
  followed by `cmake --build build-checkpoint-validation --target
  samplingRuntimeTest -j 4` failed while compiling CellShard runtime because
  committed `b69a168` `nccl_communicator.cuh` uses `local_context` before its
  definition. The test source separately rebuilt/linked against the prior
  runtime archive and passed. No repair was attempted in this status-only task.
- 2026-08-14: Completed the post-evaluator CP-BP-04 optimizer-readiness audit. The evaluator passed without code changes; CP-BP-04 is ready to begin with a proxy-plus-CPU-oracle architecture, while CP-BP-05 remains blocked on the owning/versioned frozen-plan contract.
- 2026-08-14: Completed the conceptual CP-BP-04 v1 implementation plan without source changes. Selected feature-first deterministic constrained coarsening plus bounded move/swap refinement, integer support-derived proxy batches, CPU exact-oracle checkpoints with rollback, identity/fixed-width row geometry, and an immutable semantic plan. GPU evaluator acceleration remains an expected deferred CUB/V100 workstream with explicit profiling triggers.
- 2026-08-14: Completed CP-BP-04 v1 for its supplied-candidate contract. Added deterministic score-kind-preserving normalization, membership-authoritative mutable blocks, zero-copy sampled-support proxy caches, constrained merge/move/swap search, exact evaluator rollback/shrink/blacklisting, immutable semantic `frozen_packing_plan`, adversarial/property tests, and a mutex optimizer benchmark. CP-BP-05 is ready; completed CP-BP-02 pairs still require CP-BP-03 exact scored-relation adaptation for production candidate quality.
- 2026-08-14: Opened `cellpack-packing-plan-cuda-evaluator` as a separate deferred workstream after CPU oracle share measured 77.6% at 5k features and 46.6% at 20k features. Absolute volume triggers remain low, so it does not block CP-BP-05.
- 2026-08-14: Added CP-BP-00 through CP-BP-13 as the complete data-inferred
  block-packing roadmap. Each executable step has its own pickup ledger,
  dependencies, collision boundary, acceptance criteria, and implementation
  status; no implementation was performed by this coordination pass.
- 2026-08-14: Reconciled CP-BP-01 with new dirty-worktree deterministic sampling
  and sampled-CSR materialization headers/sources/tests. These are evidence of
  active progress, but sampled support bitsets were not found and the step is
  not marked complete.
- 2026-08-14: Preliminary coordination reconciled CP-BP-04 with the completed
  `cellpack-packing-plan-evaluator` and temporarily recorded an active optimizer
  thread. The later post-evaluator audit supersedes that pickup state: exact
  supplied-plan evaluation is complete infrastructure, and CP-BP-04 is ready.
- 2026-08-14: Recorded safe independent pickups CP-BP-02, CP-BP-03, and the
  foundational reference/metric portion of CP-BP-11. Their ledgers forbid
  edits to CP-BP-01/04-owned files and identify the narrow integration seams.
- Started `cellpack-packing-plan-evaluator` on 2026-08-14 after correcting the earlier downstream physical-format interpretation. This stage evaluates a `PackingPlan` defined by row and feature permutations plus row-group and feature-block boundaries. It does not define `gene_masks`, a final sparse codec, `.cspack` serialization, preprocessing changes, plan inference, or execution kernels.
- Finished `cellpack-packing-plan-evaluator`: exact sparse occupied-tile evaluation, reusable prepared CSR support/scratch, separate reference cost policy, static-plan geometry adapter, comprehensive tests, mutex benchmark, and dated CellPack handoff notes are in place. The next stage is plan inference/refinement unless evaluator throughput is first shown to block search.
- Started `sequence-bits-dna2`: requested scope is a new `include/Cellerator/seq/dna2.cuh` GPU-native DNA 2-bit primitive with packed storage words, warp-compute bitplanes, correctness kernels, tests, primitive benchmark, and docs. This work was separate from the historical core sequence port material.
- Finished `sequence-bits-dna2`: configured Cellerator, built `sequenceDna2Test`, `sequenceDna2CudaTest`, and `sequenceDna2Bench`, ran both focused tests, and ran the primitive benchmark with `./build/sequenceDna2Bench 1048576 16 1 10`.
- Migrated `sequence-bits-dna2` out of Cellerator to the sibling Baseplane
  project. Cellerator now consumes `Baseplane::seq` for sequence bit primitive
  smoke coverage and no longer owns the sequence headers, target, tests,
  benches, or docs.
- Preserved post-umbrella Baseplane explicit-width DNA2 and benchmark work in
  sibling Baseplane commit `12c83b6`, with top-level Baseplane validation for
  `baseplaneDna2Test` and `baseplaneDna2CudaTest`.
- Validated the hard cut with sibling Baseplane: `cmake -S Cellerator -B
  Cellerator/build`, `cmake --build Cellerator/build --target
  coreSparseLayoutRuntimeTest quantizedMatrixTest exactSearchRuntimeTest -j 4`,
  and the three resulting runtime smokes passed.
- Extended `sequence-bits-dna2` validation: added `tests/seq/dna2_test_helpers.hh` for deterministic random sequence generation, changed the benchmark to use random DNA input with independent representation selection, and ran a packed-word64 vs warp-planes32 comparison matrix plus Nsight Systems profiles.
- Finished the first CPU SIMD backend pass: Highway now uses SIMD mask extraction/materialization for full 32-base ASCII pack/unpack, the CPU benchmark reports the active backend, and both Highway-enabled and scalar-only builds pass `sequenceDna2Test` plus `sequenceDna2CpuBench`.
- Added the first former-core ownership slice: the then-current core include path exposed
  sparse layout primitives/device views, and CellShard layout headers were
  compatibility shims over those types.
- Verified the former-core/CellShard wiring with CellShard package-consumer
  checks, Cellerator sparse/quantized tests, and a CellShardPreprocess build.
- Started `cellerator-sparse-ml-layout` from the supplied source layout plan. The intended first pass is behavior-preserving except for moving forward-neighbor policy/API ownership into the new the external neighbor-caller package.
- Checkpointed `cellerator-sparse-ml-layout`: moved sparse-operator/model-op code under `src/compute/ml`, moved shared host buffering to `src/compute/core`, moved cuVS/KNN scoring helpers under `src/compute/neighbors/scoring`, and removed Cellerator-owned forward-neighbor compatibility wrappers.
- Implemented `cellshard-multi-assay-archive`: CellShard now has measurement-agnostic assay descriptors and row-map helpers, Cellerator validates those semantics against the biology semantics package, and docs state that multiome execution uses coordinated single-assay CSPACK artifacts.
- Ran `todo-cleanup --partial` and cleared workstreams: dual-cuda-optimization-modes, cellshard-first-stable-release, cellshard-blocked-ell-ingest-runtime, cellshard-runtime-service-contract, quantized-blocked-ell-codecs, cellshard-user-metadata-annotations, gpu-prototype-ingest-blocked-ell, gpu-prototype-model-sparse-boundaries, gpu-prototype-neighbors-trajectory, blocked-ell-optimization-study, gpu-benchmark-sliced-preprocess-campaign, cellshard-hierarchy-reset, implement-derived-subset-and-reorder-materialization-for-cellshard-and-workbench.
- Started `cellshard-preprocess-gpu-biology-backbone` from the supplied implementation plan.
- Finished `cellshard-preprocess-gpu-biology-backbone`: moved Blocked-ELL/Sliced-ELL native preprocessing and CSR fallback ownership into CellShardPreprocess, moved preprocessing benchmarks there, and removed Cellerator preprocessing APIs and root targets.
- Checkpointed the earlier core layout migration: old `core/sparse` and `quantized` public paths were removed, matrix representation/runtime substrate/quantized packing moved under the then-current core include path, compute owns conversion and CUDA primitives, CellShard shims were updated, and focused Cellerator runtime checks passed. Standalone CellShard `cellShardMaskGroupsRuntimeTest` builds but exits 14 on its row-keep expectation.
- Checkpointed the matrix/compute boundary cleanup: `core/format` became `core/matrix`, conversion and bucket moved to `compute/matrix/convert`, `warp_reduce.cuh` moved under `src/compute/core/primitives`, and `cmake --build build -j 4` plus `./build/coreSparseLayoutRuntimeTest` passed.
- Started `cellerator-preprocess-rehome`: requested scope is a hard cutover that removes the CellShardPreprocess submodule/package and splits the moved implementation into Cellerator compute preprocessing kernels plus the Cellerator preprocessing policy/runtime API.
- Finished `cellerator-preprocess-rehome`: Cellerator now builds preprocessing compute and pipeline targets, ported preprocessing tests and benchmarks, CellShard no longer fetches or installs the old package, and the root no longer tracks the preprocessing submodule. Validation passed for the Cellerator build, focused preprocessing tests, adjacent Cellerator smoke tests, CellShard build, and CellShard package-consumer install check. `cellShardMaskGroupsRuntimeTest` still exits 14 as previously recorded under `cellerator-sparse-ml-layout`.
- Started `cellerator-python-preprocess-runtime`: Python should expose Scanpy-like Cellerator preprocessing entry points, but all omics preprocessing compute must delegate to GPU-native Cellerator/CellShard runtime paths.
- Finished `cellerator-python-preprocess-runtime`: direct `_cellerator` build, source smoke test, wheel build, and installed-wheel import smoke passed. A fixture-backed GPU session test remains a future follow-up because no stable tiny `.csh5` fixture is currently checked in.
- Started `cellerator-preprocess-scanpy-validation`: the PBMC3K `.h5ad` and `.csh5` fixtures now exist under `data/test/reference`, so the missing fixture-backed validation can compare Cellerator GPU-native preprocessing metrics against a Scanpy reference.
- Finished `cellerator-preprocess-scanpy-validation`: added `tests/validate_scanpy_preprocess.py`, exposed missing Python session metrics, fixed direct Python session device-reservation, fixed the `cell_mito_counts` alias, and passed the PBMC3K Scanpy comparison for all metric families.
- Started `cellerator-runtime-autotune`: requested scope is an optional, bounded runtime optimizer that is callable from Cellerator, defaults off in C++ mode, and is exposed in Python preprocessing as `autotune=True`.
- Finished `cellerator-runtime-autotune`: added the reusable light optimizer surface, plan-aware preprocessing compute calls, Python autotune/session metrics, README notes, and focused validation including PBMC3K `.csh5` autotune smoke.
- Finished `cellerator-sparse-ml-layout`: hard-cut the former core identity into direct Cellerator domains with no compatibility headers or `Cellerator::core` alias, updated sibling CellShard shims/package config, and validated Cellerator plus CellShard configure/build and focused Cellerator runtime tests.
- Finished `cellpack-m3-layout-selection`: CellPack now computes per-region padding/fill/index/value/output-byte metrics from packed coordinate plans, applies deterministic V100-oriented hybrid layout selection, and exposes `cellPackLayoutBench` summaries for CSR, current Blocked-ELL, and selected hybrid plan estimates.
- Started `cellpack-m4-oracle-gating`: requested scope is a narrow static-oracle gating experiment over M2/M3 CellPack plans. The prototype should select precompiled region spans through route masks/tapes and run coordinate-based CUDA SpMV forward plus transpose replay without dynamic module assembly, learned gates, real-data sweeps, or production Blocked-ELL/Sliced-ELL kernels.
- Finished `cellpack-m4-oracle-gating`: CellPack now has host route-mask and route-tape contracts, deterministic static oracle scenarios, a separate CUDA coordinate-span runtime target, focused host/CUDA tests, and `cellPackGatingBench` summaries for no-gating versus oracle-gating.
- 2026-08-14: Current thread resumed CP-BP-01 from the completed deterministic sampler and structural sampled-CSR bundle; next implementation is the pointer-first CPU/CUDA gene-support bitset primitive.
- 2026-08-14: Added include/Cellerator/compute/gene_support_bitset.hh and src/compute/dataset/gene_support_bitset.cu. The pointer-first host bundle retains immutable sampling provenance and sampled-position/global-row mapping; the CUDA builder uploads only CSR row pointers and column indices, constructs gene-major u32 support words with atomicOr, and counts unique detected cells despite duplicate entries.
- 2026-08-14: Added tests/gene_support_bitset_runtime_test.cu and normal CMake targets. Passed samplingRuntimeTest, datasetRuntimeTest, samplingMaterializationRuntimeTest, geneSupportBitsetRuntimeTest, and the opt-in geneSupportBitsetRuntimeTest --full-size-gpu smoke on Tesla V100.
- 2026-08-14: For 65,536 cells and 30,000 genes, words_per_gene=2,048 and support_bytes=245,760,000. The current CUDA convenience builder holds host and device support concurrently; peak structural storage is 2*support_bytes + 2*(genes*4) + cells*8 + (cells+1)*4 + nnz*4 + 4 bytes, excluding small provenance allocations.
- 2026-08-14: Claimed by the current serial thread after confirming no live agent owns CP-BP-02. Step 1 sampling, dataset, materialization, support-bitset, and full-size V100 smoke tests passed before edits.
- 2026-08-14: Added the Cellerator pointer-first host candidate contract and fixed SplitMix64-v1 MinHash/LSH provenance. CPU and CUDA paths omit empty genes, use stable CUB radix sorting/scans/unique, cap oversized buckets with a deterministic circular window, and return lexicographically sorted unique canonical pairs.
- 2026-08-14: Focused candidate runtime test passes CPU/GPU exact agreement. The 64-gene exhaustive fixture retained 48/48 deliberately high-overlap pairs, emitted 48/2,016 candidates, and reduced the unordered pair set by 97.619%.
- 2026-08-14: Final Step 1 regressions and CP-BP-02 CPU/CUDA tests passed. The full 65,536-cell support smoke allocated 245,760,000 bytes; no Step 1 API was changed.
- 2026-08-14: Serialized Tesla V100 sm_70 benchmark at 65,536 cells x 30,000 genes produced 105,000 candidates from 449,985,000 exhaustive pairs, retained all 105,000 constructed cluster pairs, and reduced the unordered pair space by 99.9767%. Three timed runs after one warmup measured 59.832 ms minimum and 60.904 ms median.
- 2026-08-14: Benchmark provenance reports 14,060,031 bytes of CUB scratch, 333,424,355 bytes of accounted peak device allocation, and a 534,306,000-byte conservative fixed bound excluding CUB. The result is synthetic correctness smoke, not a production recall threshold.
- 2026-08-14: CP-BP-02 completed deterministic SplitMix64-v1 global-row MinHash, configurable LSH, bounded oversized-bucket handling, CUB grouping/deduplication, canonical host candidate pairs, CPU/GPU exact tests, and a serialized V100 smoke benchmark. CP-BP-03 was not started.

## Next Actions
- Primary next implementation: CP-BP-05 frozen-plan application. It is the
  first missing transform after the completed semantic plan and unlocks
  CP-BP-06/07/08/09 and later persistence work without absorbing their record,
  tile, kernel, or serialization contracts.
- Runner-up: CP-BP-03 exact merge-cost scoring, which can independently adapt
  completed CP-BP-02 pairs into exact `candidate_relation` evidence through the
  evaluator-compatible cost seam. The CUDA evaluator remains a separate
  deferred acceleration stream.
- CP-BP-11 validation contracts/null references are also independently
  available. Read each child ledger before claiming it.
- Reactivate blocked children only when their recorded representation/API
  prerequisite lands; do not guess a downstream physical ABI.
- CP-BP-02 is closed. CP-BP-03 may consume gene_candidate_pair_view and its immutable provenance without reinterpretation; persistent device-resident Step 1 support remains an optional optimization.

## Done Criteria
- Every active workstream in `todos/` is reflected here with a current status.
- CP-BP-01 through CP-BP-13 satisfy their child done criteria, including exact
  decode/numerical equivalence, held-out/null/stability evidence, fair existing
  layout baselines, and a no-repack durable execution lifecycle.

## Historical Summary
- Recent completed work included Blocked-ELL persistence, real-data sparse replay benchmarking, quantize sparse operator kernels, workbench browse-cache updates, semantic the biology semantics package cleanup, and the initial pointer-first neighbor workspace refactor.
- Detailed historical workstream notes are preserved in git history rather than as active repo ledgers.
