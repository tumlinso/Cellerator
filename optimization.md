# Cellerator Optimization Deep Dive

## 1. Scope

### 1.1 Assumptions

- This document assumes Tesla V100 16 GB GPUs on Volta `sm_70`.
- It follows the `cuda-v100` skill path: general V100 strategy first, then kernel-mechanics reasoning.
- The codebase state inspected here is the current working tree, including in-progress migration from older `_rna` and `_models` surfaces into `src/compute/*` and `src/models/*`.

### 1.2 Method

- This pass is based on static code inspection, target layout inspection, and focused benchmark/profiler work on the sparse autograd surface.
- Nsight Systems and Nsight Compute were used on representative large `SpMV` runs in `computeAutogradBench`.
- Where a section is still call-shape guidance rather than measured truth, it is called out explicitly.

### 1.3 Design Posture

- This repo is intentionally built from low-level, performance-visible pieces.
- The goal is not to abstract kernels, sparse layouts, residency boundaries, or transfer structure away behind generic convenience layers.
- When this document recommends a refactor, it is recommending a better low-level surface or a more reusable concrete operator, not a move toward a more opaque framework.
- In this codebase, reusable building blocks should stay explicit enough that their memory traffic, synchronization points, and launch structure can still be reasoned about directly.

### 1.4 What The Estimates Mean

- `fixed overhead` means cost paid even when tensors are small: launch latency, descriptor creation, alloc/free, sync, tensor conversion, map lookup, host parsing setup.
- `streaming overhead` means bytes moved or touched: PCIe transfers, HBM passes, CPU memory copies, sort passes, tensor densification.
- When a path is described as `HBM-bound`, `PCIe-bound`, `CPU-parse-bound`, or `launch-bound`, that is an inference from the implementation shape.

## 2. Fast Cost Model

### 2.1 Common Fixed Costs On This Host Class

| Operation class | Estimated fixed overhead per call | Scales with | Notes |
| --- | ---: | --- | --- |
| CUDA kernel launch | ~`3-8 us` | launch count | Usually not the main problem until kernels are small or trains are long |
| cuSPARSE SpMV dispatch after warmup | ~`10-30 us` host-side | descriptor churn, stream state | Real cost becomes HBM traffic quickly |
| cuSPARSE descriptor create/destroy pair | ~`15-60 us` | call count | Recreating descriptors in hot loops is visible |
| CUB reduction launch path | ~`10-30 us` | row count | Usually cheap relative to full sparse passes |
| `cudaMemcpyAsync` pinned H2D/D2H over PCIe Gen3 | raw floor `bytes / 12 GB/s` | bytes transferred | `1 MB ~= 0.08 ms`, `128 MB ~= 10.7 ms` before software overhead |
| `cudaStreamSynchronize` / `cudaDeviceSynchronize` | queue-drain boundary | all queued work | Cheap only when nothing substantial is pending |
| `cudaMalloc` / `cudaFree` | `10s of us` to low `ms` | allocator state, size | Per-call allocation in steady state is unacceptable |
| Torch sparse layout conversion | O(`rows + nnz`) | metadata + value bytes | CSR<->COO conversion is a real copy, not a view |
| CPU `std::sort` on row metadata | O(`n log n`) | rows | Fine one-time, expensive inside iterative paths |

### 2.2 Repo-Wide Bottleneck Classes

1. Ingest is mostly CPU parse and PCIe staging bound.
2. Sparse preprocessing is mostly HBM-pass and dispatch bound.
3. Libtorch model paths are currently dominated by sparse layout conversion and dense materialization more than by raw GEMM math.
4. Forward-neighbor search is structurally hybrid: GPU scores candidates, but host orchestration and repeated query staging still consume too much fixed overhead.
5. Trajectory building pays one GPU scoring stage, then returns to CPU-heavy graph assembly.

## 3. Cross-Cutting Findings

### 3.1 What Is Already Good

- `src/compute/preprocess/workspace.cuh` preallocates large device slabs and avoids repeated steady-state allocation when dimensions are stable.
- `src/ingest/mtx/compressed_parts.cuh` uses pinned host staging and bulk copies instead of a stream of tiny transfers.
- The quantized backend under `src/quantized/` is explicitly custom-kernel and does not pretend sparse irregular work is a Tensor Core problem.
- NCCL is wired in as the fast path for multi-GPU gene-metric reduction.
- The torch binding layer is explicit about being a copy boundary instead of hiding expensive aliasing semantics.

### 3.2 What Is Structurally Expensive

- Several hot paths recreate library descriptors or temporary device buffers at call time instead of caching them in workspaces.
- Several model paths convert sparse CSR into other Torch layouts on every forward pass.
- Some model losses densify sparse batches, which turns a sparse training problem into a dense memory-traffic problem.
- Neighbor search repeatedly uploads query blocks and downloads intermediate top-k state per shard.
- Trajectory building uploads full latent tables to device, then downloads candidate tables and continues on CPU, so the pipeline is not end-to-end resident.

### 3.3 Divergence Versus Launches

- Divergence exists in row-length-skewed kernels such as preprocessing and neighbor search, but it is not the first-order problem in most places.
- The larger issues are repeated full-memory passes, repeated host-device boundaries, and repeated setup work.
- On V100, moderate divergence is cheaper than exploding a path into many tiny kernels or host-visible staging steps. That rule applies to current preprocess kernels and the exact neighbor kernels.

### 3.4 Memory Tier Guidance

- Keep gene metrics, row masks, and active-row counts on device global memory until the final reporting boundary.
- Keep query centroids, query latent blocks, and top-k scratch on device across neighbor-search blocks instead of rebuilding them per block.
- Keep ingest pinned slabs alive across windows and use them in double-buffered parse/convert/store pipelines if ingest throughput becomes a priority.
- Keep model-facing sparse batches in one sparse layout if possible; the current CSR->COO path is paying extra metadata traffic for no obvious end-to-end win.

### 3.5 Sparse Autograd Findings

- Large CSR `SpMV` on V100 is decisively memory-bound, not Tensor Core-bound. The hot kernel reached high DRAM pressure and low SM math utilization in Nsight Compute.
- Pair-local row sharding across the real NVLink pairs scales well for CSR `SpMV` because it keeps the workload bandwidth-heavy and avoids expensive global merges.
- Four-way feature-sharded CSR `SpMV` improves latency, but it does not drive all four V100s like a Tensor Core GEMM. The per-GPU shards become too thin and the math intensity stays low.
- If the goal is high sustained board power and Tensor Core usage, CSR `SpMV` is the wrong target. The sparse Tensor Core direction for this host class is Blocked-ELL `SpMM`.
- That makes the right storage strategy dual-path:
  - keep CSR/row-compressed as the canonical sparse storage and general bioinformatics path
  - add Blocked-ELL as the preferred resident execution layout for repeated sparse x dense-matrix projection

## 4. Build And Configuration Surface

### 4.1 `CMakeLists.txt`

- The build defaults to `sm_70`, `-O3`, `--use_fast_math`, and `-Xptxas=-dlcm=ca`, which is directionally right for V100.
- `INTERPROCEDURAL_OPTIMIZATION_RELEASE` is enabled for performance targets. That helps host code more than kernels, but it is still a reasonable default.
- The target list is useful because it exposes three major performance domains:
  - ingest and preprocessing
  - quantization backend
  - libtorch model, custom-op, and compute-native sparse runtime surfaces

### 4.2 Optimization Interpretation

- The build is already declaring Volta intent correctly.
- The major remaining wins are algorithmic and residency related, not compiler-flag related.
- The right optimization direction is usually a more explicit low-level operator or workspace, not a more abstract façade.

## 5. Ingest And Series Conversion

### 5.1 Manifest And Metadata Paths

Relevant files:

- `src/ingest/series/series_manifest.cuh`
- `src/ingest/common/metadata_table.cuh`
- `src/ingest/common/text_column.cuh`

Observations:

- These paths are synchronous, host-only, and intentionally copy-heavy.
- `manifest::load_tsv()` and `metadata_table::load_tsv()` tokenize line-by-line and append into packed text columns.
- This is acceptable control-plane code, not a GPU hot path.

Estimated overhead when called:

- Fixed cost is low.
- Cost scales linearly with input bytes and field count.
- The dominant limiter is CPU parsing and allocator traffic, not CUDA.

Performance comment:

- There is no reason to push this parsing onto GPU.
- If it becomes noticeable, the right fix is fewer temporary string copies and larger buffered reads, not device offload.

### 5.2 MTX Header Scan And Partition Planning

Relevant files:

- `src/ingest/mtx/mtx_reader.cuh`
- `src/ingest/series/series_partition.cuh`

Observations:

- `scan_row_nnz()` does a full first pass over the MTX text source.
- `plan_row_partitions_by_nnz()` then creates row windows from the scanned counts.
- This is correctly biased toward large streaming passes rather than random access.

Estimated overhead when called:

- One full source-file scan: O(file bytes).
- Partition planning: O(rows).
- `scan_row_nnz()` is entirely CPU and branch-heavy; the GPU is idle.

Interpretation:

- This is CPU-parse-bound, not launch-bound.
- The row-sorted detection is cheap compared with parsing and worth keeping because it unlocks a faster compressed-build path later.

### 5.3 COO-To-Compressed Conversion Workspace

Relevant file:

- `src/ingest/mtx/compressed_parts.cuh`

Good choices:

- One pinned host slab and one device slab are allocated and reused.
- The same workspace can service both the sorted fast path and the fallback atomic path.
- Bulk H2D/D2H copies are used instead of per-row or per-line transfers.

Important cost model:

- `build_pinned_triplet_to_compressed()` performs:
  - 3 H2D copies: row index, col index, value
  - 1 compressed-build launch train
  - 3 D2H copies: major pointer, minor index, value
  - 1 stream synchronization

Raw transfer floor per call:

- H2D bytes: `nnz * (4 + 4 + 2)`
- D2H bytes: `(rows + 1) * 4 + nnz * (4 + 2)`
- Total raw PCIe floor: approximately `(16 * nnz + 4 * (rows + 1)) / 12 GB/s`

Examples:

- `1M nnz` and `64K` rows: raw transfer floor is about `1.3-1.5 ms` before parse cost, kernel work, and sync overhead.
- `16M nnz`: raw transfer floor alone is about `21-24 ms`.

Important conclusion:

- On small windows, call setup and parse dominate.
- On large windows, PCIe plus parse dominate unless the COO->CSR conversion kernel becomes pathological.

### 5.4 End-To-End Window Conversion

Relevant function:

- `convert_transposed_byte_range_to_row_compressed_parts()`

Pipeline:

1. Parse source byte range into pinned COO arrays.
2. Upload triplets to device.
3. Build compressed form.
4. Download compressed arrays.
5. Split and store one file per part.

Estimated overhead per window:

- Fixed overhead: one conversion call, six bulk copies, one full stream sync, and one host-side split/store pass.
- Variable overhead: O(window bytes parsed + window nnz transferred + window nnz converted + output bytes written).

Dominant limiter:

- CPU parse and PCIe, not HBM.

Optimization comment:

- The current design is throughput-sensible for correctness and staging reuse.
- The next real step would be a double-buffered parser/converter pipeline:
  - CPU thread parses window `N+1` into pinned slab B
  - GPU converts window `N` from pinned slab A
  - host thread stores window `N-1`
- That would attack idle gaps without rewriting the core conversion logic.

### 5.5 Series HDF5 Conversion

Relevant file:

- `src/ingest/series/series_ingest.cuh`

Observations:

- `convert_manifest_mtx_series_to_hdf5()` is intentionally orchestration-heavy.
- It does full-dataset planning on CPU, merges global feature identity through `std::unordered_map<std::string, std::uint32_t>`, and uses many `std::vector` surfaces to build the final layout.
- That is acceptable because this is an offline ingest pipeline.

Main costs:

- repeated full scans of matrix metadata
- barcode and feature-table loads
- feature deduplication via string hash table
- repeated window conversion and file emission

Optimization comment:

- This is not where custom CUDA should go first.
- The two worthwhile throughput levers are:
  - overlapping parse/convert/store
  - reducing host string work if feature global-ID construction shows up in profiles

## 6. Sparse Preprocessing

### 6.1 Relevant Surface

- `src/compute/preprocess/types.cuh`
- `src/compute/preprocess/workspace.cuh`
- `src/compute/preprocess/kernels.cuh`
- `src/compute/preprocess/operators.cuh`
- `bench/scrna_preprocess_bench.cu`

This is the cleanest performance-oriented subsystem in the repo today.

### 6.2 Workspace Strategy

Good choices:

- one `device_workspace` per GPU
- one `distributed_workspace` for cross-device reduction
- contiguous slabs for cell metrics and gene metrics
- cached CUB and cuSPARSE temporary buffers

Why this matters:

- `cudaMalloc`/`cudaFree` inside per-part preprocessing would cost `10s-100s of us` per event and create allocator jitter.
- The current reserve model amortizes that correctly when dimensions are stable.

### 6.3 `compute_cell_metrics()`

Relevant code:

- `compute_cell_metrics_kernel` in `src/compute/preprocess/kernels.cuh`

Shape:

- warp-per-row
- row-local reduction for total counts, mitochondrial counts, max counts, and detected-gene count

Estimated overhead per call:

- Fixed: 1 kernel launch, usually `3-8 us`
- Variable: one pass over row pointers and all row nnz values

Performance interpretation:

- Dominant limiter is HBM reads from CSR `val` and `minorIdx`.
- Divergence from variable row lengths exists, but it is expected and not the primary problem.
- The warp-per-row mapping is a reasonable V100 choice as long as rows are not extremely long and skewed.

Optimization comment:

- Do not split this into smaller kernels.
- If row-length skew becomes extreme, binning rows by nnz class would be the next move, not CPU preprocessing.

### 6.4 `normalize_log1p_inplace()`

Shape:

- second pass over all row nnz
- in-place overwrite of `src.val`

Estimated overhead per call:

- Fixed: 1 launch
- Variable: reads `total_counts`, reads half values, writes half values, computes `log1p`

Interpretation:

- This is still mostly memory-pass-bound.
- Arithmetic is heavier than in `compute_cell_metrics()`, but not enough to make it compute-bound on sparse CSR data.

Optimization comment:

- Keeping normalization in-place is correct.
- The main cost is paying another full sparse-value pass, which is unavoidable unless normalization is fused into a later pass that already touches every nnz.

### 6.5 `accumulate_gene_metrics()`

This is the preprocessing hotspot.

Call sequence:

1. `expand_keep_mask_kernel` or `fill_ones_kernel`
2. CUB reduce for kept-row count
3. `add_scalar_kernel`
4. `convert_values_kernel`
5. transpose SpMV for gene sum
6. `square_values_kernel`
7. transpose SpMV for gene squared sum
8. `fill_ones_kernel`
9. transpose SpMV for detected-cell counts

Estimated fixed overhead per part:

- `6-8` explicit CUDA kernel launches
- `1` CUB reduction dispatch
- `3` cuSPARSE SpMV dispatches
- practical fixed overhead floor: roughly `60-250 us` per part before the real sparse traffic dominates

Dominant limiter:

- repeated HBM passes over the sparse payload
- repeated library dispatch
- descriptor churn in `run_spmv_transpose()`

Important structural issue:

- `run_spmv_transpose()` recreates CSR and dense-vector descriptors every time.
- That adds host overhead and makes graph capture or extremely tight steady-state loops less attractive.

Interpretation using the kernel-mechanics path:

- The problem is not primarily divergence.
- The problem is too many full-memory passes and too much dispatch structure around the same sparse matrix.

Optimization priorities:

1. Cache cuSPARSE descriptors inside the workspace for stable `src` views.
2. Reuse x/y dense-vector descriptors when dimensions stay constant.
3. Keep `active_rows` entirely on device until the end of the pipeline.
4. Benchmark whether one custom fused nnz traversal that accumulates `sum`, `sq_sum`, and `detected` via atomics beats `3x` transpose SpMV on the real sparsity distribution.

Important nuance:

- The fused custom-kernel option is not automatically better.
- If gene contention is high, the current library-backed path may still win on V100.
- This is exactly the case where Nsight Compute should decide, not aesthetics.

### 6.6 `build_gene_filter_mask()`

Shape:

- copies one scalar `active_rows` from device to host
- synchronizes the stream
- launches one filter kernel

Estimated overhead per call:

- D2H scalar copy and sync: typically `10-50 us` when the stream is otherwise idle, but effectively a hard serialization boundary when queued work is still running
- 1 kernel launch

Performance comment:

- The scalar copy is small, but the stream sync is architecturally expensive because it breaks overlap.
- This is an easy win: compute `inv_cells` on device and keep the path asynchronous.

### 6.7 Cross-GPU Reduction

Relevant code:

- `allreduce_gene_metrics()` in `src/compute/preprocess/workspace.cuh`

NCCL path:

- `4` separate allreduces:
  - `gene_sum`
  - `gene_sq_sum`
  - `gene_detected`
  - `active_rows`

Fallback path:

- synchronize all devices
- copy all arrays to pinned host memory
- reduce on host
- copy reduced arrays back to every device

Estimated fallback transfer floor for `D` devices and `C` columns:

- D2H + H2D bytes are approximately `2 * D * (3 * C * 4 + 4)`
- For `D=4`, `C=32768`, that is about `3.1 MB` total traffic, raw floor around `0.26 ms`
- For `C=200000`, that rises to about `19.2 MB`, raw floor around `1.6 ms`

Interpretation:

- NCCL is the only acceptable steady-state path.
- The fallback path is fine for correctness and tests, but it is not a scaling path.

V100 topology note:

- This code does not encode the actual fast pairs `0<->2` and `1<->3`.
- If the process layout above it is topology-unaware, the NCCL path can still pay cross-pair penalties.

Optimization priorities:

1. Flatten the three gene-metric arrays into one contiguous reduction buffer and allreduce once.
2. Ensure rank placement preserves pair-local traffic when possible.
3. Keep the fallback only as a correctness path.

### 6.8 Overall Preprocess Verdict

- Library-backed where it makes sense: yes.
- Assumes `sm_70`: yes.
- Dominant limiter: HBM traffic and dispatch structure, not occupancy collapse.
- Divergence present: yes, but not first-order.
- Best next step: fewer passes and fewer synchronization points, not “more generic abstraction”.

## 7. Torch Bindings And Libtorch Model Surface

### 7.1 Explicit Copy Boundary

Relevant file:

- `src/torch/bindings.hh`

This file is honest about cost. That is good engineering.

Actual behavior:

- always allocates brand-new CPU tensors
- widens 32-bit CSR metadata to Torch 64-bit metadata
- copies values out of CellShard storage

Estimated overhead per call:

- O(`rows + nnz`) CPU copy and widening
- one sparse CSR tensor materialization

Example footprint for exporting one part with `1M nnz`:

- column indices alone become `8 MB`
- crow indices are `8 * (rows + 1)` bytes
- values are `2 MB` in FP16 or `4 MB` in FP32

Interpretation:

- This is acceptable at an interop boundary.
- It is unacceptable inside a per-step hot loop.

### 7.2 `BalancedTimeSampler`

Relevant file:

- `src/models/developmental_time/dT_dataloader.hh`

What it does:

- bins rows by day
- fetches needed CellShard parts on demand
- assembles a CPU `torch::sparse_csr_tensor`

Main fixed costs per batch:

- mutex acquisition
- vector growth and row-span bookkeeping
- optional part fetch and drop
- CSR tensor construction

Main variable costs per batch:

- copying row-local `minorIdx` and `val` slices into Torch-owned CPU buffers

Important performance note:

- This is a control-plane/data-loader path, not a GPU execution path.
- If the same parts are fetched and dropped repeatedly, storage I/O can dominate everything else.

Optimization comment:

- The cheapest improvement is to cache hot parts longer when training locality is high.
- The more ambitious improvement is a device-resident sparse batch builder, but that is a larger redesign.

### 7.3 Developmental-Time Model Forward

Relevant file:

- `src/models/developmental_time/dT_model.hh`

Critical path:

1. `sparse_csr_batch.to(torch::kFloat32)`
2. if CSR, `to_sparse()` to COO
3. `torch::matmul(sparse_input, projection_weight_)`
4. dense MLP head

This is the single most important model-surface observation in the repo.

Performance interpretation:

- The model math itself is reasonable.
- The expensive part is the format churn before the math.

Why it hurts:

- CSR->COO conversion is a full metadata transform.
- FP16->FP32 widening touches every nonzero.
- If this happens every step, the model is paying a sparse reformat tax before every useful GEMM-like operation.

Estimated overhead when called:

- Fixed: Torch dispatcher and sparse conversion setup
- Variable: O(`rows + nnz`) for layout conversion plus sparse-dense matmul cost

Optimization priority:

- Either stay in CSR all the way into the sparse matmul path, or define a narrower custom-op boundary.
- The current path is library-backed for the dense projection, but prefaces it with expensive sparse bookkeeping.

### 7.4 Developmental-Time Inference

Relevant file:

- `src/models/developmental_time/dT_infer.hh`

Pipeline per part:

1. fetch part if needed
2. export part as CPU Torch sparse tensor
3. move tensor to `config.device`
4. run model
5. move predictions back to CPU

Interpretation:

- This is explicitly PCIe-visible inference.
- It is acceptable for offline batch inference by part.
- It is not a low-overhead streaming inference design.

Estimated overhead per part:

- export copy + index widening on CPU
- H2D transfer of sparse tensor payload
- D2H transfer of dense predictions

If a part has `N` nonzeros, raw H2D payload is roughly:

- `8 * (rows + 1) + 8 * N + value_bytes * N`

That makes metadata much more expensive than CellShard’s native 32-bit CSR representation.

### 7.5 Dense-Reduce Model

Relevant file:

- `src/models/dense_reduce/dR_model.hh`

Good part:

- forward uses sparse projection first, which is the right high-level idea.

Expensive parts:

- `maybe_corrupt_sparse_()` converts CSR to COO and coalesces in training mode
- `compute_dense_reduce_loss()` calls `batch.features.to_dense().to(torch::kFloat32)`

The loss path is the largest model-side red flag in the repo.

Why:

- A sparse minibatch becomes a dense `batch_rows x genes` tensor.
- That shifts cost from sparse nnz traffic to full dense matrix traffic and memory footprint.

Example:

- `8192 x 32768` dense FP32 target is about `1.0 GB`
- even `2048 x 32768` is about `256 MB`

Interpretation:

- This is likely to dominate both memory and time before the decoder math becomes the bottleneck.

Optimization priority:

- Keep the reconstruction loss sparse-aware.
- If exact dense reconstruction is required, compute it in tiles or only over sampled negatives plus known nonzeros.

### 7.6 Quantize Model Surface

Relevant file:

- `src/models/quantize/quantize.hh`

Observations:

- The quantizer surface is convenient, but many helper paths densify or copy tensors on CPU.
- `pack_dense_quantized_matrix_impl_()` is host-centric and appropriate for export or offline packing, not hot online training.
- Future-neighbor supervision introduces another cross-subsystem dependency where host row lookup and dense reference features can become a cost center.

Interpretation:

- This is fine as a training or export scaffold.
- The actual hot backend lives in `src/quantized/`, not in the model wrapper.

## 8. Custom Model Ops

### 8.1 Relevant Surface

- `src/compute/model_ops/model_ops.cu`

This is the correct place for CUDA-specialized Torch work.

### 8.2 Dense-Reduce Pair Loss

Observations:

- forward kernel uses atomic accumulation into scalar totals and counts
- backward kernel uses atomic updates into `grad_latent`

Interpretation:

- For modest sampled pair counts this is reasonable.
- For very large pair counts, atomic contention can become visible, especially in backward.

Estimated overhead when called:

- Fixed: 1 forward launch, 1 backward launch, several scalar host reads through `.item<>()`
- Variable: O(`pair_count * latent_dim`)

Optimization comment:

- The kernel is simple enough that launch overhead is not the main issue once pair counts are large.
- If this becomes hot, block-local reduction before global atomic add is the natural next step.

### 8.3 Developmental-Stage Bucket Loss

Observations:

- forward uses one accumulate kernel and one single-thread finalize kernel
- backward is one elementwise kernel

Interpretation:

- The single-thread finalize kernel is acceptable because bucket count is small relative to row count.
- The bigger issue is scalar extraction via `.item<>()` in autograd plumbing, which introduces synchronization at the Torch boundary.

### 8.4 Weighted Future Target

Observations:

- straightforward dense gather-weighted sum kernel
- correct custom-op boundary because it keeps the weighted blend on device

Interpretation:

- This is library-independent custom glue and belongs here.

### 8.5 Compute-Native Sparse Autograd Runtime

Relevant files:

- `src/compute/autograd/autograd.hh`
- `src/compute/autograd/runtime.cu`
- `src/compute/autograd/kernels/base_sparse.cu`
- `src/compute/autograd/kernels/dist_sparse.cu`
- `src/compute/autograd/primitives/common.cuh`

Observations:

- this surface is lower-level than a framework runtime: it is pointer-first, CSR-first, and explicit about streams, scratch, cuSPARSE caches, and fleet topology
- `autograd.hh` exposes raw-buffer contexts rather than tensor-wrapper-heavy hot paths
- `base_sparse.cu` provides the single-GPU reference kernels and cuSPARSE-backed library paths for:
  - CSR row scaling with custom backward
  - sparse value reduction with CUB
  - CSR SpMV with custom value-grad and library-backed vector-grad
  - CSR SpMM with custom value-grad and library-backed rhs-grad
- `dist_sparse.cu` launches one base copy per selected slot and performs explicit leader-merge reduction instead of hiding cross-device traffic behind a framework boundary
- the distributed policy matches the host assumptions in the repo: pair-local work first, then leader merge across the real 4-GPU topology

Interpretation:

- this is the correct direction for sparse model code that would otherwise pay Torch layout conversion and boxed autograd overhead
- this is not trying to become a generic autograd framework; it is trying to expose reusable low-level sparse building blocks without abstracting their cost model away
- the runtime is not yet a full training system, but it already demonstrates the right split:
  - library-backed sparse linear algebra where the library is clearly best
  - custom kernels for sparse glue and fused value-gradient logic
- the highest-value model-facing use is sparse projection, because `dense_reduce` and `developmental_time` both currently pay sparse layout churn around `torch::matmul`

Estimated overhead when called:

- `csr_row_scale`
  - Fixed: `1` forward launch, up to `2` backward launches
  - Variable: O(`rows + nnz`)
  - Limiter: HBM traffic over CSR value segments
- `sparse_value_sum`
  - Fixed: CUB dispatch plus scratch-size query
  - Variable: O(`nnz`)
  - Limiter: contiguous value-array bandwidth
- `csr_spmv`
  - Fixed first use: cached CSR descriptor create plus `cusparseSpMV_bufferSize`
  - Fixed steady state: dense vector descriptor create/destroy plus one SpMV call
  - Variable: O(`nnz`)
  - Limiter: HBM bandwidth for CSR walk, not launch count
- `csr_spmm`
  - Fixed: dense matrix descriptor create/destroy plus `cusparseSpMM_bufferSize`
  - Variable forward and weight-grad: O(`nnz * out_dim`) sparse-dense streaming work
  - Variable value-grad: O(`nnz * out_dim`) dot products over the touched dense feature rows
  - Limiter: usually HBM traffic once `out_dim` is moderate; on tiny `out_dim`, descriptor churn becomes visible
  - Note: the custom value-gradient path avoids recovering sparse value grads from a dense outer-product formulation

Optimization comment:

- `csr_spmv` is already on the correct backend for V100 and now caches the CSR descriptor plus workspace-size query; the next win is caching dense vector descriptors only if repeated vector shapes dominate host overhead
- the value-gradient kernel is intentionally custom because sparse value gradients are cheaper to emit directly than to recover from a dense outer-product path
- `csr_spmm` is the more important projection primitive for this repo because it matches sparse batch to hidden-feature projection; the next likely optimization is fusing bias or a lightweight epilogue only if it removes a full output pass without causing register spill problems
- this runtime should be the preferred home for future sparse training ops that currently cross the libtorch boundary only to come back to explicit CUDA

## 9. Forward-Neighbor Index And Search

### 9.1 Build Path

Relevant files:

- `src/compute/neighbors/forward_neighbors/fn_index.hh`
- `src/compute/neighbors/forward_neighbors/fn_query.hh`
- `src/compute/neighbors/forward_neighbors/forward_neighbors.cu`

Build path shape:

1. append batches on CPU
2. optional latent renormalization on CPU
3. sort by embryo/time/cell on CPU
4. segment into shard-local spans
5. build ANN centroids on CPU
6. convert latent to half and upload per shard
7. build `row_by_cell_index` hash map on CPU

Interpretation:

- This is an index-build path, so CPU sorting and vector work are acceptable.
- The one-time cost is O(`rows log rows + rows * latent_dim`), which is fine if the index is reused.

### 9.2 Query-By-Cell-Index Path

Relevant function:

- `ForwardNeighborIndex::query_batch_from_cell_indices()`

Observations:

- Performs hash lookup for every requested cell
- copies latent row back out of shard CPU storage into a fresh query buffer

Interpretation:

- Fine for moderate batch sizes.
- The CPU-side latent copy is a fixed staging cost before any GPU scoring happens.

### 9.3 Search Core

This is the main issue in the neighbor subsystem.

For each shard and each query block, the code currently does all of the following:

1. slices a query block on CPU
2. converts query latent F32 to half on CPU
3. allocates device buffers for query latent, lower bounds, upper bounds, embryo ids
4. uploads those buffers
5. allocates device buffers for best-candidate arrays
6. launches exact or ANN kernels
7. synchronizes
8. downloads candidate arrays
9. merges candidates on host into `std::vector<std::vector<Candidate>>`

Estimated fixed overhead per `(shard, query block)`:

- multiple `cudaMalloc`-backed `device_buffer` constructions unless allocator caching masks them
- 4 H2D uploads for query metadata
- 1 best-array init path
- `1-2` search kernel launches
- 1 D2H download of candidate state
- host merge work

Practical implication:

- For small and medium query batches, fixed overhead can dominate the actual similarity math.
- A query of a few hundred rows can still pay `0.1-0.4 ms` or worse per shard/block just in orchestration, before the search interval is large enough to amortize it.

### 9.4 Exact Search Kernel

Shape:

- one block per query row
- 32 threads per block
- scans candidate interval and maintains local top-k

Interpretation:

- Good when each query row has substantial candidate work.
- Occupancy is not the first-order issue; staging and repeated interval launches are.

Divergence note:

- Query intervals vary, but that is not the first problem to solve.
- The cost center is repeated block staging plus host merge, not warp control flow.

### 9.5 ANN Search Path

Key issue:

- eligible ANN centroids are rebuilt into host vectors and uploaded for every query block

Interpretation:

- This defeats index residency.
- The ANN path keeps the index on device but still re-stages a meaningful fraction of search metadata every block.

### 9.6 `same_embryo_first`

Observations:

- runs the whole search twice
- then host-merges results while deduplicating neighbors

Interpretation:

- This roughly doubles search cost for that policy.
- It is logically simple but expensive.

### 9.7 Optimization Priorities

1. Add a persistent query workspace per device:
   - query latent
   - lower/upper time bounds
   - embryo IDs
   - top-k scratch
2. Reuse that workspace across blocks instead of allocating per block.
3. Keep ANN list metadata resident on device instead of rebuilding eligible centroid arrays on host.
4. Merge shard-local top-k on device or at least download one compact final structure instead of repeated intermediate arrays.
5. If multi-GPU query traffic grows, align shards with the real NVLink pairs instead of assuming ordinal locality.

Overall verdict:

- assumes `sm_70`: yes
- recommendation class: partly custom-kernel, partly orchestration rewrite
- dominant limiter: launch/setup plus host-device staging, not raw math throughput

## 10. Trajectory Graph Construction

### 10.1 Record Table And Slab Planning

Relevant files:

- `src/compute/graph/record_table.cuh`
- `src/compute/graph/slab_index.cuh`

Observations:

- these are host-side vector-backed tables
- sorting, embryo span construction, and time slab planning are CPU work

Interpretation:

- This is fine for an analysis/build pipeline.
- The main question is whether the GPU scoring stage is large enough to justify its upload/download boundary.

### 10.2 Forward Candidate Scoring

Relevant file:

- `src/compute/graph/forward_candidates.cuh`

Pipeline:

1. upload full latent table
2. upload time vector and window bounds
3. launch one candidate-scoring kernel
4. synchronize
5. download dst, similarity, and delta-t tables

Estimated transfer footprint:

- upload latent: `rows * latent_dim * 2` bytes
- upload time: `rows * 4` bytes
- upload bounds: `rows * 8` bytes
- download candidate outputs: `rows * k * (4 + 4 + 4)` bytes

Example:

- `1M` rows, `32` latent dims, `k=4`
- latent upload alone is about `64 MB`
- candidate download is about `48 MB`

Interpretation:

- This path is dominated by PCIe unless the table is already large enough that the exact candidate scoring math dwarfs transfer time.
- Because the next steps are CPU-side anyway, the current shape is a one-shot accelerator stage, not an end-to-end GPU pipeline.

### 10.3 Graph Prune, Tree, Supernodes, DAG

Relevant files:

- `src/compute/graph/forward_prune.cuh`
- `src/compute/graph/supernode_reduce.cuh`

Observations:

- `prune_candidate_edges()` sorts candidate rows on CPU
- `build_principal_tree()` builds inbound graph and tree metadata on CPU
- `build_supernodes()` computes centroids on CPU
- `build_supernode_dag()` uses per-row `std::unordered_map`

Interpretation:

- This code is algorithmically clear.
- It is not GPU-first, and that is okay if trajectory build is offline and not repeatedly called in tight loops.

Dominant limiter:

- CPU memory traffic and allocator churn for large graphs

Optimization comment:

- If trajectory build becomes an online operation, the biggest win would be to keep candidate tables resident and move prune/reduce stages onto device.
- If it stays offline, clarity is probably the right tradeoff.

## 11. Microscaled Quantized Backend

### 11.1 Backend Shape

Relevant files:

- `src/quantized/README.md`
- `src/quantized/kernels.cuh`
- `src/quantized/packing.cuh`

This subsystem is architecturally aligned with the repo goals.

Why:

- custom-kernel where sparse irregular work warrants it
- policy metadata is compile-time templated
- one packed CSR path instead of many runtime branches

### 11.2 Kernel Mechanics

Current design:

- one thread handles one row in `quantize_block_kernel` and `dequantize_block_kernel`
- packing helpers specialize the `Bits == 8` path with vectorized `uint4` stores/loads when alignment allows

Interpretation:

- For dense reconstruction packed as row-major CSR with uniform row width, this is a good Volta design.
- The backend is mostly HBM-traffic-bound with some per-element quantization math.

Estimated overhead when called:

- Fixed: one launch per block region, usually negligible once rows are large
- Variable: O(`nnz`) reads plus packed writes/reads

Divergence note:

- If row nnz is uniform, divergence is minimal.
- If row lengths skew, one-thread-per-row can underutilize long rows. That is a row-skew issue, not a reason to abandon the backend.

### 11.3 What Not To Optimize Prematurely

- Do not replace this with a generic library path just because it feels cleaner.
- The current work is not a dense Tensor Core problem.
- The best next optimizations, if needed, are:
  - row binning by nnz class
  - wider vectorized paths for more bit widths when alignment allows
  - persistent block scheduling if row skew becomes severe

## 12. Priority Queue

### 12.1 Highest-Value Changes

1. Preprocess:
   - cache cuSPARSE descriptors in `device_workspace`
   - remove the `active_rows` D2H sync boundary
   - fuse NCCL reductions into one contiguous buffer
2. Models:
   - stop converting sparse CSR to COO every forward
   - remove or tile away `to_dense()` in dense-reduce loss
   - keep sparse layout stable across the model boundary
3. Forward neighbors:
   - persistent per-device query workspaces
   - no per-block centroid rebuild/upload
   - less host merging
4. Ingest:
   - overlap parse, convert, and store with double buffering

### 12.2 Medium-Value Changes

1. Trajectory:
   - keep candidate stage on device only if trajectory build becomes an online path
2. Quantized backend:
   - bin row widths if real row skew appears in benchmarks
3. Torch bindings:
   - preserve current explicit-copy contract, but keep it out of hot loops

### 12.3 Changes I Would Not Prioritize First

1. Compiler-flag churn beyond the current `sm_70` setup
2. Replacing clear host-side ingest parsing with harder-to-maintain GPU parsing
3. Micro-tuning divergence in preprocess before reducing full-memory passes
4. Rewriting the quantized backend into a dense-library abstraction

## 13. Subsystem Summary

### 13.1 Ingest

- library-backed or custom-kernel: mixed; parse is CPU, COO->CSR is library-backed/fallback custom
- dominant limiter: CPU parse + PCIe
- divergence as bottleneck: no
- critical intermediates should live in: pinned host slabs and reused device slabs

### 13.2 Preprocess

- library-backed or custom-kernel: mixed; custom row kernels plus cuSPARSE/CUB
- dominant limiter: HBM traffic + dispatch count
- divergence as bottleneck: secondary
- critical intermediates should live in: device global memory until pipeline end

### 13.3 Models

- library-backed or custom-kernel: mostly Torch/library-backed with custom CUDA loss ops
- dominant limiter: sparse layout conversion and dense materialization
- divergence as bottleneck: no
- critical intermediates should live in: one stable sparse layout on the target device

### 13.4 Forward Neighbors

- library-backed or custom-kernel: custom-kernel search with host orchestration
- dominant limiter: repeated staging, allocation, and host merge overhead
- divergence as bottleneck: secondary
- critical intermediates should live in: persistent device query/search workspaces

### 13.5 Trajectory

- library-backed or custom-kernel: one custom GPU scoring stage plus CPU graph assembly
- dominant limiter: PCIe for the scoring stage, then CPU memory traffic
- divergence as bottleneck: no
- critical intermediates should live in: device only if the whole downstream pipeline moves there

### 13.6 Quantized Backend

- library-backed or custom-kernel: custom-kernel
- dominant limiter: HBM traffic and row-skew sensitivity
- divergence as bottleneck: workload dependent, usually not first-order for dense-row reconstructions
- critical intermediates should live in: packed device buffers, not host scaffolding

## 14. Final Read

The repo already has the right strategic bias for Volta: explicit workspaces, explicit sparse layouts, custom kernels where library paths do not fit, and a refusal to hide expensive boundaries. The biggest remaining performance problems are not “missing CUDA” in the abstract. They are repeated format conversion, repeated device setup work, repeated host synchronization boundaries, and a few places where a sparse workload is being forced through dense or host-centric surfaces.

This should not be misread as an argument for abstracting the system upward. In this repository, the low-level building blocks are the product. They should become more reusable and better-factored where needed, but they should not be abstracted so aggressively that layout control, residency control, and cost visibility disappear.

If only one principle is carried forward from this document, it should be this:

- keep data in one useful layout
- keep it resident on device
- pay setup once
- make every full-memory pass justify itself

## 15. Inline Comment Coverage

### 15.1 Commented Active Surfaces

- `src/models/*`, `src/quantized/*`, `src/ingest/*`, `src/trajectory/*`, `src/torch/bindings.hh`, and the active non-`_TODO` CellShard residency/layout surfaces now carry concise inline comments.
- The comments are intentionally short and mostly attached to:
  - host versus device materialization boundaries
  - alloc/free or realloc/copy points
  - fixed-overhead call sites such as launches, descriptor setup, fetch/drop, and tensor export
  - places where a path is linear metadata work versus a real hot-path data pass

### 15.2 What The Inline Comments Are For

- They are meant to speed up code review and profiling triage on this V100-targeted codebase.
- They are not intended to replace measurement. Estimated overhead notes inside the code should still be treated as structural guidance until Nsight or benchmark data confirms them.
