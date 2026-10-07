# Custom Torch Ops Registry

## `resident_cuda_torch_interop`

- Purpose: provide an optional thin Torch view over Cellerator's resident CUDA
  streams, buffers, prepared CSR relation, and FP32 numeric operations.
- Owning model or component: `cellerator.cuda` owns native resident state and
  numeric execution; `cellerator.torch.cuda` is only a framework adapter.
- Status: implemented; CUDA qualification is pending.
- Python API boundary: `cellerator.torch.cuda` validates and borrows contiguous
  CUDA FP32 tensors, records borrowed storage on the active Torch stream,
  delegates `multiply_into`, `axpby_into`, and prepared CSR application to the
  native API, and exposes DLPack aliases for native-owned buffers.
- C++ binding boundary: Torch-free `cellerator.cuda` bindings over the existing
  resident Cellerator APIs. This entry adds no libtorch binding or independent
  topology/value owner.
- CUDA or library backend: existing Cellerator native FP32 numeric kernels and
  prepared CSR implementation; no Torch arithmetic or new math backend is
  introduced here.
- Input contract: CUDA float32 contiguous rank-one or rank-two tensors, all on
  one device, with `requires_grad=False`; the supplied or implicit stream must
  be PyTorch's current stream. CSR topology is contiguous NumPy `uint64` data;
  weights are a borrowed FP32 CUDA vector. For direct `Buffer.borrow`,
  `capacity_bytes` is a trusted caller precondition that must describe the
  accessible allocation extent.
- Output contract: caller-owned CUDA FP32 output tensors, or a zero-copy Torch
  alias of a native-owned buffer through DLPack. Prepared CSR output has shape
  `[destination_count, feature_width]`.
- Dtype, layout, and device assumptions: FP32 native storage and round-to-nearest
  arithmetic. The existing prepared-relation candidate permits FMA and
  reassociation under its native numeric contract, and this adapter adds no
  precision mode. Tensors are contiguous on one CUDA device. `record_stream`
  protects Torch allocator lifetime. Native arithmetic and prepared apply do
  not retain per-call borrowed views after enqueue, so direct callers preserve
  those views and owners through completion. A prepared owner retains the
  weight buffer published as its current value generation. Callers establish
  producer readiness and mutation ordering themselves.
- Backward or autograd notes: forward-only; tensors requiring gradients are
  rejected. No autograd implementation is provided.
- Distributed implications: single-device stream and buffer interop only; no
  multi-GPU behavior is claimed.
- Code location: `python/cellerator/torch/cuda.py` and the native
  `cellerator.cuda` bindings.
- Validation notes: focused CUDA adapter tests, including a producer-to-consumer
  DLPack stream handoff, are provided in `tests/bindings/python/test_resident_torch.py`;
  execution and CUDA qualification remain pending the controller-managed run.

## `state_reduce_native_runtime`

- Purpose: replace the old Torch-bound dense reducer with a native CUDA cell-identity reducer that trains on Blocked-ELL or sliced-ELL batches without libtorch or Torch custom ops.
- Owner: `src/models/state_reduce/`
- Boundary:
  public model surface in [`src/models/state_reduce/stateReduce.hh`](/home/tumlinson/Software/Repos/Cellerator/src/models/state_reduce/stateReduce.hh:1)
  native CUDA runtime in [`src/models/state_reduce/state_reduce_cuda.cu`](/home/tumlinson/Software/Repos/Cellerator/src/models/state_reduce/state_reduce_cuda.cu:1)
- Inputs:
  native Blocked-ELL or sliced-ELL batch descriptors
  optional forward-neighbor graph edges and weights
  explicit model, loss, optimizer, and distributed configs
- Outputs:
  CUDA-resident embeddings
  scalar reconstruction and graph-consistency losses
  updated native parameter buffers after training steps
- Backend: native CUDA runtime with a WMMA-capable path plus a cuSPARSE-heavy encoder path
- Backward:
  handwritten CUDA gradients for reconstruction, graph smoothing, encoder projection, decoder factors, and AdamW updates
  no Torch differentiation boundary
- Assumptions:
  Volta `sm_70`
  Blocked-ELL and sliced-ELL are the steady-state sparse layouts
  current runtime slice is single-GPU only even though the interface is NCCL-shaped for later scale-out
- Status: implemented as the replacement path; no Torch op required

## `dense_reduce_pair_losses`

- Purpose: provide reusable FP32 CUDA pairwise local-smoothness and far-separation loss evaluation with first-order gradients.
- Owner: `src/compute/operation/model_ops/`
- Boundary:
  Torch adapter in [`bindings/torch/src/model_ops.cu`](bindings/torch/src/model_ops.cu)
  Torch-free pointer API in [`include/Cellerator/compute/operation/model_ops/model_ops.hh`](include/Cellerator/compute/operation/model_ops/model_ops.hh)
  CUDA backend in [`src/compute/operation/model_ops/model_ops.cu`](src/compute/operation/model_ops/model_ops.cu)
- Inputs:
  contiguous CUDA `int64` `pair_rows`, `pair_cols`
  contiguous CUDA `float32` `latent_unit`, `developmental_time`
  scalar windows and margin
- Outputs:
  CUDA scalar `local_loss`, CUDA scalar `far_loss`
- Backend: custom CUDA kernels; the optional Torch adapter owns tensor allocation and autograd
- Backward:
  native custom backward for `latent_unit`; each nonempty loss is normalized by its contributing pair count
  this corrects the legacy adapter, which divided the forward sums but omitted the corresponding count in its backward scale; forward values are unchanged
  no gradients for pair indices or time
- Assumptions:
  `latent_unit` is row-major `[batch, latent_dim]`
  inputs share a CUDA device; pair indices must be within the latent row range
  FP32 accumulation and output; no Tensor Core or broader numerical envelope is claimed
- Status: implemented (CUDA correctness pending)

## `developmental_stage_bucket_losses`

- Purpose: provide reusable FP32 CUDA bucket ranking, anchor, and spread losses with first-order gradients.
- Owner: `src/compute/operation/model_ops/`
- Boundary:
  Torch adapter in [`bindings/torch/src/model_ops.cu`](bindings/torch/src/model_ops.cu)
  Torch-free pointer API in [`include/Cellerator/compute/operation/model_ops/model_ops.hh`](include/Cellerator/compute/operation/model_ops/model_ops.hh)
  CUDA backend in [`src/compute/operation/model_ops/model_ops.cu`](src/compute/operation/model_ops/model_ops.cu)
- Inputs:
  contiguous CUDA `float32` `stage`
  contiguous CUDA `int64` `day_buckets`
  scalar margin/std config
- Outputs:
  CUDA scalar `ranking`, `anchor`, `spread`
- Backend: custom CUDA kernels; the optional Torch adapter owns tensor allocation and autograd
- Backward:
  custom backward for `stage`
  no gradients for `day_buckets`
- Assumptions:
  inputs share a CUDA device; bucket ids are non-negative and are sized before launch
  bucket labels fit signed 32-bit indexing; scratch scales with inferred `max(label) + 1`, including empty groups
  bucket finalization is a single-thread pass over bucket statistics and all-pairs ranking is O(bucket_count^2); this favors the existing small-bucket workload
  FP32 accumulation and output; no Tensor Core or broader numerical envelope is claimed
- Status: implemented (CUDA correctness pending)

## `weighted_future_target`

- Purpose: build quantizer forward-neighbor dense targets on GPU when the reference feature table is already resident there.
- Owner: `src/compute/operation/model_ops/` (numerical gather); quantizer remains a consumer
- Boundary:
  Torch adapter in [`bindings/torch/src/model_ops.cu`](bindings/torch/src/model_ops.cu)
  Torch-free pointer API in [`include/Cellerator/compute/operation/model_ops/model_ops.hh`](include/Cellerator/compute/operation/model_ops/model_ops.hh)
  CUDA backend in [`src/compute/operation/model_ops/model_ops.cu`](src/compute/operation/model_ops/model_ops.cu)
- Inputs:
  contiguous CUDA `float32` `reference_dense`
  contiguous CUDA `int64` `neighbor_row_indices`
  contiguous CUDA `float32` `neighbor_weights`
- Outputs:
  CUDA `float32` dense target matrix
- Backend: custom CUDA kernel; the optional Torch adapter owns tensor allocation and current-stream selection
- Backward:
  not used; target is treated as supervision, not a differentiable input
- Assumptions:
  all negative neighbor indices are skipped; nonnegative indices must be within the reference row range
  inputs share a CUDA device; FP32 accumulation and output
  no Tensor Core or broader numerical envelope is claimed
- Status: implemented (CUDA correctness pending)

## `sparse_ops_runtime_v1`

- Purpose: provide a pointer-first sparse building-block library in `src/compute/sparse/ops/` for model-specific fused CUDA kernels on V100.
- Owner: `src/compute/sparse/ops/`
- Boundary:
  public runtime surface in [`src/compute/runtime/runtime.hh`](/home/tumlinson/Software/Repos/Cellerator/src/compute/runtime/runtime.hh:1)
  base single-GPU kernels in [`src/compute/sparse/ops/kernels/base_sparse.cu`](/home/tumlinson/Software/Repos/Cellerator/src/compute/sparse/ops/kernels/base_sparse.cu:1)
  distributed launch and leader-merge helpers in [`src/compute/sparse/ops/kernels/dist_sparse.cu`](/home/tumlinson/Software/Repos/Cellerator/src/compute/sparse/ops/kernels/dist_sparse.cu:1)
- Inputs:
  raw CSR metadata pointers, raw dense pointers, explicit sizes, and explicit stream / scratch state
- Outputs:
  raw dense outputs, sparse-value gradients, dense-input gradients, and explicit distributed reductions to leader devices
- Backend: custom CUDA building blocks plus CUB-backed value reduction and optional cuSPARSE float32 baselines
- Backward:
  custom backward building blocks for sparse row scaling and sparse projection paths
- Assumptions:
  Volta `sm_70`
  pair-local distributed execution uses `0 <-> 2` and `1 <-> 3`
  4-GPU reduction is hierarchical through pair leaders
  FP16 storage with FP32 accumulation is the primary path
  Blocked-ELL is the native sparse execution layout and CSR metadata is the secondary fallback layout
- Status: implemented pointer-first base and distributed reference copies

## `quantize_sparse_feature_affine`

- Purpose: move quantizer reconstruction and range gradients for sparse CUDA CSR batches into fused kernels under `src/compute/sparse/ops` instead of dense libtorch math.
- Owner: `src/models/quantize/`
- Boundary:
  model-facing wrapper in [`src/models/quantize/quantize.hh`](/home/tumlinson/Software/Repos/Cellerator/src/models/quantize/quantize.hh:1)
  fused sparse kernels in [`src/compute/sparse/ops/kernels/base_sparse.cu`](/home/tumlinson/Software/Repos/Cellerator/src/compute/sparse/ops/kernels/base_sparse.cu:1)
  low-level runtime surface in [`src/compute/runtime/runtime.hh`](/home/tumlinson/Software/Repos/Cellerator/src/compute/runtime/runtime.hh:1)
- Inputs:
  CUDA sparse CSR batch with cell-major rows and gene-major columns
  contiguous CUDA `float32` `log_scale` and `offset`
  scalar bit width, scale floor, and loss weights
- Outputs:
  CUDA scalar reconstruction loss
  CUDA scalar range loss
  CUDA `float32` gradients for `log_scale` and `offset`
- Backend: custom CUDA fused zero-baseline plus sparse-correction kernels for CSR and Blocked-ELL layouts
- Backward:
  custom backward for quantizer parameters
  no gradients for sparse batch metadata or sparse feature values in the model-facing path
- Assumptions:
  Volta `sm_70`
  sparse batches are implicit-zero expression matrices
  offset anchor in the sparse CUDA path treats the dense floor as zero
- Status: implemented for sparse CUDA reconstruction/range training path
