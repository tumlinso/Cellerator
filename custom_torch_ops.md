# Custom Torch Ops Registry

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
  no Torch autograd boundary
- Assumptions:
  Volta `sm_70`
  Blocked-ELL and sliced-ELL are the steady-state sparse layouts
  current runtime slice is single-GPU only even though the interface is NCCL-shaped for later scale-out
- Status: implemented as the replacement path; no Torch op required

## `dense_reduce_pair_losses`

- Purpose: fuse `dense_reduce` pairwise local-smoothness and far-separation loss evaluation for CUDA training.
- Owner: `src/models/dense_reduce/`
- Boundary:
  C++ caller in [`src/models/dense_reduce/dR_model.hh`](/home/tumlinson/Software/Repos/Cellerator/src/models/dense_reduce/dR_model.hh:202)
  CUDA/autograd backend in [`src/compute/model_ops/model_ops.cu`](/home/tumlinson/Software/Repos/Cellerator/src/compute/model_ops/model_ops.cu:1)
- Inputs:
  contiguous CUDA `int64` `pair_rows`, `pair_cols`
  contiguous CUDA `float32` `latent_unit`, `developmental_time`
  scalar windows and margin
- Outputs:
  CUDA scalar `local_loss`, CUDA scalar `far_loss`
- Backend: custom CUDA kernels with C++ autograd wrapper
- Backward:
  custom backward for `latent_unit`
  no gradients for pair indices or time
- Assumptions:
  `latent_unit` is row-major `[batch, latent_dim]`
  Volta `sm_70`
- Status: legacy Torch-only prototype op; superseded for new identity reduction by `state_reduce_native_runtime`

## `developmental_stage_bucket_losses`

- Purpose: keep `developmental_time` bucket ranking/anchor/spread loss on device and avoid CPU bucket discovery plus repeated masked selects.
- Owner: `src/models/developmental_time/`
- Boundary:
  C++ caller in [`src/models/developmental_time/dT_model.hh`](/home/tumlinson/Software/Repos/Cellerator/src/models/developmental_time/dT_model.hh:234)
  CUDA/autograd backend in [`src/compute/model_ops/model_ops.cu`](/home/tumlinson/Software/Repos/Cellerator/src/compute/model_ops/model_ops.cu:1)
- Inputs:
  contiguous CUDA `float32` `stage`
  contiguous CUDA `int64` `day_buckets`
  scalar margin/std config
- Outputs:
  CUDA scalar `ranking`, `anchor`, `spread`
- Backend: custom CUDA kernels with C++ autograd wrapper
- Backward:
  custom backward for `stage`
  no gradients for `day_buckets`
- Assumptions:
  bucket ids are non-negative
  bucket count is small enough that one-block bucket finalization is acceptable
  Volta `sm_70`
- Status: implemented

## `weighted_future_target`

- Purpose: build quantizer forward-neighbor dense targets on GPU when the reference feature table is already resident there.
- Owner: `src/models/quantize/`
- Boundary:
  C++ caller in [`src/models/quantize/quantize.hh`](/home/tumlinson/Software/Repos/Cellerator/src/models/quantize/quantize.hh:342)
  CUDA backend in [`src/compute/model_ops/model_ops.cu`](/home/tumlinson/Software/Repos/Cellerator/src/compute/model_ops/model_ops.cu:1)
- Inputs:
  contiguous CUDA `float32` `reference_dense`
  contiguous CUDA `int64` `neighbor_row_indices`
  contiguous CUDA `float32` `neighbor_weights`
- Outputs:
  CUDA `float32` dense target matrix
- Backend: custom CUDA kernel
- Backward:
  not used; target is treated as supervision, not a differentiable input
- Assumptions:
  invalid neighbor rows are encoded as `-1`
  Volta `sm_70`
- Status: implemented

## `compute_autograd_runtime_v1`

- Purpose: provide a pointer-first sparse building-block library in `src/compute/autograd/` for model-specific fused CUDA kernels on V100.
- Owner: `src/compute/autograd/`
- Boundary:
  public runtime surface in [`src/compute/autograd/autograd.hh`](/home/tumlinson/Software/Repos/Cellerator/src/compute/autograd/autograd.hh:1)
  base single-GPU kernels in [`src/compute/autograd/kernels/base_sparse.cu`](/home/tumlinson/Software/Repos/Cellerator/src/compute/autograd/kernels/base_sparse.cu:1)
  distributed launch and leader-merge helpers in [`src/compute/autograd/kernels/dist_sparse.cu`](/home/tumlinson/Software/Repos/Cellerator/src/compute/autograd/kernels/dist_sparse.cu:1)
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

- Purpose: move quantizer reconstruction and range gradients for sparse CUDA CSR batches into fused kernels under `src/compute/autograd` instead of dense libtorch math.
- Owner: `src/models/quantize/`
- Boundary:
  model-facing wrapper in [`src/models/quantize/quantize.hh`](/home/tumlinson/Software/Repos/Cellerator/src/models/quantize/quantize.hh:1)
  fused sparse kernels in [`src/compute/autograd/kernels/base_sparse.cu`](/home/tumlinson/Software/Repos/Cellerator/src/compute/autograd/kernels/base_sparse.cu:1)
  low-level runtime surface in [`src/compute/autograd/autograd.hh`](/home/tumlinson/Software/Repos/Cellerator/src/compute/autograd/autograd.hh:1)
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
