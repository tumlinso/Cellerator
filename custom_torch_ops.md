# Custom Torch Ops Registry

## `dense_reduce_pair_losses`

- Purpose: fuse `dense_reduce` pairwise local-smoothness and far-separation loss evaluation for CUDA training.
- Owner: `src/models/dense_reduce/`
- Boundary:
  C++ caller in [`src/models/dense_reduce/dR_model.hh`](/home/tumlinson/Software/Repos/Cellerator/src/models/dense_reduce/dR_model.hh:202)
  CUDA/autograd backend in [`src/models/custom_ops/model_ops.cu`](/home/tumlinson/Software/Repos/Cellerator/src/models/custom_ops/model_ops.cu:1)
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
- Status: implemented

## `developmental_stage_bucket_losses`

- Purpose: keep `developmental_time` bucket ranking/anchor/spread loss on device and avoid CPU bucket discovery plus repeated masked selects.
- Owner: `src/models/developmental_time/`
- Boundary:
  C++ caller in [`src/models/developmental_time/dT_model.hh`](/home/tumlinson/Software/Repos/Cellerator/src/models/developmental_time/dT_model.hh:234)
  CUDA/autograd backend in [`src/models/custom_ops/model_ops.cu`](/home/tumlinson/Software/Repos/Cellerator/src/models/custom_ops/model_ops.cu:1)
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
  CUDA backend in [`src/models/custom_ops/model_ops.cu`](/home/tumlinson/Software/Repos/Cellerator/src/models/custom_ops/model_ops.cu:1)
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
