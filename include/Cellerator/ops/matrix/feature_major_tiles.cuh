#pragma once

#include <Cellerator/layout/projection/feature_major_tiles.hh>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace cellerator::ops::matrix {

inline constexpr uint32_t feature_major_tiles_dense_width = 16;

// Use a prepared structure and matching index on the current device. Keep
// borrowed arrays live through stream completion. Writable arrays must be
// disjoint from each other and from their read-only inputs.
// Dense arrays are logical-order, contiguous row-major FP32, width 16. X is
// in tile order. Calls enqueue overwrite operations without host waits.
// Forward can use FP32 atomic accumulation between record ranges.
// The floating-point summation order can vary between runs.
cudaError_t feature_major_tiles_spmm(tensor::FeatureMajorTilesView<const __half> x,
    layout::projection::FeatureMajorTilesIndexView index, const float *w, float *y,
    cudaStream_t stream) noexcept;

// Heavy features use one 16-float partial per prepared range. The caller
// supplies this scratch with capacity measured in floats.
size_t feature_major_tiles_transpose_partial_count(
    layout::projection::FeatureMajorTilesIndexView index) noexcept;

cudaError_t feature_major_tiles_spmm_transpose(tensor::FeatureMajorTilesView<const __half> x,
    layout::projection::FeatureMajorTilesIndexView index, const float *g, float *dw,
    float *partials, size_t partial_count, cudaStream_t stream) noexcept;

// Each stored coordinate receives the inner product of its row of G and
// feature of W, even when X's execution value at that coordinate is zero.
// dx contains nnz gradients in packed value order.
cudaError_t feature_major_tiles_sampled_gradient(tensor::FeatureMajorTileStructureView structure,
    layout::projection::FeatureMajorTilesIndexView index, const float *g, const float *w,
    float *dx, cudaStream_t stream) noexcept;

// These value operations are asynchronous. The caller orders writes after
// all previous readers and orders new readers after the writes. A refresh
// imports FP16 data only. To replace training values, initialize both the
// packed FP32 master and its FP16 execution plane from the same CSR input.
cudaError_t feature_major_tiles_refresh(tensor::FeatureMajorTilesView<__half> x,
    layout::projection::FeatureMajorTilesIndexView index, const __half *csr_values,
    cudaStream_t stream) noexcept;

cudaError_t feature_major_tiles_initialize_master(tensor::FeatureMajorTilesView<__half> x,
    layout::projection::FeatureMajorTilesIndexView index, const float *csr_values,
    float *packed_master, cudaStream_t stream) noexcept;

cudaError_t feature_major_tiles_cast_master(tensor::FeatureMajorTilesView<__half> x,
    const float *packed_master, cudaStream_t stream) noexcept;

cudaError_t feature_major_tiles_sparse_sgd(tensor::FeatureMajorTilesView<__half> x,
    float *packed_master, const float *dx, float learning_rate, cudaStream_t stream) noexcept;

cudaError_t feature_major_tiles_export_gradient(tensor::FeatureMajorTileStructureView structure,
    layout::projection::FeatureMajorTilesIndexView index, const float *dx,
    float *csr_gradient, cudaStream_t stream) noexcept;

} // namespace cellerator::ops::matrix
