#pragma once

#include <Cellerator/ops/matrix/feature_major_tiles.cuh>

namespace cellerator::ops::matrix::detail {

// The legacy feature-major adapter accepts widths 1 through 16 and can
// describe narrower row tiles. Its dense rows are already in execution order.
cudaError_t enqueue_feature_major_tiles_forward(tensor::FeatureMajorTilesView<const __half> x,
    const uint32_t *logical_rows, const float *w, uint32_t dense_width, float *y,
    cudaStream_t stream) noexcept;

cudaError_t enqueue_feature_major_values(const __half *source, const uint32_t *packed_to_source,
    __half *packed, uint32_t count, cudaStream_t stream) noexcept;

} // namespace cellerator::ops::matrix::detail
