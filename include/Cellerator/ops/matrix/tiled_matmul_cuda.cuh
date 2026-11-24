#pragma once

#include <Cellerator/memory/Tile.hh>
#include <Cellerator/memory/view.hh>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace cellerator::ops::matrix {

struct TiledMatmulShape {
    size_t output_tiles = 0, cell_tiles = 0, feature_tiles = 0;
};

class TiledMatmulPlan {
public:
    TiledMatmulPlan() = default;

private:
    friend cudaError_t prepare_tiled_matmul(int, TiledMatmulShape, TiledMatmulPlan &) noexcept;
    friend cudaError_t enqueue_tiled_matmul(
        const TiledMatmulPlan &, memory::const_array_view<memory::Tile<__half>>,
        memory::const_array_view<memory::Tile<__half>>, memory::matrix_view<float>,
        cudaStream_t) noexcept;

    TiledMatmulShape shape_{};
    int device_ordinal_ = -1;
};

// Prepare positive whole-tile extents on a Volta device. Failure clears plan.
cudaError_t prepare_tiled_matmul(int device_ordinal, TiledMatmulShape shape,
                                 TiledMatmulPlan &plan) noexcept;

// Compute Y[output,cell] = W[output,feature] * X[feature,cell]. The W tile
// array is ordered [output tile][feature tile], and X [feature tile][cell tile].
// All tiles are row-major FP16. Y is row-major FP32 with exact logical extents,
// a 32-byte-aligned base, and a row stride divisible by four elements.
// The prepared device must be current and own the stream and all three views.
// Keep the disjoint device buffers live through completion on that stream.
cudaError_t enqueue_tiled_matmul(
    const TiledMatmulPlan &plan, memory::const_array_view<memory::Tile<__half>> weights,
    memory::const_array_view<memory::Tile<__half>> features, memory::matrix_view<float> output,
    cudaStream_t stream) noexcept;

} // namespace cellerator::ops::matrix
