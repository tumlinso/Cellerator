#pragma once

#include <Cellerator/memory/view.hh>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <stddef.h>

namespace cellerator::ops::matrix {

struct PanelMatmulShape {
    size_t groups = 0, panels = 0, output_tiles = 0;
};

class PanelMatmulPlan {
public:
    PanelMatmulPlan() = default;

private:
    friend cudaError_t prepare_panel_matmul(int, PanelMatmulShape, PanelMatmulPlan &) noexcept;
    friend cudaError_t enqueue_panel_matmul(
        const PanelMatmulPlan &, memory::const_array_view<__half>,
        memory::const_array_view<__half>, memory::matrix_view<float>, cudaStream_t) noexcept;

    PanelMatmulShape shape_{};
    unsigned output_bundles_ = 0;
    int device_ = -1;
};

// Prepare whole 16-cell groups and 16-output tiles on one Volta device.
// Panels contain four ordered 16-feature blocks. Failure clears the plan.
cudaError_t prepare_panel_matmul(int device, PanelMatmulShape shape,
                                PanelMatmulPlan &plan) noexcept;

// Y[output, cell] = W[output, feature] * X[feature, cell]. FP16 X is stored
// [group][panel][cell][feature], with a row stride of 64. FP16 W is stored
// [output tile][panel][input tile][output][feature], with a row stride of 16.
// Within a tile the cell, output and feature indices each range over 16 slots.
// The same panel and input tile order must be used for each group and for W.
// Y is row-major FP32 with exact logical extents, a 32-byte-aligned base and a
// row stride divisible by four elements. Zero panels overwrite Y with zeros.
// The prepared device must be current and own the stream and live views.
// Keep the disjoint device buffers live through their last stream consumer.
cudaError_t enqueue_panel_matmul(
    const PanelMatmulPlan &plan, memory::const_array_view<__half> weights,
    memory::const_array_view<__half> features, memory::matrix_view<float> output,
    cudaStream_t stream) noexcept;

} // namespace cellerator::ops::matrix
